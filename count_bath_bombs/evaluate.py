"""Evaluate the rule predictions against the manual labels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _coerce_manual(manual: pd.DataFrame) -> pd.DataFrame:
    def _as_bool(x):
        if pd.isna(x) or x == "":
            return None
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False
        return None

    manual = manual.copy()
    manual["is_pure_bath_bomb_manual"] = manual["is_pure_bath_bomb_manual"].map(_as_bool)
    manual["n_bomb_balls_manual"] = pd.to_numeric(manual["n_bomb_balls_manual"], errors="coerce")
    return manual


def _load(manual: str | Path | pd.DataFrame) -> pd.DataFrame:
    if not isinstance(manual, pd.DataFrame):
        manual = pd.read_csv(manual)
    required = {"asin", "is_pure_bath_bomb_manual", "n_bomb_balls_manual"}
    missing = required - set(manual.columns)
    if missing:
        raise ValueError(f"Manual-label file missing columns: {missing}")
    return _coerce_manual(manual)


def evaluate_against_manual(pred: pd.DataFrame, manual: str | Path | pd.DataFrame) -> dict:
    """Headline metrics: purity precision/recall + count exact/MAE."""
    m = pred.merge(_load(manual), on="asin", how="inner", suffixes=("", "_m"))
    if m.empty:
        return {"n_overlap": 0}

    labeled = m["is_pure_bath_bomb_manual"].notna()
    sub = m.loc[labeled]
    tp = ((sub["is_pure_bath_bomb"] == True) & (sub["is_pure_bath_bomb_manual"] == True)).sum()  # noqa: E712
    fp = ((sub["is_pure_bath_bomb"] == True) & (sub["is_pure_bath_bomb_manual"] == False)).sum()  # noqa: E712
    fn = ((sub["is_pure_bath_bomb"] == False) & (sub["is_pure_bath_bomb_manual"] == True)).sum()  # noqa: E712
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    pure = m[m["is_pure_bath_bomb_manual"] == True]  # noqa: E712
    both = pure[pure["n_bomb_balls_manual"].notna() & pure["n_bomb_balls"].notna()]
    exact = (both["n_bomb_balls"] == both["n_bomb_balls_manual"]).mean() if len(both) else None
    mae = (both["n_bomb_balls"] - both["n_bomb_balls_manual"]).abs().mean() if len(both) else None

    return {
        "n_overlap": int(len(m)),
        "n_purity_labeled": int(labeled.sum()),
        "purity_precision": precision,
        "purity_recall": recall,
        "n_count_eval": int(len(both)),
        "count_exact_match": exact,
        "count_mae": float(mae) if mae is not None and not pd.isna(mae) else None,
    }


def _purity_scores(sub: pd.DataFrame) -> dict:
    """Precision/recall/F1/accuracy for purity on rows with a purity label."""
    s = sub[sub["is_pure_bath_bomb_manual"].notna()]
    if s.empty:
        return {"n": 0}
    yp = s["is_pure_bath_bomb"] == True   # noqa: E712
    yg = s["is_pure_bath_bomb_manual"] == True  # noqa: E712
    tp = int((yp & yg).sum())
    fp = int((yp & ~yg).sum())
    fn = int((~yp & yg).sum())
    tn = int((~yp & ~yg).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n": int(len(s)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": (tp + tn) / len(s),
    }


def _count_scores(sub: pd.DataFrame) -> dict:
    """Exact / within-1 / MAE on rows where the label is pure and both counts exist."""
    both = sub[
        (sub["is_pure_bath_bomb_manual"] == True)  # noqa: E712
        & sub["n_bomb_balls_manual"].notna()
        & sub["n_bomb_balls"].notna()
    ]
    if both.empty:
        return {"n": 0}
    err = (both["n_bomb_balls"] - both["n_bomb_balls_manual"]).abs()
    return {
        "n": int(len(both)),
        "exact": float((err == 0).mean()),
        "within_1": float((err <= 1).mean()),
        "mae": float(err.mean()),
    }


def evaluate_report(pred: pd.DataFrame, manual: str | Path | pd.DataFrame) -> dict:
    """Full report: purity confusion/F1, count exact/within-1/MAE, breakdowns by
    model confidence (calibration) and by class label, plus error rows."""
    m = pred.merge(_load(manual), on="asin", how="inner", suffixes=("", "_m"))
    if m.empty:
        return {"n_overlap": 0}

    report: dict = {
        "n_overlap": int(len(m)),
        "purity": _purity_scores(m),
        "count": _count_scores(m),
        "by_confidence": {},
        "by_class": {},
        "errors": [],
    }

    if "count_confidence" in m.columns:
        for c in ["high", "medium", "low"]:
            block = m[m["count_confidence"] == c]
            if len(block):
                report["by_confidence"][c] = {"purity": _purity_scores(block), "count": _count_scores(block)}

    if "class_label" in m.columns:
        for s in sorted(m["class_label"].dropna().unique().tolist()):
            block = m[m["class_label"] == s]
            report["by_class"][str(s)] = {"purity": _purity_scores(block), "count": _count_scores(block)}

    for _, r in m.iterrows():
        pg = r["is_pure_bath_bomb_manual"]
        if pg is not None and (r["is_pure_bath_bomb"] == True) != (pg == True):  # noqa: E712
            report["errors"].append({
                "asin": r["asin"], "type": "purity",
                "pred": bool(r["is_pure_bath_bomb"] == True), "manual": bool(pg == True),  # noqa: E712
            })
        elif (pg == True and pd.notna(r["n_bomb_balls_manual"]) and pd.notna(r["n_bomb_balls"])  # noqa: E712
              and int(r["n_bomb_balls"]) != int(r["n_bomb_balls_manual"])):
            report["errors"].append({
                "asin": r["asin"], "type": "count",
                "pred": int(r["n_bomb_balls"]), "manual": int(r["n_bomb_balls_manual"]),
            })
    return report
