from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GOLD_COLUMNS = [
    "asin",
    "stratum",
    "is_pure_bath_bomb_gold",
    "n_bomb_balls_gold",
    "exclude_reason_gold",
    "notes",
    "annotator",
]

# The two independent labelling dimensions the review UI filters on.
CLASS_LABELS = ["pure", "craft_kit", "bundle", "substitute", "toiletry", "unclassified"]
COUNT_LABELS = ["multi_pack", "single", "pack_as_one", "extreme_count", "count_unable", "n/a"]


def class_label_for_row(row: pd.Series) -> str:
    """Classification outcome: pure / <exclude_reason> / unclassified."""
    is_pure = row.get("is_pure_bath_bomb")
    if is_pure is True:
        return "pure"
    reason = row.get("exclude_reason")
    if is_pure is False and isinstance(reason, str) and reason:
        return reason
    return "unclassified"


def count_label_for_row(row: pd.Series) -> str:
    """Counting bucket — only meaningful for pure items (else 'n/a')."""
    if row.get("is_pure_bath_bomb") is not True:
        return "n/a"
    if bool(row.get("count_unable")):
        return "count_unable"
    n = row.get("n_bomb_balls")
    if bool(row.get("seller_counts_pack_as_one")):
        return "pack_as_one"
    if n is not None and not pd.isna(n) and float(n) >= 50:
        return "extreme_count"
    if n is not None and not pd.isna(n) and float(n) > 1:
        return "multi_pack"
    if n == 1 or n == 1.0:
        return "single"
    return "count_unable"


def build_labeling_sample(
    df: pd.DataFrame,
    sample_size: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    work = df.copy()
    work["class_label"] = [class_label_for_row(r) for _, r in work.iterrows()]
    work["count_label"] = [count_label_for_row(r) for _, r in work.iterrows()]
    # `stratum` kept for backwards-compat display = classification bucket.
    work["stratum"] = work["class_label"]

    # Stratify across classification labels for balanced coverage.
    strata = work["class_label"].unique().tolist()
    per = max(1, sample_size // max(len(strata), 1))
    rng = np.random.default_rng(seed)
    parts = []
    for s in strata:
        block = work[work["class_label"] == s]
        n = min(len(block), per)
        if n == 0:
            continue
        idx = rng.choice(block.index.to_numpy(), size=n, replace=False)
        parts.append(block.loc[idx])

    sample = pd.concat(parts).drop_duplicates(subset=["asin"])
    # Top up if short
    if len(sample) < sample_size:
        remain = work[~work["asin"].isin(sample["asin"])]
        need = min(sample_size - len(sample), len(remain))
        if need:
            idx = rng.choice(remain.index.to_numpy(), size=need, replace=False)
            sample = pd.concat([sample, remain.loc[idx]])

    out = sample[
        [
            "asin",
            "stratum",
            "class_label",
            "count_label",
            "title",
            "number_of_items",
            "size",
            "is_pure_bath_bomb",
            "n_bomb_balls",
            "count_source",
            "exclude_reason",
            "seller_counts_pack_as_one",
            "keepa_main_image_url",
            "keepa_image_count",
        ]
    ].copy()
    out["is_pure_bath_bomb_gold"] = ""
    out["n_bomb_balls_gold"] = ""
    out["exclude_reason_gold"] = ""
    out["notes"] = ""
    out["annotator"] = ""
    return out.reset_index(drop=True)


def evaluate_against_gold(pred: pd.DataFrame, gold: str | Path | pd.DataFrame) -> dict:
    """`gold` may be a path or an already-loaded/filtered DataFrame."""
    if not isinstance(gold, pd.DataFrame):
        gold = pd.read_csv(gold)
    required = {"asin", "is_pure_bath_bomb_gold", "n_bomb_balls_gold"}
    missing = required - set(gold.columns)
    if missing:
        raise ValueError(f"Gold file missing columns: {missing}")

    # Coerce gold purity
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

    gold = gold.copy()
    gold["is_pure_bath_bomb_gold"] = gold["is_pure_bath_bomb_gold"].map(_as_bool)
    gold["n_bomb_balls_gold"] = pd.to_numeric(gold["n_bomb_balls_gold"], errors="coerce")

    m = pred.merge(gold, on="asin", how="inner", suffixes=("", "_g"))
    if m.empty:
        return {"n_overlap": 0}

    purity_labeled = m["is_pure_bath_bomb_gold"].notna()
    sub = m.loc[purity_labeled]
    tp = ((sub["is_pure_bath_bomb"] == True) & (sub["is_pure_bath_bomb_gold"] == True)).sum()  # noqa: E712
    fp = ((sub["is_pure_bath_bomb"] == True) & (sub["is_pure_bath_bomb_gold"] == False)).sum()  # noqa: E712
    fn = ((sub["is_pure_bath_bomb"] == False) & (sub["is_pure_bath_bomb_gold"] == True)).sum()  # noqa: E712
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    pure_gold = m[m["is_pure_bath_bomb_gold"] == True]  # noqa: E712
    both = pure_gold[pure_gold["n_bomb_balls_gold"].notna() & pure_gold["n_bomb_balls"].notna()]
    exact = (both["n_bomb_balls"] == both["n_bomb_balls_gold"]).mean() if len(both) else None
    mae = (both["n_bomb_balls"] - both["n_bomb_balls_gold"]).abs().mean() if len(both) else None

    return {
        "n_overlap": int(len(m)),
        "n_purity_labeled": int(purity_labeled.sum()),
        "purity_precision": precision,
        "purity_recall": recall,
        "n_count_eval": int(len(both)),
        "count_exact_match": exact,
        "count_mae": float(mae) if mae is not None and not pd.isna(mae) else None,
    }


def _coerce_gold(gold: pd.DataFrame) -> pd.DataFrame:
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

    gold = gold.copy()
    gold["is_pure_bath_bomb_gold"] = gold["is_pure_bath_bomb_gold"].map(_as_bool)
    gold["n_bomb_balls_gold"] = pd.to_numeric(gold["n_bomb_balls_gold"], errors="coerce")
    return gold


def _purity_scores(sub: pd.DataFrame) -> dict:
    """Precision/recall/F1/accuracy for purity on rows with a purity gold label."""
    s = sub[sub["is_pure_bath_bomb_gold"].notna()]
    if s.empty:
        return {"n": 0}
    yp = s["is_pure_bath_bomb"] == True   # noqa: E712
    yg = s["is_pure_bath_bomb_gold"] == True  # noqa: E712
    tp = int((yp & yg).sum())
    fp = int((yp & ~yg).sum())
    fn = int((~yp & yg).sum())
    tn = int((~yp & ~yg).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    acc = (tp + tn) / len(s)
    return {
        "n": int(len(s)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
    }


def _count_scores(sub: pd.DataFrame) -> dict:
    """Exact / within-1 / MAE on rows where gold is pure and both counts exist."""
    both = sub[
        (sub["is_pure_bath_bomb_gold"] == True)  # noqa: E712
        & sub["n_bomb_balls_gold"].notna()
        & sub["n_bomb_balls"].notna()
    ]
    if both.empty:
        return {"n": 0}
    err = (both["n_bomb_balls"] - both["n_bomb_balls_gold"]).abs()
    return {
        "n": int(len(both)),
        "exact": float((err == 0).mean()),
        "within_1": float((err <= 1).mean()),
        "mae": float(err.mean()),
    }


def evaluate_report(pred: pd.DataFrame, gold: str | Path | pd.DataFrame) -> dict:
    """Full model report: purity confusion/F1, count exact/within-1/MAE, plus
    breakdowns by model confidence (calibration) and by stratum, and error rows."""
    if not isinstance(gold, pd.DataFrame):
        gold = pd.read_csv(gold)
    if not {"asin", "is_pure_bath_bomb_gold", "n_bomb_balls_gold"} <= set(gold.columns):
        raise ValueError("Gold missing required columns")
    gold = _coerce_gold(gold)
    m = pred.merge(gold, on="asin", how="inner", suffixes=("", "_g"))
    if m.empty:
        return {"n_overlap": 0}

    report: dict = {
        "n_overlap": int(len(m)),
        "purity": _purity_scores(m),
        "count": _count_scores(m),
        "by_confidence": {},
        "by_stratum": {},
        "errors": [],
    }

    conf_col = "count_confidence" if "count_confidence" in m.columns else None
    if conf_col:
        for c in ["high", "medium", "low"]:
            block = m[m[conf_col] == c]
            if len(block):
                report["by_confidence"][c] = {
                    "purity": _purity_scores(block),
                    "count": _count_scores(block),
                }

    strat_col = "stratum" if "stratum" in m.columns else ("stratum_g" if "stratum_g" in m.columns else None)
    if strat_col:
        for s in sorted(m[strat_col].dropna().unique().tolist()):
            block = m[m[strat_col] == s]
            report["by_stratum"][str(s)] = {
                "purity": _purity_scores(block),
                "count": _count_scores(block),
            }

    # Error rows: purity mismatches + count mismatches on pure golds.
    for _, r in m.iterrows():
        pg = r["is_pure_bath_bomb_gold"]
        if pg is not None and (r["is_pure_bath_bomb"] == True) != (pg == True):  # noqa: E712
            report["errors"].append({
                "asin": r["asin"], "type": "purity",
                "pred": bool(r["is_pure_bath_bomb"] == True), "gold": bool(pg == True),  # noqa: E712
            })
        elif (pg == True and pd.notna(r["n_bomb_balls_gold"]) and pd.notna(r["n_bomb_balls"])  # noqa: E712
              and int(r["n_bomb_balls"]) != int(r["n_bomb_balls_gold"])):
            report["errors"].append({
                "asin": r["asin"], "type": "count",
                "pred": int(r["n_bomb_balls"]), "gold": int(r["n_bomb_balls_gold"]),
            })
    return report
