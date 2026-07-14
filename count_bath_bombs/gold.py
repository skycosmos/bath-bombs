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


def _stratum_for_row(row: pd.Series) -> str:
    title = str(row.get("title") or "").lower()
    if row.get("purity_source") == "rule_kit" or "kit" in title:
        return "kit"
    if row.get("exclude_reason") == "mixed_set":
        return "mixed_set"
    noi = row.get("number_of_items")
    title_n = row.get("cand_title")
    if (noi == 1 or noi == 1.0) and title_n is not None and title_n > 1:
        return "pack_as_one"
    if title_n is not None and title_n > 1:
        return "multi_pack"
    if row.get("needs_llm") or row.get("is_hard_case"):
        return "needs_llm"
    if row.get("n_bomb_balls") == 1:
        return "single"
    extreme = row.get("number_of_items")
    if extreme is not None and not pd.isna(extreme) and float(extreme) >= 50:
        return "extreme_count"
    return "other"


def build_labeling_sample(
    df: pd.DataFrame,
    sample_size: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    work = df.copy()
    work["stratum"] = [_stratum_for_row(r) for _, r in work.iterrows()]

    strata = work["stratum"].unique().tolist()
    per = max(1, sample_size // max(len(strata), 1))
    rng = np.random.default_rng(seed)
    parts = []
    for s in strata:
        block = work[work["stratum"] == s]
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
            "title",
            "number_of_items",
            "size",
            "is_pure_bath_bomb",
            "n_bomb_balls",
            "count_source",
            "exclude_reason",
            "seller_counts_pack_as_one",
            "is_hard_case",
        ]
    ].copy()
    out["is_pure_bath_bomb_gold"] = ""
    out["n_bomb_balls_gold"] = ""
    out["exclude_reason_gold"] = ""
    out["notes"] = ""
    out["annotator"] = ""
    return out.reset_index(drop=True)


def evaluate_against_gold(pred: pd.DataFrame, gold_path: str | Path) -> dict:
    gold = pd.read_csv(gold_path)
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
