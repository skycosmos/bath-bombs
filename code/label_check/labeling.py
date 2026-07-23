"""Pipeline 4 (support) — manual-label store, taxonomy, and review sampling.

Single reviewer: the label CSV holds one row per ASIN and saving overwrites the
previous label for that ASIN (latest wins). The Streamlit UI in
scripts/label_ui.py is the front end.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "asin", "class_label", "is_pure_bath_bomb_manual", "n_bomb_balls_manual",
    "exclude_reason_manual", "notes", "ts",
]


def now_ts() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Label store — one row per ASIN, latest wins
# --------------------------------------------------------------------------- #
def load_labels(path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p) if (p.exists() and p.stat().st_size > 0) else pd.DataFrame(columns=LABEL_COLUMNS)
    for c in LABEL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[LABEL_COLUMNS].copy()


def save_labels(path, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in LABEL_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out[LABEL_COLUMNS].to_csv(path, index=False)


def save_label(path, record: dict) -> pd.DataFrame:
    """Upsert one timestamped label for an ASIN and persist the store."""
    record = dict(record)
    record["ts"] = now_ts()
    asin = str(record["asin"])
    df = load_labels(path)
    df = df[df["asin"].astype(str) != asin].copy()
    df = pd.concat([df, pd.DataFrame([{c: record.get(c) for c in LABEL_COLUMNS}])], ignore_index=True)
    save_labels(path, df)
    return df


# --------------------------------------------------------------------------- #
# Taxonomy — the two review dimensions (labels themselves are in config)
# --------------------------------------------------------------------------- #
def class_label_for_row(row: pd.Series) -> str:
    """Classification outcome: pure / <exclude_reason> / unclassified."""
    if row.get("is_pure_bath_bomb") is True:
        return "pure"
    reason = row.get("exclude_reason")
    if isinstance(reason, str) and reason:
        return reason
    return "unclassified"


def count_label_for_row(row: pd.Series) -> str:
    """Counting bucket — only meaningful for pure items (else 'n/a')."""
    if row.get("is_pure_bath_bomb") is not True:
        return "n/a"
    if bool(row.get("count_unable")):
        return "count_unable"
    if bool(row.get("seller_counts_pack_as_one")):
        return "pack_as_one"
    n = row.get("n_bomb_balls")
    if n is not None and not pd.isna(n) and float(n) >= 50:
        return "extreme_count"
    if n is not None and not pd.isna(n) and float(n) > 1:
        return "multi_pack"
    if n == 1 or n == 1.0:
        return "single"
    return "count_unable"


# --------------------------------------------------------------------------- #
# Stratified review sample
# --------------------------------------------------------------------------- #
def build_labeling_sample(df: pd.DataFrame, sample_size: int = 300, seed: int = 42) -> pd.DataFrame:
    work = df.copy()
    work["class_label"] = [class_label_for_row(r) for _, r in work.iterrows()]
    work["count_label"] = [count_label_for_row(r) for _, r in work.iterrows()]

    strata = work["class_label"].unique().tolist()
    per = max(1, sample_size // max(len(strata), 1))
    rng = np.random.default_rng(seed)
    parts = []
    for s in strata:
        block = work[work["class_label"] == s]
        n = min(len(block), per)
        if n:
            idx = rng.choice(block.index.to_numpy(), size=n, replace=False)
            parts.append(block.loc[idx])

    sample = pd.concat(parts).drop_duplicates(subset=["asin"])
    if len(sample) < sample_size:
        remain = work[~work["asin"].isin(sample["asin"])]
        need = min(sample_size - len(sample), len(remain))
        if need:
            idx = rng.choice(remain.index.to_numpy(), size=need, replace=False)
            sample = pd.concat([sample, remain.loc[idx]])

    keep = [
        "asin", "class_label", "count_label", "title", "number_of_items", "size",
        "is_pure_bath_bomb", "n_bomb_balls", "count_source", "exclude_reason",
        "seller_counts_pack_as_one", "keepa_main_image_url", "keepa_image_count",
    ]
    keep = [c for c in keep if c in sample.columns]
    return sample[keep].reset_index(drop=True)
