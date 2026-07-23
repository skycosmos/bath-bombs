"""Manual labeling store + label taxonomy and review sampling.

Single reviewer: the manual-label CSV holds one row per ASIN, and saving a label
overwrites the previous one for that ASIN (latest wins). Evaluation of the rules
against these labels lives in evaluate.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Label store — one row per ASIN, latest label wins
# --------------------------------------------------------------------------- #
LABEL_COLUMNS = [
    "asin",
    "class_label",
    "is_pure_bath_bomb_manual",
    "n_bomb_balls_manual",
    "exclude_reason_manual",
    "notes",
    "ts",
]


def now_ts() -> float:
    return time.time()


def load_labels(path) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)
    else:
        df = pd.DataFrame(columns=LABEL_COLUMNS)
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


def upsert_label(df: pd.DataFrame, record: dict) -> pd.DataFrame:
    """Insert or replace the single label for this ASIN (latest wins)."""
    asin = str(record["asin"])
    if df is not None and len(df):
        df = df[df["asin"].astype(str) != asin].copy()
    row = {c: record.get(c) for c in LABEL_COLUMNS}
    new = pd.DataFrame([row], columns=LABEL_COLUMNS)
    if df is None or len(df) == 0:
        return new
    return pd.concat([df, new], ignore_index=True)


def save_label(path, record: dict) -> pd.DataFrame:
    """Load the store, upsert one timestamped record, save, return the store."""
    record = dict(record)
    record["ts"] = now_ts()
    df = upsert_label(load_labels(path), record)
    save_labels(path, df)
    return df


# --------------------------------------------------------------------------- #
# Label taxonomy — the two review dimensions the UI filters on
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Stratified review sample
# --------------------------------------------------------------------------- #
def build_labeling_sample(
    df: pd.DataFrame,
    sample_size: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    work = df.copy()
    work["class_label"] = [class_label_for_row(r) for _, r in work.iterrows()]
    work["count_label"] = [count_label_for_row(r) for _, r in work.iterrows()]

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
    if len(sample) < sample_size:
        remain = work[~work["asin"].isin(sample["asin"])]
        need = min(sample_size - len(sample), len(remain))
        if need:
            idx = rng.choice(remain.index.to_numpy(), size=need, replace=False)
            sample = pd.concat([sample, remain.loc[idx]])

    out = sample[
        [
            "asin",
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
    return out.reset_index(drop=True)
