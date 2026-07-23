"""Orchestration: data -> purity -> counting, in one row-wise pass."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_config
from counting import build_counter
from data import load_data
from labeling import build_labeling_sample
from purity import build_purifier


def classify_and_count(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Single pass over the frame: purity verdict + candidate counts + resolution."""
    purifier = build_purifier(cfg)
    counter = build_counter(cfg)
    records = []
    for _, row in df.iterrows():
        pr = purifier.classify(row)
        cand = counter.candidates(row)
        res = counter.resolve(pr.is_pure_bath_bomb, cand)
        records.append({
            "is_pure_bath_bomb": pr.is_pure_bath_bomb,
            "exclude_reason": pr.exclude_reason,
            "needs_review": pr.needs_review,
            "purity_source": pr.purity_source,
            **cand, **res,
        })
    return pd.concat([df, pd.DataFrame(records, index=df.index)], axis=1)


def run_pipeline(config_path=None, *, write_labeling_sample: bool = False) -> pd.DataFrame:
    cfg = load_config(config_path)

    df = load_data(cfg)                         # 1) read + consolidate
    df = classify_and_count(df, cfg)            # 2) purify + 3) count

    out_path = Path(cfg["paths"]["output_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if write_labeling_sample:
        sample = build_labeling_sample(
            df,
            sample_size=int(cfg["labeling"]["sample_size"]),
            seed=int(cfg["labeling"]["seed"]),
        )
        sample_path = Path(cfg["paths"]["labeling_sample_csv"])
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_path, index=False)
        print(f"Wrote labeling sample: {len(sample):,} rows -> {sample_path}")

    pure = int((df["is_pure_bath_bomb"] == True).sum())  # noqa: E712
    print(f"Wrote {len(df):,} rows -> {out_path}")
    print(f"Pure bath bombs: {pure:,} | excluded: {len(df) - pure:,}")
    print("Exclude reasons:", df.loc[df["is_pure_bath_bomb"] != True, "exclude_reason"]  # noqa: E712
          .value_counts().to_dict())
    return df
