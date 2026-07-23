from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from count_bath_bombs.config import load_config
from count_bath_bombs.counts import apply_candidates
from count_bath_bombs.gold import build_labeling_sample, evaluate_against_gold
from count_bath_bombs.keepa import IMAGE_URL_PREFIX, KEEPA_FIELDS, attach_keepa
from count_bath_bombs.purity import apply_purity
from count_bath_bombs.resolve import apply_resolver


def load_products(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg["paths"]["csv"]
    cols = cfg["columns_to_keep"]
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[cols].copy()


def run_pipeline(
    config_path: str | Path | None = None,
    *,
    write_labeling_sample: bool = False,
) -> pd.DataFrame:
    """Rules over the product CSV + Keepa. Counts every pure bath bomb."""
    cfg = load_config(config_path)

    df = load_products(cfg)

    keepa_cfg = cfg.get("keepa", {})
    if keepa_cfg.get("enabled", False):
        df = attach_keepa(
            df,
            cfg["paths"].get("keepa_csv"),
            image_url_prefix=keepa_cfg.get("image_url_prefix", IMAGE_URL_PREFIX),
        )
    else:
        for col in KEEPA_FIELDS:
            if col not in df.columns:
                df[col] = None

    df = apply_purity(df, cfg.get("scope", {}), cfg.get("purity", {}))
    df = apply_candidates(df)
    df = apply_resolver(df)

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

    gold_path = Path(cfg["paths"]["gold_csv"])
    if gold_path.exists() and gold_path.stat().st_size > 0:
        gold_df = pd.read_csv(gold_path)
        if len(gold_df) > 0:
            metrics = evaluate_against_gold(df, gold_path)
            metrics_path = out_path.parent / "gold_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print("Gold metrics:", metrics)

    print(f"Wrote {len(df):,} rows → {out_path}")
    print(
        "Purity counts:",
        df["is_pure_bath_bomb"].value_counts(dropna=False).to_dict(),
    )
    return df
