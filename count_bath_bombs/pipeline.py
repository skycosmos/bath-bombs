from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from count_bath_bombs.config import load_config
from count_bath_bombs.counting import extract_candidates, resolve_row
from count_bath_bombs.evaluate import evaluate_against_manual
from count_bath_bombs.keepa import IMAGE_URL_PREFIX, KEEPA_FIELDS, attach_keepa
from count_bath_bombs.manual_label import build_labeling_sample
from count_bath_bombs.purity import classify_purity


def load_products(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg["paths"]["csv"]
    cols = cfg["columns_to_keep"]
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[cols].copy()


def classify_and_count(df: pd.DataFrame, scope: dict, purity_cfg: dict) -> pd.DataFrame:
    """One row-wise pass: purity → candidate counts → resolved count."""
    records = []
    for _, row in df.iterrows():
        pr = classify_purity(row, scope, purity_cfg)
        cand = extract_candidates(row)
        res = resolve_row({"is_pure_bath_bomb": pr.is_pure_bath_bomb, **cand})
        records.append({
            "is_pure_bath_bomb": pr.is_pure_bath_bomb,
            "exclude_reason": pr.exclude_reason,
            "needs_review": pr.needs_review,
            "purity_source": pr.purity_source,
            **cand,
            **res,
        })
    return pd.concat([df, pd.DataFrame(records, index=df.index)], axis=1)


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

    df = classify_and_count(df, cfg.get("scope", {}), cfg.get("purity", {}))

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

    manual_path = Path(cfg["paths"]["manual_labels_csv"])
    if manual_path.exists() and manual_path.stat().st_size > 0:
        manual_df = pd.read_csv(manual_path)
        if len(manual_df) > 0:
            metrics = evaluate_against_manual(df, manual_path)
            metrics_path = out_path.parent / "manual_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print("Manual-label metrics:", metrics)

    print(f"Wrote {len(df):,} rows → {out_path}")
    print(
        "Purity counts:",
        df["is_pure_bath_bomb"].value_counts(dropna=False).to_dict(),
    )
    return df
