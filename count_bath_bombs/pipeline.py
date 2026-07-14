from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from count_bath_bombs.config import REPO_ROOT, load_config
from count_bath_bombs.counts import apply_candidates
from count_bath_bombs.gold import build_labeling_sample, evaluate_against_gold
from count_bath_bombs.html_extract import attach_html_extracts
from count_bath_bombs.llm import apply_llm_hard_cases
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


def _make_openai_client():
    load_dotenv(REPO_ROOT / ".env")
    try:
        from openai import OpenAI

        return OpenAI()
    except Exception:
        return None


def run_pipeline(
    config_path: str | Path | None = None,
    *,
    skip_html: bool = False,
    html_limit: int | None = None,
    write_labeling_sample: bool = False,
    enable_llm: bool | None = None,
) -> pd.DataFrame:
    """
    Default: rules + HTML only (LLM off).
    After a full parse, re-run with enable_llm=True to improve needs_llm rows.
    """
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(config_path)
    if enable_llm is not None:
        cfg["llm"]["enabled"] = enable_llm

    df = load_products(cfg)

    if not skip_html:
        df = attach_html_extracts(
            df,
            html_dir=cfg["paths"]["html_dir"],
            cache_dir=cfg["paths"]["html_cache_dir"],
            max_bullets=int(cfg["html"]["max_bullets"]),
            max_description_chars=int(cfg["html"]["max_description_chars"]),
            limit=html_limit,
        )
    else:
        for col in (
            "html_bullets",
            "html_description",
            "html_number_of_items",
            "html_unit_count",
            "html_item_package_quantity",
            "html_size",
            "html_item_weight",
        ):
            if col not in df.columns:
                df[col] = None

    df = apply_purity(df, cfg.get("scope", {}), cfg.get("purity", {}))
    df = apply_candidates(df)
    df = apply_resolver(df)

    client = _make_openai_client() if cfg["llm"].get("enabled") else None
    df = apply_llm_hard_cases(df, cfg, client=client)

    out_path = Path(cfg["paths"]["output_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    hard = df[df["is_hard_case"].fillna(False)]
    hard_path = Path(cfg["paths"]["hard_cases_csv"])
    hard.to_csv(hard_path, index=False)

    needs = df[df["needs_llm"].fillna(False)]
    needs_path = Path(cfg["paths"].get("needs_llm_csv", out_path.parent / "needs_llm.csv"))
    needs.to_csv(needs_path, index=False)

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
    print(f"needs_llm={len(needs):,} → {needs_path}")
    print(f"Hard cases: {len(hard):,} → {hard_path}")
    print(
        "Purity counts:",
        df["is_pure_bath_bomb"].value_counts(dropna=False).to_dict(),
    )
    print("LLM enabled:", bool(cfg["llm"].get("enabled")))
    return df
