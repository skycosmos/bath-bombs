#!/usr/bin/env python3
"""Run the bath-bomb unit-count pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from count_bath_bombs.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Path to config/pipeline.yml")
    p.add_argument("--skip-html", action="store_true", help="Skip HTML parse (CSV-only rules)")
    p.add_argument(
        "--html-limit",
        type=int,
        default=None,
        help="Only parse/cache HTML for the first N rows (smoke test)",
    )
    p.add_argument(
        "--labeling-sample",
        action="store_true",
        help="Also write a stratified manual-labeling sample CSV",
    )
    p.add_argument(
        "--enable-llm",
        action="store_true",
        help="Override config to enable LLM on hard cases (needs OPENAI_API_KEY)",
    )
    p.add_argument(
        "--disable-llm",
        action="store_true",
        help="Override config to disable LLM",
    )
    args = p.parse_args()

    enable_llm = None
    if args.enable_llm:
        enable_llm = True
    if args.disable_llm:
        enable_llm = False

    run_pipeline(
        args.config,
        skip_html=args.skip_html,
        html_limit=args.html_limit,
        write_labeling_sample=args.labeling_sample,
        enable_llm=enable_llm,
    )


if __name__ == "__main__":
    main()
