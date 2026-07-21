#!/usr/bin/env python3
"""Seed provisional labels from model predictions into the annotation store.

Only high-confidence rows in the **train** split are seeded (source=model_seed).
The held-out **eval** split is left untouched so evaluation stays honest — the
model is never scored on labels it produced itself. Seeds pre-fill the UI so a
human can confirm them fast, and are excluded from the human-only gold used by
scripts/eval_gold.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from count_bath_bombs.annotations import (
    load_annotations,
    save_annotations,
    seed_candidates,
    upsert_many,
)
from count_bath_bombs.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--pred", default=None, help="Predictions CSV (default: paths.output_csv)")
    p.add_argument(
        "--confidence",
        nargs="+",
        default=None,
        help="Confidence levels to seed (default: gold.seed_confidence, e.g. high)",
    )
    p.add_argument(
        "--include-eval",
        action="store_true",
        help="DANGER: also seed the held-out eval split (biases evaluation). Off by default.",
    )
    p.add_argument("--dry-run", action="store_true", help="Report how many would be seeded, write nothing")
    args = p.parse_args()

    cfg = load_config(args.config)
    pred_path = args.pred or cfg["paths"]["output_csv"]
    ann_path = cfg["paths"].get("annotations_csv", str(ROOT / "data" / "gold" / "annotations.csv"))
    eval_frac = float(cfg.get("gold", {}).get("eval_frac", 0.2))
    confidences = args.confidence or cfg.get("gold", {}).get("seed_confidence", ["high"])

    pred = pd.read_csv(pred_path, low_memory=False)
    cands = seed_candidates(
        pred,
        eval_frac=eval_frac,
        confidences=confidences,
        only_train=not args.include_eval,
    )
    print(f"{len(cands)} candidate rows (confidence in {list(confidences)}, "
          f"{'train+eval' if args.include_eval else 'train only'}).")

    if args.dry_run or not cands:
        return

    ann = load_annotations(ann_path)
    before = len(ann)
    ann = upsert_many(ann, cands)
    save_annotations(ann_path, ann)
    print(f"Wrote {len(ann) - before:+d} net annotations → {ann_path} "
          f"({len(ann)} total).")


if __name__ == "__main__":
    main()
