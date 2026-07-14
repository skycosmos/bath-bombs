#!/usr/bin/env python3
"""Evaluate pipeline predictions against manually curated gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from count_bath_bombs.config import load_config
from count_bath_bombs.gold import evaluate_against_gold


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--pred", default=None, help="Predictions CSV (default: paths.output_csv)")
    p.add_argument("--gold", default=None, help="Gold CSV (default: paths.gold_csv)")
    args = p.parse_args()

    cfg = load_config(args.config)
    pred_path = args.pred or cfg["paths"]["output_csv"]
    gold_path = args.gold or cfg["paths"]["gold_csv"]
    pred = pd.read_csv(pred_path, low_memory=False)
    metrics = evaluate_against_gold(pred, gold_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
