#!/usr/bin/env python3
"""Compare the rule predictions against the manual-label set."""

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
from count_bath_bombs.evaluate import evaluate_against_manual, evaluate_report


def _pct(x):
    return "  —  " if x is None else f"{x:6.1%}"


def _print_report(rep: dict) -> None:
    if rep.get("n_overlap", 0) == 0:
        print("No overlap between predictions and manual labels.")
        return
    p, c = rep["purity"], rep["count"]
    print(f"\n=== Rule vs manual labels ({rep['n_overlap']} labeled ASINs) ===")
    if p.get("n"):
        print("Purity classification:")
        print(f"  precision {_pct(p['precision'])}  recall {_pct(p['recall'])}  "
              f"F1 {_pct(p['f1'])}  acc {_pct(p['accuracy'])}   (n={p['n']})")
        print(f"  confusion: TP={p['tp']} FP={p['fp']} FN={p['fn']} TN={p['tn']}")
    if c.get("n"):
        print("Count (pure manual labels):")
        print(f"  exact {_pct(c['exact'])}  within±1 {_pct(c['within_1'])}  "
              f"MAE {c['mae']:.2f}   (n={c['n']})")
    if rep["by_confidence"]:
        print("By model confidence (calibration):")
        for lvl, b in rep["by_confidence"].items():
            pp, cc = b["purity"], b["count"]
            print(f"  {lvl:<6} purity acc {_pct(pp.get('accuracy'))} (n={pp.get('n',0)})  "
                  f"count exact {_pct(cc.get('exact'))} (n={cc.get('n',0)})")
    if rep["by_class"]:
        print("By class label:")
        for s, b in rep["by_class"].items():
            pp, cc = b["purity"], b["count"]
            print(f"  {s:<14} purity acc {_pct(pp.get('accuracy'))} (n={pp.get('n',0)})  "
                  f"count exact {_pct(cc.get('exact'))} (n={cc.get('n',0)})")
    if rep["errors"]:
        print(f"Errors ({len(rep['errors'])}):")
        for e in rep["errors"][:20]:
            print(f"  {e['asin']}  {e['type']}: pred={e['pred']} manual={e['manual']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--pred", default=None, help="Predictions CSV (default: paths.output_csv)")
    p.add_argument("--manual", default=None, help="Manual-label CSV (default: paths.manual_labels_csv)")
    p.add_argument(
        "--report",
        action="store_true",
        help="Print a full performance report (confusion, F1, by-confidence, by-stratum, errors)",
    )
    p.add_argument("--out", default=None, help="Write metrics JSON to this path")
    args = p.parse_args()

    cfg = load_config(args.config)
    pred_path = args.pred or cfg["paths"]["output_csv"]
    manual_path = args.manual or cfg["paths"]["manual_labels_csv"]

    pred = pd.read_csv(pred_path, low_memory=False)
    manual = pd.read_csv(manual_path) if Path(manual_path).exists() else pd.DataFrame()

    metrics = evaluate_against_manual(pred, manual)
    print(json.dumps(metrics, indent=2))

    if args.report:
        rep = evaluate_report(pred, manual)
        _print_report(rep)
        metrics = {"headline": metrics, "report": rep}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote metrics → {args.out}")


if __name__ == "__main__":
    main()
