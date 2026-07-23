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
    p.add_argument(
        "--labeling-sample",
        action="store_true",
        help="Also write a stratified manual-labeling sample CSV",
    )
    args = p.parse_args()

    run_pipeline(
        args.config,
        write_labeling_sample=args.labeling_sample,
    )


if __name__ == "__main__":
    main()
