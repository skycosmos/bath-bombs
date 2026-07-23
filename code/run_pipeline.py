#!/usr/bin/env python3
"""Run the bath-bomb data -> purity -> counting pipeline."""
from __future__ import annotations

import argparse

from pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="Path to config/config.yml")
    p.add_argument("--labeling-sample", action="store_true",
                   help="Also write a stratified manual-labeling sample CSV")
    args = p.parse_args()
    run_pipeline(args.config, write_labeling_sample=args.labeling_sample)


if __name__ == "__main__":
    main()
