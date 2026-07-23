"""Load the single YAML config and resolve relative output paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # code/filter_count/ -> repo root
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Relative paths resolve against the repo root; absolute paths are kept.
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        if value and not Path(value).is_absolute():
            paths[key] = str((REPO_ROOT / value).resolve())
    return cfg
