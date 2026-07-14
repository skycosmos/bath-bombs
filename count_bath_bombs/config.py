from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against repo root.
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        if value is None:
            continue
        p = Path(value)
        if not p.is_absolute():
            paths[key] = str((REPO_ROOT / p).resolve())
    prompt = cfg.get("llm", {}).get("prompt_path")
    if prompt:
        p = Path(prompt)
        if not p.is_absolute():
            cfg["llm"]["prompt_path"] = str((REPO_ROOT / p).resolve())
    return cfg
