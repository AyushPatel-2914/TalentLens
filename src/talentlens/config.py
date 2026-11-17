"""Load configs/config.yaml and resolve repo-relative paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# src/talentlens/config.py -> parents[2] is the repo root. In Docker (non-editable install)
# the package lives in site-packages, so TALENTLENS_ROOT points at the app folder instead.
ROOT = Path(os.environ.get("TALENTLENS_ROOT") or Path(__file__).resolve().parents[2])


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a repo-relative path from the config into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
