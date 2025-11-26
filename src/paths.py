"""Centralized filesystem paths for data and outputs.

This keeps scripts consistent whether they reference legacy symlink names
(`generated/`, `outputs/`, etc.) or the consolidated `data/` directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Root of the repository (src is one level down)
ROOT = Path(__file__).resolve().parent.parent

# Prefer consolidated data/ layout; fall back to repo root if missing.
DATA_DIR = ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = ROOT

# Common locations
GENERATED = DATA_DIR / "generated"
OUTPUTS = DATA_DIR / "outputs"
PRETRAINED = DATA_DIR / "pretrained"
LOCAL_DATA = DATA_DIR / "local_data"
TMP_RADGRAPH = DATA_DIR / "tmp_radgraph"
WANDB_RUNS = DATA_DIR / "wandb"


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_repo_root_to_sys_path() -> str:
    """Ensure the repository root is available on sys.path for script entrypoints."""
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root_str
