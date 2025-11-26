"""Lightweight config loader for YAML/JSON with env overrides."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: Path) -> Dict[str, Any]:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".yml", ".yaml"}:
            if yaml is None:
                raise ImportError("PyYAML is required to parse YAML configs.")
            return yaml.safe_load(handle) or {}
        if suffix == ".json":
            return json.load(handle) or {}
    raise ValueError(f"Unsupported config extension: {path}")


def merge_dict(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), MutableMapping):
            merge_dict(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def apply_env_overrides(cfg: MutableMapping[str, Any], prefix: str = "MEDCLIP_") -> MutableMapping[str, Any]:
    """Override config keys from environment variables with given prefix."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        key = env_key[len(prefix) :].lower()
        cfg[key] = env_val
    return cfg
