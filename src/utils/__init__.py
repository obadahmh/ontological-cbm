"""Shared helper utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Render Paths/objects to strings for logging."""
    sanitized: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def init_wandb(args, extra_config: Optional[Dict[str, Any]] = None):
    """Lazy import wandb and initialize if requested."""
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Weights & Biases logging requested, but the 'wandb' package is not installed.") from exc
    run_config = sanitize_config({**vars(args), **(extra_config or {})})
    return wandb.init(project=getattr(args, "wandb_project", None), name=getattr(args, "wandb_run_name", None), config=run_config)


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def align_data(
    pred_ids: List[str],
    pred_tensor: torch.Tensor,
    label_ids: List[str],
    label_tensor: torch.Tensor,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Align prediction rows to label rows by study id."""
    pred_map = {sid: i for i, sid in enumerate(pred_ids)}
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    aligned_ids: List[str] = []
    for sid, y in zip(label_ids, label_tensor):
        idx = pred_map.get(sid)
        if idx is None:
            continue
        xs.append(pred_tensor[idx])
        ys.append(y)
        aligned_ids.append(sid)
    if not xs:
        raise RuntimeError("No overlapping study IDs between predictions and labels.")
    return aligned_ids, torch.stack(xs), torch.stack(ys)
