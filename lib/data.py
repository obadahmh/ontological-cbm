"""Data utilities shared across scripts."""
from __future__ import annotations

import random
from typing import List, Tuple

import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_data(
    pred_ids: List[str],
    pred_tensor: torch.Tensor,
    label_ids: List[str],
    label_tensor: torch.Tensor,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Align prediction rows to label rows by shared study id."""
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
