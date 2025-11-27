"""Tiny metric helpers."""
from __future__ import annotations

from typing import Dict, Union

import torch


def compute_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: Union[float, torch.Tensor] = 0.5,
) -> Dict[str, float]:
    """Compute micro precision/recall/F1 for multi-label predictions."""
    eps = 1e-8
    thresh_tensor = threshold if isinstance(threshold, torch.Tensor) else torch.tensor(threshold)
    thresh_tensor = thresh_tensor.to(probs.device)
    preds = (probs >= thresh_tensor).float()

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "micro_precision": precision.item(),
        "micro_recall": recall.item(),
        "micro_f1": f1.item(),
    }
