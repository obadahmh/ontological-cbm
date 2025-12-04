"""Utilities for loading MedCLIP models with a shared cache location."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModel, MedCLIPVisionModelViT


def load_medclip(
    *,
    device: torch.device,
    variant: str = "resnet",
    checkpoint_dir: Path | None = None,
) -> Tuple[MedCLIPProcessor, MedCLIPModel]:
    """Load MedCLIP processor + model using the official from_pretrained workflow."""
    processor = MedCLIPProcessor()
    variant_normalized = variant.lower()
    if variant_normalized == "resnet":
        vision_cls = MedCLIPVisionModel
    elif variant_normalized == "vit":
        vision_cls = MedCLIPVisionModelViT
    else:
        raise ValueError("variant must be either 'resnet' or 'vit'")

    model = MedCLIPModel(vision_cls=vision_cls)
    input_dir = str(Path(checkpoint_dir).expanduser().resolve()) if checkpoint_dir else None
    model.from_pretrained(input_dir=input_dir)
    model = model.to(device)
    model.eval()
    return processor, model
