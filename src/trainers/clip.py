"""Utilities for CLIP fine-tuning and feature caching."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class _ClipArchiveLoader(pickle.Unpickler):
    def __init__(self, file, storage_root: Path):
        super().__init__(file)
        self._storage_root = storage_root

    def persistent_load(self, pid):
        _, storage_cls, key, _, size = pid
        return storage_cls.from_file(str(self._storage_root / key), size=size, shared=False)


def load_clip_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load a saved CLIP checkpoint (file or extracted archive) and return the state dict."""
    path = path.expanduser()
    if path.is_file():
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        archive = path / "archive"
        data_file = archive / "data.pkl"
        if not data_file.exists():
            raise RuntimeError(f"Checkpoint not found at {path}")
        with data_file.open("rb") as handle:
            loader = _ClipArchiveLoader(handle, archive / "data")
            data = loader.load()
    if isinstance(data, dict) and "model" in data:
        data = data["model"]
    if not isinstance(data, dict):
        raise RuntimeError("Unsupported checkpoint format; expected a dict of tensors.")
    return data


def clip_backbone_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract ResNet image encoder weights (excluding FC) from CLIP state dict."""
    weights: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("image_encoder.resnet."):
            continue
        new_key = key.replace("image_encoder.resnet.", "", 1)
        if new_key.startswith("fc."):
            continue
        weights[new_key] = value
    if not weights:
        raise RuntimeError("Image encoder weights not found in the supplied CLIP checkpoint.")
    return weights


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, study_ids: Sequence[str]):
        self.features = features
        self.targets = targets
        self.study_ids = list(study_ids)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx], self.study_ids[idx]


def cache_features(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    cache_path: Path,
) -> Dict[str, object]:
    """Cache backbone features for faster linear-head training."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] caching features to {cache_path}")
    model.eval()
    requires_restore = hasattr(model, "fc")
    original_fc = None
    if requires_restore:
        original_fc = model.fc
        model.fc = nn.Identity()
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    study_ids: List[str] = []
    with torch.no_grad():
        for images, batch_targets, ids in loader:
            images = images.to(device, non_blocking=True)
            feats = model(images).cpu()
            features.append(feats)
            targets.append(batch_targets.clone())
            study_ids.extend([str(study_id) if study_id is not None else "" for study_id in ids])
    if requires_restore and original_fc is not None:
        model.fc = original_fc
    cache = {
        "features": torch.cat(features, dim=0),
        "targets": torch.cat(targets, dim=0),
        "study_ids": study_ids,
    }
    torch.save(cache, cache_path)
    return cache


def compute_label_density(samples: Sequence) -> float:
    """Compute average label density over ConceptSample sequence."""
    if not samples:
        return 0.0
    stacked = torch.stack([sample.targets for sample in samples], dim=0)
    return stacked.mean().item()
