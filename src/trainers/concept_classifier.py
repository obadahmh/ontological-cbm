"""Reusable concept classifier training utilities."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.trainers.clip import cache_features, clip_backbone_weights, load_clip_state_dict
from src.utils import init_wandb, set_seed

# Backward compatibility alias
set_random_seed = set_seed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class ConceptSample:
    image_path: Path
    study_id: str
    targets: torch.Tensor


class FeatureTensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, study_ids: Sequence[str]):
        self._features = features
        self._targets = targets
        self._ids = list(study_ids)

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, idx: int):
        return self._features[idx], self._targets[idx], self._ids[idx]


class ConceptImageDataset(Dataset):
    """Torch dataset that returns (image_tensor, target_vector, study_id)."""

    def __init__(self, samples: Sequence[ConceptSample], transform: transforms.Compose):
        self._samples = list(samples)
        self._transform = transform
        self._skip_warned = False

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample = self._samples[idx]
        try:
            from PIL import Image
            with Image.open(sample.image_path) as image:
                image = image.convert("RGB")
        except Exception:
            return self._handle_corrupt_sample(idx)
        tensor = self._transform(image) if self._transform else image
        targets = sample.targets.clone()
        return tensor, targets, sample.study_id

    def _handle_corrupt_sample(self, idx: int):
        fallback_indices = list(range(len(self._samples)))
        random.shuffle(fallback_indices)
        for fallback_idx in fallback_indices:
            if fallback_idx == idx:
                continue
            try:
                from PIL import Image
                with Image.open(self._samples[fallback_idx].image_path) as image:
                    image = image.convert("RGB")
            except Exception:
                continue
            tensor = self._transform(image) if self._transform else image
            targets = self._samples[fallback_idx].targets.clone()
            return tensor, targets, self._samples[fallback_idx].study_id
        raise RuntimeError("All image samples appear to be unreadable; aborting.")


class SimpleCNN(nn.Module):
    """Minimal CNN baseline for concept prediction."""

    def __init__(self, out_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def build_model(backbone: str, num_outputs: int, pretrained: bool = True) -> nn.Module:
    name = backbone.lower()
    if name == "simple_cnn":
        return SimpleCNN(num_outputs)
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, num_outputs)
    return model


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    from torchvision import transforms as T

    train_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def split_samples(samples: Sequence[ConceptSample], train_frac: float, seed: int) -> Tuple[List[ConceptSample], List[ConceptSample]]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    pivot = max(1, int(len(indices) * train_frac))
    train_idx = indices[:pivot]
    test_idx = indices[pivot:]
    if not test_idx:
        raise RuntimeError("Hold-out split is empty; decrease --train-fraction or increase data availability.")
    return [samples[i] for i in train_idx], [samples[i] for i in test_idx]


# ============================================================================
# TRAINING CLI & UTILITIES
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test a vision model to predict SapBERT concepts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset identifier (e.g., 'chexpert' or 'mimic_cxr').")
    parser.add_argument(
        "--concepts-path",
        required=True,
        help="Path to SapBERT study_concepts.jsonl produced by build_umls_concept_bank_sapbert.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/concept_classifier",
        help="Directory to write metrics, checkpoints, and concept index.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of samples for training split.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=25,
        help=(
            "Minimum study count for a concept to be kept. "
            "For the current MIMIC-CXR SapBERT bank, 25 yields ~700 concepts; "
            "consider 10 or 50 for ablations."
        ),
    )
    parser.add_argument("--max-concepts", type=int, default=64, help="Upper bound on number of concepts to model.")
    parser.add_argument(
        "--assertion",
        default="present",
        choices=["present", "absent", "uncertain", "any"],
        help="Assertion label to treat as positive. 'any' keeps all assertions.",
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "simple_cnn"],
        help="Backbone architecture to use.",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights where available.")
    parser.add_argument(
        "--clip-backbone-checkpoint",
        type=Path,
        default=None,
        help="Optional CXR-CLIP checkpoint to extract frozen backbone features (overrides --model).",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional directory to save/load cached CLIP features when using --clip-backbone-checkpoint.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda", help="Device identifier passed to torch.device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and initialization.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional limit on total labelled samples.")
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Optional fraction (0-1] of image records to sample before matching concepts.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Optional JSONL path to write predicted per-study concepts (evaluated split).",
    )
    parser.add_argument(
        "--reference-output",
        default=None,
        help="Optional JSONL path to write reference concepts for the evaluated split.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold for keeping a predicted concept in the JSONL output.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop training if micro-F1 fails to improve for this many epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Minimum micro-F1 improvement required to reset the early stopping counter.",
    )
    parser.add_argument(
        "--early-stop-monitor",
        default="micro_f1",
        choices=[
            "micro_f1",
            "micro_precision",
            "micro_recall",
            "multilabel_accuracy",
            "subset_accuracy",
            "test_loss",
        ],
        help="Metric used for early stopping (test_loss implies minimise).",
    )
    parser.add_argument(
        "--early-stop-mode",
        choices=["max", "min"],
        default=None,
        help="Override early-stop direction (default: max for metrics, min for test_loss).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases (requires wandb package).",
    )
    parser.add_argument(
        "--wandb-project",
        default="medclip",
        help="Weights & Biases project name (used when --wandb is enabled).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional custom run name for Weights & Biases (otherwise auto-generated).",
    )
    return parser.parse_args(argv)


def _normalize_identifier(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    return text or None


def load_concept_targets(
    concepts_path: Path,
    assertion_filter: str,
    min_frequency: int,
    max_concepts: int,
) -> Tuple[List[str], Dict[str, torch.Tensor], Dict[str, int], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    concept_counts: Counter[str] = Counter()
    study_to_concepts: Dict[str, List[str]] = {}
    concept_metadata: Dict[str, Dict[str, Any]] = {}

    with concepts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            study_id = _normalize_identifier(record.get("study_id"))
            if not study_id:
                continue
            kept_concepts: List[str] = []
            for item in record.get("concepts", []):
                assertion = item.get("assertion")
                if assertion_filter != "any" and assertion != assertion_filter:
                    continue
                name = item.get("concept")
                if not name:
                    continue
                concept_counts[name] += 1
                kept_concepts.append(name)
                cui = item.get("cui")
                if cui:
                    concept_metadata.setdefault(name, {})["cui"] = cui
            if kept_concepts:
                study_to_concepts[study_id] = kept_concepts

    if not concept_counts:
        raise RuntimeError("No concepts matched the requested assertion filter.")

    frequent = [concept for concept, count in concept_counts.items() if count >= min_frequency]
    if not frequent:
        top_concept, top_count = concept_counts.most_common(1)[0]
        raise RuntimeError(
            f"No concepts met min_frequency={min_frequency}. Highest frequency concept "
            f"is '{top_concept}' with {top_count} studies. Lower --min-frequency."
        )
    frequent.sort(key=lambda name: (-concept_counts[name], name.lower()))
    if max_concepts is not None:
        frequent = frequent[:max_concepts]

    concept_to_index = {concept: idx for idx, concept in enumerate(frequent)}
    study_vectors: Dict[str, torch.Tensor] = {}
    study_positive: Dict[str, List[str]] = {}
    for study_id, names in study_to_concepts.items():
        vector = torch.zeros(len(frequent), dtype=torch.float32)
        for name in names:
            idx = concept_to_index.get(name)
            if idx is not None:
                vector[idx] = 1.0
        if vector.sum() > 0:
            study_vectors[study_id] = vector
            study_positive[study_id] = [frequent[i] for i, value in enumerate(vector.tolist()) if value > 0]

    if not study_vectors:
        raise RuntimeError("No study-level labels intersect with the selected concepts.")

    # Ensure metadata keys exist for selected concepts
    for name in frequent:
        concept_metadata.setdefault(name, {})

    return frequent, study_vectors, concept_to_index, study_positive, concept_metadata
def gather_samples_from_iterator(
    dataset_name: str,
    study_vectors: Dict[str, torch.Tensor],
    *,
    limit: Optional[int],
    subsample: Optional[float],
) -> List[ConceptSample]:
    try:
        from src.per_study import iter_image_records
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("src.per_study is required to iterate image records.") from exc

    dataset_norm = dataset_name.lower()
    if dataset_norm == "chexpert":
        dataset_norm = "chexpert_plus"
    supported = {"chexpert_plus", "mimic_cxr"}
    if dataset_norm not in supported:
        raise ValueError(f"Dataset '{dataset_name}' is not supported by the image iterator.")

    samples: List[ConceptSample] = []
    missing_images = 0
    missing_labels = 0
    all_records = list(iter_image_records(dataset_norm))
    if not all_records:
        raise RuntimeError(f"No image records found for dataset '{dataset_name}'.")

    records = all_records
    if subsample is not None:
        if subsample < 0 or subsample > 1:
            raise ValueError("--subsample must be within [0, 1].")
        if subsample > 0:
            keep = max(1, int(len(all_records) * subsample))
            random.shuffle(records)
            records = records[:keep]

    for study_key, path in records:
        candidates = [study_key]
        if ":" in study_key:
            candidates.append(study_key.split(":", 1)[-1])
        matched_key: Optional[str] = None
        target: Optional[torch.Tensor] = None
        for key in candidates:
            candidate = study_vectors.get(key)
            if candidate is not None:
                matched_key = key
                target = candidate
                break
        if target is None:
            missing_labels += 1
            continue
        path_obj = Path(str(path)).expanduser()
        if not path_obj.is_file():
            missing_images += 1
            continue
        samples.append(
            ConceptSample(
                image_path=path_obj,
                study_id=_normalize_identifier(matched_key),
                targets=target.clone(),
            )
        )
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        if missing_labels:
            print(f"[warn] unmatched study ids: {missing_labels} image records had no concept entry.")
        if missing_images:
            print(f"[warn] missing files: {missing_images} image paths were not found on disk.")
        raise RuntimeError("No samples were collected; verify dataset paths and concept ids.")
    if missing_images:
        print(f"[warn] skipped {missing_images} entries with missing image files.")
    if missing_labels:
        print(f"[warn] skipped {missing_labels} entries without concept annotations.")
    return samples


def _load_or_create_cache(
    cache_path: Path,
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    cache_path = cache_path.expanduser()
    if cache_path.exists():
        print(f"[info] loading cached features from {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    return cache_features(loader, model, device, cache_path)


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    iterator = tqdm(loader, desc=f"train {epoch:03d}", unit="batch", dynamic_ncols=True, leave=False) if tqdm else loader
    for images, targets, _ in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    threshold: float = 0.5,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[float, Dict[str, float], Optional[List[Dict[str, float]]], Optional[List[Dict[str, float]]]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    iterator = tqdm(loader, desc=f"eval {epoch:03d}", unit="batch", dynamic_ncols=True, leave=False) if tqdm else loader
    for images, targets, _ in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
    mean_loss = total_loss / max(total, 1)
    probs_tensor = torch.cat(all_probs)
    targets_tensor = torch.cat(all_targets)
    metrics = compute_metrics(probs_tensor, targets_tensor, threshold=threshold)
    preds = (probs_tensor >= threshold).float()
    metrics["multilabel_accuracy"] = (preds == targets_tensor).float().mean().item()
    metrics["subset_accuracy"] = (preds == targets_tensor).all(dim=1).float().mean().item()
    per_class = _per_class_metrics(preds, targets_tensor, class_names)
    thresholds = _tune_thresholds(probs_tensor, targets_tensor, class_names)
    metrics.update(_auc_metrics(probs_tensor, targets_tensor))
    return mean_loss, metrics, per_class, thresholds


def compute_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: Union[float, torch.Tensor] = 0.5,
) -> Dict[str, float]:
    eps = 1e-8
    thresh_tensor = threshold
    if isinstance(threshold, torch.Tensor):
        thresh_tensor = threshold.to(probs.device)
    preds = (probs >= thresh_tensor).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    label_density = targets.mean().item()

    metrics = {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "label_density": label_density,
    }
    return metrics


def _per_class_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[Sequence[str]],
) -> Optional[List[Dict[str, float]]]:
    if not class_names:
        return None
    eps = 1e-8
    rows: List[Dict[str, float]] = []
    for idx, name in enumerate(class_names):
        col_pred = preds[:, idx]
        col_target = targets[:, idx]
        tp = (col_pred * col_target).sum().item()
        fp = (col_pred * (1.0 - col_target)).sum().item()
        fn = ((1.0 - col_pred) * col_target).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1})
    return rows


def _auc_metrics(probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    result: Dict[str, float] = {}
    probs_np = probs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    try:
        result["micro_auroc"] = roc_auc_score(targets_np, probs_np, average="micro")
        result["macro_auroc"] = roc_auc_score(targets_np, probs_np, average="macro")
    except ValueError:
        pass
    try:
        result["micro_auprc"] = average_precision_score(targets_np, probs_np, average="micro")
        result["macro_auprc"] = average_precision_score(targets_np, probs_np, average="macro")
    except ValueError:
        pass
    return result


def _tune_thresholds(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[Sequence[str]],
) -> Optional[List[Dict[str, float]]]:
    if not class_names:
        return None
    probs_np = probs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    thresholds: List[Dict[str, float]] = []
    candidates = np.linspace(0.05, 0.95, 19)
    for idx, name in enumerate(class_names):
        y_true = targets_np[:, idx]
        y_scores = probs_np[:, idx]
        best_t = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (y_scores >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = threshold
        thresholds.append({"class": name, "threshold": float(best_t), "f1": float(best_f1)})
    return thresholds


@torch.no_grad()
def collect_study_predictions(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    desc: str,
) -> Dict[str, torch.Tensor]:
    model.eval()
    study_rows: Dict[str, List[torch.Tensor]] = {}
    iterator = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False) if tqdm else loader
    for images, _, study_ids in iterator:
        images = images.to(device, non_blocking=True)
        probs = torch.sigmoid(model(images)).cpu()
        for sid, row in zip(study_ids, probs):
            key = str(sid)
            study_rows.setdefault(key, []).append(row)
    aggregated: Dict[str, torch.Tensor] = {}
    for sid, rows in study_rows.items():
        if rows:
            aggregated[sid] = torch.stack(rows, dim=0).mean(dim=0)
    return aggregated


def build_sample_metadata(dataset: ConceptImageDataset) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for sample in getattr(dataset, "_samples", []):
        if not sample.study_id:
            continue
        mapping[str(sample.study_id)] = {"image_path": str(sample.image_path)}
    return mapping


def write_predictions_jsonl(
    path: Path,
    study_probs: Mapping[str, torch.Tensor],
    concept_names: Sequence[str],
    concept_metadata: Mapping[str, Dict[str, Any]],
    threshold: float,
    dataset_name: str,
    metadata_lookup: Mapping[str, Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for study_id in sorted(study_probs):
            probs = study_probs[study_id]
            concepts: List[Dict[str, Any]] = []
            for idx, prob in enumerate(probs.tolist()):
                if prob >= threshold:
                    name = concept_names[idx]
                    meta = concept_metadata.get(name, {})
                    concepts.append(
                        {
                            "concept": name,
                            "cui": meta.get("cui"),
                            "score": float(prob),
                            "assertion": "present",
                        }
                    )
            metadata = {"dataset": dataset_name, "split": "test"}
            extra = metadata_lookup.get(study_id)
            if extra and extra.get("image_path"):
                metadata.setdefault("image_path", extra["image_path"])
            record = {
                "study_key": study_id,
                "study_id": study_id,
                "concepts": concepts,
                "metadata": metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_reference_jsonl(
    path: Path,
    study_ids: Iterable[str],
    study_concepts: Mapping[str, List[str]],
    concept_metadata: Mapping[str, Dict[str, Any]],
    dataset_name: str,
    metadata_lookup: Mapping[str, Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for study_id in sorted(set(study_ids)):
            names = study_concepts.get(study_id, [])
            concepts = []
            for name in names:
                meta = concept_metadata.get(name, {})
                concepts.append({"concept": name, "cui": meta.get("cui"), "assertion": "present"})
            metadata = {"dataset": dataset_name, "split": "test"}
            extra = metadata_lookup.get(study_id)
            if extra and extra.get("image_path"):
                metadata.setdefault("image_path", extra["image_path"])
            record = {
                "study_key": study_id,
                "study_id": study_id,
                "concepts": concepts,
                "metadata": metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    clip_checkpoint = args.clip_backbone_checkpoint
    feature_cache_dir = Path(args.feature_cache).expanduser() if args.feature_cache else None
    if clip_checkpoint and args.model.lower() != "resnet50":
        raise ValueError("--clip-backbone-checkpoint currently requires --model resnet50.")
    if clip_checkpoint is None and feature_cache_dir is not None:
        raise ValueError("--feature-cache requires --clip-backbone-checkpoint to be set.")

    concepts_path = Path(args.concepts_path).expanduser()
    outputs_root = Path(args.output_dir).expanduser()
    outputs_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading SapBERT concepts from {concepts_path}")
    concept_names, study_vectors, _, study_positive, concept_metadata = load_concept_targets(
        concepts_path=concepts_path,
        assertion_filter=args.assertion,
        min_frequency=args.min_frequency,
        max_concepts=args.max_concepts,
    )
    print(
        f"[info] tracking {len(concept_names)} concepts "
        f"(min_frequency={args.min_frequency}, assertion={args.assertion})"
    )
    print(f"[info] enumerating images for dataset '{args.dataset_name}'")
    samples = gather_samples_from_iterator(
        dataset_name=args.dataset_name,
        study_vectors=study_vectors,
        limit=args.limit_samples,
        subsample=args.subsample,
    )
    print(f"[info] gathered {len(samples)} labelled samples across {len(concept_names)} concepts.")

    train_samples, test_samples = split_samples(samples, args.train_fraction, args.seed)
    print(f"[info] split samples into train={len(train_samples)} test={len(test_samples)}")
    train_tf, eval_tf = build_transforms()

    train_dataset = ConceptImageDataset(train_samples, transform=train_tf)
    test_dataset = ConceptImageDataset(test_samples, transform=eval_tf)

    print("[info] building data loaders…")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device.lower() else "cpu")
    clip_state: Optional[Dict[str, torch.Tensor]] = None
    if clip_checkpoint:
        clip_state = clip_backbone_weights(load_clip_state_dict(clip_checkpoint))
        print(f"[info] loading CLIP backbone weights from {clip_checkpoint}")

    model_name = "resnet50" if clip_checkpoint else args.model
    print(f"[info] initializing model '{model_name}' on device {device}")
    model = build_model(model_name, num_outputs=len(concept_names), pretrained=args.pretrained and not clip_checkpoint)
    model.to(device)
    if clip_state:
        incompatible = model.load_state_dict(clip_state, strict=False)
        if incompatible.missing_keys:
            print(f"[warn] missing keys when loading CLIP weights: {sorted(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[warn] unexpected keys when loading CLIP weights: {sorted(incompatible.unexpected_keys)}")
        if feature_cache_dir is None:
            for name, param in model.named_parameters():
                if name.startswith("fc"):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            print("[info] frozen CLIP backbone; training linear head only.")

    if feature_cache_dir:
        cache_root = feature_cache_dir
        train_cache = _load_or_create_cache(cache_root / "train.pt", train_loader, model, device)
        test_cache = _load_or_create_cache(cache_root / "test.pt", test_loader, model, device)
        feature_dim = train_cache["features"].shape[1]
        model = nn.Linear(feature_dim, len(concept_names)).to(device)
        train_dataset = FeatureTensorDataset(train_cache["features"], train_cache["targets"], train_cache["study_ids"])
        test_dataset = FeatureTensorDataset(test_cache["features"], test_cache["targets"], test_cache["study_ids"])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    monitor_metric = args.early_stop_monitor
    monitor_mode = args.early_stop_mode or ("min" if monitor_metric == "test_loss" else "max")
    if monitor_mode == "min":
        best_metric_value = float("inf")
    else:
        best_metric_value = float("-inf")
    best_f1 = -1.0
    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = max(0, args.early_stop_patience)
    min_delta = max(0.0, args.early_stop_min_delta)
    epochs_without_improvement = 0
    wandb_run = init_wandb(
        args,
        {
            "num_concepts": len(concept_names),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
        },
    )

    last_per_class: Optional[List[Dict[str, float]]] = None
    last_thresholds: Optional[List[Dict[str, float]]] = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch)
        test_loss, test_metrics, test_per_class, tuned_thresholds = evaluate(
            test_loader,
            model,
            criterion,
            device,
            epoch,
            threshold=args.prediction_threshold,
            class_names=concept_names,
        )
        last_per_class = test_per_class
        last_thresholds = tuned_thresholds
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, **test_metrics}
        history.append(epoch_metrics)

        monitored_value = test_loss if monitor_metric == "test_loss" else test_metrics.get(monitor_metric)
        if monitored_value is None:
            raise ValueError(f"Requested early-stop metric '{monitor_metric}' is unavailable.")
        improved = False
        if monitor_mode == "min":
            if monitored_value < best_metric_value - min_delta:
                improved = True
        else:
            if monitored_value > best_metric_value + min_delta:
                improved = True

        best_f1 = max(best_f1, test_metrics.get("micro_f1", best_f1))

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"microF1={test_metrics['micro_f1']:.4f} "
            f"precision={test_metrics['micro_precision']:.4f} "
            f"recall={test_metrics['micro_recall']:.4f} "
            f"subsetAcc={test_metrics['subset_accuracy']:.4f} "
            f"multiAcc={test_metrics['multilabel_accuracy']:.4f}"
        )

        if improved:
            best_metric_value = monitored_value
            checkpoint = {
                "model": model.state_dict(),
                "concept_names": concept_names,
                "args": vars(args),
            }
            torch.save(checkpoint, outputs_root / "best_model.pt")
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"[info] early stopping triggered after epoch {epoch:03d}")
                break

        if wandb_run:
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                **test_metrics,
            }
            # Build wandb table inline
            if test_per_class:
                try:
                    import wandb
                    columns = ["class", "precision", "recall", "f1"]
                    has_threshold = bool(tuned_thresholds)
                    if has_threshold:
                        columns.append("threshold")
                    table = wandb.Table(columns=columns)
                    threshold_map = {entry["class"]: entry for entry in tuned_thresholds} if tuned_thresholds else {}
                    for entry in test_per_class:
                        row = [entry["class"], entry["precision"], entry["recall"], entry["f1"]]
                        if has_threshold:
                            row.append(threshold_map.get(entry["class"], {}).get("threshold", None))
                        table.add_data(*row)
                    log_data["per_class_table"] = table
                except ImportError:
                    pass
            wandb_run.log(log_data)

    summary = {
        "concepts": concept_names,
        "history": history,
        "best_micro_f1": best_f1,
        "best_monitor_metric": {
            "name": monitor_metric,
            "mode": monitor_mode,
            "value": best_metric_value,
        },
        "num_training_samples": len(train_samples),
        "num_test_samples": len(test_samples),
        "per_class_metrics": last_per_class,
        "tuned_thresholds": last_thresholds,
    }
    with (outputs_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (outputs_root / "concept_index.json").open("w", encoding="utf-8") as handle:
        json.dump({idx: name for idx, name in enumerate(concept_names)}, handle, indent=2)

    if args.predictions_output or args.reference_output:
        if best_state is not None:
            model.load_state_dict(best_state)
        sample_metadata = build_sample_metadata(test_dataset)
        study_probs = collect_study_predictions(test_loader, model, device, desc="predict test")
        if args.predictions_output:
            write_predictions_jsonl(
                Path(args.predictions_output).expanduser(),
                study_probs,
                concept_names,
                concept_metadata,
                args.prediction_threshold,
                args.dataset_name,
                sample_metadata,
            )
        if args.reference_output:
            write_reference_jsonl(
                Path(args.reference_output).expanduser(),
                study_probs.keys(),
                study_positive,
                concept_metadata,
                args.dataset_name,
                sample_metadata,
            )

    print(f"[done] wrote metrics and checkpoint to {outputs_root}")
    if wandb_run:
        wandb_run.summary["best_micro_f1"] = best_f1
        wandb_run.summary["num_training_samples"] = len(train_samples)
        wandb_run.summary["num_test_samples"] = len(test_samples)
        wandb_run.finish()




# ============================================================================
# EVALUATION CLI & UTILITIES
# ============================================================================

def parse_args_eval(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse arguments for evaluation mode."""
    parser = argparse.ArgumentParser(
        description="Evaluate a concept classifier checkpoint and report multi-label metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset identifier (e.g., 'chexpert').")
    parser.add_argument(
        "--concepts-path",
        required=True,
        type=Path,
        help="Path to SapBERT study_concepts.jsonl used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the saved model checkpoint (e.g., outputs/concept_classifier/best_model.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Torch device identifier.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction used when recreating the split.")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "all"),
        help="Which split to evaluate ('all' ignores the train/test split).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional sample cap before splitting.")
    parser.add_argument("--subsample", type=float, default=None, help="Optional subsample fraction before matching.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for accuracy metrics.")
    parser.add_argument(
        "--assertion",
        default="present",
        choices=("present", "absent", "uncertain", "any"),
        help="Assertion type to treat as positive (needs to match training).",
    )
    parser.add_argument("--min-frequency", type=int, default=25, help="Minimum study count per concept.")
    parser.add_argument("--max-concepts", type=int, default=64, help="Upper bound on concepts to model.")
    parser.add_argument("--model", default=None, help="Optional override for the architecture name.")
    parser.add_argument("--predictions-output", type=Path, default=None, help="Optional JSONL path for predictions.")
    parser.add_argument("--reference-output", type=Path, default=None, help="Optional JSONL path for references.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON file to store the reported metrics.",
    )
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Force pretrained weights.")
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Force randomly initialized weights (overrides checkpoint args).",
    )
    parser.set_defaults(pretrained=None)
    return parser.parse_args(argv)


def _select_split_samples(
    samples: Sequence[ConceptSample],
    split: str,
    train_fraction: float,
    seed: int,
) -> List[ConceptSample]:
    if split == "all":
        return list(samples)
    train_samples, test_samples = split_samples(samples, train_fraction, seed)
    return train_samples if split == "train" else test_samples


@torch.no_grad()
def evaluate_loader(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    iterator = tqdm(loader, desc="eval", unit="batch", dynamic_ncols=True, leave=False) if tqdm else loader
    for images, targets, _ in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
    if total == 0:
        raise RuntimeError("No samples were seen during evaluation.")
    probs_tensor = torch.cat(all_probs, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(probs_tensor, targets_tensor, threshold=threshold)
    preds = (probs_tensor >= threshold).float()
    overall_accuracy = (preds == targets_tensor).float().mean().item()
    subset_accuracy = (preds == targets_tensor).all(dim=1).float().mean().item()
    metrics.update(
        {
            "loss": total_loss / total,
            "multilabel_accuracy": overall_accuracy,
            "subset_accuracy": subset_accuracy,
            "threshold": threshold,
            "samples": int(total),
        }
    )
    return metrics


def main_eval(argv: Optional[Sequence[str]] = None) -> None:
    """Evaluate a trained concept classifier checkpoint."""
    args = parse_args_eval(argv)
    set_seed(args.seed)

    concepts_path = args.concepts_path.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    model_name = args.model or saved_args.get("model", "resnet18")
    if args.pretrained is None:
        pretrained_flag = bool(saved_args.get("pretrained", False))
    else:
        pretrained_flag = bool(args.pretrained)

    print(f"[info] loading concepts from {concepts_path}")
    concept_names, study_vectors, _, study_positive, concept_metadata = load_concept_targets(
        concepts_path=concepts_path,
        assertion_filter=args.assertion,
        min_frequency=args.min_frequency,
        max_concepts=args.max_concepts,
    )
    print(
        f"[info] tracking {len(concept_names)} concepts "
        f"(assertion={args.assertion}, min_frequency={args.min_frequency})"
    )

    print(f"[info] enumerating images for dataset '{args.dataset_name}'")
    samples = gather_samples_from_iterator(
        dataset_name=args.dataset_name,
        study_vectors=study_vectors,
        limit=args.limit_samples,
        subsample=args.subsample,
    )
    eval_samples = _select_split_samples(samples, args.split, args.train_fraction, args.seed)
    print(f"[info] evaluating {args.split} split with {len(eval_samples)} samples")

    _, eval_tf = build_transforms()
    dataset = ConceptImageDataset(eval_samples, transform=eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device.lower() else "cpu")
    print(f"[info] initializing model '{model_name}' on {device}")
    model = build_model(name=model_name, num_outputs=len(concept_names), pretrained=pretrained_flag)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    metrics = evaluate_loader(loader, model, criterion, device, threshold=args.threshold)
    ordered_keys = [
        "loss",
        "samples",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "label_density",
        "multilabel_accuracy",
        "subset_accuracy",
        "threshold",
    ]
    print("[metrics]")
    for key in ordered_keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key}={value:.4f}")
            else:
                print(f"  {key}={value}")

    if args.metrics_output:
        metrics_output = args.metrics_output.expanduser()
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with metrics_output.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": args.dataset_name,
                    "split": args.split,
                    "model": model_name,
                    "pretrained": pretrained_flag,
                    "metrics": metrics,
                },
                handle,
                indent=2,
            )

    if args.predictions_output or args.reference_output:
        pred_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        sample_metadata = build_sample_metadata(dataset)
        study_probs = collect_study_predictions(pred_loader, model, device, desc="predict")
        if args.predictions_output:
            write_predictions_jsonl(
                args.predictions_output.expanduser(),
                study_probs,
                concept_names,
                concept_metadata,
                args.threshold,
                args.dataset_name,
                sample_metadata,
            )
        if args.reference_output:
            write_reference_jsonl(
                args.reference_output.expanduser(),
                study_probs.keys(),
                study_positive,
                concept_metadata,
                args.dataset_name,
                sample_metadata,
            )




# ============================================================================
# CLI ENTRY POINTS
# ============================================================================

def train_main(argv: Optional[Sequence[str]] = None) -> None:
    """Train concept classifier."""
    main(argv)


def eval_main(argv: Optional[Sequence[str]] = None) -> None:
    """Evaluate concept classifier."""
    main_eval(argv)
