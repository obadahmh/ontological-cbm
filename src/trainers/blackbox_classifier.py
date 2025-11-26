#!/usr/bin/env python3
"""Train and evaluate a black-box vision model that predicts the standard CheXpert/MIMIC-CXR labels."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
# Reuse existing trainer utilities (concept classifier helpers)
import src.trainers.concept_classifier as trainer
from medclip import constants as medclip_constants
from src.utils import init_wandb


DEFAULT_LABELS = tuple(medclip_constants.CHEXPERT_TASKS)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-label classifier directly on CheXpert/MIMIC labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", default="chexpert", help="Dataset identifier (for logging only).")
    parser.add_argument(
        "--labels-csv",
        required=True,
        type=Path,
        help="CSV containing image paths and label columns.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional directory to prepend to relative image paths found in the CSV.",
    )
    parser.add_argument(
        "--path-column",
        default=None,
        help="Optional override for the column that stores image paths (defaults to Path/path_to_image/image_path).",
    )
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Label columns to consume (defaults to the 14 CheXpert tasks).",
    )
    parser.add_argument(
        "--uncertainty-mode",
        choices=("zero", "positive"),
        default="zero",
        help="How to treat uncertain (-1) scores from CheXpert: map to 0 or to 1.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/label_classifier",
        help="Directory to write checkpoints and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of samples used for training.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional cap on total samples.")
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Optional random fraction [0, 1] of rows to keep before splitting.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda", help="Device identifier passed to torch.device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and sampling.")
    parser.add_argument("--model", default="resnet34", choices=["resnet18", "resnet34", "resnet50", "simple_cnn"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained weights.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained weights.")
    parser.set_defaults(pretrained=True)
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
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional metadata CSV that provides image paths when the labels CSV lacks them "
        "(e.g., mimic-cxr-2.0.0-metadata.csv.gz).",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=None,
        help="Optional positive example weight for BCE loss (applied to all classes).",
    )
    return parser.parse_args(argv)


def _closest_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    mapping = {col.lower(): col for col in columns}
    for candidate in candidates:
        resolved = mapping.get(candidate.lower())
        if resolved:
            return resolved
    return None


def _load_image_paths(
    df: pd.DataFrame,
    path_column: str,
    image_root: Optional[Path],
) -> List[Optional[Path]]:
    paths: List[Optional[Path]] = []
    for value in df[path_column]:
        raw = str(value).strip()
        if not raw or raw.lower() == "nan":
            paths.append(None)
            continue
        candidate = Path(raw)
        if not candidate.is_absolute() and image_root is not None:
            candidate = (image_root / candidate).resolve()
        paths.append(candidate.expanduser())
    return paths


def _map_label_value(value: object, mode: str) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return 0.0
    if numeric == -1.0:
        return 1.0 if mode == "positive" else 0.0
    return 1.0 if numeric > 0.0 else 0.0


def _normalize_id_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _build_mimic_relative_path(subject_id: str, study_id: str, dicom_id: str) -> str:
    subject_norm = subject_id.zfill(8)
    study_norm = study_id.zfill(8)
    top = f"p{subject_norm[:2]}"
    patient_dir = f"p{subject_norm}"
    study_dir = f"s{study_norm}"
    filename = f"{dicom_id}.jpg"
    return str(Path(top) / patient_dir / study_dir / filename)


def _load_mimic_image_paths(metadata_csv: Path) -> pd.DataFrame:
    metadata_csv = metadata_csv.expanduser()
    if not metadata_csv.exists():
        raise RuntimeError(f"Metadata CSV not found: {metadata_csv}")
    meta = pd.read_csv(metadata_csv)
    columns = {col.lower(): col for col in meta.columns}
    subject_col = columns.get("subject_id")
    study_col = columns.get("study_id")
    dicom_col = columns.get("dicom_id")
    path_col = columns.get("path")
    if not subject_col or not study_col:
        raise RuntimeError("Metadata CSV must contain subject_id and study_id columns.")
    meta["__subject_norm"] = meta[subject_col].apply(_normalize_id_value)
    meta["__study_norm"] = meta[study_col].apply(_normalize_id_value)
    if path_col:
        meta["image_path"] = meta[path_col].astype(str)
    elif dicom_col:
        meta["image_path"] = meta.apply(
            lambda row: _build_mimic_relative_path(
                row["__subject_norm"],
                row["__study_norm"],
                str(row[dicom_col]).strip(),
            )
            if row["__subject_norm"] and row["__study_norm"] and str(row[dicom_col]).strip()
            else None,
            axis=1,
        )
    else:
        raise RuntimeError("Metadata CSV must include either a 'path' or 'dicom_id' column.")
    meta = (
        meta.dropna(subset=["__subject_norm", "__study_norm", "image_path"])
        .loc[:, ["__subject_norm", "__study_norm", "image_path"]]
        .drop_duplicates()
    )
    return meta


def _stratified_indices(label_matrix: np.ndarray, keep: int, seed: int) -> List[int]:
    num_samples, num_labels = label_matrix.shape
    rng = np.random.RandomState(seed)
    selected = set()
    for label_idx in range(num_labels):
        pos_idx = np.where(label_matrix[:, label_idx] == 1)[0]
        if not len(pos_idx):
            continue
        target = max(1, int(len(pos_idx) * (keep / num_samples)))
        draw = rng.choice(pos_idx, size=min(len(pos_idx), target), replace=False)
        selected.update(draw.tolist())
    selected_list = list(selected)
    if len(selected_list) > keep:
        selected_list = rng.choice(selected_list, size=keep, replace=False).tolist()
    if len(selected_list) < keep:
        remaining = np.setdiff1d(np.arange(num_samples), np.array(selected_list))
        extra = rng.choice(remaining, size=min(len(remaining), keep - len(selected_list)), replace=False)
        selected_list.extend(extra.tolist())
    return selected_list


def gather_label_samples(
    csv_path: Path,
    *,
    image_root: Optional[Path],
    path_column: Optional[str],
    label_columns: Sequence[str],
    uncertainty_mode: str,
    limit: Optional[int],
    subsample: Optional[float],
    seed: int,
    dataset_name: Optional[str],
    metadata_csv: Optional[Path],
) -> List[trainer.ConceptSample]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"{csv_path} contains no rows.")
    column = path_column or _closest_column(df.columns, ("Path", "path", "path_to_image", "image_path"))
    dataset_norm = (dataset_name or "").lower()
    subject_col = _closest_column(df.columns, ("subject_id", "patient_id"))
    study_col = _closest_column(df.columns, ("study_id", "dicom_study_id", "study"))
    meta_path = metadata_csv
    if column is None:
        if meta_path is None and "mimic" in dataset_norm:
            candidate = csv_path.with_name("mimic-cxr-2.0.0-metadata.csv.gz")
            if candidate.exists():
                meta_path = candidate
        if meta_path is None:
            raise RuntimeError(
                "Unable to determine the image path column. Provide --metadata-csv to supply image paths."
            )
        if subject_col is None or study_col is None:
            raise RuntimeError("Labels CSV must include subject_id and study_id when metadata join is required.")
        df = df.copy()
        df["__subject_norm"] = df[subject_col].apply(_normalize_id_value)
        df["__study_norm"] = df[study_col].apply(_normalize_id_value)
        meta_paths = _load_mimic_image_paths(meta_path)
        df = df.merge(meta_paths, on=["__subject_norm", "__study_norm"], how="inner")
        if df.empty:
            raise RuntimeError("No rows matched between labels CSV and metadata CSV; verify both files.")
        column = "image_path"
    else:
        df = df.copy()
        if subject_col:
            df["__subject_norm"] = df[subject_col].apply(_normalize_id_value)
        if study_col:
            df["__study_norm"] = df[study_col].apply(_normalize_id_value)

    missing_labels = [col for col in label_columns if col not in df.columns]
    if missing_labels:
        raise RuntimeError(f"Requested label columns missing from CSV: {missing_labels}")

    label_values = [
        [_map_label_value(row[col], uncertainty_mode) for col in label_columns]
        for _, row in df.iterrows()
    ]
    label_matrix = np.array(label_values, dtype=np.float32)
    if subsample is not None:
        if not 0 < subsample <= 1:
            raise ValueError("--subsample must be in (0, 1].")
        keep = max(1, int(len(label_matrix) * subsample))
        if keep < len(label_matrix):
            indices = _stratified_indices(label_matrix, keep, seed)
            df = df.iloc[indices].reset_index(drop=True)
            label_matrix = label_matrix[indices]
    study_column = _closest_column(df.columns, ("study_id", "studyid", "Study ID", "StudyID"))
    paths = _load_image_paths(df, column, image_root)

    samples: List[trainer.ConceptSample] = []
    missing_images = 0
    records = list(df.itertuples(index=False))
    for idx, row in enumerate(records):
        image_path = paths[idx]
        if not image_path or not image_path.exists():
            missing_images += 1
            continue
        targets = torch.tensor(label_matrix[idx], dtype=torch.float32)
        row_dict = row._asdict()
        study_value = row_dict.get(study_column) if study_column else None
        samples.append(
            trainer.ConceptSample(
                image_path=image_path,
                study_id=trainer._normalize_identifier(study_value),
                targets=targets,
            )
        )
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise RuntimeError("No labelled samples were collected; verify --labels-csv and --image-root.")
    if missing_images:
        print(f"[warn] skipped {missing_images} entries whose image paths could not be resolved.")

    return samples


def _compute_label_density(samples: Sequence[trainer.ConceptSample]) -> float:
    if not samples:
        return 0.0
    stacked = torch.stack([sample.targets for sample in samples], dim=0)
    return stacked.mean().item()


def train_main(argv: Optional[Sequence[str]] = None) -> None:
    """Train blackbox label classifier."""
    args = parse_args(argv)
    trainer.set_random_seed(args.seed)
    image_root = Path(args.image_root).expanduser() if args.image_root else None
    samples = gather_label_samples(
        csv_path=args.labels_csv.expanduser(),
        image_root=image_root,
        path_column=args.path_column,
        label_columns=args.label_columns,
        uncertainty_mode=args.uncertainty_mode,
        limit=args.limit_samples,
        subsample=args.subsample,
        seed=args.seed,
        dataset_name=args.dataset_name,
        metadata_csv=args.metadata_csv,
    )

    outputs_root = Path(args.output_dir).expanduser()
    outputs_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] training {args.dataset_name} label classifier on {len(samples)} samples")

    train_samples, test_samples = trainer.split_samples(samples, args.train_fraction, args.seed)
    train_density = _compute_label_density(train_samples)
    test_density = _compute_label_density(test_samples)
    print(
        f"[info] split samples into train={len(train_samples)} (density={train_density:.4f}) "
        f"test={len(test_samples)} (density={test_density:.4f})"
    )

    train_tf, eval_tf = trainer.build_transforms()
    train_dataset = trainer.ConceptImageDataset(train_samples, transform=train_tf)
    test_dataset = trainer.ConceptImageDataset(test_samples, transform=eval_tf)

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
    print(f"[info] initializing model '{args.model}' on device {device}")
    model = trainer.build_model(args.model, num_outputs=len(args.label_columns), pretrained=args.pretrained)
    model.to(device)
    train_stack = torch.stack([sample.targets for sample in train_samples], dim=0)
    pos_freq = train_stack.mean(dim=0)
    pos_weight = (1.0 - pos_freq) / (pos_freq + 1e-8)
    if args.positive_class_weight is not None:
        pos_weight = torch.full_like(pos_weight, float(args.positive_class_weight))
    pos_weight = pos_weight.clamp(min=0.05, max=10.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3, threshold=1e-3
    )

    best_f1 = -1.0
    history: List[dict] = []
    best_state: Optional[dict] = None
    patience = max(0, args.early_stop_patience)
    min_delta = max(0.0, args.early_stop_min_delta)
    epochs_without_improvement = 0
    wandb_run = init_wandb(
        args,
        {
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "num_labels": len(args.label_columns),
            "train_label_density": train_density,
            "test_label_density": test_density,
        },
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, model, criterion, optimizer, device, epoch)
        test_loss, test_metrics, test_per_class, test_thresholds = trainer.evaluate(
            test_loader,
            model,
            criterion,
            device,
            epoch,
            threshold=0.5,
            class_names=args.label_columns,
        )
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, **test_metrics}
        history.append(epoch_metrics)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"microF1={test_metrics['micro_f1']:.4f} precision={test_metrics['micro_precision']:.4f} "
            f"recall={test_metrics['micro_recall']:.4f} "
            f"subsetAcc={test_metrics['subset_accuracy']:.4f} "
            f"multiAcc={test_metrics['multilabel_accuracy']:.4f}"
        )

        if test_metrics["micro_f1"] > best_f1 + min_delta:
            best_f1 = test_metrics["micro_f1"]
            checkpoint = {
                "model": model.state_dict(),
                "label_columns": args.label_columns,
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

        scheduler.step(test_metrics["micro_f1"])
        if wandb_run:
            table = trainer._build_wandb_table(test_per_class, test_thresholds)
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                **test_metrics,
            }
            if table is not None:
                log_data["per_class_table"] = table
            wandb_run.log(log_data)

    summary = {
        "labels": args.label_columns,
        "history": history,
        "best_micro_f1": best_f1,
        "num_training_samples": len(train_samples),
        "num_test_samples": len(test_samples),
    }
    with (outputs_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (outputs_root / "label_index.json").open("w", encoding="utf-8") as handle:
        json.dump({idx: name for idx, name in enumerate(args.label_columns)}, handle, indent=2)

    print(f"[done] wrote metrics and checkpoint to {outputs_root}")
    if wandb_run:
        wandb_run.summary["best_micro_f1"] = best_f1
        wandb_run.summary["num_training_samples"] = len(train_samples)
        wandb_run.summary["num_test_samples"] = len(test_samples)
        wandb_run.finish()


# Evaluation functions

def _parse_eval_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the blackbox label classifier.")
    parser.add_argument("--dataset-name", default="mimic_cxr")
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--label-columns", nargs="+", default=medclip_constants.CHEXPERT_TASKS)
    parser.add_argument("--uncertainty-mode", choices=("zero","positive"), default="zero")
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="resnet34", choices=["resnet18", "resnet34", "resnet50"], help="Backbone architecture for the classifier.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)  # unused but kept for CLI parity
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", nargs="+", type=float, help="Per-label thresholds.")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns [study_id,split] to filter the evaluation split.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "validation", "valid", "val", "all"),
        default="test",
        help="Split name used when --split-csv is provided.",
    )
    parser.add_argument("--predictions-output", type=Path, default=None)
    return parser.parse_args(argv)


def _build_threshold_tensor(args: argparse.Namespace, num_labels: int) -> Any:
    if hasattr(args, 'thresholds') and args.thresholds:
        if len(args.thresholds) != num_labels:
            raise ValueError("Number of thresholds must match number of labels.")
        return torch.tensor(args.thresholds, dtype=torch.float32)
    return getattr(args, 'threshold', 0.5)


def _load_split_ids(split_csv: Path, split_name: str) -> set:
    split_csv = split_csv.expanduser()
    if not split_csv.exists():
        raise RuntimeError(f"Split CSV not found: {split_csv}")
    df = pd.read_csv(split_csv)
    if "study_id" not in df.columns or "split" not in df.columns:
        raise RuntimeError("Split CSV must contain 'study_id' and 'split' columns.")
    target = split_name
    if split_name in {"validation", "valid"}:
        target = "validation"
    ids = df[df["split"].str.lower() == target.lower()]["study_id"]
    return set(str(int(val)) for val in ids.dropna())


def write_predictions(
    path: Path,
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    label_names: Sequence[str],
    threshold: Any,
) -> None:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        iterator = trainer._iter_loader(loader, desc="predict",)
        thresh_tensor = threshold
        if isinstance(threshold, torch.Tensor):
            thresh_tensor = threshold.to(device)
        for images, targets, study_ids in iterator:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= thresh_tensor).cpu()
            for study_id, vector, prob in zip(study_ids, preds, probs.cpu()):
                labels = [name for name, val in zip(label_names, vector) if val]
                handle.write(json.dumps({"study_id": str(study_id), "labels": labels}) + "\n")


def eval_main(argv: Optional[Sequence[str]] = None) -> None:
    """Evaluate blackbox label classifier."""
    args = _parse_eval_args(argv)
    samples = gather_label_samples(
        csv_path=args.labels_csv.expanduser(),
        image_root=Path(args.image_root).expanduser() if args.image_root else None,
        path_column=None,
        label_columns=list(args.label_columns),
        uncertainty_mode=args.uncertainty_mode,
        limit=None,
        subsample=args.subsample,
        seed=args.seed,
        dataset_name=args.dataset_name,
        metadata_csv=args.metadata_csv,
    )
    if args.split_csv:
        split_ids = _load_split_ids(args.split_csv, args.split)
        filtered = [sample for sample in samples if sample.study_id and sample.study_id in split_ids]
        if not filtered:
            raise RuntimeError(f"No {args.split} samples found in {args.split_csv}")
        eval_samples = filtered
    else:
        _, eval_samples = trainer.split_samples(samples, args.train_fraction, args.seed)
    _, eval_tf = trainer.build_transforms()
    test_dataset = trainer.ConceptImageDataset(eval_samples, transform=eval_tf)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device.lower() else "cpu")
    model = trainer.build_model(args.model, num_outputs=len(args.label_columns), pretrained=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        name = key
        if name.startswith("image_encoder.resnet."):
            name = name.replace("image_encoder.resnet.", "")
        if name.startswith("logit_scale") or name.startswith("image_projection") or name.startswith("text_encoder"):
            continue
        if name.startswith("fc."):
            continue
        normalized[name] = value
    model.load_state_dict(normalized, strict=False)
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    criterion = nn.BCEWithLogitsLoss()
    threshold_tensor = _build_threshold_tensor(args, len(args.label_columns))
    test_loss, test_metrics, per_class, thresholds = trainer.evaluate(
        loader,
        model,
        criterion,
        device,
        epoch=0,
        threshold=threshold_tensor,
        class_names=args.label_columns,
    )
    print(f"micro_precision={test_metrics['micro_precision']:.4f} micro_recall={test_metrics['micro_recall']:.4f}")
    print(f"micro_f1={test_metrics['micro_f1']:.4f} subset_acc={test_metrics['subset_accuracy']:.4f}")
    print(f"micro_auroc={test_metrics.get('micro_auroc'):.4f} macro_auroc={test_metrics.get('macro_auroc'):.4f}")
    if args.predictions_output:
        write_predictions(args.predictions_output.expanduser(), loader, model, device, args.label_columns, threshold_tensor)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_main(sys.argv[2:])
    else:
        train_main()
