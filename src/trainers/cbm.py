"""CBM label head trainer."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import apply_env_overrides, load_config
from src.paths import OUTPUTS, ensure_dir
from src.utils import align_data, init_wandb, set_seed


class CBMHead(nn.Module):
    """Simple MLP head for concept bottleneck classification."""

    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, num_labels),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class CBMTrainerConfig:
    input_dim: int
    num_labels: int
    device: torch.device
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    threshold: float = 0.5
    dropout: float = 0.3
    wandb_run: Any = None


class CBMTrainer:
    """Encapsulates the CBM label-head training loop."""

    def __init__(self, cfg: CBMTrainerConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.model = CBMHead(cfg.input_dim, cfg.num_labels, dropout=cfg.dropout).to(cfg.device)
        self.crit = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Tuple[List[Dict[str, float]], float, Optional[Dict[str, torch.Tensor]]]:
        best_f1 = -1.0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_no_improve = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})

            if val_metrics["micro_f1"] > best_f1 + self.cfg.early_stop_min_delta:
                best_f1 = val_metrics["micro_f1"]
                best_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.cfg.early_stop_patience and epochs_no_improve >= self.cfg.early_stop_patience:
                    print(f"[info] early stopping triggered at epoch {epoch}")
                    break

            if self.cfg.wandb_run:
                log_data = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
                self.cfg.wandb_run.log(log_data)

            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"microF1={val_metrics.get('micro_f1', float('nan')):.4f} "
                f"precision={val_metrics.get('micro_precision', float('nan')):.4f} "
                f"recall={val_metrics.get('micro_recall', float('nan')):.4f}"
            )

        return history, best_f1, best_state

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.opt.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.crit(logits, y)
            loss.backward()
            self.opt.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total = 0.0
        count = 0
        all_probs: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = self.crit(logits, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(y.cpu())
        mean_loss = total / max(count, 1)
        probs = torch.cat(all_probs)
        targets = torch.cat(all_targets)
        metrics = compute_metrics(probs, targets, threshold=self.cfg.threshold)
        return mean_loss, metrics


def compute_metrics(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    eps = 1e-8
    preds = (probs >= threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    metrics: Dict[str, float] = {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "label_density": targets.mean().item(),
        "multilabel_accuracy": (preds == targets).float().mean().item(),
        "subset_accuracy": (preds == targets).all(dim=1).float().mean().item(),
    }
    probs_np = probs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    try:
        metrics["micro_auroc"] = _safe_roc_auc(targets_np, probs_np, average="micro")
    except ValueError:
        pass
    try:
        metrics["micro_auprc"] = _safe_avg_precision(targets_np, probs_np, average="micro")
    except ValueError:
        pass

    per_class_auroc: List[float] = []
    per_class_auprc: List[float] = []
    for i in range(targets_np.shape[1]):
        y_true = targets_np[:, i]
        y_pred = probs_np[:, i]
        if y_true.max() == y_true.min():
            continue
        try:
            per_class_auroc.append(_safe_roc_auc(y_true, y_pred))
        except ValueError:
            pass
        try:
            per_class_auprc.append(_safe_avg_precision(y_true, y_pred))
        except ValueError:
            pass
    if per_class_auroc:
        metrics["macro_auroc"] = float(sum(per_class_auroc) / len(per_class_auroc))
    if per_class_auprc:
        metrics["macro_auprc"] = float(sum(per_class_auprc) / len(per_class_auprc))
    return metrics


def _safe_roc_auc(y_true, y_pred, average=None):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y_true, y_pred, average=average)


def _safe_avg_precision(y_true, y_pred, average=None):
    from sklearn.metrics import average_precision_score

    return average_precision_score(y_true, y_pred, average=average)


class ConceptDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    ids: List[str],
    train_frac: float,
    seed: int,
    split_map: Optional[Dict[str, str]] = None,
    train_split: str = "train",
    eval_split: str = "validate",
) -> Tuple[ConceptDataset, ConceptDataset]:
    import random

    if split_map:
        train_idx = [i for i, sid in enumerate(ids) if split_map.get(sid) == train_split]
        test_idx = [i for i, sid in enumerate(ids) if split_map.get(sid) == eval_split]
        if not train_idx or not test_idx:
            raise RuntimeError("Split CSV provided but produced empty train or eval set; check split labels.")
    else:
        if not 0.0 < train_frac < 1.0:
            raise ValueError("--train-fraction must be between 0 and 1.")
        indices = list(range(x.shape[0]))
        random.Random(seed).shuffle(indices)
        pivot = max(1, int(len(indices) * train_frac))
        train_idx = indices[:pivot]
        test_idx = indices[pivot:]
        if not test_idx:
            raise RuntimeError("Hold-out split is empty; decrease --train-fraction or increase data availability.")
    return ConceptDataset(x[train_idx], y[train_idx]), ConceptDataset(x[test_idx], y[test_idx])


# Default CheXpert 14 labels.
CHEXPERT_LABELS = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CBM label head from concept predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config file; CLI flags override config values.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="JSONL of concept predictions with fields: study_id, probs (list of floats).",
    )
    parser.add_argument(
        "--labels-csv",
        required=True,
        type=Path,
        help="CSV with study IDs and label columns (e.g., CheXpert labels).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["chexpert_plus", "mimic_cxr"],
        help="Optional dataset name; if set and --split-csv not provided, will use the registry split file when available.",
    )
    parser.add_argument(
        "--study-id-column",
        default="study_id",
        help="Column name in labels CSV that identifies the study.",
    )
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=None,
        help="Label columns to use from the CSV. Defaults to CheXpert14 or CheXpert5 when --chexpert5 is passed.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS / "cbm_label_head"),
        help="Directory to write metrics and checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Train split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Device string for torch.device.")
    parser.add_argument(
        "--uncertainty-mode",
        choices=("zero", "positive"),
        default="zero",
        help="How to handle -1 labels: map to 0 or to 1.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional CSV containing official splits (e.g., mimic-cxr-2.0.0-split.csv.gz).",
    )
    parser.add_argument(
        "--split-column",
        default="split",
        help="Column name in the split CSV that stores the split label.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Value in split-column to use for training data.",
    )
    parser.add_argument(
        "--eval-split",
        default="validate",
        help="Value in split-column to use for evaluation/hold-out data.",
    )
    parser.add_argument(
        "--chexpert5",
        action="store_true",
        help="Use the 5-label CheXpert subset (Enlarged Cardiomediastinum, Cardiomegaly, Edema, Consolidation, Pleural Effusion).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Stop if micro-F1 fails to improve for this many epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement in monitored metric to reset patience.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases (requires wandb package).",
    )
    parser.add_argument(
        "--wandb-project",
        default="medclip",
        help="W&B project name (used when --wandb is enabled).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional custom W&B run name.",
    )
    return parser.parse_args(argv)


def _load_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
    cfg = apply_env_overrides(cfg)
    # Merge CLI over config
    merged = {**cfg, **{k: v for k, v in vars(args).items() if v is not None}}
    args_dict = vars(args).copy()
    args_dict.update(merged)
    return argparse.Namespace(**args_dict)


def _map_label(value: Any, mode: str) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if val == -1.0:
        return 1.0 if mode == "positive" else 0.0
    return 1.0 if val > 0 else 0.0


def load_predictions(jsonl_path: Path) -> Tuple[List[str], torch.Tensor]:
    def _norm(s: str) -> str:
        # Strip known prefixes like "mimic_cxr:study" and keep digits.
        if s is None:
            return ""
        text = str(s)
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or text

    study_ids: List[str] = []
    rows: List[List[float]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            study_ids.append(_norm(rec["study_id"]))
            rows.append([float(x) for x in rec["probs"]])
    if not rows:
        raise RuntimeError(f"No predictions found in {jsonl_path}")
    return study_ids, torch.tensor(rows, dtype=torch.float32)


def load_split_map(split_csv: Path, split_col: str, id_col: str) -> Dict[str, str]:
    split_csv = split_csv.expanduser()
    if not split_csv.exists():
        raise RuntimeError(f"Split CSV not found: {split_csv}")
    df = pd.read_csv(split_csv, usecols=[id_col, split_col])
    if id_col not in df.columns or split_col not in df.columns:
        raise RuntimeError(f"Split CSV must contain columns '{id_col}' and '{split_col}'.")
    return {str(sid): str(split) for sid, split in zip(df[id_col], df[split_col])}


def load_labels(
    csv_path: Path,
    study_id_col: str,
    label_cols: Sequence[str],
    mode: str,
) -> Tuple[List[str], torch.Tensor]:
    df = pd.read_csv(csv_path)
    label_cols = list(label_cols)
    if study_id_col not in df.columns:
        raise RuntimeError(f"study-id column '{study_id_col}' not in labels CSV.")
    missing = [col for col in label_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Label columns missing from CSV: {missing}")
    ids = df[study_id_col].astype(str).tolist()
    ids = ["".join(ch for ch in sid if ch.isdigit()) or sid for sid in ids]
    labels = df[label_cols].map(lambda v: _map_label(v, mode)).astype(float).values
    return ids, torch.tensor(labels, dtype=torch.float32)


def train_main(argv: Optional[Sequence[str]] = None) -> None:
    """Train CBM label head from concept predictions."""
    args = _load_config(_parse_args(argv))
    set_seed(args.seed)

    # Resolve label columns default
    if args.label_columns is None:
        if args.chexpert5:
            args.label_columns = [
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pleural Effusion",
            ]
        else:
            args.label_columns = list(CHEXPERT_LABELS)

    output_dir = ensure_dir(Path(args.output_dir).expanduser())

    pred_ids, pred_tensor = load_predictions(args.predictions.expanduser())
    label_ids, label_tensor = load_labels(args.labels_csv.expanduser(), args.study_id_column, args.label_columns, args.uncertainty_mode)
    aligned_ids, x, y = align_data(pred_ids, pred_tensor, label_ids, label_tensor)

    split_map = None
    if args.split_csv:
        split_map = load_split_map(args.split_csv, args.split_column, args.study_id_column)
    elif args.dataset:
        # Optional convenience: attempt to use a registry split map if available.
        try:
            from src.datasets import registry_split_map  # type: ignore
            split_map, _ = registry_split_map(args.dataset, args.split_column, args.study_id_column)
        except Exception:
            split_map = None

    train_ds, test_ds = split_dataset(
        x,
        y,
        aligned_ids,
        args.train_fraction,
        args.seed,
        split_map=split_map,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device.lower() else "cpu")
    wandb_run = init_wandb(args, {"num_concepts": x.shape[1], "num_labels": len(args.label_columns)})
    trainer_cfg = CBMTrainerConfig(
        input_dim=x.shape[1],
        num_labels=len(args.label_columns),
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        wandb_run=wandb_run,
    )
    trainer = CBMTrainer(trainer_cfg)

    history, best_f1, best_state = trainer.train(train_loader, test_loader, epochs=args.epochs)
    if best_state:
        torch.save({"model": best_state, "args": vars(args)}, output_dir / "best_cbm_label_head.pt")

    metrics = {
        "history": history,
        "best_micro_f1": best_f1,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "label_columns": list(args.label_columns),
    }
    with (output_dir / "cbm_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[done] wrote metrics/checkpoint to {output_dir}")
    if wandb_run:
        wandb_run.summary["best_micro_f1"] = best_f1
        wandb_run.summary["train_samples"] = len(train_ds)
        wandb_run.summary["test_samples"] = len(test_ds)
        wandb_run.finish()


def eval_main(argv: Optional[Sequence[str]] = None) -> None:
    """Evaluate CBM label head (currently runs evaluation as part of training)."""
    print("[info] CBM evaluation is integrated into training; use 'train' mode.")
    train_main(argv)
