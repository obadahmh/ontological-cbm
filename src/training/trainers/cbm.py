"""CBM label head trainer - self contained and minimal."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.utils import align_data, set_seed
from src.training.utils.metrics import compute_metrics

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


class CBMHead(nn.Module):
    """Two-layer MLP head for concept-to-label prediction."""

    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConceptDataset(Dataset):
    """Dataset wrapper for concept tensors and label tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def split_data(x: torch.Tensor, y: torch.Tensor, train_fraction: float, seed: int) -> Tuple[Dataset, Dataset]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=generator)
    pivot = max(1, int(len(indices) * train_fraction))
    train_idx = indices[:pivot]
    val_idx = indices[pivot:]
    if len(val_idx) == 0:
        raise RuntimeError("Validation split is empty; reduce train_fraction or provide more samples.")
    return ConceptDataset(x[train_idx], y[train_idx]), ConceptDataset(x[val_idx], y[val_idx])


def load_predictions(path: Path) -> Tuple[List[str], torch.Tensor]:
    """Load concept probabilities from a JSONL file."""
    ids: List[str] = []
    rows: List[torch.Tensor] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            ids.append(str(record["study_id"]))
            rows.append(torch.tensor(record["probs"], dtype=torch.float32))
    if not rows:
        raise RuntimeError(f"No prediction rows found in {path}")
    return ids, torch.stack(rows)


def load_labels(
    path: Path,
    study_id_column: str,
    label_columns: Sequence[str],
    uncertainty_mode: str = "zero",
) -> Tuple[List[str], torch.Tensor]:
    """Load study labels from a CSV file."""
    rows: List[torch.Tensor] = []
    ids: List[str] = []

    def _normalize_label(raw: str) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if value == -1.0:
            return 1.0 if uncertainty_mode == "positive" else 0.0
        return 1.0 if value > 0 else 0.0

    with path.expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in label_columns if col not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"Missing label columns in {path}: {missing}")
        for row in reader:
            ids.append(str(row[study_id_column]))
            rows.append(torch.tensor([_normalize_label(row[col]) for col in label_columns], dtype=torch.float32))
    if not rows:
        raise RuntimeError(f"No label rows found in {path}")
    return ids, torch.stack(rows)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    threshold: float,
) -> Tuple[Optional[dict], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_state: Optional[dict] = None
    best_f1 = -1.0
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        probs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                probs.append(torch.sigmoid(model(x)).cpu())
                targets.append(y)
        val_probs = torch.cat(probs)
        val_targets = torch.cat(targets)
        metrics = compute_metrics(val_probs, val_targets, threshold=threshold)
        mean_loss = running_loss / max(len(train_loader), 1)
        print(f"[{epoch:03d}] loss={mean_loss:.4f} f1={metrics['micro_f1']:.4f}")

        if metrics["micro_f1"] > best_f1 + 1e-4:
            best_f1 = metrics["micro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_state, best_f1


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CBM label head.")
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL of concept predictions.")
    parser.add_argument("--labels-csv", required=True, type=Path, help="CSV with study labels.")
    parser.add_argument("--study-id-column", default="study_id", help="Column name for study identifiers.")
    parser.add_argument("--label-columns", nargs="+", default=None, help="Label columns to use; defaults to CheXpert14.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of data for training.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for metrics.")
    parser.add_argument("--uncertainty-mode", choices=("zero", "positive"), default="zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cbm"))
    return parser.parse_args(argv)


def train_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    set_seed(args.seed)

    label_columns = args.label_columns or list(CHEXPERT_LABELS)
    pred_ids, pred_tensor = load_predictions(args.predictions)
    label_ids, label_tensor = load_labels(args.labels_csv, args.study_id_column, label_columns, args.uncertainty_mode)
    _, x, y = align_data(pred_ids, pred_tensor, label_ids, label_tensor)

    train_ds, val_ds = split_data(x, y, args.train_fraction, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBMHead(x.shape[1], y.shape[1], dropout=args.dropout).to(device)
    best_state, best_f1 = train(model, train_loader, val_loader, args.epochs, args.lr, device, args.threshold)

    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, args.output_dir / "best_model.pt")
    with (args.output_dir / "cbm_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_micro_f1": best_f1,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "label_columns": list(label_columns),
            },
            handle,
            indent=2,
        )
    print(f"Best F1: {best_f1:.4f}")


def eval_main(argv: Optional[Sequence[str]] = None) -> None:
    """Evaluation is integrated into training for this minimal version."""
    train_main(argv)


if __name__ == "__main__":
    train_main()
