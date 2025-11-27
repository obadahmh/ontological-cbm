#!/usr/bin/env python3
"""Train a simple linear classifier from feature tensors (concepts or labels)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.data import set_seed
from lib.metrics import compute_metrics


class TensorDataset(Dataset):
    """Tiny dataset wrapper for paired tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class Classifier(nn.Module):
    """Single linear head with optional activation on inputs."""

    def __init__(self, input_dim: int, num_outputs: int, activation: str = "none"):
        super().__init__()
        acts = {
            "none": nn.Identity(),
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
        }
        if activation not in acts:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = acts[activation]
        self.fc = nn.Linear(input_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        return self.fc(x)


def _load_tensor(path: Path) -> torch.Tensor:
    path = path.expanduser()
    if path.suffix.lower() in {".pt", ".pth"}:
        loaded = torch.load(path)
        return loaded if isinstance(loaded, torch.Tensor) else torch.tensor(loaded)
    if path.suffix.lower() in {".npz", ".npy"}:
        array = np.load(path)
        if isinstance(array, np.lib.npyio.NpzFile):
            if "arr_0" not in array:
                raise RuntimeError(f"{path} must contain an 'arr_0' entry.")
            array = array["arr_0"]
        return torch.tensor(array)
    raise ValueError(f"Unsupported tensor format: {path}")


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
    return TensorDataset(x[train_idx], y[train_idx]), TensorDataset(x[val_idx], y[val_idx])


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
        probs = []
        targets = []
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


def _mode_defaults(mode: str) -> Dict[str, object]:
    if mode == "label":
        return {
            "epochs": 5,
            "batch_size": 16,
            "lr": 2e-4,
            "output_dir": Path("outputs/label_classifier"),
        }
    return {
        "epochs": 10,
        "batch_size": 8,
        "lr": 1e-3,
        "output_dir": Path("outputs/concept_classifier"),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight feature classifier.")
    parser.add_argument("--mode", choices=["concept", "label"], default="concept", help="Classifier mode controls defaults.")
    parser.add_argument("--features", required=True, type=Path, help="Path to feature tensor (.pt or .npz).")
    parser.add_argument("--labels", required=True, type=Path, help="Path to label tensor (.pt or .npz).")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of data for training.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--activation", choices=["none", "relu", "tanh"], default="none")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def train_main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    defaults = _mode_defaults(args.mode)

    for key in ("epochs", "batch_size", "lr", "output_dir"):
        if getattr(args, key) is None:
            value = defaults[key]
            setattr(args, key, value if key != "output_dir" else Path(value))
    args.output_dir = Path(args.output_dir).expanduser()

    set_seed(args.seed)

    features = _load_tensor(args.features).float()
    labels = _load_tensor(args.labels).float()
    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError("features and labels must be 2D tensors.")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows.")

    train_ds, val_ds = split_data(features, labels, args.train_fraction, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(features.shape[1], labels.shape[1], activation=args.activation).to(device)
    best_state, best_f1 = train(model, train_loader, val_loader, args.epochs, args.lr, device, args.threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, args.output_dir / "best_model.pt")
    metrics_name = "concept_classifier_metrics.json" if args.mode == "concept" else "label_classifier_metrics.json"
    with (args.output_dir / metrics_name).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_micro_f1": best_f1,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
            },
            handle,
            indent=2,
        )
    print(f"Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train_main(sys.argv[1:])
