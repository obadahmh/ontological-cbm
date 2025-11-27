#!/usr/bin/env python3
"""Evaluate saved models (CBM or feature classifier)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.data import set_seed
from lib.metrics import compute_metrics
from training.train_cbm import CBMHead, ConceptDataset, align_data, load_labels, load_predictions
from training.train_classifier import Classifier, TensorDataset, _load_tensor

def _eval_cbm(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    label_columns = args.label_columns or []
    pred_ids, pred_tensor = load_predictions(args.predictions)
    label_ids, label_tensor = load_labels(args.labels_csv, args.study_id_column, label_columns, args.uncertainty_mode)
    _, x, y = align_data(pred_ids, pred_tensor, label_ids, label_tensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBMHead(x.shape[1], y.shape[1], dropout=args.dropout).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()
    loader = DataLoader(ConceptDataset(x, y), batch_size=args.batch_size)
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            probs.append(torch.sigmoid(model(features)).cpu())
            targets.append(labels)
    val_probs = torch.cat(probs)
    val_targets = torch.cat(targets)
    return compute_metrics(val_probs, val_targets, threshold=args.threshold)


def _eval_classifier(args: argparse.Namespace) -> Dict[str, float]:
    defaults = {"concept": {"batch_size": 8, "lr": 1e-3}, "label": {"batch_size": 16, "lr": 2e-4}}
    batch_size = args.batch_size or defaults[args.mode]["batch_size"]
    set_seed(args.seed)
    features = _load_tensor(args.features).float()
    labels = _load_tensor(args.labels).float()
    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError("features and labels must be 2D tensors.")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    first_weight = next(iter(state_dict.values()))
    output_dim, input_dim = first_weight.shape
    model = Classifier(input_dim, output_dim, activation=args.activation).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs.append(torch.sigmoid(model(x)).cpu())
            targets.append(y)
    all_probs = torch.cat(probs)
    all_targets = torch.cat(targets)
    return compute_metrics(all_probs, all_targets, threshold=args.threshold)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved models.")
    parser.add_argument("target", choices=["cbm", "concept", "label"], help="Model type to evaluate.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for rebuilding the model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation", choices=["none", "relu", "tanh"], default="none")

    # CBM-specific
    parser.add_argument("--predictions", type=Path, help="JSONL of concept predictions (CBM).")
    parser.add_argument("--labels-csv", type=Path, help="CSV with study labels (CBM).")
    parser.add_argument("--study-id-column", default="study_id", help="Column name for study identifiers.")
    parser.add_argument("--label-columns", nargs="+", default=None, help="Label columns to use.")
    parser.add_argument("--uncertainty-mode", choices=("zero", "positive"), default="zero")

    # Classifier-specific
    parser.add_argument("--features", type=Path, help="Feature tensor (.pt/.npz).")
    parser.add_argument("--labels", type=Path, help="Label tensor (.pt/.npz).")
    parser.add_argument("--mode", choices=["concept", "label"], default="concept")

    parsed = parser.parse_args(argv)

    if parsed.target == "cbm":
        if not parsed.predictions or not parsed.labels_csv or not parsed.label_columns:
            raise SystemExit("CBM eval requires --predictions, --labels-csv, and --label-columns.")
        metrics = _eval_cbm(parsed)
        out_path = Path(parsed.checkpoint).with_suffix(".cbm_eval.json")
    else:
        if not parsed.features or not parsed.labels:
            raise SystemExit("Classifier eval requires --features and --labels.")
        metrics = _eval_classifier(parsed)
        suffix = ".concept_eval.json" if parsed.target == "concept" else ".label_eval.json"
        out_path = Path(parsed.checkpoint).with_suffix(suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
