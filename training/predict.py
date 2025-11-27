#!/usr/bin/env python3
"""Run inference for a trained classifier or CBM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_cbm import CBMHead
from training.train_classifier import Classifier


class TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx]


def _load_tensor(path: Path) -> torch.Tensor:
    path = path.expanduser()
    if path.suffix.lower() in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu")
        return loaded if isinstance(loaded, torch.Tensor) else torch.tensor(loaded)
    if path.suffix.lower() in {".npz", ".npy"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr["arr_0"]
        return torch.tensor(arr)
    raise ValueError(f"Unsupported tensor format: {path}")


def _infer_classifier_shapes(state_dict: dict) -> tuple[int, int]:
    first = next(iter(state_dict.values()))
    output_dim, input_dim = first.shape
    return input_dim, output_dim


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for classifiers or CBM.")
    parser.add_argument("--target", choices=["cbm", "concept", "label"], required=True)
    parser.add_argument("--features", required=True, type=Path, help="Feature tensor (.pt/.npz) for inference.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Optional JSONL to write probabilities.")
    parser.add_argument("--activation", choices=["none", "relu", "tanh"], default="none", help="Activation used in the classifier.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = _load_tensor(args.features).float()
    loader = DataLoader(TensorDataset(feats), batch_size=args.batch_size)

    state = torch.load(args.checkpoint, map_location=device)
    state_dict = state.get("model", state) if isinstance(state, dict) else state

    if args.target == "cbm":
        input_dim = feats.shape[1]
        output_dim = list(state_dict.values())[-1].shape[0]
        model = CBMHead(input_dim, output_dim, dropout=0.0).to(device)
    else:
        input_dim, output_dim = _infer_classifier_shapes(state_dict)
        model = Classifier(input_dim, output_dim, activation=args.activation).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu())
    all_probs = torch.cat(probs)

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for i, row in enumerate(all_probs.tolist()):
                handle.write(json.dumps({"index": i, "probs": row}) + "\n")
    else:
        out_path = args.checkpoint.with_suffix(".probs.pt")
        torch.save(all_probs, out_path)
        print(f"Wrote probabilities to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
