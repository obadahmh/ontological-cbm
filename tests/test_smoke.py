import json
from pathlib import Path

import torch

from training.train_cbm import train_main as train_cbm
from training.train_classifier import train_main as train_classifier


def test_cbm_smoke(tmp_path):
    preds = [
        {"study_id": "a", "probs": [0.1, 0.9]},
        {"study_id": "b", "probs": [0.8, 0.2]},
    ]
    pred_path = tmp_path / "preds.jsonl"
    pred_path.write_text("\n".join(json.dumps(row) for row in preds), encoding="utf-8")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("study_id,label1,label2\n" "a,1,0\n" "b,0,1\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    train_cbm(
        [
            "--predictions",
            str(pred_path),
            "--labels-csv",
            str(labels_csv),
            "--label-columns",
            "label1",
            "label2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--train-fraction",
            "0.5",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "cbm_metrics.json").exists()


def test_classifier_smoke(tmp_path: Path):
    features = torch.randn(4, 3)
    labels = torch.zeros(4, 2)
    labels[:2, 0] = 1.0
    labels[2:, 1] = 1.0

    feat_path = tmp_path / "features.pt"
    labels_path = tmp_path / "labels.pt"
    torch.save(features, feat_path)
    torch.save(labels, labels_path)

    output_dir = tmp_path / "out"
    train_classifier(
        [
            "--mode",
            "label",
            "--features",
            str(feat_path),
            "--labels",
            str(labels_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--train-fraction",
            "0.5",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "label_classifier_metrics.json").exists()
