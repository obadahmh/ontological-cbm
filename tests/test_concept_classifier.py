from pathlib import Path

import torch

from training.train_classifier import train_main


def test_concept_classifier_smoke(tmp_path: Path):
    features = torch.randn(4, 3)
    labels = torch.zeros(4, 2)
    labels[:2, 0] = 1.0
    labels[2:, 1] = 1.0

    feat_path = tmp_path / "features.pt"
    labels_path = tmp_path / "labels.pt"
    torch.save(features, feat_path)
    torch.save(labels, labels_path)

    output_dir = tmp_path / "out"
    train_main(
        [
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

    assert (output_dir / "concept_classifier_metrics.json").exists()


def test_label_classifier_smoke(tmp_path: Path):
    features = torch.randn(4, 3)
    labels = torch.zeros(4, 2)
    labels[:2, 0] = 1.0
    labels[2:, 1] = 1.0

    feat_path = tmp_path / "features.pt"
    labels_path = tmp_path / "labels.pt"
    torch.save(features, feat_path)
    torch.save(labels, labels_path)

    output_dir = tmp_path / "label_out"
    train_main(
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
