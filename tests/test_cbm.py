import json
from pathlib import Path

from src.training.trainers.cbm import train_main


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
    train_main(
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
