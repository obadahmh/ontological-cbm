# MedCLIP Toolkit (Slim)

Small, self-contained training loops for MedCLIP experiments. The trainers operate on pre-computed tensors or simple CSV/JSON files so the repo stays lightweight and easy to read.

## Quickstart

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Run a trainer (use `--` to forward args after the subcommand):
   ```bash
   # CBM label head from concept predictions + label CSV
   python scripts/train_cli.py cbm -- \
     --predictions preds.jsonl \
     --labels-csv labels.csv \
     --label-columns label1 label2 \
     --epochs 1

   # Concept classifier on feature + label tensors
   python scripts/train_cli.py concept-classifier -- \
     --features features.pt \
     --labels labels.pt \
     --epochs 1

   # Direct label classifier baseline
   python scripts/train_cli.py blackbox-classifier -- \
     --features features.pt \
     --labels labels.pt \
     --epochs 1
   ```

Each trainer writes metrics and the best checkpoint into `outputs/...` (configurable via `--output-dir`).

## Data expectations
- **CBM**: `preds.jsonl` with `{"study_id": "...", "probs": [...]}` rows and a CSV of labels with `study_id` plus label columns.
- **Concept/blackbox classifiers**: `.pt` or `.npz` tensors shaped `[num_samples, dim]` for features and `[num_samples, num_labels]` for labels.

## Layout
- `scripts/train_cli.py`, `scripts/eval_cli.py` – entry points dispatching to trainers.
- `src/training/` – training stack: trainers (`cbm.py`, `feature_classifier.py`), and training utils.
- `src/extraction/` – concept extraction pipeline: concepts/, dataset iteration helpers, per-study utilities, and ID normalization.

## Testing
Run the lightweight smoke tests:
```bash
pytest
```
