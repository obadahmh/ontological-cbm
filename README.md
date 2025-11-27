# MedCLIP Toolkit

Small, self-contained training loops for MedCLIP experiments. The trainers operate on pre-computed tensors or simple CSV/JSON files so the repo stays lightweight and easy to read.

## Quickstart

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Run a trainer:
   ```bash
   # CBM label head from concept predictions + label CSV
   python training/train_cbm.py \
     --predictions preds.jsonl \
     --labels-csv labels.csv \
     --label-columns label1 label2 \
     --epochs 1

   # Concept classifier on feature + label tensors
   python training/train_classifier.py \
     --mode concept \
     --features features.pt \
     --labels labels.pt \
     --epochs 1

   # Direct label classifier baseline
   python training/train_classifier.py \
     --mode label \
     --features features.pt \
     --labels labels.pt \
     --epochs 1
   ```

Each trainer writes metrics and the best checkpoint into `outputs/...` (configurable via `--output-dir`).

## Data expectations
- **CBM**: `preds.jsonl` with `{"study_id": "...", "probs": [...]}` rows and a CSV of labels with `study_id` plus label columns.
- **Concept/blackbox classifiers**: `.pt` or `.npz` tensors shaped `[num_samples, dim]` for features and `[num_samples, num_labels]` for labels.

## Layout
- `training/` – flat training scripts (`train_cbm.py`, `train_classifier.py`, `eval.py`, `predict.py`).
- `concept-extraction/` – concept extraction scripts and helpers (`convert_reports.py`, `build_concept_bank.py`, `render_report.py`, etc.).
- `lib/` – shared helpers (data seeding/alignment, metrics, identifiers, CLIP utilities).

## Testing
Run the lightweight smoke tests:
```bash
pytest
```
