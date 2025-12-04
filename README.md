# MedCLIP Toolkit

This repository bundles small, self-contained scripts around MedCLIP-style experiments on chest X-ray data. It focuses on concept-based models (CBMs) built from UMLS/SNOMED concepts and simple linear baselines trained on pre-computed features, so that most heavy lifting (feature extraction, entity linking) stays outside the training loop.

At a high level the code supports:

- building UMLS-linked concept banks from radiology reports (ontology–concept-distillation pipeline),
- pruning and vectorizing concept banks into tensors suitable for CBM training,
- caching image features from CLIP/MedCLIP backbones, and
- training, evaluating, and running inference for lightweight heads on top of these features.

## Installation

Clone the repo and install the Python dependencies:

```bash
pip install -r requirements.txt
```

The concept-extraction pipeline expects additional resources that are not distributed here (UMLS release, SapBERT checkpoints, RadGraph model weights, etc.). Paths to those resources are configured via `cfg/paths.yml` and `cfg/umls_sapbert.yml` (see below).

## Configuring paths

Machine-specific filesystem paths live in YAML, with optional environment-variable overrides:

- Copy `cfg/paths.example.yml` to `cfg/paths.yml` (or `paths.local.yml`) and edit it to point at your local datasets and outputs.
- The loader in `lib/constants.py` exposes derived paths such as `DATA_ROOT`, `OUTPUTS`, `MIMIC_JPG_ROOT`, and UMLS/SapBERT locations.
- Environment variables `DATA_ROOT`, `OUTPUTS_ROOT`, and `UMLS_ROOT` override the corresponding entries in the YAML if set.

The UMLS/SapBERT-related keys in `cfg/paths.yml` and the semantic filters in `cfg/umls_sapbert.yml` are used by the concept-extraction scripts.

## Typical workflows

### 1. Build a concept bank from reports

For a dataset with radiology reports (e.g. CheXpert+ or MIMIC-CXR), you can construct a concept bank using the ontology–concept-distillation pipeline:

```bash
python concept_extraction/build_concept_bank.py \
  --dataset chexpert_plus \
  --output-dir outputs/snomed_mimic_bank
```

This script runs RadGraph + SapBERT-based entity linking over report text, then writes:

- `concept_inventory.json` (canonical concept inventory),
- `study_concepts.jsonl` (per-study concepts), and
- `concept_bank.meta.json` (summary metadata).

Alternatively, `concept_extraction/convert_reports.py` exposes a similar pipeline for an explicit CSV of reports via `--csv-path`.

### 2. Prune the concept bank and build CBM tensors

Given a concept bank and per-study concepts, you can prune to a CBM-friendly subset and optionally emit a dense concept label matrix:

```bash
python concept_extraction/concept_bank_pruning.py \
  --inventory-path outputs/snomed_mimic_bank/concept_inventory.json \
  --study-concepts-path outputs/snomed_mimic_bank/study_concepts.jsonl \
  --output-dir outputs/snomed_mimic_bank_pruned \
  --emit-label-matrix \
  --label-mode binary
```

The script applies frequency, assertion, category, and name-based filters, and can:

- write `concept_inventory.pruned.json` and `concept_bank.pruned.meta.json`,
- save a `concept_index.json` (mapping concepts to indices), and
- optionally produce `labels.npz` with shape `[num_studies, num_concepts]` for CBM training.

### 3. Cache image features (CLIP / MedCLIP)

To keep training loops simple, the repository assumes image features are pre-computed. Utilities in `lib/clip.py` and `lib/medclip_utils.py` provide helpers for:

- loading CLIP checkpoints and extracting backbone weights (`load_clip_state_dict`, `clip_backbone_weights`), and
- loading MedCLIP models via the `medclip` library (`load_medclip`), then caching features with `cache_features`.

The resulting feature tensors (and matching label tensors) can be stored as `.pt` or `.npz` files and fed directly into the training scripts below.

### 4. Train CBMs and baseline classifiers

With precomputed tensors, training reduces to short CLI calls.

**CBM label head from concept probabilities + label CSV**

```bash
python training/train_cbm.py \
  --predictions preds.jsonl \
  --labels-csv labels.csv \
  --label-columns label1 label2 \
  --epochs 1
```

Here `preds.jsonl` contains rows like `{"study_id": "...", "probs": [...]}`; `labels.csv` contains a `study_id` column plus one or more label columns. The script splits into train/validation, optimizes a small MLP (`CBMHead`), and writes metrics and the best checkpoint to `outputs/cbm` by default (configurable via `--output-dir`).

**Concept or label classifier on feature tensors**

```bash
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

Features and labels are `.pt`/`.npz` tensors shaped `[num_samples, dim]` and `[num_samples, num_labels]`, respectively. The script trains a single linear head with optional activation and writes metrics and checkpoints under `outputs/concept_classifier` or `outputs/label_classifier`.

### 5. Evaluate and run inference

Given a trained head, you can evaluate or generate predictions:

```bash
# Evaluate a CBM head
python training/eval.py \
  cbm \
  --checkpoint outputs/cbm/best_model.pt \
  --predictions preds.jsonl \
  --labels-csv labels.csv \
  --label-columns label1 label2

# Evaluate a feature classifier
python training/eval.py \
  concept \
  --checkpoint outputs/concept_classifier/best_model.pt \
  --features features.pt \
  --labels labels.pt
```

For forward-only inference on new feature tensors:

```bash
python training/predict.py \
  --target concept \
  --features new_features.pt \
  --checkpoint outputs/concept_classifier/best_model.pt \
  --output-jsonl outputs/preds.jsonl
```

## Quickstart (tensors only)

If you only care about lightweight model training on precomputed tensors:

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train a head from concepts or features as shown above.
3. Inspect metrics and checkpoints in `outputs/...`.

Each trainer writes metrics and the best checkpoint into `outputs/...` by default (override with `--output-dir`).

## Repository layout

- `concept_extraction/` – concept extraction scripts, ontology–concept-distillation pipeline, and helpers
  (`build_concept_bank.py`, `concept_bank_pruning.py`, `convert_reports.py`, `render_report.py`, etc.).
- `training/` – flat training and evaluation scripts (`train_cbm.py`, `train_classifier.py`, `eval.py`, `predict.py`).
- `lib/` – shared utilities for paths, data alignment, metrics, CLIP/MedCLIP integration, and identifiers.
- `cfg/` – path and UMLS/SapBERT configuration (`paths.example.yml`, `paths.yml`, `umls_sapbert.yml`).
- `notebooks/` – exploratory notebooks for analysing concept banks and CBMs
  (`concept_bank_analysis.ipynb`, `cbm_analysis.ipynb`, `faiss_vector_explorer.ipynb`, etc.).
- `templates/` – Jinja2 HTML templates for concept-level report visualisation.
- `sample_reports/` – small sample of report data with linked concepts, useful for inspection and debugging.
- `outputs/` – default location for generated artifacts (concept banks, cached features, checkpoints, reports).

## Testing

For a quick smoke check of the training scripts:

```bash
pytest
```

The tests in `tests/test_smoke.py` exercise the CBM and feature-classifier training loops on tiny synthetic inputs.
