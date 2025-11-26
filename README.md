# MedCLIP Toolkit

Utilities and scripts built around the MedCLIP vision–language model for chest X-ray analysis. It includes the core `medclip` package, data utilities, training/eval scripts, and analysis notebooks.

**Key Features:**
- 🎯 **Unified entry points**: All training and evaluation through `scripts/train.py` and `scripts/eval.py`
- 📦 **Consolidated modules**: All trainer logic organized in `src/trainers/`
- 🔧 **Minimal overhead**: Shared utilities in `src/utils/` (seeding, logging, data alignment)
- 📊 **Built-in metrics**: Precision, recall, F1, AUROC, early stopping, W&B integration

## Quick Start

1. **Create a conda environment**
   ```bash
   conda create -n medclip python=3.10 -y
   conda activate medclip
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure dataset locations**

   Scripts expect chest X-ray datasets (MIMIC-CXR, CheXpert Plus, IU X-Ray) to live under a shared root. Set the environment variable before running any tooling:
   ```bash
   export MEDCLIP_DATASETS_ROOT=/path/to/your/datasets
   ```
   You can also provide dataset-specific overrides via `datasets.yaml` or command line flags depending on the script.

4. **Place artifacts/weights**

   All heavy artifacts now live under `data/` (with symlinks at the repo root for backward compatibility):
   - `data/pretrained/` – downloaded weights (symlinked from `pretrained/`)
   - `data/generated/` – concept banks, embeddings, heads (symlinked from `generated/`)
   - `data/outputs/` – cached features, CBM heads, probes (symlinked from `outputs/`)
   - `data/local_data/` – small helper CSVs
   - `data/wandb/`, `data/tmp_radgraph/` – run/cache folders

   You can keep using the legacy paths (`generated/`, `outputs/`, etc.); they now point into `data/`.

5. **Run training and evaluation**

   All training and evaluation tasks use unified entry points:

   ```bash
   # See available trainers
   python scripts/train.py --help
   python scripts/eval.py --help

   # Train a CBM label head from concept predictions
   python scripts/train.py cbm -- \
     --predictions /path/to/concept_predictions.jsonl \
     --labels-csv data/local_data/chexpert-5x200-val-meta.csv \
     --chexpert5 \
     --output-dir data/outputs/cbm_label_head_demo

   # Evaluate CBM
   python scripts/eval.py cbm -- \
     --predictions /path/to/concept_predictions.jsonl \
     --labels-csv data/local_data/chexpert-5x200-val-meta.csv

   # Train concept classifier (vision model -> concepts)
   python scripts/train.py concept-classifier -- \
     --dataset-name mimic_cxr \
     --concepts-path data/generated/concept_bank_sapbert_mimic/study_concepts.jsonl \
     --output-dir data/outputs/concept_classifier_demo

   # Evaluate concept classifier
   python scripts/eval.py concept-classifier -- \
     --checkpoint data/outputs/concept_classifier_demo/best_model.pt \
     --dataset-name mimic_cxr \
     --concepts-path data/generated/concept_bank_sapbert_mimic/study_concepts.jsonl

   # Train blackbox classifier (baseline)
   python scripts/train.py blackbox-classifier -- \
     --dataset-name mimic_cxr \
     --labels-csv /path/to/labels.csv \
     --image-root /path/to/images

   # Fine-tune CLIP backbone (uses shared CLIP utilities)
   python scripts/clip/finetune_clip.py \
     --dataset mimic_cxr \
     --labels-csv /path/to/labels.csv \
     --image-root /path/to/images \
     --clip-checkpoint /path/to/r50_m.tar \
     --output-dir data/outputs/clip_finetune_example
   ```

   **Note:** Use `--` to separate the trainer selection from trainer-specific arguments.

## Repository Structure

### Core Directories

- **`medclip/`** – Core model package (models, datasets, trainers, utilities)
- **`src/`** – Shared utilities:
  - `src/trainers/` – Consolidated trainer modules (CBM, concept classifier, blackbox classifier)
  - `src/utils/` – Common utilities (random seeding, W&B init, data alignment, ID normalization)
  - Dataset iterators, text/embed helpers, ontology tools
- **`scripts/`** – Entry points and utilities:
  - `train.py` – Unified training entry point (cbm, concept-classifier, blackbox-classifier)
  - `eval.py` – Unified evaluation entry point (cbm, concept-classifier, blackbox-classifier)
  - Concept bank utilities (build, curate, distill, report generation)
  - `clip/` – CLIP fine-tuning scripts
- **`analysis/`** – Jupyter notebooks and analysis helpers
- **`cfg/`** – Configuration files (concept bank templates, YAML/JSON configs)
- **`templates/`** – LaTeX/HTML templates for reporting
- **`reports/`** – Generated HTML reports
- **`data/`** – Artifacts and caches:
  - `data/pretrained/` – Downloaded model weights
  - `data/generated/` – Concept banks, embeddings
  - `data/outputs/` – Training outputs, cached features, checkpoints
  - `data/local_data/` – Helper CSVs and metadata
  - `data/wandb/`, `data/tmp_radgraph/` – Run logs and temporary files

### Legacy and Other

- **`legacy/`** – Archived scripts/configs from earlier iterations
- **`ontology-concept-distillation/`** – Separate research prototype
- **`docs/`** – Documentation notes

### Architecture Overview

**Training/Evaluation Flow:**
```
scripts/train.py  →  src/trainers/{cbm,concept_classifier,blackbox_classifier}.py
scripts/eval.py   →  src/trainers/{cbm,concept_classifier,blackbox_classifier}.py
```

Each trainer module contains:
- Data loading and preprocessing
- Training loops with early stopping
- Evaluation metrics (precision, recall, F1, AUROC, etc.)
- CLI argument parsing
- W&B logging integration

**Shared Utilities:**
- `src/utils/` – Random seeding, config sanitization, W&B initialization, data alignment
- `src/trainers/clip.py` – CLIP backbone utilities (feature caching, weight loading)

No formal test suite is provided yet; add targeted checks around new scripts or training workflows as needed.

## Environment Variables

- `MEDCLIP_DATASETS_ROOT` – overrides the default dataset root path
- `UMLS_API_KEY` – API key for the UMLS REST client (optional unless you interact with UMLS services)

## License

Refer to the original MedCLIP project for licensing details. This toolkit is provided without warranty; use it responsibly when handling clinical data.
