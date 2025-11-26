import os
from pathlib import Path


# ---- set this to wherever your datasets actually live on THIS machine ----
def _resolve_datasets_root() -> Path:
    """Resolve dataset root from environment variables, with a sensible default."""
    candidate = os.getenv("DATASETS_ROOT")
    if candidate:
        return Path(candidate).expanduser().resolve()
    # Fallback: assume datasets are under ~/datasets
    return (Path.home() / "datasets").expanduser().resolve()


DATASETS_ROOT = _resolve_datasets_root()

# ------------ MIMIC-CXR ------------
# Preferred (cleanest): sectioned CSV with columns [subject_id, study_id, impression, findings]
MIMIC_SECTIONED_CSV = DATASETS_ROOT / "mimic-cxr-reports" / "mimic_cxr_sectioned.csv.gz"

# Images (JPG derivative)
# Adjust paths here to match your local MIMIC-CXR-JPG layout.
MIMIC_JPG_META_CSV  = DATASETS_ROOT / "mimic-cxr-jpg" / "2.1.0" / "mimic-cxr-2.0.0-metadata.csv.gz"
MIMIC_CHEXP_CSV     = DATASETS_ROOT / "mimic-cxr-jpg" / "2.1.0" / "mimic-cxr-2.0.0-chexpert.csv.gz"
MIMIC_JPG_ROOT      = DATASETS_ROOT / "mimic-cxr-jpg" / "2.1.0" / "files"
MIMIC_JPG_SPLIT_CSV = DATASETS_ROOT / "mimic-cxr-jpg" / "2.1.0" / "mimic-cxr-2.0.0-split.csv.gz"

# (Optional) if you DON'T have the sectioned CSV and want to parse raw TXT later:
# MIMIC_REPORTS_ROOT = DATASETS_ROOT / "mimic-cxr-reports" / "files"

# ------------ CheXpert Plus ------------
# Main CSV that has report sections and image paths
CHEXPERT_PLUS_CSV   = DATASETS_ROOT / "CheXpert" / "df_chexpert_plus_240401.csv"
CHEXPERT_PLUS_ROOT = DATASETS_ROOT / "CheXpert" / "PNG"

# (optional overrides if your CSV uses different headers)
CHEXPERT_PLUS_PATH_COL      = "path_to_image"         # or "image_path" / "path"
CHEXPERT_PLUS_IMPRESS_COL   = "section_impression"    # or "impression"
CHEXPERT_PLUS_FINDINGS_COL  = "section_findings"      # or "findings"
CHEXPERT_PLUS_STUDY_ID_COL  = "study_id"              # leave None to derive from path
