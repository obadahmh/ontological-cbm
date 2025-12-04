"""Path configuration loader.

Machine-specific paths are defined in YAML:
  - paths.example.yml (tracked sample)
  - paths.local.yml (gitignored, user-specific)

Environment variables override YAML when present:
  DATA_ROOT, OUTPUTS_ROOT, UMLS_ROOT
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError as exc:
    raise ImportError("PyYAML is required to load path configuration.") from exc

ROOT = Path(__file__).resolve().parent.parent


def _load_yaml() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for candidate in (ROOT / "cfg" / "paths.example.yml", ROOT / "cfg" / "paths.yml"):
        if candidate.exists():
            loaded = yaml.safe_load(candidate.read_text()) or {}
            if isinstance(loaded, dict):
                cfg.update(loaded)
    return cfg


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, {})
    return cur if cur else default


_CFG = _load_yaml()

# Roots (env overrides YAML)
DATA_ROOT = Path(os.getenv("DATA_ROOT", _get(_CFG, "data", "root", default=(Path.home() / "datasets")))).expanduser().resolve()
OUTPUTS = Path(os.getenv("OUTPUTS_ROOT", _get(_CFG, "outputs", "root", default=ROOT / "outputs"))).expanduser()

# MIMIC-CXR
MIMIC_JPG_META_CSV = Path(_get(_CFG, "data", "mimic_jpg_meta_csv", default=DATA_ROOT / "mimic-cxr-jpg" / "2.1.0" / "mimic-cxr-2.0.0-metadata.csv.gz"))
MIMIC_CHEXP_CSV = Path(_get(_CFG, "data", "mimic_chexp_csv", default=DATA_ROOT / "mimic-cxr-jpg" / "2.1.0" / "mimic-cxr-2.0.0-chexpert.csv.gz"))
MIMIC_JPG_ROOT = Path(_get(_CFG, "data", "mimic_jpg_root", default=DATA_ROOT / "mimic-cxr-jpg" / "2.1.0" / "files"))
MIMIC_JPG_SPLIT_CSV = Path(_get(_CFG, "data", "mimic_jpg_split_csv", default=DATA_ROOT / "mimic-cxr-jpg" / "2.1.0" / "mimic-cxr-2.0.0-split.csv.gz"))

# CheXpert Plus
CHEXPERT_PLUS_CSV = Path(_get(_CFG, "data", "chexpert_plus_csv", default=DATA_ROOT / "CheXpert" / "df_chexpert_plus_240401.csv"))
CHEXPERT_PLUS_ROOT = Path(_get(_CFG, "data", "chexpert_plus_root", default=DATA_ROOT / "CheXpert" / "PNG"))
CHEXPERT_PLUS_PATH_COL = _get(_CFG, "data", "chexpert_plus_path_col", default="path_to_image")
CHEXPERT_PLUS_IMPRESS_COL = _get(_CFG, "data", "chexpert_plus_impress_col", default="section_impression")
CHEXPERT_PLUS_FINDINGS_COL = _get(_CFG, "data", "chexpert_plus_findings_col", default="section_findings")
CHEXPERT_PLUS_STUDY_ID_COL = _get(_CFG, "data", "chexpert_plus_study_id_col", default="study_id")

# UMLS / SapBERT
UMLS_ROOT = Path(os.getenv("UMLS_ROOT", _get(_CFG, "umls", "root", default=""))).expanduser()
UMLS_STRINGS_TSV = Path(_get(_CFG, "umls", "strings_tsv", default=""))
UMLS_STY_TSV = Path(_get(_CFG, "umls", "semantic_types_tsv", default=""))
UMLS_FAISS_INDEX = Path(_get(_CFG, "umls", "faiss_index", default=""))
UMLS_SAPBERT = Path(_get(_CFG, "umls", "sapbert_checkpoint", default=""))

# Pretrained weights (e.g., MedCLIP cache)
PRETRAINED_DIR = (ROOT / "pretrained").expanduser()
MEDCLIP_RESNET_DIR = PRETRAINED_DIR / "medclip-resnet"
MEDCLIP_VIT_DIR = PRETRAINED_DIR / "medclip-vit"
