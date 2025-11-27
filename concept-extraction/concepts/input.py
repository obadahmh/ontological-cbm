"""Shared input utilities for concept extraction pipelines."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

try:
    from radgraph import utils as radgraph_utils  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    radgraph_utils = None

from concept_extraction.per_study import make_study_key

DEFAULT_TEXT_COLUMN = "section_impression"
DERIVED_PATIENT_COLUMN = "__derived_patient_id"
DERIVED_STUDY_COLUMN = "__derived_study_id"

DATASET_PRESETS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "chexpert_plus": {
        "text": ("section_impression", "section_summary", "section_findings", "report", "impression"),
        "patient": (DERIVED_PATIENT_COLUMN, "deid_patient_id", "patient_id", "subject_id"),
        "study": (DERIVED_STUDY_COLUMN, "study_id", "dicom_id", "path_to_image", "path_to_dcm"),
    },
    "mimic_cxr": {
        "text": ("findings", "impression", "report"),
        "patient": ("subject_id", "patient_id"),
        "study": ("study_id", "dicom_id"),
    },
}

DATASET_ID_COLUMNS: Dict[str, List[List[str]]] = {
    "chexpert_plus": [
        [DERIVED_PATIENT_COLUMN, DERIVED_STUDY_COLUMN],
        ["deid_patient_id", DERIVED_STUDY_COLUMN],
        ["deid_patient_id", "path_to_image"],
        ["study_id"],
        ["dicom_id"],
        ["path_to_image"],
        ["path_to_dcm"],
    ],
    "mimic_cxr": [["study_id"], ["dicom_id"]],
}


def expand_path(value: Optional[str]) -> Optional[Path]:
    """Expand user and environment variables."""
    if not value:
        return None
    return Path(os.path.expandvars(str(value))).expanduser()


def _parse_mimic_record_id(record_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (patient_id, study_id) from paths like p10/p10659857/s59206984.txt."""
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    for part in Path(record_id).parts:
        lowered = part.lower()
        token = lowered.split(".")[0]
        if token.startswith("p") and len(token) > 1 and token[1:].isdigit():
            candidate = token[1:]
            if patient_id is None or len(candidate) > len(patient_id):
                patient_id = candidate
        if token.startswith("s") and len(token) > 1 and token[1:].isdigit():
            study_id = token[1:]
    return patient_id, study_id


def normalize_record_id(row: Mapping[str, object], id_columns: Sequence[str]) -> str:
    """Build a stable identifier string using the requested columns."""
    tokens: List[str] = []
    for name in id_columns:
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            tokens.append(text)
    if tokens:
        return "::".join(tokens)
    fallback = row.get("Index")
    return f"row{fallback}" if fallback is not None else "row"


def load_annotation_payload(path: Path) -> List[dict]:
    """Load RadGraph annotation JSON into a list of dict entries."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        items: List[dict] = []
        for key, value in data.items():
            def _append(entry: Any) -> None:
                if isinstance(entry, dict):
                    cloned = dict(entry)
                    cloned.setdefault("__record_id", key)
                    items.append(cloned)

            if isinstance(value, list):
                for entry in value:
                    _append(entry)
            elif isinstance(value, dict):
                _append(value)
        return items
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Unsupported annotation format in {path}")


def build_annotation_index(
    payload: Iterable[dict],
    dataset: Optional[str] = None,
) -> Dict[str, Dict[str, List[dict]]]:
    """Index RadGraph annotations by normalized text and record identifier."""
    text_index: Dict[str, List[dict]] = defaultdict(list)
    record_index: Dict[str, List[dict]] = defaultdict(list)

    def _normalize_text(text: str) -> str:
        if radgraph_utils is not None:
            return radgraph_utils.radgraph_xl_preprocess_report(text).strip()
        return text.strip()

    dataset_norm = dataset.lower() if dataset else None

    def _record_key_variants(record_id: Optional[str]) -> List[str]:
        if not record_id:
            return []
        candidates = {record_id.strip()}
        pid = None
        sid = None
        if dataset_norm == "mimic_cxr":
            pid, sid = _parse_mimic_record_id(record_id)
            if sid is None:
                try:
                    _, sid = _pid_sid_from_path(record_id)
                except Exception:
                    sid = None
            if sid:
                sid_text = str(sid).strip()
                sid_text = sid_text[1:] if sid_text.lower().startswith("s") else sid_text
                sid_norm = sid_text.lstrip("0") or "0"
                candidates.update(
                    {
                        sid_text,
                        sid_norm,
                        f"s{sid_norm}",
                        make_study_key("mimic_cxr", sid=sid_norm),
                    }
                )
        elif dataset_norm == "chexpert_plus":
            try:
                pid, sid = _pid_sid_from_path(record_id)
            except Exception:
                pid = sid = None
            if pid and sid:
                pid_text = str(pid).strip()
                sid_text = str(sid).strip()
                candidates.add(make_study_key("chexpert_plus", pid=pid_text, sid=sid_text))
        return [c for c in candidates if c]

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        doc = entry.get("0")
        if isinstance(doc, dict):
            text = doc.get("text")
            if isinstance(text, str):
                key = _normalize_text(text)
                if key:
                    text_index[key].append(entry)
                    continue
        text = entry.get("text")
        if isinstance(text, str):
            key = _normalize_text(text)
            if key:
                text_index[key].append(entry)
        record_id = entry.get("__record_id")
        if isinstance(record_id, str):
            for variant in _record_key_variants(record_id):
                record_index[variant].append(entry)
    return {"text": dict(text_index), "record": dict(record_index)}


try:
    from concept_extraction.dataset_iter import _pid_sid_from_path  # type: ignore
except Exception:  # pragma: no cover - fallback parser
    def _pid_sid_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
        parts = Path(str(path)).parts
        pid_val: Optional[str] = None
        sid_val: Optional[str] = None
        for part in parts:
            lowered = part.lower()
            if lowered.startswith("patient") and len(part) > len("patient"):
                pid_val = part[len("patient") :]
            if lowered.startswith("study") and len(part) > len("study"):
                sid_val = part[len("study") :]
        return pid_val, sid_val


def derive_study_key(
    dataset: Optional[str],
    row: Mapping[str, Any],
    patient_column: Optional[str],
    study_column: Optional[str],
    fallback: Optional[str] = None,
) -> Optional[str]:
    if not dataset:
        return None
    dataset_norm = dataset.lower()
    if dataset_norm == "chexpert_plus":
        if not patient_column or not study_column:
            raise ValueError("chexpert_plus requires --patient-column and --study-column for study_key.")
        pid_raw = row.get(patient_column)
        sid_raw = row.get(study_column)
        if pid_raw is None or sid_raw is None or pd.isna(pid_raw) or pd.isna(sid_raw):
            return None
        pid = str(pid_raw).strip()
        sid = str(sid_raw).strip()
        if not pid or not sid:
            return None
        return make_study_key(dataset_norm, pid=pid, sid=sid)
    if dataset_norm == "mimic_cxr":
        if not study_column:
            raise ValueError("mimic_cxr requires --study-column for study_key.")
        sid_raw = row.get(study_column)
        if sid_raw is None or pd.isna(sid_raw):
            return None
        sid = str(sid_raw).strip()
        if not sid:
            return None
        return make_study_key(dataset_norm, sid=sid)
    if fallback:
        return fallback
    return None


def _normalize_mimic_reports_dir(path: Path) -> Path:
    files_dir = path / "files"
    if files_dir.is_dir():
        return files_dir
    return path


def resolve_mimic_reports_dir(cli_value: Optional[str]) -> Path:
    candidates: List[Optional[Path]] = [
        expand_path(cli_value),
        expand_path(os.environ.get("MIMIC_CXR_REPORTS_DIR")),
        expand_path(os.environ.get("MIMIC_CXR_REPORTS_ROOT")),
    ]
    for cand in candidates:
        if cand and cand.exists():
            return _normalize_mimic_reports_dir(cand)
    default_root = Path.home() / "datasets" / "mimic-cxr-reports"
    if default_root.exists():
        return _normalize_mimic_reports_dir(default_root)
    raise SystemExit(
        "Unable to locate MIMIC-CXR reports directory. "
        "Provide --mimic-reports-dir or set MIMIC_CXR_REPORTS_DIR."
    )


def load_mimic_reports_dataframe(path_hint: Optional[str], quiet: bool, *, iter_progress) -> Tuple[pd.DataFrame, str, str, str, str]:
    reports_root = resolve_mimic_reports_dir(path_hint)
    if not reports_root.exists():
        raise SystemExit(f"MIMIC-CXR reports directory not found: {reports_root}")
    if not quiet:
        print(f"[info] scanning MIMIC-CXR reports under {reports_root}")

    pattern = reports_root.glob("p*/p*/s*.txt")
    iterator = iter_progress(pattern, total=None, desc="reading reports", unit="report")
    rows: List[Dict[str, Any]] = []
    for report_path in iterator:
        try:
            text = report_path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError as exc:
            if not quiet:
                print(f"[warn] failed to read {report_path}: {exc}", file=sys.stderr)
            continue
        if not text:
            continue
        patient_token = report_path.parent.name
        study_token = report_path.stem
        subject_id = patient_token[1:] if patient_token.lower().startswith("p") else patient_token
        study_id = study_token[1:] if study_token.lower().startswith("s") else study_token
        rows.append(
            {
                "subject_id": subject_id,
                "study_id": study_id,
                "report_text": text,
                "report_path": str(report_path),
            }
        )

    if not rows:
        raise SystemExit(f"No report texts were discovered under {reports_root}")
    df = pd.DataFrame(rows)
    return df, "report_text", "subject_id", "study_id", "report_text"


def infer_dataset_csv_path(dataset: Optional[str]) -> Optional[Path]:
    if dataset == "chexpert_plus":
        env_candidates = [
            expand_path(os.environ.get("CHEXPERT_PLUS_CSV")),
            expand_path(os.environ.get("CHEXPERT_CSV")),
        ]
        for cand in env_candidates:
            if cand and cand.exists():
                return cand
        search_roots = []
        for env_name in ("MEDCLIP_DATASETS_ROOT", "DATASETS_ROOT"):
            env_val = os.environ.get(env_name)
            if env_val:
                search_roots.append(Path(env_val).expanduser())
        search_roots.append(Path.home() / "datasets")
        candidates = []
        for root in search_roots:
            if not root or not root.exists():
                continue
            for name in ("CheXpert", "chexpert", "CheXpert-v1.0", "CheXpert-v1.0-small"):
                probe = root / name
                if probe.exists():
                    candidates.extend(
                        sorted(probe.glob("df_chexpert_plus*.csv")) + sorted(probe.glob("train*.csv"))
                    )
        for cand in candidates:
            if cand.exists():
                return cand
    return None


def _add_derived_ids(frame: pd.DataFrame) -> None:
    path_columns = [col for col in ("path_to_image", "path_to_dcm") if col in frame.columns]
    if not path_columns:
        return

    def _extract(row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        for col in path_columns:
            raw = row.get(col)
            if isinstance(raw, str) and raw.strip():
                pid, sid = _pid_sid_from_path(raw)
                if pid or sid:
                    return pid, sid
        return None, None

    derived = frame.apply(lambda row: _extract(row), axis=1, result_type="expand")
    frame[DERIVED_PATIENT_COLUMN] = derived[0].astype(object)
    frame[DERIVED_STUDY_COLUMN] = derived[1].astype(object)


def _choose_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def apply_dataset_defaults(
    dataset: Optional[str],
    frame: pd.DataFrame,
    text_column: Optional[str],
    patient_column: Optional[str],
    study_column: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not dataset:
        return text_column, patient_column, study_column
    presets = DATASET_PRESETS.get(dataset)
    if not presets:
        return text_column, patient_column, study_column
    text_column = text_column or _choose_column(frame.columns, presets.get("text", ()))
    patient_column = patient_column or _choose_column(frame.columns, presets.get("patient", ()))
    study_column = study_column or _choose_column(frame.columns, presets.get("study", ()))
    return text_column, patient_column, study_column


def resolve_id_columns(frame: pd.DataFrame, dataset: Optional[str]) -> List[str]:
    """Pick a reasonable set of identifier columns."""
    preferred = list(DATASET_ID_COLUMNS.get(dataset, []))
    preferred.extend(
        [
            ["study_id"],
            ["dicom_id"],
            ["id"],
            ["patient_id", "study_id"],
            ["subject_id", "study_id"],
        ]
    )
    seen: set[Tuple[str, ...]] = set()
    for columns in preferred:
        key = tuple(columns)
        if key in seen:
            continue
        seen.add(key)
        if all(col in frame.columns for col in columns):
            return columns
    if not len(frame.columns):
        raise SystemExit("Input CSV must contain at least one column.")
    return [frame.columns[0]]


def load_csv_dataframe(
    csv_path: Path,
    dataset: Optional[str],
    text_column: Optional[str],
    patient_column: Optional[str],
    study_column: Optional[str],
    default_text_column: str = DEFAULT_TEXT_COLUMN,
) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str], str]:
    csv_path = csv_path.expanduser()
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    _add_derived_ids(df)
    text_column = text_column or default_text_column
    text_column, patient_column, study_column = apply_dataset_defaults(dataset, df, text_column, patient_column, study_column)
    if not text_column:
        raise SystemExit("Unable to determine report text column; specify --text-column.")
    if text_column not in df.columns:
        raise SystemExit(f"Column '{text_column}' not found in CSV.")
    if dataset == "chexpert_plus" and (not patient_column or not study_column):
        raise SystemExit("Could not infer patient/study columns for chexpert_plus; provide --patient-column/--study-column.")
    if dataset == "mimic_cxr" and not study_column:
        raise SystemExit("Could not infer study column for mimic_cxr; provide --study-column.")
    section_label = text_column
    return df, text_column, patient_column, study_column, section_label


def prepare_input_dataframe(
    *,
    csv_path: Optional[str],
    dataset: Optional[str],
    text_column: Optional[str],
    patient_column: Optional[str],
    study_column: Optional[str],
    mimic_reports_dir: Optional[str],
    quiet: bool,
    iter_progress,
    default_text_column: str = DEFAULT_TEXT_COLUMN,
) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str], str]:
    if csv_path:
        csv_path_obj = Path(csv_path).expanduser()
        return load_csv_dataframe(csv_path_obj, dataset, text_column, patient_column, study_column, default_text_column)
    if not dataset:
        raise SystemExit("Provide --dataset or --csv-path to select input data.")
    if dataset == "mimic_cxr":
        return load_mimic_reports_dataframe(mimic_reports_dir, quiet, iter_progress=iter_progress)
    inferred_csv = infer_dataset_csv_path(dataset)
    if inferred_csv:
        if not quiet:
            print(f"[info] inferred CSV for dataset '{dataset}': {inferred_csv}")
        return load_csv_dataframe(inferred_csv, dataset, text_column, patient_column, study_column, default_text_column)
    raise SystemExit(
        f"Could not determine source data for dataset '{dataset}'. "
        "Provide --csv-path to specify the report table explicitly."
    )


__all__ = [
    "DEFAULT_TEXT_COLUMN",
    "DERIVED_PATIENT_COLUMN",
    "DERIVED_STUDY_COLUMN",
    "DATASET_PRESETS",
    "DATASET_ID_COLUMNS",
    "expand_path",
    "normalize_record_id",
    "load_annotation_payload",
    "build_annotation_index",
    "derive_study_key",
    "resolve_id_columns",
    "load_csv_dataframe",
    "prepare_input_dataframe",
    "infer_dataset_csv_path",
    "load_mimic_reports_dataframe",
    "resolve_mimic_reports_dir",
    "apply_dataset_defaults",
]
