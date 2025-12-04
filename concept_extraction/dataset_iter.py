from pathlib import Path
import pandas as pd
from typing import Iterator, Tuple, Optional, List
from functools import lru_cache
from concept_extraction.identifiers import (
    normalize_patient_id,
    normalize_study_id,
    patient_dir_variants,
    pid_sid_from_path,
    study_dir_variants,
)
from lib.constants import (
    # edit these in your repo; they're the only "config"
    MIMIC_JPG_META_CSV,
    MIMIC_JPG_ROOT,
    CHEXPERT_PLUS_CSV,
    CHEXPERT_PLUS_ROOT,
)


# -------- MIMIC-CXR --------
def iter_mimic_reports() -> Iterator[Tuple[str, str]]:
    """
    Yields (study_id, report_text) from MIMIC-CXR by scanning raw report .txt files.
    """
    # Import here to avoid circular dependency on per_study -> dataset_iter.
    from concept_extraction.concepts.input import load_mimic_reports_dataframe

    iter_progress = (lambda it, **kw: tqdm(it, **kw)) if 'tqdm' in globals() and tqdm is not None else (lambda it, **kw: it)
    df, text_col, _, sid_col, _ = load_mimic_reports_dataframe(path_hint=None, quiet=True, iter_progress=iter_progress)
    for sid, rpt in zip(df[sid_col].astype(str), df[text_col].astype(str)):
        if rpt:
            yield sid, rpt

def _derive_mimic_path(row: pd.Series) -> Optional[Path]:
    subject = row.get("subject_id")
    study = row.get("study_id")
    dicom = row.get("dicom_id")
    if pd.isna(subject) or pd.isna(study) or pd.isna(dicom):
        return None
    subject = str(int(subject)).zfill(8)
    study = str(int(study)).zfill(8)
    dicom = str(dicom).strip()
    if not dicom:
        return None
    rel = Path(f"p{subject[:2]}") / f"p{subject}" / f"s{study}" / f"{dicom}.jpg"
    return Path(MIMIC_JPG_ROOT) / rel


def iter_mimic_images() -> Iterator[Tuple[str, str]]:
    """
    Yields (study_id, absolute_image_path) from MIMIC-CXR-JPG metadata.
    """
    meta = Path(MIMIC_JPG_META_CSV)
    if not meta.exists():
        raise FileNotFoundError(f"MIMIC_JPG_META_CSV not found: {meta}")
    df = pd.read_csv(meta)
    cols = {c.lower(): c for c in df.columns}
    path_col = cols.get("path")
    if path_col is not None:
        df = df.dropna(subset=[cols.get("study_id", "study_id"), path_col]).drop_duplicates()
        root = Path(MIMIC_JPG_ROOT)
        for sid, rel in zip(df[cols.get("study_id", "study_id")].astype(str), df[path_col].astype(str)):
            yield sid, str((root / rel).resolve())
        return

    required = {"study_id", "subject_id", "dicom_id"}
    missing = [col for col in required if col not in cols]
    if missing:
        raise KeyError(f"MIMIC metadata missing required columns: {missing}")
    study_series = df[cols["study_id"]].astype(str)
    paths = df.apply(_derive_mimic_path, axis=1)
    for sid, path in zip(study_series, paths):
        if path and path.exists():
            yield sid, str(path.resolve())

# -------- CheXpert Plus --------

def iter_chexpert_plus_reports() -> Iterator[Tuple[str, str, str]]:
    csv_path = Path(CHEXPERT_PLUS_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CheXpert Plus CSV not found: {csv_path}. "
            "Set CHEXPERT_PLUS_CSV or MEDCLIP_DATASETS_ROOT appropriately."
        )

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    path_col = cols.get("path_to_image") or cols.get("image_path") or cols.get("path")
    if not path_col:
        raise ValueError("CheXpert Plus CSV needs a path column (path_to_image/image_path/path).")

    # derive patient_id / study_id
    if "patient_id" in cols and "study_id" in cols:
        df["patient_id"] = df[cols["patient_id"]].astype(str)
        df["study_id"]   = df[cols["study_id"]].astype(str)
    else:
        tmp = df[path_col].astype(str).map(pid_sid_from_path)
        df["patient_id"] = tmp.map(lambda t: t[0])
        df["study_id"]   = tmp.map(lambda t: t[1])

    imp_col = cols.get("section_impression") or cols.get("impression")
    fnd_col = cols.get("section_findings")   or cols.get("findings")
    imp = df[imp_col].fillna("").astype(str).str.strip() if imp_col else ""
    fnd = df[fnd_col].fillna("").astype(str).str.strip() if fnd_col else ""
    df["report"] = imp.where(imp.ne(""), fnd)

    # one report per (patient_id, study_id): first non-empty wins
    agg = (df[["patient_id","study_id","report"]]
           .dropna(subset=["patient_id","study_id"])
           .groupby(["patient_id","study_id"], as_index=False)["report"]
           .agg(lambda s: next((x for x in s if x), "")))
    for pid, sid, rpt in agg.itertuples(index=False):
        if rpt:
            yield pid, sid, rpt

@lru_cache(maxsize=1)
def _load_chexpert_plus_table() -> Tuple[pd.DataFrame, str]:
    csv_path = Path(CHEXPERT_PLUS_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CheXpert Plus CSV not found: {csv_path}. "
            "Set CHEXPERT_PLUS_CSV or MEDCLIP_DATASETS_ROOT appropriately."
        )

    root = Path(CHEXPERT_PLUS_ROOT)
    if not root.exists():
        raise FileNotFoundError(
            f"CheXpert Plus image root not found: {root}. "
            "Set MEDCLIP_DATASETS_ROOT or CHEXPERT_PLUS_ROOT to the correct directory."
        )

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    path_col = cols.get("path_to_image") or cols.get("image_path") or cols.get("path")
    if not path_col:
        raise ValueError("CheXpert Plus CSV needs a path column (path_to_image/image_path/path).")

    if "patient_id" in cols and "study_id" in cols:
        df["patient_id"] = df[cols["patient_id"]].astype(str).str.strip()
        df["study_id"] = df[cols["study_id"]].astype(str).str.strip()
    else:
        tmp = df[path_col].astype(str).map(pid_sid_from_path)
        df["patient_id"] = tmp.map(lambda t: t[0]).fillna("")
        df["study_id"] = tmp.map(lambda t: t[1]).fillna("")

    df[path_col] = df[path_col].astype(str).str.strip()
    table = df[["patient_id", "study_id", path_col]].copy()
    table[path_col] = table[path_col].astype(str).str.strip()
    table["patient_id"] = table["patient_id"].astype(str).map(normalize_patient_id).fillna("")
    table["study_id"] = table["study_id"].astype(str).map(normalize_study_id).fillna("")
    return table, path_col


def _resolve_chexpert_path(rel_path: str) -> Optional[str]:
    root = Path(CHEXPERT_PLUS_ROOT)
    q = Path(rel_path)

    candidates: List[Path] = []
    if q.is_absolute():
        candidates.append(q)
        parts = q.parts
        for split in ("train", "valid", "test"):
            if split in parts:
                rebased = root / Path(*parts[parts.index(split):])
                candidates.append(rebased)
    else:
        candidates.append(root / q)

    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    resolved = []
    for cand in candidates:
        cand = cand.expanduser()
        if cand.exists():
            resolved.append(cand)
            continue
        stem = cand.with_suffix("")
        for ext in exts:
            alt = Path(str(stem) + ext)
            if alt.exists():
                resolved.append(alt)
                break
    if resolved:
        return str(Path(resolved[0]).resolve(strict=False))
    return None


def _fallback_chexpert_dirs(pid: str, sid: str) -> List[str]:
    root = Path(CHEXPERT_PLUS_ROOT)
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    hits: List[str] = []
    for split in ("train", "valid", "test"):
        split_root = root / split
        if not split_root.exists():
            continue
        for p_dir in patient_dir_variants(pid):
            for s_dir in study_dir_variants(sid):
                study_dir = split_root / p_dir / s_dir
                if not study_dir.exists() or not study_dir.is_dir():
                    continue
                for path in study_dir.iterdir():
                    if path.is_file() and path.suffix in exts:
                        hits.append(str(path.resolve(strict=False)))
    return sorted(set(hits))


def iter_chexpert_plus_images() -> Iterator[Tuple[str, str, str]]:
    table, path_col = _load_chexpert_plus_table()
    for pid, sid, rel in table.itertuples(index=False, name=None):
        if pid and sid and isinstance(rel, str) and rel:
            resolved = _resolve_chexpert_path(rel)
            if resolved:
                yield pid, sid, resolved


def lookup_chexpert_plus_images(pid: str, sid: str) -> List[str]:
    table, path_col = _load_chexpert_plus_table()
    pid_norm = normalize_patient_id(pid)
    sid_norm = normalize_study_id(sid)
    subset = table[(table["patient_id"] == pid_norm) & (table["study_id"] == sid_norm)]
    out: List[str] = []
    for rel in subset[path_col]:
        resolved = _resolve_chexpert_path(rel)
        if resolved:
            out.append(resolved)
    if not out:
        out = _fallback_chexpert_dirs(pid_norm, sid_norm)
    return sorted(set(out))


@lru_cache(maxsize=1)
def _chexpert_plus_report_dict():
    df = pd.read_csv(CHEXPERT_PLUS_CSV)
    cols = {c.lower(): c for c in df.columns}
    path_col = cols.get("path_to_image") or cols.get("image_path") or cols.get("path")
    if not path_col:
        raise ValueError("CheXpert Plus CSV needs a path column (path_to_image/image_path/path).")

    if "patient_id" in cols and "study_id" in cols:
        df["patient_id"] = df[cols["patient_id"]].astype(str)
        df["study_id"] = df[cols["study_id"]].astype(str)
    else:
        tmp = df[path_col].astype(str).map(pid_sid_from_path)
        df["patient_id"] = tmp.map(lambda t: t[0])
        df["study_id"] = tmp.map(lambda t: t[1])

    # Limit extraction to Findings and Impression style columns.
    impression_keys = [
        "section_impression",
        "impression",
        "impression_text",
        "impression_section",
        "section_summary",
        "summary",
    ]
    findings_keys = [
        "section_findings",
        "findings",
        "findings_text",
        "section_narrative",
        "narrative",
    ]

    def resolve_columns(pref_keys):
        seen = []
        for key in pref_keys:
            col = cols.get(key)
            if col and col not in seen:
                seen.append(col)
        return seen

    impression_cols = resolve_columns(impression_keys)
    findings_cols = resolve_columns(findings_keys)

    def gather_text(row, columns):
        snippets = []
        for col in columns:
            val = row.get(col, "") if hasattr(row, "get") else getattr(row, col, "")
            if pd.isna(val):
                continue
            text = str(val).strip()
            if text:
                snippets.append(text)
        return snippets

    def extract_clinical_sections(text):
        """Extract only IMPRESSION and FINDINGS sections from full report."""
        if not text:
            return None

        text = str(text).strip()

        # Define section headers to extract (case-insensitive)
        target_sections = {'IMPRESSION', 'FINDINGS'}

        # Define section headers to skip
        skip_sections = {
            'NARRATIVE', 'CLINICAL HISTORY', 'HISTORY', 'COMPARISON',
            'TECHNIQUE', 'CLINICAL DATA', 'PROCEDURE COMMENTS',
            'ACCESSION NUMBER', 'SUMMARY', 'END OF IMPRESSION'
        }

        lines = text.split('\n')
        extracted = []
        current_section = None
        section_content = []

        for line in lines:
            line_upper = line.strip().upper()

            # Check if this line is a section header
            is_section_header = False
            for section in target_sections | skip_sections:
                if line_upper.startswith(section + ':') or line_upper == section:
                    is_section_header = True

                    # Save previous section if it was a target
                    if current_section in target_sections and section_content:
                        content = '\n'.join(section_content).strip()
                        if content:
                            extracted.append(f"{current_section}:\n{content}")

                    # Start new section
                    if section in target_sections:
                        current_section = section
                        section_content = []
                    else:
                        current_section = None
                        section_content = []
                    break

            # Add line to current section if it's a target
            if not is_section_header and current_section in target_sections:
                section_content.append(line)

        # Save last section if it was a target
        if current_section in target_sections and section_content:
            content = '\n'.join(section_content).strip()
            if content:
                extracted.append(f"{current_section}:\n{content}")

        return '\n\n'.join(extracted) if extracted else None

    def assemble_report(row):
        # First try to extract from full report column (contains complete text)
        report_col = cols.get("report")
        if report_col:
            full_report = row.get(report_col, "") if hasattr(row, "get") else getattr(row, report_col, "")
            if not pd.isna(full_report):
                extracted = extract_clinical_sections(full_report)
                if extracted:
                    return extracted

        # Fallback to sectioned columns if extraction failed
        sections = []
        imp_texts = gather_text(row, impression_cols)
        find_texts = gather_text(row, findings_cols)
        if imp_texts:
            sections.append("IMPRESSION:\n" + "\n".join(imp_texts))
        if find_texts:
            sections.append("FINDINGS:\n" + "\n".join(find_texts))
        assembled = "\n\n".join(sections).strip()

        return assembled

    df["report"] = df.apply(assemble_report, axis=1)

    agg = (df[["patient_id", "study_id", "report"]]
             .dropna(subset=["patient_id", "study_id"])
             .groupby(["patient_id", "study_id"], as_index=False)["report"]
             .agg(lambda s: next((x for x in s if x), "")))

    mapping = {}
    for pid, sid, rpt in agg.itertuples(index=False):
        key = (normalize_patient_id(str(pid)), normalize_study_id(str(sid)))
        mapping[key] = rpt
    return mapping


@lru_cache(maxsize=1)
def _mimic_cxr_report_dict():
    mapping = {}
    for sid, text in iter_mimic_reports():
        sid_norm = normalize_study_id(str(sid))
        mapping[sid_norm] = text
    return mapping


def lookup_report_text(dataset: str, pid: Optional[str], sid: str) -> Optional[str]:
    dataset = (dataset or "").lower()
    # Handle dataset name aliases
    dataset_aliases = {
        "chexpert": "chexpert_plus",
        "mimic": "mimic_cxr",
    }
    dataset = dataset_aliases.get(dataset, dataset)

    sid_norm = normalize_study_id(sid)
    if dataset == "chexpert_plus":
        mapping = _chexpert_plus_report_dict()
        return mapping.get((normalize_patient_id(pid), sid_norm))
    if dataset == "mimic_cxr":
        mapping = _mimic_cxr_report_dict()
        return mapping.get(sid_norm)
    raise KeyError(f"Unknown dataset for report lookup: {dataset}")

            
# -------- Public convenience --------
def iter_reports(dataset: str):
    if dataset == "mimic_cxr":       return iter_mimic_reports()
    if dataset == "chexpert_plus":   return iter_chexpert_plus_reports()
    raise KeyError(f"Unknown dataset: {dataset}")

def iter_images(dataset: str):
    if dataset == "mimic_cxr":       return iter_mimic_images()
    if dataset == "chexpert_plus":   return iter_chexpert_plus_images()
    raise KeyError(f"Unknown dataset: {dataset}")
