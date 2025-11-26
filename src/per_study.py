"""Utilities for working with per-study concept extractions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from src.dataset_iter import iter_images, _load_chexpert_plus_table, _fallback_chexpert_dirs
from src.utils.identifiers import pid_sid_from_path
from src.constants import CHEXPERT_PLUS_ROOT


def _normalize_identifier(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip()
    return value or "unknown"


def make_study_key(dataset: str, *, pid: Optional[str] = None, sid: Optional[str] = None) -> str:
    dataset_norm = (dataset or "").lower()
    if dataset_norm == "chexpert_plus":
        return f"{dataset_norm}:patient{_normalize_identifier(pid)}/study{_normalize_identifier(sid)}"
    if dataset_norm == "mimic_cxr":
        return f"{dataset_norm}:study{_normalize_identifier(sid)}"
    raise KeyError(f"Unsupported dataset: {dataset}")


def load_per_study(path: Path, *, linked_only: bool = True) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            concepts = entry.get("concepts", [])
            if linked_only:
                concepts = [c for c in concepts if c.get("linked") and c.get("cui")]
            entry["concepts"] = concepts
            study_key = entry.get("study_key")
            if study_key:
                records[study_key] = entry
    return records


def load_per_study_concepts(path: Path, *, linked_only: bool = True) -> Dict[str, List[str]]:
    records = load_per_study(path, linked_only=linked_only)
    return {
        study_key: [c.get("cui") for c in entry.get("concepts", []) if c.get("cui")]
        for study_key, entry in records.items()
    }


def _study_key_short(study_key: str) -> str:
    return study_key.split(":", 1)[-1]


def _is_allowed(study_key: str, allowed: Optional[Set[str]]) -> bool:
    if not allowed:
        return True
    if study_key in allowed:
        return True
    return _study_key_short(study_key) in allowed


def iter_image_records(dataset: str,
                       allowed_keys: Optional[Set[str]] = None) -> Iterator[Tuple[str, str]]:
    dataset_norm = (dataset or "").lower()
    allowed = set(allowed_keys) if allowed_keys else None
    if dataset_norm == "chexpert_plus":
        table, path_col = _load_chexpert_plus_table()
        root = Path(CHEXPERT_PLUS_ROOT)
        suffix_cache: Dict[str, str] = {}
        ext_candidates = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")
        iterator = table.itertuples(index=False, name=None)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(table), desc="chexpert images", unit="img")
        for pid, sid, rel_path in iterator:
            rel_str = str(rel_path)
            pid_disp, sid_disp = pid_sid_from_path(rel_str)
            pid_key = pid_disp if pid_disp is not None else str(pid)
            sid_key = sid_disp if sid_disp is not None else str(sid)
            study_key = make_study_key(dataset_norm, pid=pid_key, sid=sid_key)
            if not _is_allowed(study_key, allowed):
                continue
            path = Path(rel_str)
            if not path.is_absolute():
                path = root / path
            path = path.expanduser()
            resolved: Optional[Path] = None
            if path.exists():
                resolved = path
                suffix = path.suffix.lower()
                if suffix:
                    suffix_cache.setdefault(suffix, suffix)
            else:
                suffix = path.suffix.lower()
                cached_ext = suffix_cache.get(suffix)
                if cached_ext is not None:
                    if cached_ext and cached_ext != suffix:
                        try:
                            resolved = path.with_suffix(cached_ext)
                        except ValueError:
                            resolved = path
                    else:
                        resolved = path
                if resolved is None:
                    stem = path.with_suffix("")
                    for ext in ext_candidates:
                        try:
                            candidate = stem.with_suffix(ext)
                        except ValueError:
                            continue
                        if candidate.exists():
                            resolved = candidate
                            suffix_cache[suffix or ext.lower()] = candidate.suffix.lower()
                            break
            if resolved is None:
                fallbacks = _fallback_chexpert_dirs(pid_key, sid_key)
                if fallbacks:
                    resolved = Path(fallbacks[0]).expanduser()
                    suffix_cache[(path.suffix.lower() or resolved.suffix.lower())] = resolved.suffix.lower()
            if resolved is not None:
                yield study_key, str(resolved)
        return
    for record in iter_images(dataset_norm):
        if dataset_norm == "mimic_cxr":
            sid, image_path = record
            study_key = make_study_key(dataset_norm, sid=sid)
        else:
            raise KeyError(f"Unsupported dataset: {dataset}")
        if _is_allowed(study_key, allowed):
            yield study_key, image_path


def iter_image_records_for_keys(dataset: str, study_keys: Iterable[str]) -> Iterator[Tuple[str, str]]:
    allowed = set(study_keys)
    for study_key, image_path in iter_image_records(dataset, allowed_keys=allowed):
        yield study_key, image_path
try:  # optional dependency for progress display
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None
