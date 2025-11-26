"""Common helpers for normalizing patient/study identifiers and parsing paths."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

_PID_STUDY_PAT = re.compile(r"patient(\d+)/study(\d+)", re.I)
_PID_PREFIX_PAT = re.compile(r"patient0*(\d+)", re.I)
_STUDY_PREFIX_PAT = re.compile(r"study0*(\d+)", re.I)


def normalize_patient_id(pid: Optional[str]) -> Optional[str]:
    """Strip prefixes/leading zeros and return a stable patient identifier."""
    if pid is None:
        return None
    text = str(pid).strip()
    if not text:
        return None
    match = _PID_PREFIX_PAT.match(text)
    return match.group(1) if match else text.lstrip("0") or "0"


def normalize_study_id(sid: Optional[str]) -> Optional[str]:
    """Strip prefixes/leading zeros and return a stable study identifier."""
    if sid is None:
        return None
    text = str(sid).strip()
    if not text:
        return None
    match = _STUDY_PREFIX_PAT.match(text)
    return match.group(1) if match else text.lstrip("0") or "0"


def pid_sid_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract patient_id/study_id from a CheXpert-style path."""
    match = _PID_STUDY_PAT.search(str(path))
    if not match:
        return None, None
    return match.group(1), match.group(2)


def patient_dir_variants(pid: str) -> List[str]:
    """Return folder naming variants for a patient id (zero-padded/non-padded)."""
    variants = {f"patient{pid}"}
    if pid.isdigit():
        pid_int = int(pid)
        variants.add(f"patient{pid_int:05d}")
        variants.add(f"patient{pid_int:06d}")
    return sorted(variants)


def study_dir_variants(sid: str) -> List[str]:
    """Return folder naming variants for a study id (zero-padded/non-padded)."""
    variants = {f"study{sid}"}
    if sid.isdigit():
        sid_int = int(sid)
        variants.add(f"study{sid_int:05d}")
        variants.add(f"study{sid_int:06d}")
    return sorted(variants)
