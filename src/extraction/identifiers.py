"""Helpers for normalizing patient/study identifiers and parsing paths."""
from __future__ import annotations

import math
import re
from typing import Any, List, Optional, Tuple

import numpy as np

_PID_STUDY_PAT = re.compile(r"patient(\d+)/study(\d+)", re.I)
_PID_PREFIX_PAT = re.compile(r"patient0*(\d+)", re.I)
_STUDY_PREFIX_PAT = re.compile(r"study0*(\d+)", re.I)


def normalize_patient_id(pid: Optional[str]) -> Optional[str]:
    text = str(pid).strip() if pid is not None else ""
    if not text:
        return None
    match = _PID_PREFIX_PAT.match(text)
    return match.group(1) if match else text.lstrip("0") or "0"


def normalize_study_id(sid: Optional[str]) -> Optional[str]:
    text = str(sid).strip() if sid is not None else ""
    if not text:
        return None
    match = _STUDY_PREFIX_PAT.match(text)
    return match.group(1) if match else text.lstrip("0") or "0"


def pid_sid_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    match = _PID_STUDY_PAT.search(str(path))
    if not match:
        return None, None
    return match.group(1), match.group(2)


def patient_dir_variants(pid: str) -> List[str]:
    variants = {f"patient{pid}"}
    if pid.isdigit():
        pid_int = int(pid)
        variants.update({f"patient{pid_int:05d}", f"patient{pid_int:06d}"})
    return sorted(variants)


def study_dir_variants(sid: str) -> List[str]:
    variants = {f"study{sid}"}
    if sid.isdigit():
        sid_int = int(sid)
        variants.update({f"study{sid_int:05d}", f"study{sid_int:06d}"})
    return sorted(variants)


def normalize_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        return text[:-2]
    return text
