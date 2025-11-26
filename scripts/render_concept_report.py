#!/usr/bin/env python3
"""Render an HTML report comparing reference concepts and model predictions per study."""
from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from PIL import Image
from jinja2 import Environment, FileSystemLoader, select_autoescape
from tqdm import tqdm
from scispacy.umls_semantic_type_tree import construct_umls_tree_from_tsv
from scispacy.linking_utils import DEFAULT_UMLS_TYPES_PATH

# Ensure repository root is on sys.path before importing src.*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
from src import constants as data_constants
from src.extraction.dataset_iter import lookup_report_text
from src.extraction.identifiers import pid_sid_from_path
from src.extraction.per_study import iter_image_records


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_concept_index(path: Path) -> Optional[List[str]]:
    """Load a concept index mapping position -> name."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:  # pragma: no cover - defensive
        print(f"[warn] unable to read concept index at {path}: {err}")
        return None

    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        try:
            max_idx = max(int(k) for k in data.keys())
        except ValueError:
            max_idx = len(data) - 1
        names: List[str] = []
        for idx in range(max_idx + 1):
            key = str(idx)
            if key in data:
                names.append(str(data[key]))
        return names

    print(f"[warn] unsupported concept index format at {path}")
    return None


def _guess_concept_index(predictions_path: Path) -> Optional[Path]:
    """Best-effort guess of a concept index JSON relative to predictions."""

    candidates = [
        predictions_path.with_name("concept_index.json"),
        predictions_path.parent / "concept_index.json",
        predictions_path.parent.parent / "concept_index.json",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def _convert_cbm_entries(
    entries: List[Dict],
    *,
    concept_names: Optional[List[str]],
    threshold: float,
    dataset: Optional[str],
) -> List[Dict]:
    """Convert CBM-style probability vectors to concept lists in-place."""

    warned_missing_names = False
    for entry in entries:
        if "concepts" in entry or "probs" not in entry:
            continue

        probs = entry.get("probs") or []
        concepts: List[Dict[str, Any]] = []
        for idx, prob in enumerate(probs):
            try:
                score = float(prob)
            except (TypeError, ValueError):
                continue
            if score < threshold:
                continue
            name = None
            if concept_names and idx < len(concept_names):
                name = concept_names[idx]
            else:
                name = f"concept_{idx}"
                if not warned_missing_names and concept_names is None:
                    print("[warn] CBM entries detected but no concept index provided; falling back to numbered names.")
                    warned_missing_names = True
            concepts.append(
                {
                    "concept": name,
                    "preferred_name": name,
                    "cui": None,
                    "score": score,
                    "assertion": "present",
                }
            )
        entry["concepts"] = concepts
        if "study_key" not in entry and entry.get("study_id"):
            entry["study_key"] = entry["study_id"]
        if dataset and "dataset" not in entry:
            entry["dataset"] = dataset

    return entries


def load_relations_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cui = record.get("cui")
            if not cui:
                continue
            cache[cui] = {
                "name": record.get("name"),
                "parents": record.get("parents", []) or [],
                "children": record.get("children", []) or [],
                "siblings": record.get("siblings", []) or [],
            }
    return cache


def study_key_short(key: str) -> str:
    return key.split(":", 1)[-1]


def _sanitize_study_key(key: str) -> str:
    if not key:
        return key
    cleaned = key
    cleaned = cleaned.replace("patientpatient", "patient", 1)
    cleaned = cleaned.replace("studystudy", "study", 1)
    return cleaned


DATASET_ROOTS = {
    "chexpert_plus": data_constants.CHEXPERT_PLUS_ROOT,
    "mimic_cxr": data_constants.MIMIC_JPG_ROOT,
}

def _normalize_dataset_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name_lower = name.lower()
    if name_lower == "chexpert":
        return "chexpert_plus"
    return name_lower

CHEXPERT_REPORT_CACHE: Dict[str, str] = {}
CHEXPERT_REPORT_LOADED = False


def _ensure_chexpert_report_cache() -> None:
    global CHEXPERT_REPORT_LOADED
    if CHEXPERT_REPORT_LOADED:
        return
    csv_path = getattr(data_constants, "CHEXPERT_PLUS_CSV", None)
    if not csv_path or not Path(csv_path).exists():
        CHEXPERT_REPORT_LOADED = True
        return
    try:
        import pandas as pd  # type: ignore
    except ImportError:  # pragma: no cover
        print("[warn] pandas not available; cannot load full CheXpert reports.")
        CHEXPERT_REPORT_LOADED = True
        return

    desired_cols = [
        "path_to_image",
        "report",
        "section_narrative",
        "section_history",
        "section_clinical_history",
        "section_findings",
        "section_impression",
        "section_summary",
    ]
    try:
        header = pd.read_csv(csv_path, nrows=0)
        available = list(header.columns)
    except Exception:
        available = desired_cols
    usecols = [col for col in desired_cols if col in available]
    if "path_to_image" not in usecols:
        usecols.append("path_to_image")

    try:
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=5000):
            for _, row in chunk.iterrows():
                path_value = str(row.get("path_to_image", "")).strip()
                pid = sid = ""
                if path_value:
                    pid, sid = pid_sid_from_path(path_value)
                pid = _strip_prefix_casefold(pid or str(row.get("deid_patient_id", "")), "patient")
                sid = _strip_prefix_casefold(sid or str(row.get("study_id", "")), "study")
                if not sid:
                    order_val = row.get("patient_report_date_order")
                    if isinstance(order_val, (int, float)) and not isinstance(order_val, bool):
                        sid = str(int(order_val))
                    elif isinstance(order_val, str) and order_val.strip():
                        sid = order_val.strip()
                if not pid or not sid:
                    continue
                short_key = f"patient{pid}/study{sid}"
                if short_key in CHEXPERT_REPORT_CACHE:
                    continue
                text = str(row.get("report", "") or "").strip()
                if not text:
                    sections: List[str] = []
                    for col, label in (
                        ("section_narrative", "NARRATIVE"),
                        ("section_history", "HISTORY"),
                        ("section_clinical_history", "CLINICAL HISTORY"),
                        ("section_findings", "FINDINGS"),
                        ("section_impression", "IMPRESSION"),
                        ("section_summary", "SUMMARY"),
                    ):
                        if col in chunk.columns:
                            value = row.get(col)
                            if isinstance(value, str) and value.strip():
                                sections.append(f"{label}:\n{value.strip()}")
                    text = "\n\n".join(sections)
                if text:
                    CHEXPERT_REPORT_CACHE[short_key] = _filter_report_sections(text)
    except Exception as exc:
        print(f"[warn] failed to build CheXpert report cache: {exc}")
    CHEXPERT_REPORT_LOADED = True


def _lookup_chexpert_report(short_key: str, image_path: Optional[str]) -> Optional[str]:
    _ensure_chexpert_report_cache()
    key = _sanitize_study_key(short_key)
    text = CHEXPERT_REPORT_CACHE.get(key)
    if text or not image_path:
        return text
    pid, sid = pid_sid_from_path(image_path)
    if pid and sid:
        alt = f"patient{_strip_prefix_casefold(pid, 'patient')}/study{_strip_prefix_casefold(sid, 'study')}"
        text = CHEXPERT_REPORT_CACHE.get(alt)
    return text


TEMPLATE_NAME = "concept_report.html"


def collect_unique_cuis(*entry_maps: Dict[str, Dict]) -> Set[str]:
    cuis: Set[str] = set()
    for entry_map in entry_maps:
        for entry in entry_map.values():
            for concept in entry.get("concepts", []):
                cui = concept.get("cui")
                # Only add valid CUI strings (not None or empty)
                if cui and isinstance(cui, str) and cui.strip():
                    cuis.add(cui)
    return cuis


def _resolve_kb_jsonl_path(path_hint: Optional[Path]) -> Optional[Path]:
    """Resolve a knowledge base JSONL file from a hint path."""

    if path_hint is None:
        return None

    candidate = path_hint.expanduser()
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        for pattern in ("umls_*.jsonl", "*.jsonl"):
            matches = sorted(candidate.glob(pattern))
            if matches:
                # Prioritize base UMLS files over relation/cache files
                base_files = [m for m in matches if '_relation' not in m.name.lower() and '_cache' not in m.name.lower()]
                return base_files[0] if base_files else matches[0]

        # Handle common scispaCy linker layout where JSONL lives alongside the index directory.
        if (candidate / "concept_aliases.json").exists():
            parent = candidate.parent
            for pattern in ("umls_*.jsonl", "*.jsonl"):
                matches = sorted(parent.glob(pattern))
                if matches:
                    # Prioritize base UMLS files over relation/cache files
                    base_files = [m for m in matches if '_relation' not in m.name.lower() and '_cache' not in m.name.lower()]
                    return base_files[0] if base_files else matches[0]

    return None


def _guess_default_kb_path() -> Optional[Path]:
    """Guess a local UMLS KB path from common installation layouts."""

    env_hints = [
        os.environ.get("UMLS_LOCAL_KB"),
        os.environ.get("SCISPACY_LINKER_PATH"),
    ]
    hint_paths = [Path(hint).expanduser() for hint in env_hints if hint]
    hint_paths.extend(
        [
            Path("~/umls_linker/index").expanduser(),
            Path("~/umls_linker/umls_2025AA.jsonl").expanduser(),
            Path("~/umls_linker").expanduser(),
        ]
    )

    for hint in hint_paths:
        kb_path = _resolve_kb_jsonl_path(hint)
        if kb_path and kb_path.exists():
            return kb_path

    return None


def _find_mrrel_in_dir(directory: Path) -> Optional[Path]:
    patterns = [
        "MRREL.RRF",
        "mrrel.rrf",
        "MRREL*.RRF",
        "mrrel*.rrf",
        "MRREL.RRF.gz",
        "MRREL.RRF.txt",
        "mrrel.tsv",
        "mrrel.txt",
    ]
    for pattern in patterns:
        for candidate in sorted(directory.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def _guess_mrrel_path(kb_entry: Path) -> Optional[Path]:
    """Best-effort lookup for an MRREL file near the knowledge base entry."""

    search_dirs: List[Path] = []
    if kb_entry.is_dir():
        search_dirs.extend([kb_entry, kb_entry / "META"])
    else:
        base_dir = kb_entry.parent
        search_dirs.extend([base_dir, base_dir / "META"])
        if base_dir.parent:
            search_dirs.extend([base_dir.parent, base_dir.parent / "META"])

    seen: Set[Path] = set()
    for directory in search_dirs:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        if directory.exists() and directory.is_dir():
            candidate = _find_mrrel_in_dir(directory)
            if candidate:
                return candidate

    return None


def _normalize_relation_priority(relation: str, rela: str) -> Tuple[Optional[int], Optional[int]]:
    relation = relation.upper()
    rela_lower = rela.lower()
    parent_priority = {"PAR": 0, "RB": 1}
    child_priority = {"CHD": 0, "RN": 1}

    if relation in parent_priority:
        return parent_priority[relation], None
    if relation in child_priority:
        return None, child_priority[relation]
    if rela_lower == "isa":
        return 2, None
    if rela_lower == "inverse_isa":
        return None, 2
    return None, None


def _load_mrrel_neighbors(
    mrrel_path: Path,
    target_cuis: Set[str],
    *,
    max_relations: int,
    max_depth: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], Set[str]]:
    """Collect parent, child, and sibling CUIs with optional ancestor expansion."""

    if max_relations == 0 or not mrrel_path.exists():
        return {}, {}, {}, set()

    unlimited = max_relations < 0
    parents: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    children: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    parent_counts: Dict[str, int] = defaultdict(int)
    child_counts: Dict[str, int] = defaultdict(int)
    related_ids: Set[str] = set(target_cuis)

    order_counter = 0

    def _add_relation(
        mapping: Dict[str, List[Tuple[int, str]]],
        counts: Dict[str, int],
        key: str,
        value: str,
    ) -> None:
        nonlocal order_counter
        existing = {entry[1] for entry in mapping[key]}
        if value in existing:
            return
        if not unlimited and counts[key] >= max_relations:
            return
        mapping[key].append((order_counter, value))
        counts[key] += 1
        order_counter += 1

    expanded: Set[str] = set()
    to_expand: Set[str] = set(target_cuis)
    depth = 0

    progress_desc = "MRREL scan"
    try:
        total_bytes = mrrel_path.stat().st_size
    except OSError:
        total_bytes = None

    while to_expand and (max_depth < 0 or depth <= max_depth):
        current_targets = to_expand
        to_expand = set()
        try:
            with mrrel_path.open("rb") as handle:
                progress_bar = None
                if depth == 0 and total_bytes:
                    progress_bar = tqdm(
                        total=total_bytes,
                        desc=progress_desc,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    )
                try:
                    for raw_line in handle:
                        if progress_bar:
                            progress_bar.update(len(raw_line))
                        line = raw_line.decode("utf-8", errors="ignore").rstrip("\n")
                        parts = line.split("|")
                        if len(parts) < 5:
                            continue
                        cui1 = parts[0]
                        relation = parts[3]
                        cui2 = parts[4]
                    rela = parts[7] if len(parts) > 7 else ""

                    parent_priority, child_priority = _normalize_relation_priority(relation, rela)
                    if parent_priority is None and child_priority is None:
                        continue

                    if parent_priority is not None:
                        child_id, parent_id = cui1, cui2
                    else:
                        child_id, parent_id = cui2, cui1

                    if child_id in current_targets:
                        _add_relation(parents, parent_counts, child_id, parent_id)
                        related_ids.add(parent_id)
                        if parent_id not in expanded and (max_depth < 0 or depth < max_depth):
                            to_expand.add(parent_id)

                    if parent_id in current_targets:
                        _add_relation(children, child_counts, parent_id, child_id)
                        related_ids.add(child_id)

                    if not unlimited:
                        all_parents_met = all(
                            parent_counts.get(target, 0) >= max_relations
                            for target in current_targets
                        )
                        all_children_met = all(
                            child_counts.get(target, 0) >= max_relations
                            for target in current_targets
                        )
                        if all_parents_met and all_children_met:
                            break
                finally:
                    if progress_bar:
                        progress_bar.close()

        except FileNotFoundError:
            return {}, {}, {}, set()

        expanded.update(current_targets)
        depth += 1

    # Normalize ordering and trim to requested limits.
    normalized_parents: Dict[str, List[str]] = {}
    for cui, entries in parents.items():
        entries.sort(key=lambda item: item[0])
        trimmed = entries if unlimited or max_relations < 0 else entries[:max_relations]
        normalized_parents[cui] = [entry[1] for entry in trimmed]

    normalized_children: Dict[str, List[str]] = {}
    for cui, entries in children.items():
        entries.sort(key=lambda item: item[0])
        trimmed = entries if unlimited or max_relations < 0 else entries[:max_relations]
        normalized_children[cui] = [entry[1] for entry in trimmed]

    siblings: Dict[str, List[str]] = {}
    for cui, parent_list in normalized_parents.items():
        if not parent_list:
            continue
        related: List[str] = []
        seen: Set[str] = set()
        for parent_id in parent_list:
            for child_id in normalized_children.get(parent_id, []):
                if child_id == cui or child_id in seen:
                    continue
                related.append(child_id)
                seen.add(child_id)
                if not unlimited and max_relations > 0 and len(related) >= max_relations:
                    break
            if not unlimited and max_relations > 0 and len(related) >= max_relations:
                break
        if related:
            siblings[cui] = related
            related_ids.update(related)

    return normalized_parents, normalized_children, siblings, related_ids


def build_cui_context_from_local(
    kb_path: Optional[Path],
    cuis: Set[str],
    *,
    types_path: Optional[str] = None,
    mrrel_path: Optional[Path] = None,
    max_parents: int = 3,
    ancestor_depth: int = 2,
    relations_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not cuis:
        return {}

    if max_parents == 0:
        parent_limit = 0
    elif max_parents < 0:
        parent_limit = -1
    else:
        parent_limit = max_parents

    kb_hint = kb_path.expanduser() if kb_path else None
    kb_file = _resolve_kb_jsonl_path(kb_hint) if kb_hint else None
    if kb_file is None:
        kb_file = _guess_default_kb_path()
    if kb_file is None or not kb_file.exists():
        print("[warn] Unable to locate local UMLS knowledge base (JSONL).")
        return {}

    kb_file = kb_file.expanduser()

    parent_map: Dict[str, List[str]] = {}
    child_map: Dict[str, List[str]] = {}
    sibling_map: Dict[str, List[str]] = {}
    related_neighbors: Set[str] = set()

    if relations_cache:
        for payload in relations_cache.values():
            for bucket in (payload.get("parents", []), payload.get("children", []), payload.get("siblings", [])):
                for entry in bucket:
                    cui = entry.get("cui") if isinstance(entry, dict) else None
                    if cui and cui not in cuis:
                        related_neighbors.add(cui)
    else:
        relation_file: Optional[Path] = None
        if mrrel_path:
            mrrel_candidate = mrrel_path.expanduser()
            if mrrel_candidate.is_dir():
                relation_file = _find_mrrel_in_dir(mrrel_candidate)
            else:
                relation_file = mrrel_candidate
        if relation_file is None:
            relation_file = _guess_mrrel_path(mrrel_path.expanduser() if mrrel_path else kb_file)
        if relation_file and relation_file.exists() and relation_file.is_file():
            parent_map, child_map, sibling_map, related_neighbors = _load_mrrel_neighbors(
                relation_file,
                set(cuis),
                max_relations=parent_limit,
                max_depth=ancestor_depth,
            )

    target_ids = set(cuis) | related_neighbors
    remaining = set(target_ids)
    raw_records: Dict[str, Dict[str, Any]] = {}

    try:
        with kb_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                concept_id = record.get("concept_id") or record.get("cui")
                if concept_id in remaining:
                    raw_records[concept_id] = record
                    remaining.remove(concept_id)
                    if not remaining:
                        break
    except FileNotFoundError:
        print(f"[warn] Unable to read UMLS KB file: {kb_file}")
        return {}

    if not raw_records:
        return {}

    tree = None
    type_source = types_path or os.environ.get("UMLS_TYPES_TSV") or DEFAULT_UMLS_TYPES_PATH
    try:
        tree = construct_umls_tree_from_tsv(type_source)
    except Exception as err:  # pragma: no cover - defensive
        print(f"[warn] unable to load semantic type tree ({type_source}): {err}")
        tree = None

    def _node_info(concept_id: str) -> Dict[str, Optional[str]]:
        record = raw_records.get(concept_id)
        name = None
        if record is None:
            cache_entry = relations_cache.get(concept_id) if relations_cache else None
            if cache_entry:
                name = cache_entry.get("name")
        else:
            name = record.get("canonical_name") or record.get("preferred_name") or record.get("name")
        return {"cui": concept_id, "name": name}

    def _collect_ancestors(focus_id: str, depth_limit: int) -> Tuple[Set[str], Dict[str, int]]:
        ancestors: Set[str] = set()
        depth_map: Dict[str, int] = {}
        stack: List[Tuple[str, int]] = [(focus_id, 0)]
        visited: Set[Tuple[str, int]] = set()
        while stack:
            node_id, level = stack.pop()
            key = (node_id, level)
            if key in visited:
                continue
            visited.add(key)
            if depth_limit >= 0 and level >= depth_limit:
                continue
            for parent_id in effective_parent_map.get(node_id, []) or []:
                if parent_id not in ancestors or level + 1 < depth_map.get(parent_id, level + 1):
                    ancestors.add(parent_id)
                    depth_map[parent_id] = level + 1
                stack.append((parent_id, level + 1))
        return ancestors, depth_map

    def _build_tree_node(
        node_id: str,
        *,
        focus_id: str,
        focus_path: Set[str],
        visited: Set[str],
    ) -> Dict[str, Any]:
        if node_id in visited:
            node = _node_info(node_id)
            node["is_self"] = node_id == focus_id
            node["on_path"] = node_id in focus_path
            node["children"] = []
            return node
        visited = set(visited)
        visited.add(node_id)
        node = _node_info(node_id)
        node["is_self"] = node_id == focus_id
        node["on_path"] = node_id in focus_path
        child_nodes: List[Dict[str, Any]] = []
        for child_id in effective_children_map.get(node_id, []) or []:
            if child_id in focus_path or child_id == focus_id:
                child_node = _build_tree_node(
                    child_id,
                    focus_id=focus_id,
                    focus_path=focus_path,
                    visited=visited,
                )
            else:
                child_node = _node_info(child_id)
                child_node["is_self"] = child_id == focus_id
                child_node["on_path"] = False
                child_node["children"] = []
            if "children" not in child_node:
                child_node["children"] = []
            child_nodes.append(child_node)
        node["children"] = child_nodes
        return node

    def _collect_breadcrumbs(nodes: List[Dict[str, Any]]) -> List[List[Dict[str, Optional[str]]]]:
        paths: List[List[Dict[str, Optional[str]]]] = []

        def _walk(node: Dict[str, Any], current_path: List[Dict[str, Optional[str]]]) -> None:
            entry = {"cui": node.get("cui"), "name": node.get("name")}
            new_path = current_path + [entry]
            if node.get("is_self"):
                paths.append(new_path)
            for child in node.get("children", []) or []:
                if child.get("on_path") or child.get("is_self"):
                    _walk(child, new_path)

        for root_node in nodes:
            _walk(root_node, [])
        return paths

    relation_map: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    effective_parent_map: Dict[str, List[str]] = {}
    effective_children_map: Dict[str, List[str]] = defaultdict(list)

    if relations_cache:
        all_nodes: Set[str] = set(target_ids) | set(relations_cache.keys())
        for payload in relations_cache.values():
            for bucket in (payload.get("parents", []), payload.get("children", []), payload.get("siblings", [])):
                for entry in bucket or []:
                    cui = entry.get("cui") if isinstance(entry, dict) else None
                    if cui:
                        all_nodes.add(cui)

        for cui in all_nodes:
            cache_entry = relations_cache.get(cui, {})
            parents_payload: List[Dict[str, Optional[str]]] = []
            for parent in cache_entry.get("parents", []) or []:
                parent_cui = parent.get("cui") if isinstance(parent, dict) else None
                if not parent_cui:
                    continue
                parent_name = parent.get("name") if isinstance(parent, dict) else None
                if not parent_name:
                    parent_name = _node_info(parent_cui).get("name")
                parents_payload.append({"cui": parent_cui, "name": parent_name})

            children_payload: List[Dict[str, Optional[str]]] = []
            for child in cache_entry.get("children", []) or []:
                child_cui = child.get("cui") if isinstance(child, dict) else None
                if not child_cui:
                    continue
                child_name = child.get("name") if isinstance(child, dict) else None
                if not child_name:
                    child_name = _node_info(child_cui).get("name")
                children_payload.append({"cui": child_cui, "name": child_name})

            siblings_payload: List[Dict[str, Optional[str]]] = []
            for sibling in cache_entry.get("siblings", []) or []:
                sib_cui = sibling.get("cui") if isinstance(sibling, dict) else None
                if not sib_cui:
                    continue
                sibling_name = sibling.get("name") if isinstance(sibling, dict) else None
                if not sibling_name:
                    sibling_name = _node_info(sib_cui).get("name")
                siblings_payload.append({"cui": sib_cui, "name": sibling_name})

            relation_map[cui] = {
                "parents": parents_payload,
                "children": children_payload,
                "siblings": siblings_payload,
            }

            parent_ids = [entry["cui"] for entry in parents_payload if entry.get("cui")]
            if parent_ids:
                seen_parent_ids: List[str] = []
                for pid in parent_ids:
                    if pid not in seen_parent_ids:
                        seen_parent_ids.append(pid)
                effective_parent_map[cui] = seen_parent_ids
                for parent_id in parent_ids:
                    if cui not in effective_children_map[parent_id]:
                        effective_children_map[parent_id].append(cui)

            for child_entry in children_payload:
                child_cui = child_entry.get("cui")
                if child_cui and child_cui not in effective_children_map[cui]:
                    effective_children_map[cui].append(child_cui)
                if child_cui:
                    plist = effective_parent_map.setdefault(child_cui, [])
                    if cui not in plist:
                        plist.append(cui)

        if parent_limit >= 0:
            for cui, entries in effective_parent_map.items():
                effective_parent_map[cui] = entries[:parent_limit]
            for cui, entries in effective_children_map.items():
                unique_children: List[str] = []
                seen_children: Set[str] = set()
                for child_id in entries:
                    if child_id in seen_children:
                        continue
                    seen_children.add(child_id)
                    unique_children.append(child_id)
                effective_children_map[cui] = unique_children[:parent_limit]
    else:
        GENERIC_NAME_KEYWORDS = (
            "(mesh category)",
            "mesh tree",
            "mesh heading",
        )
        GENERIC_NAME_EXACT = {
            "topical descriptor",
            "topical descriptors",
            "geographicals",
            "persons",
            "disciplines and occupations",
            "anthropology, education, sociology and social phenomena",
            "information science",
            "analytical, diagnostic and therapeutic techniques and equipment",
            "organisms",
        }

        def _is_generic_cui(cui_id: Optional[str]) -> bool:
            if not cui_id:
                return False
            record = raw_records.get(cui_id)
            if not record:
                return False
            name = record.get("canonical_name") or record.get("preferred_name") or record.get("name")
            if not name:
                return False
            name_lower = name.strip().lower()
            if name_lower in GENERIC_NAME_EXACT:
                return True
            for keyword in GENERIC_NAME_KEYWORDS:
                if keyword in name_lower:
                    return True
            return False

        def _collect_meaningful_parents(
            node_id: str,
            depth: int = 0,
            seen: Optional[Set[str]] = None,
        ) -> List[str]:
            seen = set(seen or set())
            results: List[str] = []
            for parent_id in parent_map.get(node_id, []) or []:
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                if ancestor_depth >= 0 and depth >= ancestor_depth:
                    continue
                if _is_generic_cui(parent_id):
                    results.extend(_collect_meaningful_parents(parent_id, depth + 1, seen))
                else:
                    results.append(parent_id)
            deduped: List[str] = []
            seen_ids: Set[str] = set()
            for parent_id in results:
                if parent_id not in seen_ids:
                    deduped.append(parent_id)
                    seen_ids.add(parent_id)
            return deduped

        candidate_nodes = set(parent_map.keys()) | set(child_map.keys()) | target_ids

        for node_id in candidate_nodes:
            meaningful = _collect_meaningful_parents(node_id)
            if meaningful:
                effective_parent_map[node_id] = meaningful
                for parent_id in meaningful:
                    if node_id not in effective_children_map[parent_id]:
                        effective_children_map[parent_id].append(node_id)

        for node_id in candidate_nodes:
            for child_id in child_map.get(node_id, []) or []:
                if _is_generic_cui(child_id):
                    continue
                if child_id not in effective_children_map[node_id]:
                    effective_children_map[node_id].append(child_id)
                if not _is_generic_cui(node_id):
                    parent_list = effective_parent_map.setdefault(child_id, [])
                    if node_id not in parent_list:
                        parent_list.append(node_id)

        relation_map = {}
        all_nodes = set(candidate_nodes) | set(effective_parent_map.keys()) | set(effective_children_map.keys())
        for cui in all_nodes:
            parents_payload = [
                _node_info(parent_cui)
                for parent_cui in effective_parent_map.get(cui, [])
            ]
            children_payload = [
                _node_info(child_cui)
                for child_cui in effective_children_map.get(cui, [])
            ]
            sibling_ids: List[str] = []
            seen_siblings: Set[str] = set()
            for parent_cui in effective_parent_map.get(cui, []):
                for child_cui in effective_children_map.get(parent_cui, []) or []:
                    if child_cui == cui or child_cui in seen_siblings:
                        continue
                    seen_siblings.add(child_cui)
                    sibling_ids.append(child_cui)
            if parent_limit >= 0:
                sibling_ids = sibling_ids[:parent_limit]
            siblings_payload = [_node_info(sibling_cui) for sibling_cui in sibling_ids]

            relation_map[cui] = {
                "parents": parents_payload,
                "children": children_payload,
                "siblings": siblings_payload,
            }

        for payload in relation_map.values():
            for bucket in (payload["parents"], payload["children"], payload["siblings"]):
                for entry in bucket:
                    cui = entry.get("cui")
                    if cui and cui not in cuis:
                        related_neighbors.add(cui)

    context: Dict[str, Dict[str, Any]] = {}
    for cui, record in raw_records.items():
        types: List[str] = record.get("types", []) or []
        aliases: List[str] = record.get("aliases", []) or []
        canonical = record.get("canonical_name")
        synonyms: List[str] = []
        for alias in aliases:
            if alias and alias != canonical and alias not in synonyms:
                synonyms.append(alias)

        semantic_types: List[str] = []
        semantic_parent_entries: List[Dict[str, Optional[str]]] = []
        for type_id in types:
            label = type_id
            node = None
            if tree:
                try:
                    node = tree.get_node_from_id(type_id)
                    label = f"{node.full_name} ({type_id})"
                except KeyError:
                    label = type_id
            if label not in semantic_types:
                semantic_types.append(label)

            if node and tree:
                parent = tree.get_parent(node)
                steps = 0
                limit = parent_limit if parent_limit >= 0 else float("inf")
                while parent and steps < limit:
                    entry = {"cui": parent.type_id, "name": parent.full_name}
                    if entry not in semantic_parent_entries:
                        semantic_parent_entries.append(entry)
                    parent = tree.get_parent(parent)
                    steps += 1

        relation_data = relation_map.get(cui, {})

        relation_parent_entries: List[Dict[str, Optional[str]]] = []
        for parent_entry in relation_data.get("parents", []) or []:
            parent_cui = parent_entry.get("cui")
            if not parent_cui:
                continue
            parent_name = parent_entry.get("name") or _node_info(parent_cui).get("name")
            relation_parent_entries.append({"cui": parent_cui, "name": parent_name})

        combined_parents = relation_parent_entries + semantic_parent_entries

        # Deduplicate parent entries while preserving order.
        seen_parent_keys: Set[Tuple[Optional[str], Optional[str]]] = set()
        deduped_parents: List[Dict[str, Optional[str]]] = []
        for entry in combined_parents:
            key = (entry.get("cui"), entry.get("name"))
            if key in seen_parent_keys:
                continue
            seen_parent_keys.add(key)
            deduped_parents.append(entry)
        parent_entries = deduped_parents

        if parent_limit >= 0:
            parent_entries = parent_entries[:parent_limit]

        child_entries: List[Dict[str, Optional[str]]] = []
        for child_entry in relation_data.get("children", []) or []:
            child_cui = child_entry.get("cui")
            if not child_cui:
                continue
            child_name = child_entry.get("name") or _node_info(child_cui).get("name")
            child_entries.append({"cui": child_cui, "name": child_name})
        if parent_limit >= 0:
            child_entries = child_entries[:parent_limit]

        sibling_entries: List[Dict[str, Optional[str]]] = []
        for sibling_entry in relation_data.get("siblings", []) or []:
            sib_cui = sibling_entry.get("cui")
            if not sib_cui:
                continue
            sib_name = sibling_entry.get("name") or _node_info(sib_cui).get("name")
            sibling_entries.append({"cui": sib_cui, "name": sib_name})
        if parent_limit >= 0:
            sibling_entries = sibling_entries[:parent_limit]

        ancestors, ancestor_depths = _collect_ancestors(cui, ancestor_depth)
        focus_path_nodes = ancestors | {cui}

        ordered_ancestors = sorted(ancestor_depths.items(), key=lambda item: (-item[1], item[0]))
        root_candidates: List[str] = []
        seen_roots: Set[str] = set()
        for ancestor_id, _depth in ordered_ancestors:
            parents_of_ancestor = effective_parent_map.get(ancestor_id, []) or []
            if not any(parent in focus_path_nodes for parent in parents_of_ancestor):
                if ancestor_id not in seen_roots:
                    root_candidates.append(ancestor_id)
                    seen_roots.add(ancestor_id)

        if not root_candidates:
            root_candidates = [cui]
        elif cui not in focus_path_nodes:
            root_candidates.append(cui)

        ontology_tree: List[Dict[str, Any]] = []
        for root_id in root_candidates:
            ontology_tree.append(
                _build_tree_node(
                    root_id,
                    focus_id=cui,
                    focus_path=focus_path_nodes,
                    visited=set(),
                )
            )

        breadcrumbs = _collect_breadcrumbs(ontology_tree)

        context[cui] = {
            "semantic_types": semantic_types,
            "parents": parent_entries,
            "children": child_entries,
            "siblings": sibling_entries,
            "ontology_tree": ontology_tree,
            "breadcrumbs": breadcrumbs,
            "synonyms": synonyms,
        }

    return context


def build_cui_context_map(
    cuis: Set[str],
    *,
    kb_path: Optional[Path] = None,
    types_path: Optional[str] = None,
    mrrel_path: Optional[Path] = None,
    max_parents: int = 3,
    ancestor_depth: int = 2,
    relations_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Create a context map for CUIs preferring local data over network calls."""

    context = build_cui_context_from_local(
        kb_path,
        cuis,
        types_path=types_path,
        mrrel_path=mrrel_path,
        max_parents=max_parents,
        ancestor_depth=ancestor_depth,
        relations_cache=relations_cache,
    )

    return context


def _template_env() -> Environment:
    templates_dir = Path(__file__).resolve().parents[1] / "templates"
    loader = FileSystemLoader(str(templates_dir))
    env = Environment(
        loader=loader,
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
        auto_reload=True,  # Always reload templates to pick up changes
    )
    env.globals.update(zip=zip)  # Make zip available in templates if needed
    return env


def resolve_image_path(path_str: str, dataset: Optional[str]) -> Optional[Path]:
    path = Path(path_str)
    if path.exists():
        return path
    if dataset:
        root = DATASET_ROOTS.get(dataset.lower())
        if root:
            path_lower = path_str.lower()
            root_name = root.name.lower()
            if root_name in path_lower:
                idx = path_lower.index(root_name) + len(root_name)
                suffix = path_str[idx:].lstrip("/\\")
                candidate = root / suffix
                if candidate.exists():
                    return candidate
            candidate = root / Path(path_str).name
            if candidate.exists():
                return candidate
    return None


def encode_resolved_image(resolved: Path, max_size: int = 512) -> str:
    from io import BytesIO

    img = Image.open(resolved).convert("RGB")
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_image(path_str: str, dataset: Optional[str], max_size: int = 512) -> Optional[str]:
    try:
        resolved = resolve_image_path(path_str, dataset)
        if resolved is None:
            raise FileNotFoundError(f"{path_str} not found")
        return encode_resolved_image(resolved, max_size=max_size)
    except Exception as err:
        print(f"[warn] could not encode image {path_str}: {err}")
        return None


def _normalize_concept_format(concept: Dict) -> Dict:
    """Normalize concept from various formats to standard format with CUI."""
    # If already has CUI and preferred_name, return as-is
    if concept.get("cui") and (concept.get("preferred_name") or concept.get("name")):
        return concept

    # Handle format with 'concept' field (text name) instead of CUI
    normalized = concept.copy()

    # Extract concept name from various fields
    concept_name = concept.get("concept") or concept.get("preferred_name") or concept.get("name")
    if concept_name:
        # Use concept name as fallback identifier if no CUI
        if not normalized.get("cui"):
            normalized["cui"] = None  # Mark as having no CUI
        if not normalized.get("preferred_name") and not normalized.get("name"):
            normalized["preferred_name"] = concept_name

    # Ensure score is present (default to 1.0 for reference concepts without scores)
    if "score" not in normalized:
        normalized["score"] = 1.0 if concept.get("assertion") == "present" else 0.0

    return normalized


def build_reference_map(entries: Iterable[Dict]) -> Dict[str, Dict]:
    ref: Dict[str, Dict] = {}
    for entry in entries:
        key = entry.get("study_key") or entry.get("study_id")
        if not key:
            continue
        key_text = str(key)
        key_text = _sanitize_study_key(key_text)

        # Normalize entry format
        normalized_entry = entry.copy()
        normalized_entry["study_key"] = key_text

        # Extract dataset from metadata if present
        if "dataset" not in normalized_entry and "metadata" in normalized_entry:
            metadata = normalized_entry["metadata"]
            if isinstance(metadata, dict) and "dataset" in metadata:
                normalized_entry["dataset"] = metadata["dataset"]

        # Normalize concepts format
        if "concepts" in normalized_entry:
            normalized_entry["concepts"] = [
                _normalize_concept_format(c) for c in normalized_entry["concepts"]
            ]

        ref[study_key_short(key_text)] = normalized_entry
    return ref


def build_prediction_map(entries: Iterable[Dict]) -> Dict[str, Dict]:
    pred: Dict[str, Dict] = {}
    for entry in entries:
        key = entry.get("study_key") or entry.get("study_id")
        if not key:
            continue
        key_text = str(key)
        key_text = _sanitize_study_key(key_text)

        # Normalize entry format
        normalized_entry = entry.copy()
        normalized_entry["study_key"] = key_text

        # Extract dataset from metadata if present
        if "dataset" not in normalized_entry and "metadata" in normalized_entry:
            metadata = normalized_entry["metadata"]
            if isinstance(metadata, dict) and "dataset" in metadata:
                normalized_entry["dataset"] = metadata["dataset"]

        # Normalize concepts format
        if "concepts" in normalized_entry:
            normalized_entry["concepts"] = [
                _normalize_concept_format(c) for c in normalized_entry["concepts"]
            ]

        pred[study_key_short(key_text)] = normalized_entry
    return pred


REPORT_CACHE: Dict[str, Optional[str]] = {}


def fetch_report_text(
    study_key: str,
    dataset: Optional[str],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    dataset_norm = _normalize_dataset_name(dataset)
    short_key = study_key_short(study_key)
    cache_key = f"{dataset_norm}:{short_key}"
    if cache_key in REPORT_CACHE:
        return REPORT_CACHE[cache_key]
    patient_id = None
    study_id = short_key
    if "/" in short_key:
        patient_id, study_id = short_key.split("/", 1)
    try:
        text = lookup_report_text(dataset_norm, patient_id, study_id) if dataset_norm else None
    except Exception as err:
        print(f"[warn] could not fetch report for {study_key}: {err}")
        text = None
    meta_image_path = None
    if metadata and isinstance(metadata, Mapping):
        meta_image_path = metadata.get("image_path")
    if dataset_norm == "chexpert_plus":
        fallback = _lookup_chexpert_report(short_key, meta_image_path)
        if fallback:
            if not text or len(fallback) > len(text):
                text = fallback
    if text:
        text = _filter_report_sections(text)
    REPORT_CACHE[cache_key] = text
    return text


def build_image_map(dataset: str, allowed_keys: Optional[Set[str]] = None) -> Dict[str, List[Dict[str, str]]]:
    mapping: Dict[str, List[Dict[str, str]]] = {}
    dataset_norm = _normalize_dataset_name(dataset)
    dataset_norm = dataset_norm or ""
    records = iter_image_records(dataset_norm, allowed_keys=allowed_keys)
    for study_key, image_path in records:
        short_key = study_key_short(study_key)
        info: Dict[str, str] = {"path": image_path, "dataset": dataset_norm}
        normalized_key = _normalize_study_key(short_key, image_path)
        mapping.setdefault(short_key, []).append(info)
        if normalized_key != short_key:
            mapping.setdefault(normalized_key, []).append(info)
    return mapping


STATUS_ORDER = {"FN": 0, "FP": 1, "TP": 2, "TN": 3}
PATIENT_ID_RE = re.compile(r"patient(\d+)", re.IGNORECASE)
STUDY_ID_RE = re.compile(r"study(\d+)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^\s*([A-Z][A-Z\s/]+)\s*:\s*(.*)$", re.IGNORECASE)
WANTED_SECTIONS = ("FINDINGS", "IMPRESSION")


def _strip_prefix_casefold(value: str, prefix: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    pref = prefix.lower()
    lowered = text.lower()
    if lowered.startswith(pref):
        return text[len(prefix) :]
    return text


def _normalize_study_key(short_key: str, path_str: Optional[str]) -> str:
    if not path_str:
        return short_key
    patient_match = PATIENT_ID_RE.search(path_str)
    study_match = STUDY_ID_RE.search(path_str)
    if not patient_match:
        return short_key
    patient_id = patient_match.group(1)
    study_id = None
    if study_match:
        study_id = study_match.group(1)
    else:
        if "/" in short_key:
            tail = short_key.split("/", 1)[-1]
            match = STUDY_ID_RE.search(tail)
            if match:
                study_id = match.group(1)
    if study_id:
        return f"patient{patient_id}/study{study_id}"
    return f"patient{patient_id}"


def _parse_sections(text: str) -> Tuple[Dict[str, List[str]], List[str]]:
    sections: Dict[str, List[str]] = defaultdict(list)
    order: List[str] = []
    current: Optional[str] = None
    for raw_line in text.splitlines():
        match = SECTION_PATTERN.match(raw_line)
        if match:
            header = match.group(1).strip().upper()
            content = match.group(2)
            current = header
            if header not in sections:
                order.append(header)
            if content:
                sections[header].append(match.group(2))
            continue
        if current:
            sections[current].append(raw_line)
    return sections, order


def _filter_report_sections(text: str) -> str:
    if not text:
        return text
    sections, _ = _parse_sections(text)
    chunks: List[str] = []
    for header in WANTED_SECTIONS:
        content = sections.get(header)
        if content:
            chunk = f"{header.title()}:\n" + "\n".join(content).strip()
            chunks.append(chunk)
    return "\n\n".join(chunks) if chunks else text


def _format_score(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}"


def _safe_mean(values: Iterable[Optional[float]]) -> float:
    nums = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not nums:
        return 0.0
    return float(sum(nums) / len(nums))


def _concept_display_name(concept: Dict) -> str:
    name = concept.get("preferred_name") or concept.get("name") or concept.get("label")
    if name:
        return str(name)
    span = concept.get("span")
    if isinstance(span, str) and span.strip():
        return span.strip()
    mentions = concept.get("mentions")
    if isinstance(mentions, list) and mentions:
        for mention in mentions:
            text = mention.get("text") or mention.get("span_text")
            if text:
                return str(text)
    return "—"


def _concept_identifier(concept: Dict, fallback_idx: int) -> str:
    cui = concept.get("cui") or concept.get("concept_id")
    if cui:
        return f"cui:{cui}"
    name = _concept_display_name(concept)
    if name and name != "—":
        return f"name:{name.lower()}"
    span = concept.get("span")
    if isinstance(span, str) and span.strip():
        return f"span:{span.strip().lower()}"
    return f"anon:{fallback_idx}"


def _extract_spans(concept: Dict) -> List[str]:
    spans: List[str] = []
    span_field = concept.get("span")
    if isinstance(span_field, str) and span_field.strip():
        spans.append(span_field.strip())
    elif isinstance(span_field, list):
        spans.extend(str(s).strip() for s in span_field if str(s).strip())
    mentions = concept.get("mentions")
    if isinstance(mentions, list):
        for mention in mentions:
            if isinstance(mention, dict):
                text = mention.get("text") or mention.get("span_text")
                if isinstance(text, str) and text.strip():
                    spans.append(text.strip())
            elif isinstance(mention, str) and mention.strip():
                spans.append(mention.strip())
    extra_spans = concept.get("spans")
    if isinstance(extra_spans, list):
        for item in extra_spans:
            if isinstance(item, dict):
                text = item.get("text") or item.get("surface")
                if text and str(text).strip():
                    spans.append(str(text).strip())
            elif isinstance(item, str) and item.strip():
                spans.append(item.strip())
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for span in spans:
        norm = span.lower()
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(span)
    return unique


def merge_concepts_with_status(
    ref_concepts: Iterable[Dict],
    pred_concepts: Iterable[Dict],
    global_vocabulary: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    ref_map: Dict[str, Dict[str, Any]] = {}
    for idx, concept in enumerate(ref_concepts):
        key = _concept_identifier(concept, fallback_idx=idx)
        entry = ref_map.setdefault(
            key,
            {
                "concept": concept,
                "score": None,
                "spans": [],
            },
        )
        score = concept.get("score")
        if isinstance(score, (int, float)):
            prev = entry.get("score")
            entry["score"] = float(score if prev is None else max(prev, float(score)))
        entry["spans"].extend(_extract_spans(concept))

    pred_map: Dict[str, Dict[str, Any]] = {}
    for idx, concept in enumerate(pred_concepts):
        key = _concept_identifier(concept, fallback_idx=idx)
        entry = pred_map.setdefault(
            key,
            {
                "concept": concept,
                "score": None,
            },
        )
        score = concept.get("score")
        if isinstance(score, (int, float)):
            prev = entry.get("score")
            entry["score"] = float(score if prev is None else max(prev, float(score)))

    rows: List[Dict[str, Any]] = []
    # If global vocabulary provided, include all concepts (enables TN)
    if global_vocabulary:
        all_keys = global_vocabulary
    else:
        all_keys = set(ref_map.keys()) | set(pred_map.keys())

    for key in all_keys:
        ref_info = ref_map.get(key)
        pred_info = pred_map.get(key)

        # Handle TN case (concept not in either ref or pred)
        if not ref_info and not pred_info:
            # This is a TN - concept from global vocabulary not present in this study
            # Strip the prefix (e.g., "name:", "cui:", "span:") from the key
            if ":" in key:
                name = key.split(":", 1)[1]
            else:
                name = key
            concept_source = {"preferred_name": name, "cui": None}
            cui = None
            status = "TN"
        else:
            concept_source = ref_info["concept"] if ref_info else pred_info["concept"]
            name = _concept_display_name(concept_source)
            cui = concept_source.get("cui") or concept_source.get("concept_id")
            status = "TP" if ref_info and pred_info else ("FN" if ref_info else "FP")
        ref_score = ref_info.get("score") if ref_info else None
        pred_score = pred_info.get("score") if pred_info else None
        sort_score = pred_score if pred_score is not None else (ref_score if ref_score is not None else 0.0)
        ref_spans = ref_info.get("spans", []) if ref_info else []
        pred_spans = pred_info.get("spans", []) if pred_info else []
        synonyms: List[str] = []
        for info in (ref_info, pred_info):
            if info:
                syns = info.get("concept", {}).get("synonyms", [])
                if syns:
                    for s in syns:
                        if isinstance(s, str) and s.strip() and s not in synonyms:
                            synonyms.append(s.strip())
        rows.append(
            {
                "concept_key": key,
                "status": status,
                "status_order": STATUS_ORDER[status],
                "cui": cui,
                "name": name,
                "ref_score": ref_score,
                "pred_score": pred_score,
                "ref_present": bool(ref_info),
                "pred_present": bool(pred_info),
                "ref_spans": ref_spans,
                "pred_spans": pred_spans,
                "sort_score": sort_score,
                "synonyms": synonyms,
            }
        )

    rows.sort(key=lambda r: (r["status_order"], -r["sort_score"]))
    return rows


def compute_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    counts = Counter(row["status"] for row in rows)
    tp = counts.get("TP", 0)
    fp = counts.get("FP", 0)
    fn = counts.get("FN", 0)
    precision = tp / (tp + fp) if tp + fp > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _summarize_spans(spans: Iterable[str], max_items: int = 2) -> str:
    cleaned = [s for s in spans if isinstance(s, str) and s.strip()]
    if not cleaned:
        return "—"
    display = ", ".join(cleaned[:max_items])
    if len(cleaned) > max_items:
        display += " …"
    return display


def format_concept_rows(
    rows: Iterable[Dict[str, Any]],
    context_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for row in rows:
        status = row.get("status", "")
        cui = row.get("cui") or "—"
        context = context_map.get(cui) if context_map and isinstance(cui, str) else {}
        semantic_types = context.get("semantic_types", []) if isinstance(context, dict) else []
        parents = context.get("parents", []) if isinstance(context, dict) else []
        children = context.get("children", []) if isinstance(context, dict) else []
        siblings = context.get("siblings", []) if isinstance(context, dict) else []
        ontology_tree = context.get("ontology_tree", []) if isinstance(context, dict) else []
        breadcrumbs = context.get("breadcrumbs", []) if isinstance(context, dict) else []
        synonyms = row.get("synonyms", []) or []
        has_ontology = bool(ontology_tree)
        formatted.append(
            {
                "status": status,
                "status_class": f"status-{status.lower()}" if status else "",
                "cui": cui,
                "name": row.get("name") or "—",
                "ref_score": row.get("ref_score"),
                "ref_score_display": _format_score(row.get("ref_score")),
                "pred_score": row.get("pred_score"),
                "pred_score_display": _format_score(row.get("pred_score")),
                "ref_present": row.get("ref_present", False),
                "pred_present": row.get("pred_present", False),
                "evidence_display": _summarize_spans(row.get("ref_spans", [])),
                "concept_key": row.get("concept_key"),
                "semantic_types": semantic_types,
                "parents": parents,
                "children": children,
                "siblings": siblings,
                "ontology_tree": ontology_tree,
                "breadcrumbs": breadcrumbs,
                "has_ontology": has_ontology,
                "synonyms": synonyms,
                "synonyms_display": _summarize_spans(synonyms, max_items=3),
            }
        )
    return formatted


def highlight_report_text(report_text: Optional[str], rows: Iterable[Dict[str, Any]]) -> Optional[str]:
    if not report_text:
        return None
    span_lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not row.get("ref_spans"):
            continue
        status = row.get("status", "TP")
        name = row.get("name") or ""
        cui = row.get("cui") or ""
        for span in row["ref_spans"]:
            if not span:
                continue
            span_lookup.setdefault(
                span.lower(),
                {
                    "span": span,
                    "statuses": set(),
                    "concepts": set(),
                    "cuis": set(),
                },
            )
            entry = span_lookup[span.lower()]
            entry["statuses"].add(status)
            if name:
                entry["concepts"].add(name)
            if cui:
                entry["cuis"].add(cui)
    if not span_lookup:
        return html.escape(report_text)

    # Sort spans by length (desc) to avoid partial matches shadowing longer phrases
    sorted_spans = sorted(span_lookup.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(span) for span in sorted_spans), re.IGNORECASE)
    result_parts: List[str] = []
    last_idx = 0
    for match in pattern.finditer(report_text):
        start, end = match.span()
        result_parts.append(html.escape(report_text[last_idx:start]))
        matched_text = match.group(0)
        lookup = span_lookup.get(matched_text.lower())
        if not lookup:
            normalized = matched_text.lower()
            for key, entry in span_lookup.items():
                if key == normalized:
                    lookup = entry
                    break
        if not lookup:
            safe_text = html.escape(matched_text)
            result_parts.append(safe_text)
            last_idx = end
            continue
        statuses = sorted(lookup.get("statuses", []))
        primary_status = statuses[0].lower() if statuses else "tp"
        concept_list = sorted(lookup.get("concepts", []))
        cui_list = sorted(lookup.get("cuis", []))
        data_attrs = []
        if concept_list:
            data_attrs.append(f"data-concepts='{html.escape('; '.join(concept_list))}'")
        if cui_list:
            data_attrs.append(f"data-cuis='{html.escape('; '.join(cui_list))}'")
        data_attr_str = " ".join(data_attrs)
        safe_text = html.escape(matched_text)
        result_parts.append(
            f"<mark class='concept-highlight status-{primary_status}' {data_attr_str}>{safe_text}</mark>"
        )
        last_idx = end
    result_parts.append(html.escape(report_text[last_idx:]))
    return "".join(result_parts)


def prepare_image_assets(
    image_infos: List[Dict[str, str]],
    dataset_label: Optional[str],
    max_main: int = 512,
    max_thumb: int = 140,
) -> Tuple[List[Dict[str, Optional[str]]], Optional[str]]:
    assets: List[Dict[str, Optional[str]]] = []
    first_thumb: Optional[str] = None
    for idx, info in enumerate(image_infos[:6]):
        path_str = info.get("path") or info.get("image_path")
        if not path_str:
            continue
        dataset = info.get("dataset") or dataset_label
        resolved = resolve_image_path(path_str, dataset)
        resolved_str = str(resolved) if resolved else None
        base_b64 = encode_image(path_str, dataset, max_size=max_main)
        thumb_b64 = encode_image(path_str, dataset, max_size=max_thumb)
        cam_path = info.get("cam_path")
        cam_b64 = None
        cam_thumb = None
        if cam_path:
            cam_b64 = encode_image(cam_path, dataset_label, max_size=max_main)
            if cam_b64 is None:
                cam_b64 = encode_image(cam_path, None, max_size=max_main)
            cam_thumb = encode_image(cam_path, dataset_label, max_size=max_thumb)
            if cam_thumb is None:
                cam_thumb = encode_image(cam_path, None, max_size=max_thumb)
        filename_source = resolved_str or path_str
        filename = Path(filename_source).name if filename_source else "Image"
        raw_display = resolved_str or path_str or "—"
        raw_href: Optional[str] = None
        if resolved:
            try:
                raw_href = resolved.resolve().as_uri()
            except Exception:
                raw_href = resolved_str
        elif path_str:
            try:
                raw_href = Path(path_str).resolve().as_uri()
            except Exception:
                raw_href = path_str
        asset = {
            "path": path_str,
            "resolved": resolved_str,
            "dataset": dataset,
            "base_b64": base_b64,
            "thumb_b64": thumb_b64,
            "cam_b64": cam_b64,
            "cam_thumb_b64": cam_thumb,
            "view": info.get("view"),
            "filename": filename,
            "raw_display": raw_display,
            "raw_href": raw_href,
            "index": idx,
        }
        assets.append(asset)
        if first_thumb is None:
            first_thumb = cam_thumb or thumb_b64
    return assets, first_thumb


def render_html(studies: List[str],
                reference: Dict[str, Dict],
                predictions: Dict[str, Dict],
                images: Dict[str, List[Dict[str, str]]],
                output_path: Path,
                context_map: Optional[Dict[str, Dict[str, Any]]] = None,
                default_dataset: Optional[str] = None,
                global_vocabulary: Optional[Set[str]] = None) -> None:
    env = _template_env()
    template = env.get_template(TEMPLATE_NAME)

    study_contexts: List[Dict[str, Any]] = []
    concept_aggregate: Dict[str, Dict[str, Any]] = {}
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []
    ref_counts: List[int] = []
    pred_counts: List[int] = []
    total_counts = Counter()

    for study in tqdm(studies, desc="render", unit="study"):
        ref_entry = reference.get(study, {})
        pred_entry = predictions.get(study, {})
        dataset_label = ref_entry.get("dataset") or pred_entry.get("dataset") or default_dataset
        dataset_label = _normalize_dataset_name(dataset_label)
        image_infos = images.get(study, [])

        ref_concepts = ref_entry.get("concepts", [])
        pred_concepts = pred_entry.get("concepts", [])
        merged_rows = merge_concepts_with_status(ref_concepts, pred_concepts, global_vocabulary=global_vocabulary)
        formatted_rows = format_concept_rows(merged_rows, context_map=context_map)
        metrics = compute_metrics(merged_rows)
        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        f1_values.append(metrics["f1"])
        tp_count = int(metrics["tp"])
        fp_count = int(metrics["fp"])
        fn_count = int(metrics["fn"])
        ref_count = sum(1 for row in merged_rows if row.get("ref_present"))
        pred_count = sum(1 for row in merged_rows if row.get("pred_present"))

        total_counts["tp"] += tp_count
        total_counts["fp"] += fp_count
        total_counts["fn"] += fn_count
        ref_counts.append(ref_count)
        pred_counts.append(pred_count)

        concept_terms: set[str] = set()
        for row in merged_rows:
            key = row["concept_key"]
            agg_entry = concept_aggregate.setdefault(
                key,
                {
                    "name": row.get("name"),
                    "cui": row.get("cui"),
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                },
            )
            status_key = row.get("status", "").lower()
            if status_key in ("tp", "fp", "fn"):
                agg_entry[status_key] += 1
            name = row.get("name")
            cui = row.get("cui")
            if isinstance(name, str) and name.strip():
                concept_terms.add(name.lower())
            if isinstance(cui, str) and cui.strip():
                concept_terms.add(cui.lower())
            for span in row.get("ref_spans") or []:
                if span:
                    concept_terms.add(span.lower())
        for row_fmt in formatted_rows:
            for syn in row_fmt.get("synonyms", []) or []:
                if isinstance(syn, str) and syn.strip():
                    concept_terms.add(syn.lower())
            for parent in row_fmt.get("parents", []) or []:
                if isinstance(parent, dict):
                    pname = parent.get("name")
                    if isinstance(pname, str) and pname.strip():
                        concept_terms.add(pname.lower())

        meta_source = None
        if ref_entry and isinstance(ref_entry.get("metadata"), Mapping):
            meta_source = ref_entry["metadata"]
        elif pred_entry and isinstance(pred_entry.get("metadata"), Mapping):
            meta_source = pred_entry["metadata"]
        report_text = fetch_report_text(study, dataset_label, metadata=meta_source)
        if not report_text:
            if not dataset_label:
                print(f"[warn] Cannot fetch report for {study}: no dataset specified")
            else:
                print(f"[debug] No report text found for study={study}, dataset={dataset_label}")
        highlighted_report = highlight_report_text(report_text, merged_rows) if report_text else None

        # Detect which sections are present in the report
        has_impression = bool(report_text and "IMPRESSION:" in report_text.upper())
        has_findings = bool(report_text and "FINDINGS:" in report_text.upper())

        image_assets, nav_thumb = prepare_image_assets(image_infos, dataset_label)

        study_contexts.append(
            {
                "study": study,
                "dataset": dataset_label,
                "concept_rows": formatted_rows,
                "metrics": metrics,
                "metrics_display": {
                    "f1": f"{metrics['f1']:.3f}",
                    "precision": f"{metrics['precision']:.3f}",
                    "recall": f"{metrics['recall']:.3f}",
                },
                "counts": {
                    "tp": tp_count,
                    "fp": fp_count,
                    "fn": fn_count,
                    "ref": ref_count,
                    "pred": pred_count,
                },
                "counts_display": {
                    "tp": f"TP {tp_count}",
                    "fp": f"FP {fp_count}",
                    "fn": f"FN {fn_count}",
                },
                "report_html": highlighted_report,
                "has_report": bool(highlighted_report),
                "has_impression": has_impression,
                "has_findings": has_findings,
                "images": image_assets,
                "nav_thumb": nav_thumb,
                "concept_terms": "|".join(sorted(concept_terms)),
            }
        )

    if not study_contexts:
        empty_html = """
        <!doctype html>
        <html><head><meta charset='utf-8'><title>Concept Report</title>
        <style>body { font-family: Arial, sans-serif; padding:40px; background:#f8fafc; color:#0f172a; }
        .empty { padding:40px; border:2px dashed #cbd5f5; border-radius:12px; background:#fff; max-width:520px; margin:80px auto; text-align:center; font-size:18px; }
        </style></head><body><div class='empty'>No studies available for rendering.</div></body></html>
        """
        output_path.write_text(empty_html, encoding="utf-8")
        print(f"[ok] wrote report to {output_path}")
        return

    study_contexts.sort(key=lambda ctx: (ctx["metrics"]["f1"], ctx["study"]))
    low_overlap_count = sum(1 for ctx in study_contexts if ctx["metrics"]["f1"] < 0.5)

    summary = {
        "total_studies": len(study_contexts),
        "mean_precision": _safe_mean(precision_values),
        "mean_recall": _safe_mean(recall_values),
        "mean_f1": _safe_mean(f1_values),
        "mean_ref_count": _safe_mean(ref_counts),
        "mean_pred_count": _safe_mean(pred_counts),
        "totals": {
            "tp": total_counts.get("tp", 0),
            "fp": total_counts.get("fp", 0),
            "fn": total_counts.get("fn", 0),
        },
        "low_overlap": low_overlap_count,
    }

    summary_display = {
        "mean_f1": f"{summary['mean_f1']:.3f}",
        "mean_precision": f"{summary['mean_precision']:.3f}",
        "mean_recall": f"{summary['mean_recall']:.3f}",
        "mean_ref_count": f"{summary['mean_ref_count']:.1f}",
        "mean_pred_count": f"{summary['mean_pred_count']:.1f}",
    }

    concept_errors: List[Dict[str, Any]] = []
    for entry in concept_aggregate.values():
        total = entry["tp"] + entry["fp"] + entry["fn"]
        if total == 0:
            continue
        errors = entry["fp"] + entry["fn"]
        error_rate = errors / total if total else 0.0
        concept_errors.append(
            {
                "name": entry.get("name"),
                "cui": entry.get("cui"),
                "error_rate": error_rate,
                "errors": errors,
                "tp": entry["tp"],
                "fp": entry["fp"],
                "fn": entry["fn"],
                "support": total,
                "error_rate_display": f"{error_rate:.2f}",
            }
        )
    concept_errors.sort(key=lambda row: (row["error_rate"], row["errors"], row["support"]), reverse=True)
    top_errors = concept_errors[:5]

    context = {
        "studies": study_contexts,
        "summary": summary,
        "summary_display": summary_display,
        "top_errors": top_errors,
        "filters": {
            "low_f1_threshold": 0.5,
        },
    }

    html_doc = template.render(**context)
    output_path.write_text(html_doc, encoding="utf-8")
    print(f"[ok] wrote report to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render concept comparison report.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., chexpert_plus, mimic_cxr)")
    parser.add_argument("--reference_per_study", required=True, help="Reference per-study concepts JSONL")
    parser.add_argument("--predictions_per_study", required=True, help="Predicted per-study concepts JSONL")
    parser.add_argument(
        "--cbm_concept_index",
        default=None,
        help="Optional concept_index.json to map CBM probability vectors to concept names (auto-guessed if omitted).",
    )
    parser.add_argument(
        "--cbm_reference_threshold",
        type=float,
        default=0.5,
        help="Threshold for treating CBM ground-truth probabilities as present concepts.",
    )
    parser.add_argument(
        "--cbm_pred_threshold",
        type=float,
        default=0.5,
        help="Threshold for treating CBM prediction probabilities as present concepts.",
    )
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of studies to include")
    parser.add_argument("--include_cui_context", action="store_true",
                        help="Include semantic types and parent links from a local UMLS knowledge base")
    parser.add_argument("--umls_local_kb", default="~/umls_linker/index",
                        help="Path to a local UMLS JSONL knowledge base (default: ~/umls_linker/index)")
    parser.add_argument("--umls_mrrel_path", default=None,
                        help="Optional path or directory containing MRREL relations (used for ontology context)")
    parser.add_argument("--umls_types_path", default=None,
                        help="Optional semantic type tree TSV (defaults to scispaCy bundled data)")
    parser.add_argument("--cui_context_max_parents", type=int, default=3,
                        help="Maximum number of parent concepts to show per CUI (<=0 for unlimited)")
    parser.add_argument("--cui_context_ancestor_depth", type=int, default=2,
                        help="Ancestor depth (levels) to traverse for ontology context (<=0 for unlimited)")
    parser.add_argument("--umls_relations_cache", default=None,
                        help="Optional JSONL cache of precomputed MRREL adjacency (skips live MRREL scan)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Clear any cached report data to ensure we use the latest code
    from src.extraction.dataset_iter import _chexpert_plus_report_dict, _mimic_cxr_report_dict
    _chexpert_plus_report_dict.cache_clear()
    _mimic_cxr_report_dict.cache_clear()
    REPORT_CACHE.clear()

    concept_names: Optional[List[str]] = None
    if args.cbm_concept_index:
        concept_names = _load_concept_index(Path(args.cbm_concept_index).expanduser())
        if concept_names:
            print(f"[info] loaded concept index with {len(concept_names)} entries from {args.cbm_concept_index}")
    if concept_names is None:
        guessed_index = _guess_concept_index(Path(args.predictions_per_study))
        if guessed_index:
            concept_names = _load_concept_index(guessed_index)
            if concept_names:
                print(f"[info] using concept index from {guessed_index}")

    ref_entries = load_jsonl(Path(args.reference_per_study))
    pred_entries = load_jsonl(Path(args.predictions_per_study))

    if concept_names is None and (
        any("probs" in entry and "concepts" not in entry for entry in ref_entries)
        or any("probs" in entry and "concepts" not in entry for entry in pred_entries)
    ):
        print("[warn] CBM-style probabilities detected but no concept index found; names will default to concept_{i}.")

    ref_entries = _convert_cbm_entries(
        ref_entries,
        concept_names=concept_names,
        threshold=args.cbm_reference_threshold,
        dataset=args.dataset,
    )
    pred_entries = _convert_cbm_entries(
        pred_entries,
        concept_names=concept_names,
        threshold=args.cbm_pred_threshold,
        dataset=args.dataset,
    )

    reference_map = build_reference_map(ref_entries)
    prediction_map = build_prediction_map(pred_entries)

    context_map: Dict[str, Dict[str, Any]] = {}
    if args.include_cui_context:
        cuis = collect_unique_cuis(reference_map, prediction_map)

        if not cuis:
            print("[info] No CUIs found in concepts. Skipping CUI context loading.")
            print("[info] Note: Concepts without CUI codes will be matched by name only.")
        else:
            print(f"[info] Found {len(cuis)} unique CUIs to look up")
            kb_path = Path(args.umls_local_kb).expanduser() if args.umls_local_kb else None
            mrrel_path = Path(args.umls_mrrel_path).expanduser() if args.umls_mrrel_path else None
            types_path = args.umls_types_path
            relations_cache: Optional[Dict[str, Dict[str, Any]]] = None
            if args.umls_relations_cache:
                cache_path = Path(args.umls_relations_cache).expanduser()
                if cache_path.exists():
                    try:
                        relations_cache = load_relations_cache(cache_path)
                        print(f"[info] loaded relations cache ({len(relations_cache)} CUIs) from {cache_path}")
                    except Exception as err:  # pragma: no cover - defensive
                        print(f"[warn] failed to load relations cache {cache_path}: {err}")
                else:
                    print(f"[warn] relations cache not found: {cache_path}")
            context_map = build_cui_context_map(
                cuis,
                kb_path=kb_path,
                types_path=types_path,
                mrrel_path=mrrel_path,
                max_parents=args.cui_context_max_parents,
                ancestor_depth=args.cui_context_ancestor_depth,
                relations_cache=relations_cache,
            )
            if not context_map:
                print(
                    "[warn] No CUI context could be assembled from the local knowledge base."
                )
                print(f"[info] Tried KB path: {kb_path}")
            else:
                print(f"[info] Successfully loaded context for {len(context_map)} CUIs")

    study_keys = sorted(set(reference_map.keys()) | set(prediction_map.keys()))
    if args.limit:
        study_keys = study_keys[: args.limit]

    # Collect global vocabulary (all unique concepts across dataset)
    global_vocabulary: Set[str] = set()
    for entry in list(reference_map.values()) + list(prediction_map.values()):
        for concept in entry.get("concepts", []):
            key = _concept_identifier(concept, fallback_idx=0)
            global_vocabulary.add(key)
    print(f"[info] Global vocabulary: {len(global_vocabulary)} unique concepts")

    image_map = build_image_map(args.dataset, allowed_keys=set(study_keys))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_html(study_keys, reference_map, prediction_map, image_map, output_path, context_map=context_map, default_dataset=args.dataset, global_vocabulary=global_vocabulary)


if __name__ == "__main__":
    main()
