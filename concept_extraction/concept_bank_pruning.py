#!/usr/bin/env python3
"""Prune a RadGraph/SapBERT-derived concept bank for CBM training.

This script post-processes:
  - a concept inventory (concept_inventory.json), and
  - per-study concepts (study_concepts.jsonl)

by applying configurable filters (frequency, category, assertions, name regexes),
then writes:
  - a pruned concept inventory,
  - a pruned per-study concepts file,
  - a concept index mapping concepts to integer ids, and optionally
  - a dense label matrix (study x concept) for CBM training.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

# Ensure repository root is on sys.path when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.paths import OUTPUTS  # noqa: E402
from concept_extraction.concepts import aggregation as concept_agg  # noqa: E402


Inventory = Dict[str, Dict[str, Any]]

# Heuristic pattern for concepts that encode overall normality / lack of acute disease.
NORMALITY_PATTERN = re.compile(
    r"(?:\bnormal\b|no acute|no significant|no abnormal|unremarkable)", re.IGNORECASE
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_bank_dir = OUTPUTS / "snomed_mimic_bank"
    default_inventory = default_bank_dir / "concept_inventory.json"
    default_study_concepts = default_bank_dir / "study_concepts.jsonl"
    parser = argparse.ArgumentParser(
        description=(
            "Prune a SapBERT-linked concept bank and per-study concepts to a "
            "smaller, CBM-friendly subset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inventory-path",
        type=str,
        default=str(default_inventory),
        help="Path to concept_inventory.json produced by build_concept_bank.py.",
    )
    parser.add_argument(
        "--study-concepts-path",
        type=str,
        default=str(default_study_concepts),
        help="Path to study_concepts.jsonl produced by build_concept_bank.py.",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=None,
        help=(
            "Optional path to concept_bank.meta.json. "
            "If omitted, will look next to --inventory-path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS / "snomed_mimic_bank_pruned"),
        help="Directory where pruned artifacts will be written.",
    )

    # Concept-level filters.
    parser.add_argument(
        "--include-category",
        action="append",
        choices=["finding", "anatomy_scaffold", "device"],
        default=None,
        help=(
            "Concept categories to keep. "
            "If omitted, only 'finding' concepts are retained."
        ),
    )
    parser.add_argument(
        "--min-total-occurrences",
        type=int,
        default=10,
        help="Minimum total instances (all assertions) required to keep a concept.",
    )
    parser.add_argument(
        "--min-present-occurrences",
        type=int,
        default=1,
        help="Minimum number of 'present' instances required to keep a concept.",
    )
    parser.add_argument(
        "--min-present-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of 'present' instances over total required to keep a concept. "
            "Set to 0.0 to disable."
        ),
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of concepts to keep; if set, the top-K concepts "
            "by total frequency (after other filters) are retained."
        ),
    )
    parser.add_argument(
        "--name-include-regex",
        type=str,
        default=None,
        help="Optional regex; only concepts whose canonical_name matches are kept.",
    )
    parser.add_argument(
        "--name-exclude-regex",
        type=str,
        default=None,
        help="Optional regex; concepts whose canonical_name matches are removed.",
    )

    parser.add_argument(
        "--vocab-allow-path",
        type=str,
        default=None,
        help=(
            "Optional JSON/YAML file listing allowed CUIs and/or concept names. "
            "If provided, concepts not present in this allowlist are dropped."
        ),
    )
    parser.add_argument(
        "--vocab-allow-mode",
        type=str,
        choices=["auto", "cui", "name", "both"],
        default="auto",
        help=(
            "How to interpret entries in --vocab-allow-path. "
            "'auto' treats C*-style tokens as CUIs and others as names."
        ),
    )
    parser.add_argument(
        "--drop-normality-concepts",
        action="store_true",
        help=(
            "If set, drop concepts whose canonical_name suggests overall normality or lack "
            "of acute pathology (e.g. 'normal study', 'no acute cardiopulmonary disease')."
        ),
    )

    # Label matrix options.
    parser.add_argument(
        "--emit-label-matrix",
        action="store_true",
        help=(
            "If set, emit a dense label matrix (labels.npz) with shape "
            "[num_studies, num_concepts] for CBM training."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multi_assertion"],
        default="binary",
        help=(
            "Encoding for labels when --emit-label-matrix is set. "
            "'binary' treats selected assertions as positive vs not; "
            "'multi_assertion' encodes absent/uncertain/present as integer codes."
        ),
    )
    parser.add_argument(
        "--label-positive-assertion",
        dest="label_positive_assertions",
        action="append",
        choices=["present", "absent", "uncertain"],
        default=None,
        help=(
            "Assertion values to treat as positive when building the label matrix. "
            "Can be specified multiple times. Defaults to ['present']."
        ),
    )

    parser.add_argument(
        "--normalize-locations",
        action="store_true",
        help=(
            "If set, normalize location strings in per-study concepts to a coarse vocabulary "
            "(e.g. left/right/upper/lower/heart/lungs/pleura/mediastinum). "
            "Labels remain per-concept (location-agnostic)."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce stdout logging.",
    )
    return parser.parse_args(argv)


def load_inventory(path: Path) -> Inventory:
    if not path.exists():
        raise SystemExit(f"Concept inventory not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise SystemExit(f"Unexpected inventory format in {path}: expected object at top level.")
    return dict(payload)


def load_vocab_allowlist(path_str: Optional[str], mode: str) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """Load an allowlist of CUIs and/or canonical names from JSON or YAML."""
    if not path_str:
        return None, None
    path = Path(path_str).expanduser()
    if not path.exists():
        raise SystemExit(f"Vocab allowlist file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            raise SystemExit(f"Failed to parse vocab allowlist {path}: {exc}") from exc

    allowed_cuis: Set[str] = set()
    allowed_names: Set[str] = set()

    def _is_cui_like(token: str) -> bool:
        return bool(re.fullmatch(r"C\d+", token.strip()))

    def add_token(token: str) -> None:
        token = token.strip()
        if not token:
            return
        is_cui = _is_cui_like(token)
        if mode == "cui":
            if is_cui:
                allowed_cuis.add(token)
            return
        if mode == "name":
            allowed_names.add(token)
            return
        if mode == "both":
            if is_cui:
                allowed_cuis.add(token)
            else:
                allowed_names.add(token)
            return
        # auto
        if is_cui:
            allowed_cuis.add(token)
        else:
            allowed_names.add(token)

    def add_entry(entry: Any) -> None:
        if isinstance(entry, str):
            add_token(entry)
        elif isinstance(entry, Mapping):
            cui = entry.get("cui")
            name = entry.get("name") or entry.get("canonical_name")
            if cui:
                add_token(str(cui))
            if name:
                allowed_names.add(str(name).strip())

    if isinstance(data, list):
        for item in data:
            add_entry(item)
    elif isinstance(data, Mapping):
        for key in ("cui", "cuis", "names", "concepts"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    add_entry(item)
            elif isinstance(value, str):
                add_entry(value)
    else:
        raise SystemExit(f"Unsupported vocab allowlist format in {path}")

    return (allowed_cuis or None), (allowed_names or None)


def compile_regex(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise SystemExit(f"Invalid regex pattern {pattern!r}: {exc}") from exc


def filter_inventory(
    inventory: Inventory,
    args: argparse.Namespace,
    allowed_cuis: Optional[Set[str]],
    allowed_names: Optional[Set[str]],
) -> Inventory:
    include_categories = set(args.include_category or ["finding"])
    name_include_pattern = compile_regex(args.name_include_regex)
    name_exclude_pattern = compile_regex(args.name_exclude_regex)

    pruned_inventory: Inventory = {}
    for canonical_name, entry in inventory.items():
        category = entry.get("category")
        if category not in include_categories:
            continue
        stats = entry.get("stats", {}) or {}
        total_count = int(stats.get("total", 0))
        assertion_counts = stats.get("by_assertion", {}) or {}
        present_count = int(assertion_counts.get("present", 0))

        if total_count < int(args.min_total_occurrences):
            continue
        if present_count < int(args.min_present_occurrences):
            continue
        if args.min_present_fraction > 0.0:
            if total_count == 0:
                continue
            present_fraction = present_count / float(total_count)
            if present_fraction < float(args.min_present_fraction):
                continue

        if name_include_pattern is not None and not name_include_pattern.search(canonical_name):
            continue
        if name_exclude_pattern is not None and name_exclude_pattern.search(canonical_name):
            continue

        if args.drop_normality_concepts and NORMALITY_PATTERN.search(canonical_name):
            continue

        if allowed_cuis or allowed_names:
            cui = entry.get("cui")
            if allowed_cuis and cui not in allowed_cuis:
                continue
            if allowed_names and canonical_name not in allowed_names:
                continue

        pruned_inventory[canonical_name] = entry

    if args.max_concepts is not None and len(pruned_inventory) > int(args.max_concepts):
        # Keep the top-K concepts by total count after all other filters.
        sorted_names = sorted(
            pruned_inventory.keys(),
            key=lambda name: (-int(pruned_inventory[name].get("stats", {}).get("total", 0)), name),
        )
        keep_count = int(args.max_concepts)
        keep_names = set(sorted_names[:keep_count])
        pruned_inventory = {name: pruned_inventory[name] for name in sorted_names if name in keep_names}

    return pruned_inventory


def build_concept_index(pruned_inventory: Inventory) -> Tuple[List[str], Dict[str, int], Dict[str, Any]]:
    """Build ordered concept list, name->index mapping, and index payload."""
    sorted_names: List[str] = sorted(
        pruned_inventory.keys(),
        key=lambda name: (-int(pruned_inventory[name].get("stats", {}).get("total", 0)), name),
    )
    name_to_index: Dict[str, int] = {name: index for index, name in enumerate(sorted_names)}
    concept_records: List[Dict[str, Any]] = []
    for index, name in enumerate(sorted_names):
        entry = pruned_inventory[name]
        concept_records.append(
            {
                "index": index,
                "name": name,
                "cui": entry.get("cui"),
                "category": entry.get("category"),
                "semantic_type": entry.get("semantic_type"),
                "stats": entry.get("stats"),
            }
        )
    index_payload: Dict[str, Any] = {"concepts": concept_records}
    return sorted_names, name_to_index, index_payload


def load_source_meta(meta_path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not meta_path.exists():
        return None, None
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return None, None
    radgraph_version = payload.get("radgraph_version")  # type: ignore[assignment]
    linker_model = payload.get("linker_model")  # type: ignore[assignment]
    return radgraph_version, linker_model


def normalize_location_value(raw: Optional[str]) -> Optional[str]:
    """Normalize free-form location text to a coarse vocabulary."""
    if not raw:
        return None
    text = raw.lower()

    laterality = None
    if "bilateral" in text:
        laterality = "bilateral"
    elif "right" in text:
        laterality = "right"
    elif "left" in text:
        laterality = "left"

    region = None
    if any(token in text for token in ("apex", "apical", "upper")):
        region = "upper"
    elif any(token in text for token in ("base", "basilar", "lower")):
        region = "lower"

    structure = None
    if any(token in text for token in ("cardiac", "heart")):
        structure = "heart"
    elif any(token in text for token in ("lung", "pulmonary")):
        structure = "lungs"
    elif "pleural" in text:
        structure = "pleura"
    elif "mediastin" in text:
        structure = "mediastinum"
    elif any(token in text for token in ("hilar", "hilum")):
        structure = "hilum"
    elif any(token in text for token in ("abdomen", "abdominal")):
        structure = "abdomen"

    parts = [part for part in (laterality, region, structure) if part]
    if not parts:
        stripped = text.strip()
        return stripped or None
    return " ".join(parts)


def prune_study_concepts(
    source_path: Path,
    destination_path: Path,
    name_to_index: Mapping[str, int],
    positive_assertions: Sequence[str],
    emit_label_matrix: bool,
    label_mode: str,
    normalize_locations: bool,
    quiet: bool,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Stream through study_concepts.jsonl, filter to pruned concepts, and optionally build labels."""
    positive_set = {value.lower() for value in positive_assertions}

    study_ids: List[str] = []
    row_positive_indices: List[List[int]] = []
    row_state_maps: List[Dict[int, str]] = []

    studies_kept = 0

    with source_path.open("r", encoding="utf-8") as source_handle, destination_path.open(
        "w", encoding="utf-8"
    ) as dest_handle:
        for raw_line in source_handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            concepts = record.get("concepts") or []
            filtered_concepts: List[Dict[str, Any]] = []
            positive_indices_for_study: List[int] = []
            state_for_study: Dict[int, str] = {}

            for concept_entry in concepts:
                concept_name = concept_entry.get("concept")
                if concept_name not in name_to_index:
                    continue
                if normalize_locations:
                    concept_entry["location"] = normalize_location_value(concept_entry.get("location"))
                filtered_concepts.append(concept_entry)
                assertion_value = (concept_entry.get("assertion") or "").lower()
                concept_index = name_to_index[concept_name]
                if emit_label_matrix:
                    if label_mode == "binary":
                        if assertion_value in positive_set:
                            positive_indices_for_study.append(concept_index)
                    else:
                        if assertion_value in concept_agg.ASSERTION_ALLOWED:
                            prev = state_for_study.get(concept_index)
                            prev_score = concept_agg.ASSERTION_PRECEDENCE.get(prev, -1) if prev else -1
                            curr_score = concept_agg.ASSERTION_PRECEDENCE.get(assertion_value, -1)
                            if curr_score > prev_score:
                                state_for_study[concept_index] = assertion_value

            if not filtered_concepts:
                continue

            record["concepts"] = filtered_concepts
            dest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            studies_kept += 1

            if emit_label_matrix:
                study_ids.append(str(record.get("study_id")))
                if label_mode == "binary":
                    # Deduplicate indices while preserving deterministic order.
                    if positive_indices_for_study:
                        dedup_indices = sorted(set(positive_indices_for_study))
                    else:
                        dedup_indices = []
                    row_positive_indices.append(dedup_indices)
                else:
                    row_state_maps.append(state_for_study)

    label_payload: Optional[Dict[str, Any]] = None
    if emit_label_matrix:
        if np is None:
            raise SystemExit(
                "numpy is required to emit the label matrix but is not installed. "
                "Install it or rerun without --emit-label-matrix."
            )
        num_studies = len(study_ids)
        num_concepts = len(name_to_index)
        if not quiet:
            print(f"[info] building dense label matrix of shape ({num_studies}, {num_concepts})")
        if label_mode == "binary":
            labels = np.zeros((num_studies, num_concepts), dtype=bool)
            for study_index, indices in enumerate(row_positive_indices):
                if indices:
                    labels[study_index, indices] = True
        else:
            # Integer codes: 0=none, 1=absent, 2=uncertain, 3=present.
            labels = np.zeros((num_studies, num_concepts), dtype=np.int8)
            for study_index, state_map in enumerate(row_state_maps):
                for concept_index, assertion_value in state_map.items():
                    code = 0
                    if assertion_value == "absent":
                        code = 1
                    elif assertion_value == "uncertain":
                        code = 2
                    elif assertion_value == "present":
                        code = 3
                    if code:
                        labels[study_index, concept_index] = code
        label_payload = {
            "labels": labels,
            "study_ids": study_ids,
        }

    return studies_kept, label_payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.label_positive_assertions:
        positive_assertions = [value.lower() for value in args.label_positive_assertions]
    else:
        positive_assertions = ["present"]

    inventory_path = Path(args.inventory_path).expanduser()
    study_concepts_path = Path(args.study_concepts_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_cuis: Optional[Set[str]]
    allowed_names: Optional[Set[str]]
    allowed_cuis, allowed_names = load_vocab_allowlist(args.vocab_allow_path, args.vocab_allow_mode)

    if not args.quiet:
        print(f"[info] loading concept inventory from {inventory_path}")
    inventory = load_inventory(inventory_path)
    pruned_inventory = filter_inventory(inventory, args, allowed_cuis, allowed_names)
    if not pruned_inventory:
        raise SystemExit("No concepts remained after pruning; adjust your thresholds and try again.")

    sorted_names, name_to_index, index_payload = build_concept_index(pruned_inventory)

    meta_path = Path(args.meta_path).expanduser() if args.meta_path else inventory_path.with_name(
        "concept_bank.meta.json"
    )
    radgraph_version, linker_model = load_source_meta(meta_path)
    concept_count = len(pruned_inventory)
    instance_count = sum(int(entry.get("stats", {}).get("total", 0)) for entry in pruned_inventory.values())
    meta_payload = concept_agg.build_meta_payload(
        concept_count=concept_count,
        instance_count=instance_count,
        radgraph_version=radgraph_version,
        linker_model=linker_model,
    )

    inventory_out_path = output_dir / "concept_inventory.pruned.json"
    meta_out_path = output_dir / "concept_bank.pruned.meta.json"
    index_out_path = output_dir / "concept_index.json"
    study_out_path = output_dir / "study_concepts.pruned.jsonl"

    if not args.quiet:
        print(f"[info] writing pruned concept inventory with {concept_count} concepts to {inventory_out_path}")
    with inventory_out_path.open("w", encoding="utf-8") as handle:
        json.dump(pruned_inventory, handle, ensure_ascii=False, indent=2)

    with meta_out_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, ensure_ascii=False, indent=2)

    with index_out_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, ensure_ascii=False, indent=2)

    if not study_concepts_path.exists():
        raise SystemExit(f"Study concepts file not found: {study_concepts_path}")
    if not args.quiet:
        print(f"[info] pruning per-study concepts from {study_concepts_path}")
    emit_label_matrix = bool(args.emit_label_matrix)
    studies_kept, label_payload = prune_study_concepts(
        study_concepts_path,
        study_out_path,
        name_to_index,
        positive_assertions=positive_assertions,
        emit_label_matrix=emit_label_matrix,
        label_mode=args.label_mode,
        normalize_locations=args.normalize_locations,
        quiet=args.quiet,
    )

    if emit_label_matrix and label_payload is not None:
        if np is None:
            raise SystemExit(
                "numpy is required to emit the label matrix but is not installed. "
                "Install it or rerun without --emit-label-matrix."
            )
        labels = label_payload["labels"]
        study_ids = np.array(label_payload["study_ids"])
        concept_names = np.array(sorted_names)
        cuis = np.array([pruned_inventory[name].get("cui") for name in sorted_names])
        labels_out_path = output_dir / "labels.npz"
        if not args.quiet:
            print(f"[info] writing label matrix to {labels_out_path}")
        extra = {"label_mode": np.array(args.label_mode)}
        if args.label_mode == "multi_assertion":
            extra["assertion_codes"] = np.array(["none", "absent", "uncertain", "present"])
        np.savez_compressed(
            labels_out_path,
            labels=labels,
            study_ids=study_ids,
            concept_names=concept_names,
            cuis=cuis,
            **extra,
        )

    if not args.quiet:
        print(f"[done] pruned concept inventory   : {inventory_out_path}")
        print(f"[done] pruned concept meta       : {meta_out_path}")
        print(f"[done] concept index             : {index_out_path}")
        print(f"[done] pruned per-study concepts : {study_out_path}")
        print(f"[info] studies with at least one pruned concept: {studies_kept}")


if __name__ == "__main__":
    main(sys.argv[1:])
