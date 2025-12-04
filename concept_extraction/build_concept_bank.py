#!/usr/bin/env python3
"""Build a SapBERT-linked UMLS concept bank following the ontology-concept-distillation pipeline.

Pipeline summary:
  1. Detect entities + assertions with RadGraph-XL (via ClinicalEntityLinker fixes).
  2. Restrict the SapBERT FAISS index to ontology strings allowed by semantic-type filters.
  3. Link each mention to a CUI (SapBERT + FAISS + semantic-type gate).
  4. Canonicalize linked mentions into per-study CUI sets and aggregate a concept inventory.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml
import sys

# Ensure repository root is on sys.path when invoked as a script (e.g., python concept_extraction/build_concept_bank.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from lib.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
from concept_extraction.concepts import aggregation as concept_agg
from concept_extraction.concepts import input as concept_input
from concept_extraction.concepts import ner as concept_ner

ASSERTION_ALLOWED = concept_agg.ASSERTION_ALLOWED
ASSERTION_PRECEDENCE = concept_agg.ASSERTION_PRECEDENCE
DEVICE_STY = concept_agg.DEVICE_STY
FINDING_STY = concept_agg.FINDING_STY
ANATOMY_STY = concept_agg.ANATOMY_STY
RADGRAPH_VERSION = "radgraph-xl"
DEFAULT_TEXT_COLUMN = concept_input.DEFAULT_TEXT_COLUMN

DATASET_PRESETS = concept_input.DATASET_PRESETS
DATASET_ID_COLUMNS = concept_input.DATASET_ID_COLUMNS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a UMLS/SNOMED-CT concept bank by running the ontology-concept-distillation "
            "entity-linking pipeline end-to-end on report text."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional CSV file containing free-text reports. "
        "If omitted, a dataset-specific default will be used when available.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Column with report text (auto-inferred for known datasets; defaults to section_impression otherwise).",
    )
    parser.add_argument(
        "--config-path",
        default="cfg/paths.yml",
        help="YAML config containing SapBERT/UMLS resource paths (defaults to cfg/paths.yml, 'umls' section).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for concept_inventory + study_concepts outputs.")
    parser.add_argument(
        "--annotation-path",
        action="append",
        dest="annotation_paths",
        default=None,
        help="Optional RadGraph annotation JSON to reuse (can be specified multiple times).",
    )
    parser.add_argument(
        "--dataset",
        choices=["chexpert_plus", "mimic_cxr"],
        default=None,
        help="Dataset slug used to derive study IDs compatible with the rest of the codebase.",
    )
    parser.add_argument("--patient-column", default=None, help="Patient column (required for --dataset chexpert_plus).")
    parser.add_argument("--study-column", default=None, help="Study column (required for dataset-aware study IDs).")
    parser.add_argument(
        "--mimic-reports-dir",
        default=None,
        help="Override path to the MIMIC-CXR reports root (defaults to ~/datasets/mimic-cxr-reports).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    return parser.parse_args(argv)


def iter_progress(iterable: Iterable[Any], *, total: Optional[int], desc: str, unit: str) -> Iterable[Any]:
    if tqdm is None or total is None or total <= 0:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


apply_dataset_defaults = concept_input.apply_dataset_defaults
resolve_id_columns = concept_input.resolve_id_columns
normalize_record_id = concept_input.normalize_record_id
load_annotation_payload = concept_input.load_annotation_payload
build_annotation_index = concept_input.build_annotation_index
derive_study_key = concept_input.derive_study_key
aggregate_mentions = concept_agg.aggregate_mentions
finalize_study_entries = concept_agg.finalize_study_entries
update_inventory = concept_agg.update_inventory
write_json = concept_agg.write_json
build_meta_payload = concept_agg.build_meta_payload


def prepare_input_dataframe(args) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str], str]:
    return concept_input.prepare_input_dataframe(
        csv_path=args.csv_path,
        dataset=args.dataset,
        text_column=args.text_column,
        patient_column=args.patient_column,
        study_column=args.study_column,
        mimic_reports_dir=args.mimic_reports_dir,
        quiet=args.quiet,
        iter_progress=iter_progress,
        default_text_column=DEFAULT_TEXT_COLUMN,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config_path).expanduser()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    df, text_column, patient_column, study_column, section_label = prepare_input_dataframe(args)
    if not args.quiet:
        print(
            f"[info] loaded {len(df):,} reports "
            f"(text_column='{text_column}', patient_column='{patient_column}', study_column='{study_column}')"
        )
    id_columns = resolve_id_columns(df, args.dataset)

    annotation_index = None
    if args.annotation_paths:
        combined_text = defaultdict(list)
        combined_record = defaultdict(list)
        total = 0
        for path_str in args.annotation_paths:
            ann_path = Path(path_str).expanduser()
            if not ann_path.exists():
                raise SystemExit(f"Annotation file not found: {ann_path}")
            payload = load_annotation_payload(ann_path)
            index_part = build_annotation_index(payload, dataset=args.dataset)
            for key, values in index_part["text"].items():
                combined_text[key].extend(values)
                total += len(values)
            for key, values in index_part["record"].items():
                combined_record[key].extend(values)
        annotation_index = {"text": dict(combined_text), "record": dict(combined_record)}
        if not args.quiet:
            print(f"[info] loaded {total:,} precomputed RadGraph docs from {len(args.annotation_paths)} file(s)")
    elif not args.quiet:
        print("[info] no precomputed RadGraph annotations supplied; will run RadGraph model on the fly.")

    with config_path.open("r", encoding="utf-8") as handle:
        config_payload = yaml.safe_load(handle) or {}
    umls_cfg = config_payload.get("umls", config_payload)
    # Merge in semantic filters from umls_sapbert.yml if the main config lacks them.
    semantic_path = Path("cfg/umls_sapbert.yml")
    if semantic_path.exists():
        semantic_cfg = yaml.safe_load(semantic_path.read_text()) or {}
        umls_cfg.setdefault("allowed_tuis", semantic_cfg.get("allowed_tuis"))
        umls_cfg.setdefault("sources", semantic_cfg.get("sources"))
        umls_cfg.setdefault("allowed_sources", semantic_cfg.get("allowed_sources"))
        if semantic_cfg.get("radlex_csv_path") and not umls_cfg.get("radlex_csv_path"):
            umls_cfg["radlex_csv_path"] = semantic_cfg["radlex_csv_path"]
    sapbert_model_id = umls_cfg.get("sapbert_model_id")

    min_link_score = umls_cfg.get("min_link_score")
    min_link_score = 0.8 if min_link_score is None else float(min_link_score)
    if not args.quiet:
        print("[info] initializing ClinicalEntityLinker and SapBERT resources…")
    # Write UMLs config to a temp file to keep linker input clean
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        yaml.safe_dump(umls_cfg, tmp)
        umls_cfg_path = Path(tmp.name)

    linker = concept_ner.create_linker(
        umls_cfg_path,
        annotation_index=annotation_index,
        min_link_score=min_link_score,
    )
    if sapbert_model_id is None:
        sapbert_model_id = getattr(linker.__class__, "SAPBERT_MODEL_ID", None)

    if not args.quiet:
        print("[info] linking reports and aggregating concept bank artifacts…")

    study_output_path = output_dir / "study_concepts.jsonl"
    inventory: Dict[str, Dict[str, Any]] = {}
    instance_total = 0
    studies_written = 0

    total_rows = len(df)
    row_iter = iter_progress(df.itertuples(index=True), total=total_rows, desc="processing reports", unit="report")

    with study_output_path.open("w", encoding="utf-8") as study_sink:
        for row in row_iter:
            record = row._asdict()
            raw_text = record.get(text_column, "")
            text = "" if pd.isna(raw_text) else str(raw_text).strip()
            if not text:
                continue
            record_id = normalize_record_id(record, id_columns)
            study_key = None
            if args.dataset:
                study_key = derive_study_key(args.dataset, record, patient_column, study_column, fallback=record_id)
            study_id = study_key or record_id
            record_lookup = study_key or record.get("report_path") or record_id

            try:
                mentions = linker(text, record_id=record_lookup)
            except Exception as exc:  # pragma: no cover - runtime guard
                if not args.quiet:
                    print(f"[warn] failed to process record_id={record_id}: {exc}", file=sys.stderr)
                continue

            entry_map = aggregate_mentions(mentions, linker)
            finalized = finalize_study_entries(entry_map.values())
            if not finalized:
                continue

            update_inventory(inventory, entry_map.values(), section_label)
            instance_total += len(finalized)
            studies_written += 1

            metadata = {"radgraph_version": RADGRAPH_VERSION, "sections": [section_label]}
            if study_key:
                metadata["study_key"] = study_key
            payload = {
                "study_id": study_id,
                "record_id": record_id,
                "concepts": finalized,
                "metadata": metadata,
            }
            study_sink.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if not inventory:
        raise SystemExit("No concepts were produced; verify that texts are non-empty and linking succeeded.")

    inventory_json = {
        name: {
            "canonical_name": data["canonical_name"],
            "aliases": sorted(alias for alias in data["aliases"] if alias),
            "cui": data["cui"],
            "vocab": data["vocab"],
            "semantic_type": data["semantic_type"],
            "category": data["category"],
            "locations_supported": sorted(data["locations_supported"]),
            "assertions_supported": sorted(
                data["assertions_supported"], key=lambda val: -ASSERTION_PRECEDENCE[val]
            ),
            "examples": sorted(data["examples"])[:8],
            "stats": data["stats"],
            "provenance": {
                "sources": sorted(data["provenance"]["sources"]),
                "notes": data["provenance"]["notes"],
            },
        }
        for name, data in sorted(inventory.items())
    }

    inventory_path = output_dir / "concept_inventory.json"
    write_json(inventory_path, inventory_json)
    meta_payload = build_meta_payload(
        concept_count=len(inventory_json),
        instance_count=instance_total,
        radgraph_version=RADGRAPH_VERSION,
        linker_model=sapbert_model_id,
    )
    meta_path = output_dir / "concept_bank.meta.json"
    write_json(meta_path, meta_payload)

    if not args.quiet:
        print(f"[info] processed {studies_written:,} studies with {instance_total:,} concept instances.")
        print(f"[done] per-study concepts   : {study_output_path}")
        print(f"[done] concept inventory   : {inventory_path}")
        print(f"[done] metadata + summary  : {meta_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
