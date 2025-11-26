#!/usr/bin/env python3
"""Convert free-text radiology reports into UMLS CUIs using the ontology-concept-distillation pipeline.

The pipeline mirrors https://github.com/Felix-012/ontology-concept-distillation:
  1. RadGraph is used to extract anatomy/observation mentions and their assertions.
  2. Mention strings are embedded with SapBERT and linked to UMLS CUIs via a FAISS index.
  3. For each study we export the linked CUIs (grouped by assertion) together with mention metadata.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import itertools

import pandas as pd
import yaml
import sys

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from src.paths import add_repo_root_to_sys_path

add_repo_root_to_sys_path()
from src.concepts import aggregation as concept_agg
from src.concepts import input as concept_input
from src.concepts import linking as concept_linking

normalize_record_id = concept_input.normalize_record_id
load_annotation_payload = concept_input.load_annotation_payload
build_annotation_index = concept_input.build_annotation_index
derive_study_key = concept_input.derive_study_key
build_concept_summary = concept_agg.build_concept_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link report texts to UMLS CUIs using the ontology-concept-distillation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-path", required=True, help="CSV file containing free-text reports.")
    parser.add_argument(
        "--text-column",
        default="section_findings",
        help="Column in the CSV that stores the free-text report.",
    )
    parser.add_argument(
        "--id-column",
        action="append",
        dest="id_columns",
        default=[],
        help="Column used to build a stable record identifier (can be specified multiple times).",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to ontology-concept-distillation style YAML config with UMLS resources.",
    )
    parser.add_argument(
        "--annotations-path",
        action="append",
        dest="annotation_paths",
        default=None,
        help="RadGraph annotation JSON file (can be provided multiple times).",
    )
    parser.add_argument(
        "--output-jsonl",
        default="generated/cui_links/report_cuis.jsonl",
        help="Where to write the JSONL output (one record per row).",
    )
    parser.add_argument(
        "--unmatched-csv",
        default=None,
        help="Optional CSV to store records without linked CUIs or with empty text.",
    )
    parser.add_argument(
        "--assertion",
        action="append",
        dest="assertions",
        choices=["present", "absent", "uncertain", "na"],
        help="If provided, only keep CUIs whose assertion matches these values.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N rows for quick inspection.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include empty-text rows in the unmatched CSV, if provided.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU inference (disable GPU utilisation even if available).",
    )
    parser.add_argument(
        "--faiss-fp16",
        action="store_true",
        help="Store FAISS GPU index in float16 for lower memory use (default: float32).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.8,
        help="Minimum SapBERT similarity required to keep a linked CUI.",
    )
    parser.add_argument(
        "--stop-term",
        action="append",
        dest="stop_terms",
        default=None,
        help="Mention text to ignore (case-insensitive). Can be provided multiple times.",
    )
    parser.add_argument(
        "--sentence-model",
        default=None,
        help="Optional spaCy model name to use for sentence splitting.",
    )
    parser.add_argument(
        "--dataset",
        choices=["chexpert_plus", "mimic_cxr"],
        default=None,
        help="Dataset slug to include study_key in the output JSONL.",
    )
    parser.add_argument(
        "--patient-column",
        default=None,
        help="Column name for patient identifier (required with --dataset chexpert_plus).",
    )
    parser.add_argument(
        "--study-column",
        default=None,
        help="Column name for study identifier (required with dataset-aware study_key).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-record progress logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser()
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    config_path = Path(args.config_path).expanduser()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    output_path = Path(args.output_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.id_columns:
        args.id_columns = ["study_id"]

    if not args.quiet:
        print(f"[info] loading reports from {csv_path}")
    df = pd.read_csv(csv_path)
    if args.text_column not in df.columns:
        raise SystemExit(f"Column '{args.text_column}' not found in CSV.")
    if args.dataset == "chexpert_plus":
        if not args.patient_column or not args.study_column:
            raise SystemExit("chexpert_plus dataset requires --patient-column and --study-column.")
    if args.dataset == "mimic_cxr" and not args.study_column:
        raise SystemExit("mimic_cxr dataset requires --study-column.")

    annotation_index = None
    if args.annotation_paths:
        combined_text: Dict[str, List[dict]] = defaultdict(list)
        combined_record: Dict[str, List[dict]] = defaultdict(list)
        total_entries = 0
        for ann_path_str in args.annotation_paths:
            ann_path = Path(ann_path_str).expanduser()
            if not ann_path.exists():
                raise SystemExit(f"Annotation file not found: {ann_path}")
            payload = load_annotation_payload(ann_path)
            index_part = build_annotation_index(payload, dataset=args.dataset)
            for key, entries in index_part["text"].items():
                combined_text[key].extend(entries)
                total_entries += len(entries)
            for key, entries in index_part["record"].items():
                combined_record[key].extend(entries)
        annotation_index = {
            "text": dict(combined_text),
            "record": dict(combined_record),
        }
        if not args.quiet:
            print(
                f"[info] loaded {total_entries:,} precomputed RadGraph annotations from "
                f"{len(args.annotation_paths)} file(s)"
            )

    linker = concept_linking.create_linker(
        config_path,
        use_gpu=(not args.cpu_only),
        sentence_model=args.sentence_model,
        faiss_fp16=args.faiss_fp16 or None,
        min_link_score=args.min_score,
        stop_terms=args.stop_terms,
        annotation_index=annotation_index,
    )

    if not args.quiet:
        print("[info] linking reports to CUIs…")

    total_rows = min(len(df), args.limit) if args.limit else len(df)
    processed = 0
    matched = 0
    unmatched: List[Dict[str, Any]] = []

    row_iter = df.itertuples(index=True)
    if args.limit:
        row_iter = itertools.islice(row_iter, args.limit)

    if not args.quiet and tqdm is not None:
        row_iter = tqdm(row_iter, total=total_rows, desc="Linking reports", unit="report")

    with output_path.open("w", encoding="utf-8") as sink:
        for row in row_iter:
            processed += 1

            series = row._asdict()
            text_raw = series.get(args.text_column, "")
            text = "" if pd.isna(text_raw) else str(text_raw).strip()
            record_id = normalize_record_id(series, args.id_columns)
            study_key = None
            if args.dataset:
                try:
                    study_key = derive_study_key(
                        args.dataset, series, args.patient_column, args.study_column, record_id
                    )
                    if study_key is None and not args.quiet:
                        print(f"[warn] unable to derive study_key for record_id={record_id}")
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc

            if not text:
                if args.include_empty:
                    unmatched.append(
                        {
                            "record_id": record_id,
                            "row_index": row.Index,
                            "reason": "empty_text",
                        }
                    )
                continue

            lookup_id = study_key or record_id
            try:
                mentions = linker(text, record_id=lookup_id)
            except Exception as exc:  # pragma: no cover - runtime safety
                print(f"[warn] failed to process record_id={record_id}: {exc}", file=sys.stderr)
                unmatched.append(
                    {
                        "record_id": record_id,
                        "row_index": row.Index,
                        "reason": f"exception:{type(exc).__name__}",
                        "message": str(exc),
                    }
                )
                continue

            concept_summary, mention_payload = build_concept_summary(
                mentions,
                linker.cui2str,
                assertion_filter=args.assertions,
            )

            unique_cuis = sorted(
                {
                    item["cui"]
                    for bucket in concept_summary.values()
                    for item in bucket
                    if item.get("cui")
                }
            )
            concepts_flat: List[Dict[str, Any]] = []
            for assertion, entries in concept_summary.items():
                for entry in entries:
                    if not entry.get("cui"):
                        continue
                    concept_flat = {
                        "cui": entry["cui"],
                        "assertion": assertion,
                        "preferred_name": entry.get("preferred_name"),
                        "mention_texts": entry.get("mention_texts"),
                        "linked": True,
                        "score_max": entry.get("score_max"),
                        "score_mean": entry.get("score_mean"),
                    }
                    concepts_flat.append(concept_flat)

            if unique_cuis:
                matched += 1
            else:
                unmatched.append(
                    {
                        "record_id": record_id,
                        "row_index": row.Index,
                        "reason": "no_cui",
                    }
                )

            payload = {
                "record_id": record_id,
                "row_index": row.Index,
                "unique_cuis": unique_cuis,
                "concepts_by_assertion": concept_summary,
                "mentions": mention_payload,
                "concepts": concepts_flat,
            }
            if study_key:
                payload["study_key"] = study_key
            sink.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if not args.quiet and tqdm is None and processed % 50 == 0:
                print(f"[info] processed={processed} matched={matched}")

    if args.unmatched_csv and unmatched:
        unmatched_path = Path(args.unmatched_csv).expanduser()
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmatched).to_csv(unmatched_path, index=False)
        if not args.quiet:
            print(f"[info] wrote unmatched records to {unmatched_path}")

    if not args.quiet:
        print(
            f"[done] processed={processed} matched={matched} unmatched={processed - matched} "
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
