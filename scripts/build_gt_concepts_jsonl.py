#!/usr/bin/env python3
"""Convert ground-truth study_concepts.jsonl into a CBM-friendly JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ground-truth study_concepts.jsonl into a CBM-compatible JSONL with probs vectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--study-concepts",
        required=True,
        type=Path,
        help="Path to study_concepts.jsonl (ground-truth concepts).",
    )
    parser.add_argument(
        "--concept-index",
        required=True,
        type=Path,
        help="Path to concept_index.json (maps index->concept_name).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path for CBM input (fields: study_id, probs).",
    )
    parser.add_argument(
        "--assertion",
        default="present",
        choices=["present", "absent", "uncertain", "any"],
        help="Assertion label to treat as positive. 'any' keeps all assertions.",
    )
    return parser.parse_args(argv)


def load_concept_index(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {v: int(k) for k, v in payload.items()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    study_path = args.study_concepts.expanduser()
    concept_index_path = args.concept_index.expanduser()
    output_path = args.output.expanduser()

    concept_to_idx = load_concept_index(concept_index_path)
    out = output_path.open("w", encoding="utf-8")

    kept_assertions = None if args.assertion == "any" else {args.assertion}
    total = 0
    written = 0

    with study_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            total += 1
            rec = json.loads(line)
            sid = str(rec.get("study_id") or rec.get("record_id") or rec.get("study_key") or "")
            if not sid:
                continue
            vec: List[float] = [0.0] * len(concept_to_idx)
            for item in rec.get("concepts", []):
                assertion = (item.get("assertion") or "").lower()
                if kept_assertions is not None and assertion not in kept_assertions:
                    continue
                name = item.get("concept")
                idx = concept_to_idx.get(name)
                if idx is not None:
                    vec[idx] = 1.0
            out.write(json.dumps({"study_id": sid, "probs": vec}) + "\n")
            written += 1
    out.close()
    print(f"[done] wrote {written:,} records from {total:,} input lines to {output_path}")


if __name__ == "__main__":
    main()

