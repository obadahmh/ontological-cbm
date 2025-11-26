#!/usr/bin/env python3
"""Curate a UMLS/SNOMED concept inventory into a cleaner schema.

This script reads a concept_inventory.json produced by build_umls_concept_bank_sapbert.py
and writes a curated concept_schema.json with more human-friendly labels:

- Prefer shortest synonym that does NOT contain:
  - NOS / \"not otherwise specified\" / \"unspecified\"
  - coordination tokens like \" or \", \" and \", '/', '&', or parentheses.
- Fall back to a cleaned canonical_name where we strip trailing
  SNOMED-style qualifiers like \", NOS\" or \"(disorder)\".

The output schema is keyed by the original canonical_name so it can be joined
back to study_concepts.jsonl, and each entry contains:
  - id: stable identifier (uses CUI when available)
  - cui / cuis: original CUI (plus 1-element list)
  - label: curated display label
  - synonyms: de-duplicated list of cleaned synonyms (includes label)
  - semantic_type: copied from the inventory
  - assertions_supported: copied from the inventory
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _load_inventory(path: Path) -> Mapping[str, Dict[str, Any]]:
    path = path.expanduser()
    if not path.exists():
        raise SystemExit(f"Inventory file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Inventory JSON must be an object at top level: {path}")
    return payload


BAD_LABEL_PATTERNS = (
    "not otherwise specified",
    "unspecified",
)

PRONOUN_TOKENS = {
    "he",
    "she",
    "it",
    "they",
    "his",
    "her",
    "their",
}

# Single-word labels that are usually too generic to be helpful on their own.
GENERIC_SINGLE_WORDS = {
    "both",
    "either",
    "neither",
    "left",
    "right",
    "r",
    "l",
    "size",
    "area",
    "cycle",
    "free",
    "vessel",
    "tube",
    "silhouette",
    "gastric",
    "normal",
    "moderate",
    "mild",
    "large",
    "small",
    "tiny",
    "raised",
    "increased",
    "stable",
    "unchanged",
    "clear",
}


def _has_bad_patterns(text: str) -> bool:
    lowered = text.lower()
    # Avoid explicit NOS variants.
    tokens = re.findall(r"[A-Za-z0-9]+", lowered)
    if "nos" in tokens:
        return True
    for phrase in BAD_LABEL_PATTERNS:
        if phrase in lowered:
            return True
    # Avoid obviously coordinated / multi-option labels.
    if " or " in lowered or " and " in lowered:
        return True
    if "/" in lowered or "&" in lowered:
        return True
    if "(" in text or ")" in text:
        return True
    return False


def _clean_trailing_qualifiers(text: str) -> str:
    """Strip trailing NOS / unspecified and SNOMED semantic tags."""
    cleaned = text.strip()
    # Remove trailing SNOMED semantic tags in parentheses, e.g. "(disorder)".
    cleaned = re.sub(
        r"\s*\((disorder|finding|body structure|procedure|qualifier value|"
        r"observable entity|situation|regime/therapy|morphologic abnormality)\)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove trailing NOS / unspecified variants like "Heart, NOS" or "Pleural effusion NOS".
    cleaned = re.sub(
        r"\s*,?\s*(nos|not otherwise specified|unspecified)\.?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Normalise whitespace and trailing commas.
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    return cleaned


def _is_composite_name(text: str) -> bool:
    """Return True if the name looks like a coordinated multi-option label."""
    lowered = text.lower()
    if " or " in lowered or " and " in lowered:
        return True
    if "/" in lowered or "&" in lowered:
        return True
    if "(" in text or ")" in text:
        return True
    return False


def _simplify_canonical(text: str) -> str:
    """Heuristically simplify a canonical SNOMED/UMLS name."""
    cleaned = _clean_trailing_qualifiers(" ".join(str(text or "").strip().split()))
    # Drop very generic trailing tokens like "part", "structure", "field", "finding", "value".
    cleaned = re.sub(
        r"\b(part|structure|field|finding|value)\b\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or text.strip()


def _choose_label(canonical_name: str, aliases: Iterable[str]) -> str:
    """Pick a concise, human-readable label."""

    # If the canonical name is a simple, single concept, prefer a cleaned version of it.
    if not _is_composite_name(canonical_name):
        simplified = _simplify_canonical(canonical_name)
        if simplified:
            return simplified
        return canonical_name

    # For highly composite canonical names (with OR/AND/parentheses), fall back to aliases.
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").strip().split())

    candidates: List[str] = []
    for alias in aliases:
        alias = _normalize(alias)
        if not alias:
            continue
        if _has_bad_patterns(alias):
            continue
        cleaned_alias = _clean_trailing_qualifiers(alias)
        if cleaned_alias:
            candidates.append(cleaned_alias)

    # De-duplicate while preserving order.
    unique: List[str] = []
    seen: set[str] = set()
    for cand in candidates:
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)

    if not unique:
        # Last resort: cleaned canonical, even if composite.
        simplified = _simplify_canonical(canonical_name)
        return simplified or canonical_name

    def _score(label: str) -> Tuple[int, int, int, str]:
        """Lower score is better."""
        norm = label.strip()
        tokens = norm.lower().split()
        if any(tok in PRONOUN_TOKENS for tok in tokens):
            return (3, len(tokens), len(norm), norm.lower())
        single = len(tokens) == 1
        generic_single = single and tokens[0] in GENERIC_SINGLE_WORDS
        generic_flag = 2 if generic_single else (1 if single else 0)
        return (generic_flag, len(tokens), len(norm), norm.lower())

    best = min(unique, key=_score)
    return best


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = _clean_trailing_qualifiers(lowered)
    # Replace non-alphanumeric with underscores.
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "concept"


def curate_inventory(
    inventory_path: Path,
    output_path: Path,
) -> None:
    inventory = _load_inventory(inventory_path)
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    curated: Dict[str, Dict[str, Any]] = {}

    for idx, (canonical_name, data) in enumerate(inventory.items(), start=1):
        if not isinstance(data, Mapping):
            continue
        aliases = list(data.get("aliases") or [])
        canonical_name = str(canonical_name)
        cui = data.get("cui")
        semantic_type = data.get("semantic_type")
        assertions_supported = sorted(data.get("assertions_supported") or [])

        label = _choose_label(canonical_name, aliases)
        # Build synonym set (include cleaned canonical + aliases + label).
        syn_set = set()
        for raw in [canonical_name] + aliases:
            text = str(raw or "").strip()
            if not text:
                continue
            cleaned = _clean_trailing_qualifiers(text)
            if cleaned:
                syn_set.add(cleaned)
        syn_set.add(label)
        synonyms = sorted(syn_set, key=lambda s: s.lower())

        if isinstance(cui, str) and cui.strip():
            concept_id = cui.strip()
            cuis = [concept_id]
        else:
            slug = _slugify(label or canonical_name)
            concept_id = f"auto_{slug}_{idx:05d}"
            cuis = []

        curated[canonical_name] = {
            "id": concept_id,
            "cui": concept_id if cuis else None,
            "cuis": cuis,
            "label": label,
            "synonyms": synonyms,
            "semantic_type": semantic_type,
            "assertions_supported": assertions_supported,
        }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(curated, handle, ensure_ascii=False, indent=2)

    print(f"[done] wrote curated concept schema for {len(curated):,} entries to {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate a concept_inventory.json into a human-friendly concept_schema.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=None,
        help="Directory containing concept_inventory.json (default: generated/concept_bank_sapbert_mimic).",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Path to concept_inventory.json (overrides --bank-dir if provided).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for concept_schema.json (default: <bank-dir>/concept_schema.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    bank_dir: Optional[Path] = args.bank_dir
    if bank_dir is None and args.inventory is None:
        # Default to the main SapBERT MIMIC concept bank if present.
        default_bank = Path("generated/concept_bank_sapbert_mimic")
        if default_bank.exists():
            bank_dir = default_bank
        else:
            raise SystemExit("Provide --bank-dir or --inventory (no default bank directory found).")

    inventory_path: Optional[Path] = args.inventory
    output_path: Optional[Path] = args.output

    if bank_dir is not None:
        bank_dir = bank_dir.expanduser()
        if not bank_dir.exists():
            raise SystemExit(f"Concept bank directory not found: {bank_dir}")
        if inventory_path is None:
            inventory_path = bank_dir / "concept_inventory.json"
        if output_path is None:
            output_path = bank_dir / "concept_schema.json"

    if inventory_path is None:
        raise SystemExit("Unable to resolve concept inventory path; provide --bank-dir or --inventory.")
    if output_path is None:
        raise SystemExit("Unable to resolve output path; provide --bank-dir or --output.")

    curate_inventory(inventory_path, output_path)


if __name__ == "__main__":
    main()
