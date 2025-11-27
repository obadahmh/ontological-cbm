"""Aggregation utilities for mention->concept processing and inventory building."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

ASSERTION_ALLOWED = {"present", "absent", "uncertain"}
ASSERTION_PRECEDENCE = {"present": 2, "uncertain": 1, "absent": 0}
DEVICE_STY = {"T074", "T075", "T121", "T122", "T123", "T203"}
FINDING_STY = {"T033", "T037", "T040", "T041", "T042", "T047", "T048", "T049", "T184", "T191"}
ANATOMY_STY = {"T017", "T021", "T023", "T029", "T030", "T031", "T082"}


def canonicalize_location(mods: Sequence[str]) -> Optional[str]:
    tokens: List[str] = []
    for mod in mods:
        cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in mod)
        normalized = " ".join(cleaned.split())
        if normalized:
            tokens.append(normalized)
    if not tokens:
        return None
    return " ".join(tokens)


def classify_semantic_type(sty_codes: Iterable[str]) -> Tuple[str, str]:
    sty_set = set(sty_codes)
    if any(code in DEVICE_STY for code in sty_set):
        return "Device", "device"
    if any(code in FINDING_STY for code in sty_set):
        return "Finding", "finding"
    if any(code in ANATOMY_STY for code in sty_set):
        return "BodyStructure", "anatomy_scaffold"
    return "Finding", "finding"


def build_span_payload(mention) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"surface": getattr(mention, "text", None)}
    span = getattr(mention, "span", None)
    if span:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        if start is not None and start >= 0:
            payload["start"] = start
        if end is not None and end >= 0:
            payload["end"] = end
    mods = getattr(mention, "mods", None)
    if mods:
        payload["mods"] = list(mods)
    score = getattr(mention, "score", None)
    if score is not None:
        payload["score"] = float(score)
    return payload


def mention_to_concept(
    mention,
    linker,
) -> Optional[Dict[str, Any]]:
    assertion = (getattr(mention, "assertion", "") or "").lower()
    if assertion not in ASSERTION_ALLOWED:
        return None
    cui = getattr(mention, "cui", None) or getattr(mention, "cui_surface", None) or getattr(mention, "cui_text", None)
    if not cui:
        return None
    canonical_name = (getattr(linker, "cui2str", {}) or {}).get(cui) or (getattr(mention, "text", "") or "").strip()
    if not canonical_name:
        return None
    sty_codes = getattr(linker, "cui2sty", {}).get(cui, ())
    if isinstance(sty_codes, str):
        sty_codes = (sty_codes,)
    semantic_type, category = classify_semantic_type(sty_codes)
    mods = getattr(mention, "mods", None) or []
    location = canonicalize_location(mods)
    return {
        "canonical_name": canonical_name,
        "cui": cui,
        "assertion": assertion,
        "location": location,
        "semantic_type": semantic_type,
        "category": category,
        "source_span": build_span_payload(mention),
        "alias": (getattr(mention, "text", None) or "").strip(),
    }


def aggregate_mentions(
    mentions: Sequence[Any],
    linker,
) -> Dict[Tuple[str, Optional[str], str], Dict[str, Any]]:
    entries: Dict[Tuple[str, Optional[str], str], Dict[str, Any]] = {}
    for mention in mentions:
        concept = mention_to_concept(mention, linker)
        if not concept:
            continue
        key = (concept["cui"], concept["location"], concept["assertion"])
        entry = entries.setdefault(
            key,
            {
                "concept": concept["canonical_name"],
                "assertion": concept["assertion"],
                "location": concept["location"],
                "cui": concept["cui"],
                "landmarks": [],
                "source_spans": [],
                "_aliases": set(),
                "_category": concept["category"],
                "_semantic_type": concept["semantic_type"],
            },
        )
        entry["source_spans"].append(concept["source_span"])
        if concept["alias"]:
            entry["_aliases"].add(concept["alias"])
    return entries


def finalize_study_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    finalized: List[Dict[str, Any]] = []
    for entry in entries:
        if not entry["source_spans"]:
            continue
        payload = {
            "concept": entry["concept"],
            "assertion": entry["assertion"],
            "location": entry["location"],
            "cui": entry["cui"],
            "landmarks": entry["landmarks"],
            "source_spans": entry["source_spans"],
        }
        finalized.append(payload)
    finalized.sort(key=lambda item: (item["concept"], item["location"] or "", item["assertion"]))
    return finalized


def update_inventory(
    inventory: MutableMapping[str, Dict[str, Any]],
    entries: Iterable[Dict[str, Any]],
    section_label: str,
) -> None:
    for entry in entries:
        canonical_name = entry["concept"]
        if not canonical_name:
            continue
        cui = entry["cui"]
        semantic_type = entry.get("_semantic_type")
        category = entry.get("_category")
        inv_entry = inventory.setdefault(
            canonical_name,
            {
                "canonical_name": canonical_name,
                "aliases": set(),
                "cui": cui,
                "vocab": "SNOMEDCT_US" if cui else None,
                "semantic_type": semantic_type,
                "category": category,
                "locations_supported": set(),
                "assertions_supported": set(),
                "examples": set(),
                "stats": {
                    "total": 0,
                    "by_assertion": {"present": 0, "absent": 0, "uncertain": 0},
                },
                "provenance": {"sources": set(), "notes": None},
            },
        )
        inv_entry["cui"] = inv_entry["cui"] or cui
        if semantic_type:
            inv_entry["semantic_type"] = semantic_type
        if category:
            inv_entry["category"] = category
        aliases = entry.get("_aliases") or set()
        inv_entry["aliases"].update(alias for alias in aliases if alias)
        for span in entry.get("source_spans", []):
            surface = span.get("surface")
            if surface:
                inv_entry["aliases"].add(surface)
                inv_entry["examples"].add(surface)
        location = entry.get("location")
        if location:
            inv_entry["locations_supported"].add(location)
        assertion = entry.get("assertion")
        if assertion:
            inv_entry["assertions_supported"].add(assertion)
            inv_entry["stats"]["by_assertion"][assertion] += 1
        inv_entry["stats"]["total"] += 1
        inv_entry["provenance"]["sources"].add(section_label)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_meta_payload(
    *,
    concept_count: int,
    instance_count: int,
    radgraph_version: Optional[str],
    linker_model: Optional[str],
) -> Dict[str, Any]:
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "bank_version": "0.2.0",
        "created_at": created_at,
        "radgraph_version": radgraph_version,
        "linker_model": linker_model,
        "concept_count": concept_count,
        "instance_count": instance_count,
    }
    return {k: v for k, v in payload.items() if v is not None}


def build_concept_summary(
    mentions: Iterable[Any],
    cui_to_name: Mapping[str, str],
    assertion_filter: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Aggregate mention-level CUIs grouped by assertion."""
    filter_set = {a.lower() for a in assertion_filter} if assertion_filter else None
    concept_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    mention_payload: List[Dict[str, Any]] = []

    for mention in mentions:
        mention_payload.append(
            {
                "text": getattr(mention, "text", None),
                "mods": getattr(mention, "mods", None),
                "assertion": getattr(mention, "assertion", None),
                "category": getattr(mention, "category", None),
                "span": getattr(getattr(mention, "span", None), "to_tuple", lambda: (None, None))(),
                "cui": getattr(mention, "cui", None),
                "cui_surface": getattr(mention, "cui_surface", None),
                "cui_text": getattr(mention, "cui_text", None),
                "score": getattr(mention, "score", None),
                "score_surface": getattr(mention, "score_surface", None),
                "score_text": getattr(mention, "score_text", None),
                "preferred_name": cui_to_name.get(getattr(mention, "cui", "") or "", None),
            }
        )

        cui = getattr(mention, "cui", None)
        if not cui:
            continue
        assertion = (getattr(mention, "assertion", None) or "na").lower()
        if filter_set and assertion not in filter_set:
            continue

        bucket = concept_map.setdefault(assertion, {})
        concept = bucket.setdefault(
            cui,
            {
                "cui": cui,
                "preferred_name": cui_to_name.get(cui),
                "mention_texts": set(),
                "scores": [],
            },
        )
        surface = getattr(mention, "text", None)
        if surface:
            concept["mention_texts"].add(surface)
        score_val = getattr(mention, "score", None)
        if score_val is not None:
            concept["scores"].append(float(score_val))

    summary: Dict[str, List[Dict[str, Any]]] = {}
    for assertion, concepts in concept_map.items():
        items: List[Dict[str, Any]] = []
        for concept in concepts.values():
            scores = concept["scores"]
            items.append(
                {
                    "cui": concept["cui"],
                    "preferred_name": concept["preferred_name"],
                    "mention_texts": sorted(concept["mention_texts"]),
                    "score_max": max(scores) if scores else None,
                    "score_mean": (sum(scores) / len(scores)) if scores else None,
                }
            )
        summary[assertion] = sorted(items, key=lambda x: x["cui"])

    return summary, mention_payload


__all__ = [
    "ASSERTION_ALLOWED",
    "ASSERTION_PRECEDENCE",
    "canonicalize_location",
    "classify_semantic_type",
    "build_span_payload",
    "mention_to_concept",
    "aggregate_mentions",
    "finalize_study_entries",
    "update_inventory",
    "write_json",
    "build_meta_payload",
    "build_concept_summary",
    "DEVICE_STY",
    "FINDING_STY",
    "ANATOMY_STY",
]
