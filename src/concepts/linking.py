"""Linking helpers that wrap ClinicalEntityLinker construction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

def create_linker(
    config_path: Path,
    *,
    annotation_index: Optional[Mapping[str, Any]] = None,
    min_link_score: Optional[float] = None,
    radgraph_utils: Any = None,
    **kwargs: Any,
):
    """Construct a ClinicalEntityLinker with optional RadGraph utils injection."""
    import src.ner as ner_module  # Deferred import to allow sys.path mutation by callers

    if radgraph_utils is None:
        try:  # pragma: no cover - optional dependency
            from radgraph import utils as radgraph_utils  # type: ignore
        except Exception:  # pragma: no cover
            radgraph_utils = None

    if radgraph_utils is not None and not hasattr(ner_module, "radgraph_utils"):
        ner_module.radgraph_utils = radgraph_utils  # type: ignore[attr-defined]

    ClinicalEntityLinker = ner_module.ClinicalEntityLinker
    kwargs = dict(kwargs)
    if min_link_score is not None:
        kwargs.setdefault("min_link_score", min_link_score)
    return ClinicalEntityLinker.from_config(
        Path(config_path).expanduser(),
        annotation_index=annotation_index,
        **kwargs,
    )


__all__ = ["create_linker"]
