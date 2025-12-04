#!/usr/bin/env python3
"""SapBERT-backed ClinicalEntityLinker shared utilities."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import yaml
import pandas as pd

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

@dataclass
class Span:
    """Character-level span in the original note (best-effort)."""

    start: int
    end: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.start, self.end


@dataclass
class Mention:
    """Mention extracted by RadGraph and linked through SapBERT."""

    text: str
    span: Span
    category: str
    assertion: str
    mods: List[str] = field(default_factory=list)
    cui: Optional[str] = None
    cui_surface: Optional[str] = None
    cui_text: Optional[str] = None
    score: Optional[float] = None
    score_surface: Optional[float] = None
    score_text: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["span"] = self.span.to_tuple()
        return payload


def _expand_path(path: Union[Path, str, None]) -> Optional[Path]:
    """Expand environment variables and user symbols in a filesystem path."""
    if path is None:
        return None
    if isinstance(path, Path):
        path = str(path)
    expanded = os.path.expandvars(str(path))
    return Path(expanded).expanduser()


class ClinicalEntityLinker:
    """Facade that wraps RadGraph + SapBERT FAISS index for CUI linking."""

    SAPBERT_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    SENTENCE_MODELS = (
        "en_core_sci_lg",
        "en_core_sci_md",
        "en_core_web_sm",
    )
    DEFAULT_ALLOWED_TYPES = {
        "Observation": {"T047", "T046", "T033"},
        "Anatomy": {"T017", "T023", "T029", "T030", "T082"},
    }
    DEFAULT_STOP_MENTION_TERMS = {
        "minimal",
        "slightly",
        "stable",
        "unchanged",
        "improved",
        "standard",
        "position",
        "tube",
        "tip",
        "right",
        "left",
        "bilateral",
        "in place",
    }

    def __init__(
        self,
        *,
        mrconso_path: Union[Path, str],
        mrsty_path: Union[Path, str],
        mrrel_path: Optional[Union[Path, str]] = None,
        index_dir: Optional[Union[Path, str]] = None,
        index_file: Optional[Union[Path, str]] = None,
        mapping_file: Optional[Union[Path, str]] = None,
        sapbert_model_id: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        sentence_model: Optional[str] = None,
        max_length: int = 25,
        faiss_fp16: Optional[bool] = None,
        min_link_score: float = 0.8,
        stop_terms: Optional[Iterable[str]] = None,
        annotation_index: Optional[Mapping[str, List[dict]]] = None,
        allowed_tuis: Optional[Iterable[str]] = None,
        radlex_csv_path: Optional[Union[Path, str]] = None,
        allowed_sources: Optional[Iterable[str]] = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required. Install it with `pip install torch`.") from exc

        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faiss is required. Install faiss-cpu or faiss-gpu.") from exc

        try:
            import numpy as np  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy is required. Install it with `pip install numpy`.") from exc

        try:
            import spacy
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("spaCy is required. Install it with `pip install spacy`.") from exc

        try:
            from radgraph import RadGraph
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("radgraph is required. Install it with `pip install radgraph`.") from exc

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required. Install it with `pip install transformers`."
            ) from exc

        self._torch = torch
        self._faiss = faiss
        self._np = __import__("numpy")
        self.max_length = max_length

        mrconso_path = _expand_path(mrconso_path) if mrconso_path is not None else None
        mrsty_path = _expand_path(mrsty_path) if mrsty_path is not None else None
        mrrel_path = _expand_path(mrrel_path) if mrrel_path is not None else None

        cache_env = os.getenv("MEDCLIP_CACHE_DIR")
        default_cache_root = _expand_path(cache_env) if cache_env else (Path.home() / ".cache/medclip")
        if index_dir is None:
            index_dir = default_cache_root / "sapbert"
        else:
            index_dir = _expand_path(index_dir)
        if index_file is None:
            index_file = index_dir / "sapbert.index"
        else:
            index_file = _expand_path(index_file)
        if mapping_file is None:
            mapping_file = index_dir / "sapbert_id2cui.json"
        else:
            mapping_file = _expand_path(mapping_file)

        if radlex_csv_path is not None:
            radlex_csv_path = _expand_path(radlex_csv_path)
        self._radlex_csv_path = radlex_csv_path

        if self._radlex_csv_path is None:
            if not mrconso_path.exists():
                raise FileNotFoundError(f"MRCONSO file not found: {mrconso_path}")
            if not mrsty_path.exists():
                raise FileNotFoundError(f"MRSTY file not found: {mrsty_path}")
            if mrrel_path is not None and not mrrel_path.exists():
                raise FileNotFoundError(f"MRREL file not found: {mrrel_path}")
        else:
            if not self._radlex_csv_path.exists():
                raise FileNotFoundError(f"RadLex CSV not found: {self._radlex_csv_path}")

        index_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = index_file
        self._mapping_file = mapping_file

        faiss_fp16_env = os.getenv("MEDCLIP_FAISS_FP16")
        if faiss_fp16 is None:
            if faiss_fp16_env is not None:
                faiss_fp16 = faiss_fp16_env.lower() in {"1", "true", "yes", "on"}
            else:
                faiss_fp16 = False  # safer default for broader GPU compatibility
        self._faiss_use_fp16 = bool(faiss_fp16)

        sapbert_model_id = sapbert_model_id or self.SAPBERT_MODEL_ID
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = bool(use_gpu and torch.cuda.is_available())
        if use_gpu and not self.use_gpu:
            print("[warn] GPU requested but CUDA not available; using CPU.", file=sys.stderr)
        self._device = torch.device("cuda" if self.use_gpu else "cpu")
        self._gpu_device_id = torch.cuda.current_device() if self.use_gpu else None
        self._gpu_resources = None
        self._RadGraphClass = RadGraph
        self.radgraph = None
        self._radgraph_device = 0 if self.use_gpu else -1
        self._annotation_by_text: Optional[Dict[str, List[dict]]] = None
        self._annotation_by_record: Optional[Dict[str, List[dict]]] = None
        self._record_miss_logged = 0
        if annotation_index:
            if isinstance(annotation_index, dict) and (
                "text" in annotation_index or "__text_index" in annotation_index
            ):
                text_key = "text" if "text" in annotation_index else "__text_index"
                record_key = "record" if "record" in annotation_index else "__record_index"
                self._annotation_by_text = dict(annotation_index.get(text_key, {}))
                self._annotation_by_record = dict(annotation_index.get(record_key, {}))
            else:
                self._annotation_by_text = dict(annotation_index)
                self._annotation_by_record = {}
        else:
            self._annotation_by_text = None
            self._annotation_by_record = None
        self._annotation_index = self._annotation_by_text
        self._missing_annotation_warned = False
        if allowed_tuis is None and self._radlex_csv_path is None:
            raise ValueError("allowed_tuis must be provided in the config.")
        self._allowed_tuis = {str(sty) for sty in allowed_tuis} if allowed_tuis else set()
        self._allowed_sources = {s.strip().upper() for s in allowed_sources} if allowed_sources else None
        # Keep a per-category mapping for backwards-compatible filtering logic.
        self.allowed_types = {
            "Observation": set(self._allowed_tuis),
            "Anatomy": set(self._allowed_tuis),
        }
        self.min_link_score = float(min_link_score)
        stop_term_source = stop_terms or self.DEFAULT_STOP_MENTION_TERMS
        if isinstance(stop_term_source, str):
            stop_term_source = [stop_term_source]
        self.stop_mention_terms = {term.strip().lower() for term in stop_term_source if term}

        # Sentence splitter for RadGraph input.
        nlp_model = sentence_model
        loaded = False
        if nlp_model:
            try:
                self._nlp = spacy.load(nlp_model)
                loaded = True
            except OSError:
                print(f"[warn] spaCy model '{nlp_model}' not found. Falling back.", file=sys.stderr)
        if not loaded:
            for candidate in self.SENTENCE_MODELS:
                try:
                    self._nlp = spacy.load(candidate)
                    loaded = True
                    break
                except OSError:
                    continue
        if not loaded:
            self._nlp = spacy.blank("en")
            if "sentencizer" not in self._nlp.pipe_names:
                self._nlp.add_pipe("sentencizer")

        # RadGraph and SapBERT encoder.
        if self._annotation_index is None:
            self._ensure_radgraph()
        dtype = torch.float16 if self.use_gpu else torch.float32
        model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        self.sapbert = AutoModel.from_pretrained(sapbert_model_id, **model_kwargs)
        self.sapbert.to(self._device)
        self.sapbert.eval()
        self.sapbert_tokenizer = AutoTokenizer.from_pretrained(sapbert_model_id)

        # Load resources (UMLS or RadLex) and build/read FAISS index.
        if self._radlex_csv_path:
            print(f"[info] loading RadLex synonyms from {self._radlex_csv_path}")
            self.syns, self.cui2sty = self._load_radlex_synonyms(self._radlex_csv_path)
        else:
            self.syns = self._load_umls_synonyms(mrconso_path)
            self.cui2sty = self._load_mrsty(mrsty_path)
        self.syns = self._filter_synonyms_by_semantic_type(self.syns, self._allowed_tuis)
        self.cui2str = dict(self.syns.values)
        need_rebuild = (not index_file.exists()) or (not mapping_file.exists())
        existing_mapping: Optional[Dict[str, str]] = None
        if not need_rebuild:
            try:
                existing_mapping = self._load_id_mapping(mapping_file)
                if len(existing_mapping) != len(self.syns):
                    need_rebuild = True
                    existing_mapping = None
            except Exception:
                need_rebuild = True
                existing_mapping = None
        if need_rebuild:
            self._build_faiss_index(index_dir, index_file, mapping_file)
        self.index = self._load_faiss_index(index_file)
        self.id2cui = existing_mapping or self._load_id_mapping(mapping_file)
        self._mrrel_path = mrrel_path

    @classmethod
    def from_config(cls, config_path: Path, **kwargs: Any) -> ClinicalEntityLinker:
        config_path = Path(config_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        use_radlex = bool(config.get("radlex_csv_path") or kwargs.get("radlex_csv_path"))
        if use_radlex:
            required_keys = set()
        else:
            required_keys = {"mrconso_path", "mrsty_path", "allowed_tuis"}
        missing = {k for k in required_keys if not config.get(k)}
        if missing:
            raise KeyError(f"Config missing required keys: {', '.join(sorted(missing))}")

        params = {
            "mrconso_path": config.get("mrconso_path"),
            "mrsty_path": config.get("mrsty_path"),
            "mrrel_path": config.get("mrrel_path"),
            "index_dir": config.get("index_dir"),
            "index_file": config.get("index_file"),
            "mapping_file": config.get("mapping_file"),
            "sapbert_model_id": config.get("sapbert_model_id"),
            "faiss_fp16": config.get("faiss_fp16"),
            "min_link_score": config.get("min_link_score"),
            "stop_terms": config.get("stop_terms"),
            "allowed_tuis": config.get("allowed_tuis"),
            "radlex_csv_path": config.get("radlex_csv_path"),
            "allowed_sources": config.get("sources") or config.get("allowed_sources"),
        }
        params.update(kwargs)
        return cls(**params)

    # -------------------- Helpers -------------------- #

    @staticmethod
    def _load_id_mapping(mapping_file: Path) -> Dict[str, str]:
        with mapping_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {str(k): str(v) for k, v in data.items()}

    def _load_faiss_index(self, index_file: Path):
        cpu_index = self._faiss.read_index(str(index_file))
        self._cpu_index = cpu_index
        if self.use_gpu:
            gpu_index = self._try_move_index_to_gpu(cpu_index)
            if gpu_index is not None:
                return gpu_index
            # failed to move – fall back to CPU
            self.use_gpu = False
            self._gpu_resources = None
        return cpu_index

    def _try_move_index_to_gpu(self, cpu_index):
        try:
            res = self._faiss.StandardGpuResources()
            if hasattr(res, "setDefaultStreamTorch"):
                res.setDefaultStreamTorch()
            opts = self._faiss.GpuClonerOptions()
            opts.useFloat16 = self._faiss_use_fp16
            opts.storeTransposed = True
            opts.indicesOptions = self._faiss.INDICES_32_BIT
            device_id = 0 if self._gpu_device_id is None else int(self._gpu_device_id)
            gpu_index = self._faiss.index_cpu_to_gpu(res, device_id, cpu_index, opts)
            self._gpu_resources = res
            dtype_name = "float16" if self._faiss_use_fp16 else "float32"
            print(f"[info] loaded FAISS index on GPU (dtype={dtype_name}, transposed=True)")
            return gpu_index
        except Exception as exc:  # pragma: no cover - GPU optional
            print(
                f"[warn] Failed to move FAISS index to GPU ({exc}); continuing with CPU index.",
                file=sys.stderr,
            )
            return None

    def _ensure_radgraph(self) -> None:
        if self.radgraph is None:
            if self._RadGraphClass is None:
                raise RuntimeError("RadGraph class unavailable.")
            self.radgraph = self._RadGraphClass(cuda=self._radgraph_device)

    def _build_faiss_index(self, index_dir: Path, index_file: Path, mapping_file: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        id2cui: Dict[int, str] = {}
        names: List[str] = []
        for cui, name in self.syns.itertuples(index=False):
            id2cui[len(names)] = cui
            names.append(name)

        vectors: List[Any] = []
        step_indices = range(0, len(names), 128)
        iterator = step_indices
        if tqdm is not None:
            iterator = tqdm(
                step_indices,
                desc="Building SapBERT index",
                unit="batch",
                total=(len(names) + 127) // 128,
            )
        for i in iterator:
            batch = names[i : i + 128]
            vec = self._encode_text(batch)
            vectors.append(vec)
        mat = self._np.concatenate(vectors, axis=0)

        index = self._faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        self._faiss.write_index(index, str(index_file))
        with mapping_file.open("w", encoding="utf-8") as handle:
            json.dump({str(k): v for k, v in id2cui.items()}, handle)
        print(f"[info] Built FAISS index with {len(mat):,} vectors at {index_file}")

    def _load_umls_synonyms(self, rrf_path: Path) -> pd.DataFrame:
        rrf_path = Path(rrf_path)
        compression = "gzip" if rrf_path.suffix == ".gz" else None
        col_names = [
            "CUI",
            "LAT",
            "TS",
            "LUI",
            "STT",
            "SUI",
            "ISPREF",
            "AUI",
            "SAUI",
            "SCUI",
            "SDUI",
            "SAB",
            "TTY",
            "CODE",
            "STR",
            "SRL",
            "SUPPRESS",
            "CVF",
            "NaN",
        ]
        df = pd.read_csv(
            rrf_path,
            sep="|",
            header=None,
            names=col_names,
            usecols=["CUI", "LAT", "TS", "SAB", "STR"],
            dtype=str,
            compression=compression,
        )
        df = df.loc[(df["LAT"] == "ENG") & (df["TS"] != "S"), ["CUI", "SAB", "STR"]]
        if self._allowed_sources:
            before = len(df)
            df = df[df["SAB"].str.upper().isin(self._allowed_sources)]
            removed = before - len(df)
            print(
                f"[info] restricted UMLS synonyms to sources={sorted(self._allowed_sources)} "
                f"(removed {removed:,} rows)."
            )
        df = df.loc[:, ["CUI", "STR"]]
        df.dropna(subset=["CUI", "STR"], inplace=True)
        return df.reset_index(drop=True)

    def _load_mrsty(self, mrsty_path: Path) -> Dict[str, Any]:
        mrsty_path = Path(mrsty_path)
        compression = "gzip" if mrsty_path.suffix == ".gz" else None
        df = pd.read_csv(
            mrsty_path,
            sep="|",
            header=None,
            names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "EMPTY"],
            usecols=["CUI", "TUI"],
            dtype=str,
            compression=compression,
        )
        result: Dict[str, Any] = {}
        for cui, tui in df.itertuples(index=False):
            if cui in result:
                current = result[cui]
                if isinstance(current, set):
                    current.add(tui)
                else:
                    result[cui] = {current, tui}
            else:
                result[cui] = tui
        return result

    def _filter_synonyms_by_semantic_type(
        self, df: pd.DataFrame, allowed_tuis: set[str]
    ) -> pd.DataFrame:
        if not allowed_tuis:
            return df

        def has_allowed_tui(cui: str) -> bool:
            sty_codes = self.cui2sty.get(cui, ())
            if isinstance(sty_codes, str):
                return sty_codes in allowed_tuis
            return any(sty in allowed_tuis for sty in sty_codes)

        mask = df["CUI"].map(has_allowed_tui)
        filtered = df.loc[mask].reset_index(drop=True)
        removed = len(df) - len(filtered)
        if removed:
            print(
                f"[info] filtered out {removed:,} UMLS synonyms outside allowed semantic types."
            )
        if filtered.empty:
            raise RuntimeError(
                "All UMLS synonyms were filtered out. "
                "Adjust the allowed semantic types or verify the MRSTY file."
            )
        return filtered

    def _load_radlex_synonyms(self, csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load RadLex synonyms from a CSV export with columns:
        Class ID, Preferred Label, Synonyms, Obsolete, CUI, Semantic Types
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(
            csv_path,
            usecols=["Class ID", "Preferred Label", "Synonyms", "Obsolete", "CUI", "Semantic Types"],
            dtype=str,
            keep_default_na=False,
        )
        df = df[df["Obsolete"].str.upper() != "TRUE"]
        syn_rows = []
        cui2sty: Dict[str, Any] = {}
        for _, row in df.iterrows():
            cui = row.get("CUI") or row.get("Class ID") or ""
            cui = cui.strip()
            if cui.startswith("http://") or cui.startswith("https://"):
                cui = cui.rsplit("/", 1)[-1]
            pref = row.get("Preferred Label", "").strip()
            syns = row.get("Synonyms", "").split("|") if row.get("Synonyms") else []
            terms = [t.strip() for t in [pref, *syns] if t and t.strip()]
            for term in terms:
                syn_rows.append({"CUI": cui, "STR": term})
            sty_field = row.get("Semantic Types", "")
            if sty_field:
                stys = {s.strip() for s in sty_field.split("|") if s.strip()}
                if stys:
                    cui2sty[cui] = stys if len(stys) > 1 else next(iter(stys))
        syn_df = pd.DataFrame(syn_rows)
        if syn_df.empty:
            raise RuntimeError(f"No RadLex terms loaded from {csv_path}")
        return syn_df.reset_index(drop=True), cui2sty

    def _unwrap_annotations(self, payload) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            return list(payload.values())
        return list(payload)

    def _resolve_annotation_docs(
        self, note: str, sentences: List[str], record_id: Optional[str] = None
    ) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        ann_sent_docs: Optional[List[Dict[str, Any]]] = None
        ann_note_docs: Optional[List[Dict[str, Any]]] = None
        candidates: List[dict] = []

        doc: Optional[Dict[str, Any]] = None

        if record_id and self._annotation_by_record:
            record_key = str(record_id).strip()
            if record_key:
                candidates = self._annotation_by_record.get(record_key, [])
                if candidates:
                    payload = candidates[0]
                    doc = payload.get("0") if isinstance(payload.get("0"), dict) else payload
                if doc is not None and isinstance(doc.get("entities"), dict):
                    if self._record_miss_logged == 0 and isinstance(record_key, str):
                        print(f"[info] matched precomputed RadGraph annotation by record_id={record_key}")
                        self._record_miss_logged = -1  # sentinel to avoid duplicate log
                    ann_sent_docs = [doc]
                    ann_note_docs = [doc]

        if (ann_sent_docs is None or ann_note_docs is None) and self._annotation_index is not None:
            if radgraph_utils is None:  # pragma: no cover - safety
                raise RuntimeError("radgraph package is required to use precomputed annotations.")
            preprocessed = radgraph_utils.radgraph_xl_preprocess_report(note).strip()
            candidates = self._annotation_index.get(preprocessed, [])
            if candidates:
                payload = candidates[0]
                doc = None
                if isinstance(payload, dict):
                    inner = payload.get("0")
                    doc = inner if isinstance(inner, dict) else payload
                if doc is not None and isinstance(doc.get("entities"), dict):
                    ann_sent_docs = [doc]
                    ann_note_docs = [doc]

        if ann_sent_docs is None or ann_note_docs is None:
            if (self._annotation_index is not None or self._annotation_by_record) and not candidates:
                if self._record_miss_logged >= 0 and self._record_miss_logged < 5 and record_id:
                    print(f"[warn] no precomputed RadGraph annotation for record_id={record_id!r}; running model.")
                    self._record_miss_logged += 1
                if not self._missing_annotation_warned:
                    print("[warn] no precomputed RadGraph annotation found; running RadGraph model.")
                    self._missing_annotation_warned = True
            self._ensure_radgraph()
            ann_sent = self.radgraph(sentences)
            ann_note = self.radgraph([note]) if len(sentences) > 1 else ann_sent
            ann_sent_docs = self._unwrap_annotations(ann_sent)
            ann_note_docs = self._unwrap_annotations(ann_note)

        if not ann_sent_docs or not ann_note_docs:
            return None
        return ann_sent_docs, ann_note_docs

    # -------------------- Inference -------------------- #

    def __call__(self, note: str, record_id: Optional[str] = None) -> List[Mention]:
        mentions = self._infer_mentions(note, record_id=record_id)
        self._link_mentions_batch(mentions)
        return mentions

    def _infer_mentions(self, note: str, record_id: Optional[str] = None) -> List[Mention]:
        if not note:
            return []

        small_note = len(note.split()) < 3
        alt_assertion = self._negation_tool(note) if small_note else None

        sentences = [s.text.strip() for s in self._nlp(note).sents if s.text.strip()]
        if not sentences:
            sentences = [note.strip()]

        resolved = self._resolve_annotation_docs(note, sentences, record_id=record_id)
        if resolved is None:
            return []
        ann_sent_docs, ann_note_docs = resolved
        note_entities = ann_note_docs[0]["entities"] if ann_note_docs else {}

        note_label_map = {
            ent["tokens"].lower(): ent["label"]
            for ent in note_entities.values()
            if isinstance(ent, dict)
        }

        mentions: List[Mention] = []
        for ann_doc in ann_sent_docs:
            entities = ann_doc.get("entities", {})
            if not isinstance(entities, MutableMapping):
                continue

            # harmonise sentence-level labels with the full-note context
            for ent in entities.values():
                tok = ent.get("tokens", "").lower()
                if tok in note_label_map:
                    ent["label"] = note_label_map[tok]

            # propagate negations through relations (matches upstream script)
            for ent in entities.values():
                label = ent.get("label", "")
                if not isinstance(label, str):
                    continue
                if label.endswith("definitely absent"):
                    for rel in ent.get("relations", []):
                        rel_ent = entities.get(rel[1])
                        if not rel_ent:
                            continue
                        if rel_ent.get("label", "").endswith("definitely present"):
                            rel_has_present = any(
                                entities.get(rr[1], {}).get("label", "").endswith("definitely present")
                                for rr in rel_ent.get("relations", [])
                            )
                            if not rel_has_present:
                                rel_ent["label"] = rel_ent["label"].replace(
                                    "definitely present", "definitely absent"
                                )

            for ent in entities.values():
                label = ent.get("label", "")
                mention_text = ent.get("tokens", "")
                if not mention_text:
                    continue
                mods = [
                    entities.get(rel[1], {}).get("tokens", "")
                    for rel in ent.get("relations", [])
                    if entities.get(rel[1], {}).get("label", "").startswith("Anatomy")
                ]
                mods = [m for m in mods if m]

                source_lower = note.lower()
                char_start = source_lower.find(mention_text.lower())
                char_end = char_start + len(mention_text) if char_start != -1 else -1

                assertion = {
                    "Observation::definitely present": "present",
                    "Observation::definitely absent": "absent",
                    "Observation::uncertain": "uncertain",
                    "Anatomy::definitely present": "present",
                    "Anatomy::definitely absent": "absent",
                    "Anatomy::uncertain": "uncertain",
                }.get(label, "na")

                if alt_assertion and assertion in {"na", "present"}:
                    if alt_assertion in {"present", "absent", "uncertain"}:
                        assertion = alt_assertion

                mentions.append(
                    Mention(
                        text=mention_text,
                        span=Span(char_start, char_end),
                        category=label,
                        assertion=assertion,
                        mods=mods,
                    )
                )
        return mentions

    def _link_mentions_batch(self, mentions: List[Mention], top_k: int = 128) -> None:
        if not mentions:
            return

        surf_strings: List[str] = []
        string_to_idx: Dict[str, int] = {}

        for mention in mentions:
            mods = mention.mods or []
            surf_str = " ".join(mods + [mention.text]).strip()
            text_str = mention.text.strip()
            for token in (surf_str, text_str):
                if token not in string_to_idx:
                    string_to_idx[token] = len(surf_strings)
                    surf_strings.append(token)
            mention._surface_idx = string_to_idx[surf_str]  # type: ignore[attr-defined]
            mention._text_idx = string_to_idx[text_str]  # type: ignore[attr-defined]

        vectors = self._encode_text_batched(surf_strings)
        sims, idxs = self._faiss_search_with_retry(vectors, top_k)

        def select(row_idx: int, category: str) -> Tuple[Optional[str], Optional[float]]:
            for idx, sim in zip(idxs[row_idx], sims[row_idx]):
                cui = self.id2cui.get(str(int(idx)))
                if not cui:
                    continue
                sty_codes = self.cui2sty.get(cui, ())
                if isinstance(sty_codes, str):
                    sty_codes = (sty_codes,)
                if self._allowed_tuis and not any(sty in self._allowed_tuis for sty in sty_codes):
                    continue
                if sim < self.min_link_score:
                    continue
                return cui, float(sim)
            return None, None

        for mention in mentions:
            cui_surface, score_surface = select(mention._surface_idx, mention.category)  # type: ignore[attr-defined]
            cui_text, score_text = select(mention._text_idx, mention.category)  # type: ignore[attr-defined]

            mention.cui_surface = cui_surface
            mention.score_surface = score_surface
            mention.cui_text = cui_text
            mention.score_text = score_text

            mention.cui = cui_surface or cui_text
            mention.score = score_surface if mention.cui == cui_surface else score_text

            mention_text_norm = mention.text.strip().lower()
            score_value = mention.score if mention.score is not None else -1.0
            if score_value < self.min_link_score:
                mention.cui = None
                mention.score = None
            elif mention_text_norm in self.stop_mention_terms:
                mention.cui = None
                mention.score = None

    def link_phrases(
        self,
        phrases: Sequence[str],
        *,
        top_k: int = 16,
        allowed_stys: Optional[Sequence[str]] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        if not phrases:
            return {}
        unique_phrases: List[str] = []
        index_map: Dict[str, int] = {}
        for phrase in phrases:
            key = phrase.strip().lower()
            if key not in index_map:
                index_map[key] = len(unique_phrases)
                unique_phrases.append(phrase)
        vectors = self._encode_text_batched(unique_phrases)
        sims, idxs = self._faiss_search_with_retry(vectors, top_k)
        allowed_set = {sty for sty in allowed_stys} if allowed_stys else None
        threshold = self.min_link_score if min_score is None else float(min_score)
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        for phrase in phrases:
            key = phrase.strip().lower()
            idx = index_map[key]
            best: Optional[Dict[str, Any]] = None
            for candidate_idx, sim in zip(idxs[idx], sims[idx]):
                cui = self.id2cui.get(str(int(candidate_idx)))
                if not cui:
                    continue
                score = float(sim)
                if score < threshold:
                    continue
                sty_codes = self.cui2sty.get(cui, ())
                if isinstance(sty_codes, str):
                    sty_codes = (sty_codes,)
                if allowed_set and not any(sty in allowed_set for sty in sty_codes):
                    continue
                best = {
                    "cui": cui,
                    "preferred_name": self.cui2str.get(cui, ""),
                    "score": score,
                    "sty_codes": list(sty_codes),
                }
                break
            results[phrase] = best
        return results

    def _faiss_search_with_retry(self, vectors, top_k: int):
        # Chunk large batches to avoid CUBLAS limits on GPU
        max_batch_size = 8  # Very conservative batch size to avoid CUBLAS errors with large indices
        if len(vectors) > max_batch_size and self.use_gpu:
            all_sims = []
            all_idxs = []
            for i in range(0, len(vectors), max_batch_size):
                chunk = vectors[i:i + max_batch_size]
                try:
                    chunk_sims, chunk_idxs = self.index.search(chunk, top_k)
                    all_sims.append(chunk_sims)
                    all_idxs.append(chunk_idxs)
                except Exception as exc:
                    # Fall back to CPU for this chunk and all remaining
                    if not self.use_gpu:
                        raise
                    print(f"[warn] FAISS GPU search failed at chunk {i}; falling back to CPU index.")
                    self.use_gpu = False
                    self.index = self._cpu_index
                    # Retry this chunk on CPU
                    chunk_sims, chunk_idxs = self.index.search(chunk, top_k)
                    all_sims.append(chunk_sims)
                    all_idxs.append(chunk_idxs)
            return self._np.vstack(all_sims), self._np.vstack(all_idxs)

        try:
            return self.index.search(vectors, top_k)
        except Exception as exc:
            if not self.use_gpu:
                raise
            error_text = str(exc)
            if "CUBLAS_STATUS" in error_text and not self._faiss_use_fp16:
                print("[warn] FAISS GPU search failed; retrying with float16 index.")
                self._faiss_use_fp16 = True
                new_index = self._try_move_index_to_gpu(self._cpu_index)
                if new_index is not None:
                    self.index = new_index
                    return self.index.search(vectors, top_k)
            print("[warn] FAISS GPU search failed; falling back to CPU index.")
            self.use_gpu = False
            self.index = self._cpu_index
            return self.index.search(vectors, top_k)

    def _encode_text_batched(self, texts: List[str]):
        batches = []
        for i in range(0, len(texts), 256):
            batch = texts[i : i + 256]
            batches.append(self._encode_text(batch))
        return self._np.vstack(batches)

    def _encode_text(self, text: Sequence[str]) -> Any:
        with self._torch.no_grad():
            toks = self.sapbert_tokenizer(
                list(text),
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                padding="max_length",
            )
            toks = {k: v.to(self._device) for k, v in toks.items()}
            vec = self.sapbert(**toks)[0][:, 0, :]
            vec = self._torch.nn.functional.normalize(vec, dim=1)
            return vec.cpu().numpy().astype("float32")

    @staticmethod
    def _negation_tool(note: str) -> str:
        words = note.lower().split()
        absence = {"no", "not", "none", "without", "absent"}
        uncertain = {"maybe", "possible", "unclear", "could", "might", "suspect"}
        if any(word in uncertain for word in words):
            return "uncertain"
        if any(word in absence for word in words):
            return "absent"
        return "present"


def create_linker(
    config_path: Path,
    *,
    annotation_index: Optional[Mapping[str, Any]] = None,
    min_link_score: Optional[float] = None,
    radgraph_utils: Any = None,
    **kwargs: Any,
) -> ClinicalEntityLinker:
    """
    Construct a ClinicalEntityLinker with optional RadGraph utils injection.

    Args:
        config_path: YAML config with SapBERT/FAISS paths.
        annotation_index: Optional precomputed RadGraph annotations.
        min_link_score: Optional override for link score threshold.
        radgraph_utils: Optional radgraph.utils module (if caller wants to share cached utils).
        **kwargs: Passed through to ClinicalEntityLinker.from_config.
    """
    if radgraph_utils is not None and not hasattr(sys.modules[__name__], "radgraph_utils"):
        setattr(sys.modules[__name__], "radgraph_utils", radgraph_utils)

    kwargs = dict(kwargs)
    kwargs.setdefault("min_link_score", 0.8 if min_link_score is None else min_link_score)

    return ClinicalEntityLinker.from_config(
        Path(config_path).expanduser(),
        annotation_index=annotation_index,
        **kwargs,
    )


__all__ = ["ClinicalEntityLinker", "Mention", "Span", "create_linker"]
