#!/usr/bin/env python3
from __future__ import annotations
import csv
import gzip
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type

import pandas as pd

# ---------- optional YAML (for datasets.yaml) ----------
try:
    import yaml  # pip install pyyaml
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# ======================================================
# Registry + common types
# ======================================================

_LOADER_REGISTRY: Dict[str, Type["BaseLoader"]] = {}

def register_loader(name: str):
    def deco(cls):
        _LOADER_REGISTRY[name] = cls
        cls.NAME = name
        return cls
    return deco

def build_loader(name: str, **kwargs) -> "BaseLoader":
    if name not in _LOADER_REGISTRY:
        raise KeyError(f"Unknown dataset loader '{name}'. Available: {list(_LOADER_REGISTRY)}")
    return _LOADER_REGISTRY[name](**kwargs)

@dataclass
class Sample:
    image: Optional[str]
    caption: str
    subject_id: Optional[int] = None
    study_id: Optional[int] = None
    dicom_id: Optional[str] = None
    meta: Optional[dict] = None

class BaseLoader:
    """
    All loaders should return a tidy DataFrame with at least:
    ['image','caption','subject_id','study_id','dicom_id']
    """
    def dataframe(self) -> pd.DataFrame:
        raise NotImplementedError

    def iter_samples(self) -> Iterable[Sample]:
        df = self.dataframe()
        for r in df.itertuples(index=False):
            yield Sample(
                image=getattr(r, "image", None),
                caption=getattr(r, "caption", "") or "",
                subject_id=getattr(r, "subject_id", None),
                study_id=getattr(r, "study_id", None),
                dicom_id=getattr(r, "dicom_id", None),
                meta={k: getattr(r, k) for k in df.columns if k not in {
                    "image","caption","subject_id","study_id","dicom_id"
                }},
            )

# ======================================================
# Utilities
# ======================================================

_ENV_PAT = re.compile(r"\$\{([^}]+)\}")

def _expand_env_vars(v: str) -> str:
    """Expand ${ENV_VAR} in strings; leave unchanged if missing."""
    if not isinstance(v, str):
        return v
    return _ENV_PAT.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)

def _open_text(path: Path):
    """Open path as text; transparently handles .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def _normalize_nl(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")

def _clean_text(s: str) -> str:
    s = _normalize_nl(s).strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s

def _choose_caption(impression: str, findings: str, prefer: str) -> str:
    imp, fin = (impression or "").strip(), (findings or "").strip()
    if prefer == "impression_then_findings":
        return imp if imp else fin
    if prefer == "findings_then_impression":
        return fin if fin else imp
    return f"{fin}\n\nimpression: {imp}" if (fin and imp) else (fin or imp)

def _maybe_abs(root: Optional[Path], p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    q = Path(p)
    if q.is_absolute() or root is None or str(root) == "":
        return str(q)
    return str((root / q).resolve())

# ======================================================
# MIMIC-CXR
# ======================================================

@register_loader("mimic_cxr")
class MIMICCXRLoader(BaseLoader):
    """
    MIMIC-CXR captions + optional JPG paths.

    Arguments:
      sectioned_csv  : CSV/GZ with (subject_id,study_id,impression,findings)
      reports_root   : root folder containing per-study TXT reports (…/files/pXX/pYYYY/sZZZZ.txt)
      prefer         : "impression_then_findings" (default) | "findings_then_impression" | "both"
      jpg_metadata_csv: MIMIC-CXR-JPG metadata CSV/GZ (has columns: subject_id,study_id,dicom_id,path,ViewPosition)
      jpg_root       : root to prepend to jpg 'path'
      split_csv      : optional split CSV/GZ with columns (subject_id,study_id,split)
      split          : optional split name to keep (e.g., "train"|"valid"|"test")
    """
    SECTION_ALIASES = {
        "impression": {"impression","impressions","conclusion","conclusions","opinion"},
        "findings": {"findings","findings and impression","findings/impression"},
        "history": {"history","indication","reason for exam","clinical information"},
        "technique": {"technique","exam","examination"},
        "comparison": {"comparison","comparisons"},
    }
    HEADER_LINE = re.compile(r"^\s*\*{0,3}\s*final report\s*\*{0,3}\s*$", re.I)
    SEC_LINE    = re.compile(r"^\s*([A-Z][A-Z /()-]{1,40})\s*:\s*$")

    def __init__(
        self,
        sectioned_csv: Optional[str | Path] = None,
        reports_root: Optional[str | Path] = None,
        prefer: str = "impression_then_findings",
        jpg_metadata_csv: Optional[str | Path] = None,
        jpg_root: Optional[str | Path] = None,
        split_csv: Optional[str | Path] = None,
        split: Optional[str] = None,
    ):
        if not sectioned_csv and not reports_root:
            raise ValueError("Provide either sectioned_csv or reports_root for MIMIC-CXR.")
        self.sectioned_csv = Path(sectioned_csv) if sectioned_csv else None
        self.reports_root = Path(reports_root) if reports_root else None
        self.prefer = prefer
        self.jpg_metadata_csv = Path(jpg_metadata_csv) if jpg_metadata_csv else None
        self.jpg_root = Path(jpg_root) if jpg_root else None
        self.split_csv = Path(split_csv) if split_csv else None
        self.split = split

    def dataframe(self) -> pd.DataFrame:
        # --- reports ---
        if self.sectioned_csv and self.sectioned_csv.exists():
            with _open_text(self.sectioned_csv) as f:
                df = pd.read_csv(f)
            for col in ("impression","findings"):
                df[col] = df[col].fillna("").map(_clean_text) if col in df.columns else ""
            df["subject_id"] = df["subject_id"].astype(int)
            df["study_id"]  = df["study_id"].astype(int)
        else:
            rows = []
            for p in Path(self.reports_root).rglob("s*.txt"):
                try:
                    sid, stid = int(p.parent.name[1:]), int(p.stem[1:])
                except Exception:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
                sec = self._sectionize_report(text)
                rows.append({
                    "subject_id": sid,
                    "study_id": stid,
                    "findings": _clean_text(sec.get("findings","")),
                    "impression": _clean_text(sec.get("impression","")),
                })
            df = pd.DataFrame(rows)

        df["caption"] = df.apply(
            lambda r: _choose_caption(r.get("impression",""), r.get("findings",""), self.prefer),
            axis=1,
        )

        df["dicom_id"] = None
        df["image"] = None

        # --- optional: join JPG metadata for image paths ---
        if self.jpg_metadata_csv is not None and self.jpg_metadata_csv.exists():
            with _open_text(self.jpg_metadata_csv) as f:
                meta = pd.read_csv(f)
            keep = [c for c in ["subject_id","study_id","dicom_id","path","ViewPosition"] if c in meta.columns]
            meta = meta[keep].drop_duplicates()
            df = df.merge(meta, on=["subject_id","study_id"], how="left")
            if "path" in df.columns and self.jpg_root is not None:
                df["image"] = df["path"].apply(lambda p: _maybe_abs(self.jpg_root, p) if isinstance(p,str) else None)

        # --- optional split filtering ---
        if self.split_csv is not None and self.split:
            with _open_text(self.split_csv) as f:
                spl = pd.read_csv(f)[["subject_id","study_id","split"]].drop_duplicates()
            df = df.merge(spl, on=["subject_id","study_id"], how="left")
            df = df[df["split"] == self.split].copy()

        # ensure columns
        for c in ["image","caption","subject_id","study_id","dicom_id","findings","impression"]:
            if c not in df.columns:
                df[c] = None
        return df[["image","caption","subject_id","study_id","dicom_id","findings","impression"]]

    @classmethod
    def _sectionize_report(cls, text: str) -> Dict[str,str]:
        lines = _normalize_nl(text).split("\n")
        i = 1 if (lines and cls.HEADER_LINE.match(lines[0])) else 0
        headers = []
        for idx in range(i, len(lines)):
            m = cls.SEC_LINE.match(lines[idx])
            if m: headers.append((idx, m.group(1).strip().lower()))
        if not headers:
            return {"findings": "\n".join(lines[i:]).strip()}
        headers.append((len(lines), "END"))
        out: Dict[str,str] = {}
        for (start, raw), (end, _) in zip(headers[:-1], headers[1:]):
            canon = cls._canonical_section(raw)
            if not canon: 
                continue
            chunk = "\n".join(lines[start+1:end]).strip()
            if chunk:
                out[canon] = (out.get(canon,"") + ("\n" if canon in out else "") + chunk)
        return out

    @classmethod
    def _canonical_section(cls, raw: str) -> Optional[str]:
        raw_norm = raw.replace("/", " ").replace("-", " ").strip().lower()
        for canon, aliases in cls.SECTION_ALIASES.items():
            if raw_norm in aliases or raw_norm == canon:
                return canon
        return None

# ======================================================
# CheXpert / CheXpert Plus
# ======================================================

@register_loader("chexpert")
class CheXpertLoader(BaseLoader):
    """
    CheXpert labels CSV + image paths.
    If you have CheXpert-Plus style reports, pass reports_csv (or a CSV that contains
    section_impression / section_findings / caption / Report).

    Args:
      labels_csv   : path to train.csv (must contain 'Path' column)
      images_root  : prepend to 'Path' if relative
      reports_csv  : optional CSV to join for captions
    """
    PAT_RE = re.compile(r"patient(\d+)/study(\d+)")

    def __init__(
        self,
        labels_csv: str | Path,
        images_root: Optional[str | Path] = None,
        reports_csv: Optional[str | Path] = None,
    ):
        self.labels_csv = Path(labels_csv)
        self.images_root = Path(images_root) if images_root else None
        self.reports_csv = Path(reports_csv) if reports_csv else None
        self._warned_missing_images = False
        self._suffix_resolution_cache: Dict[str, str] = {}

    def dataframe(self) -> pd.DataFrame:
        with _open_text(self.labels_csv) as f:
            df = pd.read_csv(f)

        path_column = None
        for candidate in ("Path", "path", "path_to_image", "image_path"):
            if candidate in df.columns:
                path_column = candidate
                break
        if path_column is None:
            raise ValueError("CheXpert CSV must contain a 'Path' or 'path_to_image' column.")

        df["image"] = df[path_column].astype(str)
        if "Path" not in df.columns:
            df["Path"] = df[path_column].astype(str)
        df["image"] = df["image"].apply(self._resolve_image_path)
        # derive subject/study from path when possible
        def _sid_study(p: str) -> Tuple[Optional[int], Optional[int]]:
            m = self.PAT_RE.search(p)
            if not m: return None, None
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                return None, None
        tmp = df["Path"].astype(str).apply(_sid_study).tolist()
        df["subject_id"] = [t[0] for t in tmp]
        df["study_id"]   = [t[1] for t in tmp]
        df["dicom_id"]   = None

        # captions from a CheXpert-Plus style CSV (optional)
        caption = None
        if self.reports_csv and self.reports_csv.exists():
            with _open_text(self.reports_csv) as f:
                rep = pd.read_csv(f)
            keys = []
            if {"subject_id","study_id"}.issubset(rep.columns): keys = ["subject_id","study_id"]
            elif {"Path"}.issubset(rep.columns): keys = ["Path"]
            elif {"path_to_image"}.issubset(rep.columns): keys = ["path_to_image"]
            elif {"image"}.issubset(rep.columns): keys = ["image"]
            if not keys:
                rep["__row__"] = range(len(rep))
                df["__row__"] = range(len(df))
                keys = ["__row__"]
            rep_cols = [c for c in ["Report","caption","section_impression","section_findings",
                                    "Path","path_to_image","image","subject_id","study_id","__row__"]
                        if c in rep.columns]
            df = df.merge(rep[rep_cols], on=keys, how="left")
            if "Report" in df.columns:
                caption = df["Report"]
            elif "caption" in df.columns:
                caption = df["caption"]
            else:
                imp = df["section_impression"] if "section_impression" in df.columns else pd.Series([""]*len(df))
                fin = df["section_findings"]   if "section_findings"   in df.columns else pd.Series([""]*len(df))
                caption = [ _choose_caption(i, f, "impression_then_findings") for i,f in zip(imp, fin) ]
        else:
            caption = [""] * len(df)

        df["caption"] = pd.Series(caption).fillna("").map(_clean_text)
        return df[["image","caption","subject_id","study_id","dicom_id"]]

    def _resolve_image_path(self, rel_path: str) -> str:
        rel = (rel_path or "").strip()
        if not rel:
            return rel

        if self.images_root is None:
            return rel

        candidate = _maybe_abs(self.images_root, rel)
        if not candidate:
            return rel

        suffix = Path(rel).suffix.lower()
        cached = self._suffix_resolution_cache.get(suffix)
        if cached is not None:
            if cached != suffix:
                base = Path(candidate).with_suffix("") if suffix else Path(candidate)
                try:
                    return str(base.with_suffix(cached))
                except ValueError:
                    return str(candidate)
            return str(candidate)

        resolved = self._validate_or_swap_extension(candidate)
        if resolved:
            resolved_suffix = Path(resolved).suffix.lower()
            self._suffix_resolution_cache[suffix] = resolved_suffix
            return resolved

        self._suffix_resolution_cache[suffix] = suffix
        return str(candidate)

    def _validate_or_swap_extension(self, candidate: str) -> Optional[str]:
        try:
            path = Path(candidate)
        except TypeError:
            return None
        if path.is_file():
            return str(path)

        suffix = path.suffix.lower()
        swap_order = [".png", ".jpg", ".jpeg"]
        if suffix in swap_order:
            swap_order = [ext for ext in swap_order if ext != suffix] + [suffix]
        base = path.with_suffix("") if suffix else path
        for ext in swap_order:
            try:
                alt = base.with_suffix(ext)
            except ValueError:
                continue
            if alt.is_file():
                return str(alt)

        if not self._warned_missing_images:
            print(
                "[warn] Some CheXpert image paths could not be resolved. "
                "Ensure images_root points to the directory containing image files.",
                file=sys.stderr,
            )
            self._warned_missing_images = True
        return None

# ======================================================
# IU X-Ray (OpenI)
# ======================================================

@register_loader("iuxray")
class IUXRayLoader(BaseLoader):
    """
    IU X-Ray / OpenI.

    Preferred input: JSON where each item has 'findings','impression','images'[{'path':...}].
    Args:
      json_path   : path to JSON (preferred)
      images_root : prepend to relative image paths
      xml_root    : optional directory of XML reports
      csv_path    : optional CSV with report/image columns
    """
    def __init__(
        self,
        json_path: Optional[str | Path] = None,
        images_root: Optional[str | Path] = None,
        xml_root: Optional[str | Path] = None,
        csv_path: Optional[str | Path] = None,
    ):
        self.json_path  = Path(json_path) if json_path else None
        self.images_root = Path(images_root) if images_root else None
        self.xml_root   = Path(xml_root) if xml_root else None
        self.csv_path   = Path(csv_path) if csv_path else None
        if not any([self.json_path, self.xml_root, self.csv_path]):
            raise ValueError("Provide one of: json_path, xml_root, or csv_path for IU X-Ray.")

    def dataframe(self) -> pd.DataFrame:
        if self.json_path and self.json_path.exists():
            data = json.loads(Path(self.json_path).read_text(encoding="utf-8"))
            rows = []
            for item in data:
                fin = _clean_text(item.get("findings",""))
                imp = _clean_text(item.get("impression",""))
                caption = _choose_caption(imp, fin, "impression_then_findings")
                imgs = item.get("images", []) or []
                if not imgs:
                    rows.append({"image": None, "caption": caption})
                else:
                    for im in imgs:
                        p = im.get("path") or im.get("filepath") or im.get("image")
                        if not p: 
                            continue
                        imgp = Path(p)
                        if self.images_root is not None and not imgp.is_absolute():
                            imgp = (self.images_root / imgp).resolve()
                        rows.append({"image": str(imgp), "caption": caption})
            df = pd.DataFrame(rows)
            df["subject_id"] = None; df["study_id"] = None; df["dicom_id"] = None
            return df[["image","caption","subject_id","study_id","dicom_id"]]

        if self.csv_path and self.csv_path.exists():
            with _open_text(self.csv_path) as f:
                df = pd.read_csv(f)
            cap = None
            if "Report" in df.columns: cap = df["Report"]
            elif "caption" in df.columns: cap = df["caption"]
            else:
                imp = df["impression"] if "impression" in df.columns else pd.Series([""]*len(df))
                fin = df["findings"] if "findings" in df.columns else pd.Series([""]*len(df))
                cap = [ _choose_caption(i, f, "impression_then_findings") for i,f in zip(imp, fin) ]
            df["caption"] = pd.Series(cap).fillna("").map(_clean_text)
            if "path" in df.columns:
                images = df["path"].astype(str)
                if self.images_root is not None:
                    images = images.apply(lambda p: _maybe_abs(self.images_root, p))
                df["image"] = images
            elif "image" in df.columns:
                images = df["image"].astype(str)
                if self.images_root is not None:
                    images = images.apply(lambda p: _maybe_abs(self.images_root, p))
                df["image"] = images
            else:
                df["image"] = None
            df["subject_id"] = None; df["study_id"] = None; df["dicom_id"] = None
            return df[["image","caption","subject_id","study_id","dicom_id"]]

        # XML fallback (best-effort)
        rows = []
        xml_files = list(Path(self.xml_root).rglob("*.xml"))
        for xp in xml_files:
            try:
                root = ET.parse(xp).getroot()
            except Exception:
                continue
            findings, impression = "", ""
            for ab in root.findall(".//Abstract/AbstractText"):
                label = (ab.attrib.get("Label","") or "").lower()
                txt = "".join(ab.itertext()).strip()
                if label == "findings": findings = txt
                if label == "impression": impression = txt
            caption = _choose_caption(_clean_text(impression), _clean_text(findings), "impression_then_findings")
            rows.append({"image": None, "caption": caption})
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["image","caption"])
        df["subject_id"] = None; df["study_id"] = None; df["dicom_id"] = None
        return df[["image","caption","subject_id","study_id","dicom_id"]]

# ======================================================
# Config via ENV / YAML
# ======================================================

def _load_dataset_cfg(name: str) -> dict:
    """
    Priority:
      1) Env vars (e.g., MIMIC_CXR_SECTIONED, CHEXPERT_CSV, etc.)
      2) DATASETS_YAML (or ./datasets.yaml or ~/.datasets.yaml), with ${ENV} expansion
      3) Defaults under DATASETS_ROOT (if set)
    """
    cfg = {}
    env = os.environ

    if name == "mimic_cxr":
        cfg.update({
            "sectioned_csv": env.get("MIMIC_CXR_SECTIONED"),
            "reports_root": env.get("MIMIC_CXR_REPORTS_ROOT"),
            "jpg_metadata_csv": env.get("MIMIC_CXR_JPG_METADATA"),
            "jpg_root": env.get("MIMIC_CXR_JPG_ROOT"),
            "split_csv": env.get("MIMIC_CXR_SPLIT_CSV"),
            "split": env.get("MIMIC_CXR_SPLIT"),
        })
    elif name == "chexpert":
        cfg.update({
            "labels_csv": env.get("CHEXPERT_CSV"),
            "images_root": env.get("CHEXPERT_IMAGES_ROOT"),
            "reports_csv": env.get("CHEXPERT_REPORTS_CSV"),  # if you have CheXpert-Plus style text
        })
    elif name == "iuxray":
        cfg.update({
            "json_path": env.get("IUXRAY_JSON"),
            "images_root": env.get("IUXRAY_IMAGES_ROOT"),
            "xml_root": env.get("IUXRAY_XML_ROOT"),
            "csv_path": env.get("IUXRAY_CSV"),
        })

    # YAML (optional)
    yaml_path = env.get("DATASETS_YAML")
    if _HAVE_YAML:
        for y in [yaml_path, "./datasets.yaml", str(Path.home() / ".datasets.yaml")]:
            if y and Path(y).exists():
                with open(y, "r", encoding="utf-8") as f:
                    ycfg = yaml.safe_load(f) or {}
                if name in ycfg:
                    ysec = {k: _expand_env_vars(v) for k, v in (ycfg[name] or {}).items()}
                    cfg = {**cfg, **ysec}
                break

    # Defaults under MEDCLIP_DATASETS_ROOT/DATASETS_ROOT (if any)
    root_env = env.get("MEDCLIP_DATASETS_ROOT") or env.get("DATASETS_ROOT")
    root = Path(root_env).expanduser().resolve() if root_env else None
    if root:
        if name == "mimic_cxr":
            cfg.setdefault("sectioned_csv", str(root / "mimic-cxr-reports/mimic_cxr_sectioned.csv.gz"))
            cfg.setdefault("reports_root", str(root / "mimic-cxr-reports/files"))
            cfg.setdefault("jpg_metadata_csv", str(root / "mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz"))
            cfg.setdefault("jpg_root", str(root / "mimic-cxr-jpg/files"))
            cfg.setdefault("split_csv", str(root / "mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz"))
            cfg.setdefault("split", None)
        elif name == "chexpert":
            cfg.setdefault("labels_csv", str(root / "CheXpert-v1.0/train.csv"))
            cfg.setdefault("images_root", str(root))  # 'Path' in CSV is relative to this
        elif name == "iuxray":
            cfg.setdefault("json_path", str(root / "iu_xray/indiana_reports.json"))
            cfg.setdefault("images_root", str(root / "iu_xray/images"))
    return cfg

def build_loader_from_env(name: str) -> BaseLoader:
    cfg = _load_dataset_cfg(name)
    if name == "mimic_cxr":
        if not (cfg.get("sectioned_csv") or cfg.get("reports_root")):
            raise SystemExit("mimic_cxr: set MIMIC_CXR_SECTIONED or MIMIC_CXR_REPORTS_ROOT (env/YAML).")
        return MIMICCXRLoader(
            sectioned_csv=cfg.get("sectioned_csv"),
            reports_root=cfg.get("reports_root"),
            jpg_metadata_csv=cfg.get("jpg_metadata_csv"),
            jpg_root=cfg.get("jpg_root"),
            split_csv=cfg.get("split_csv"),
            split=cfg.get("split"),
            prefer="impression_then_findings",
        )
    if name == "chexpert":
        if not cfg.get("labels_csv"):
            raise SystemExit("chexpert: set CHEXPERT_CSV (env/YAML).")
        return CheXpertLoader(
            labels_csv=cfg["labels_csv"],
            images_root=cfg.get("images_root"),
            reports_csv=cfg.get("reports_csv"),
        )
    if name == "iuxray":
        if not any([cfg.get("json_path"), cfg.get("xml_root"), cfg.get("csv_path")]):
            raise SystemExit("iuxray: set IUXRAY_JSON (preferred) or IUXRAY_XML_ROOT or IUXRAY_CSV.")
        return IUXRayLoader(
            json_path=cfg.get("json_path"),
            images_root=cfg.get("images_root"),
            xml_root=cfg.get("xml_root"),
            csv_path=cfg.get("csv_path"),
        )
    return build_loader(name, **cfg)

# ======================================================
# Export helpers (to feed the miner/embeder)
# ======================================================

def export_prepared_csvs(loader: BaseLoader, out_dir: Path) -> Tuple[Path, Path]:
    """
    Writes:
      reports.csv with columns: study_id,report
      images.csv  with columns: study_id,path
    Returns (reports_csv, images_csv). Missing ones are still returned as paths (may not exist).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = loader.dataframe()

    # reports.csv
    reports_csv = out_dir / "reports.csv"
    rep = df.loc[df["caption"].astype(str).str.len() > 0, ["study_id","caption"]].dropna(subset=["study_id"])
    if not rep.empty:
        rep = rep.copy()
        rep["study_id"] = rep["study_id"].astype(str)
        with open(reports_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["study_id","report"])
            w.writeheader()
            for r in rep.itertuples(index=False):
                w.writerow({"study_id": str(r.study_id), "report": r.caption})
    else:
        # touch empty file for consistency
        reports_csv.write_text("study_id,report\n", encoding="utf-8")

    # images.csv
    images_csv = out_dir / "images.csv"
    imgs = df.loc[df["image"].notna(), ["study_id","image"]].dropna(subset=["study_id"])
    if not imgs.empty:
        imgs = imgs.copy()
        imgs["study_id"] = imgs["study_id"].astype(str)
        with open(images_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["study_id","path"])
            w.writeheader()
            for r in imgs.itertuples(index=False):
                w.writerow({"study_id": str(r.study_id), "path": r.image})
    else:
        images_csv.write_text("study_id,path\n", encoding="utf-8")

    return reports_csv, images_csv

# ======================================================
# CLI
# ======================================================

def _expand_env_in_str(s: str) -> str:
    return _expand_env_vars(s)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dataset loaders → export standardized CSVs for the miner.")
    ap.add_argument("--dataset", required=True, choices=list(_LOADER_REGISTRY.keys()))
    ap.add_argument("--export_dir", default="${MEDCLIP_DATASETS_ROOT}/_prepared")
    args = ap.parse_args()

    export_root = Path(_expand_env_in_str(args.export_dir)).resolve() / args.dataset
    loader = build_loader_from_env(args.dataset)
    rep_csv, img_csv = export_prepared_csvs(loader, export_root)

    print("[dataset_loaders] wrote:")
    print(" ", rep_csv)
    print(" ", img_csv)

if __name__ == "__main__":
    main()
