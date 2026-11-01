#!/usr/bin/env python3
"""Build the ECG / not-ECG gate dataset (binary: ``ecg`` vs ``not_ecg``).

Positives are ECG sheets the classifier is meant to see: clean12 renders and
their synthetic photographs (the training distribution), ECG-Image-Database
generated / scanned / phone-photographed printouts, and MEETI renders (a
different 12-row layout). Negatives are COCO photos, RVL-CDIP documents
(50 per class), ChartQA charts (line charts first), CORD receipts, synthetic
ECG-look-alike paper (``synth_negatives.py``) and anything in
``data_raw/self_collected/``.

Every kept image is re-encoded identically (EXIF orientation applied, alpha
composited on white, RGB, longest side <= 1600 px, JPEG q95) so file format,
alpha or EXIF cannot separate the classes. Native size still differs by source
(COCO 640 px, ChartQA ~300-850 px, ECG sheets 1000-1600 px), so the trainer
must resize every image to one fixed input size, never crop at native scale.
Splits are 70/15/15, grouped by the unit that produces near-duplicates
(patient, ECG record, receipt, chart image, self_collected subfolder) and
stratified by source, so every source is in every split.

Outputs under --root:
  data_raw/<source>/          sampled originals of downloaded negatives
  data_raw/self_collected/    group phone photos (optional, all negatives)
  images/<label>/<source>/    processed images the manifest points to
  manifest.csv (+ .sha256)    one row per image
  build_report.json           counts, sources, licences, duplicate audit

Downloads are expected in data_raw/_downloads (see DOWNLOADS); the build is
deterministic for a given --seed and set of inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_negatives import KINDS, render  # noqa: E402

DATASET_ROOT = Path("/media/lindtseyxvii/Data/CARDIOSENTRY_DATASET")
DEFAULT_ROOT = DATASET_ROOT / "ecg_gate"
CLEAN12_SAMPLES = Path.home() / "cardiologist_samples_clean12"
ECG_IMAGE_DB = DATASET_ROOT / "ECG-Image-Database"
MEETI = DATASET_ROOT / "MEETI"

MAX_SIDE = 1600
JPEG_QUALITY = 95
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = np.array([0.70, 0.15, 0.15])
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
HEIF_EXTS = {".heic", ".heif"}

QUOTAS = {
    "ecgimgdb_d0": 450,       # images, one per record, variants round-robin
    "ecgimgdb_d1": 300,       # images, one per record, variants round-robin
    "ecgimgdb_d2_records": 200,  # records, two photographs each
    "meeti": 600,             # images, one per patient
    "coco_val2017": 800,
    "chartqa": 500,
    "synthetic_per_kind": 125,
}

DOWNLOADS = {
    "coco_val2017": ["coco_val2017.zip"],
    "rvl_cdip": ["rvl_cdip_100pc_train.parquet"],
    "chartqa": ["chartqa_test.parquet", "chartqa_val.parquet"],
    "cord_v2": ["cord_v2_test.parquet", "cord_v2_validation.parquet", "cord_v2_train0.parquet"],
}

RVL_CLASSES = ("letter", "form", "email", "handwritten", "advertisement", "scientific report",
               "scientific publication", "specification", "file folder", "news article", "budget",
               "invoice", "presentation", "questionnaire", "resume", "memo")

SOURCES = {
    "clean12_samples": {"label": "ecg", "origin": str(CLEAN12_SAMPLES),
                        "note": "HEEDB clean12 renders (training renderer) + synthetic photographs, all 992"},
    "ecgimgdb_d0": {"label": "ecg", "origin": str(ECG_IMAGE_DB / "D0"),
                    "url": "https://doi.org/10.48550/arXiv.2409.16612", "license": "see ECG-Image-Database terms"},
    "ecgimgdb_d1": {"label": "ecg", "origin": str(ECG_IMAGE_DB / "D1"),
                    "url": "https://doi.org/10.48550/arXiv.2409.16612", "license": "see ECG-Image-Database terms"},
    "ecgimgdb_d2": {"label": "ecg", "origin": str(ECG_IMAGE_DB / "D2"),
                    "url": "https://doi.org/10.48550/arXiv.2409.16612", "license": "see ECG-Image-Database terms"},
    "meeti": {"label": "ecg", "origin": str(MEETI), "license": "PhysioNet credentialed (MIMIC-IV-ECG derived)"},
    "coco_val2017": {"label": "not_ecg", "url": "http://images.cocodataset.org/zips/val2017.zip",
                     "license": "COCO terms; images under their Flickr licences"},
    "rvl_cdip": {"label": "not_ecg", "url": "https://huggingface.co/datasets/jordyvl/rvl_cdip_100_examples_per_class",
                 "note": "parquet mirror of aharley/rvl_cdip (train split = 50 images x 16 classes); "
                         "the original loader needs the 38.8 GB tarball",
                 "license": "RVL-CDIP / IIT-CDIP research terms"},
    "chartqa": {"label": "not_ecg", "url": "https://huggingface.co/datasets/ahmed-masry/ChartQA",
                "note": "data/test + data/val parquet of the same repo instead of 'ChartQA Dataset.zip'",
                "license": "GPL-3.0"},
    "cord_v2": {"label": "not_ecg", "url": "https://huggingface.co/datasets/naver-clova-ix/cord-v2",
                "note": "test + validation + train shard 0 parquet", "license": "CC BY 4.0"},
    "synthetic_lookalike": {"label": "not_ecg", "origin": "synth_negatives.py",
                            "note": "ECG-style grid paper without traces, graph paper with smooth curves, "
                                    "notebook grids with text, smooth curves on ECG grid; half photo-treated"},
    "self_collected": {"label": "not_ecg", "note": "group phone photos; included when present"},
}


@dataclass
class Item:
    item_id: str
    label: str
    source: str
    subsource: str
    group_id: str
    origin: str          # file path, or a spec for synthetic images


# ---------------------------------------------------------------- positives

def clean12_items() -> list[Item]:
    items = []
    with (CLEAN12_SAMPLES / "index.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            group = f"heedb_patient_{row['patient_key']}"
            for col, sub in (("clean_image", "clean"), ("augmented_images", "synthetic_photo")):
                for rel in filter(None, row[col].split(";")):
                    path = CLEAN12_SAMPLES / rel
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    items.append(Item(f"clean12_{Path(rel).stem}", "ecg", "clean12_samples", sub, group, str(path)))
    return items


def _ecgimgdb_records(part: str) -> dict[str, list[Path]]:
    records = {}
    for rec in sorted((ECG_IMAGE_DB / part).iterdir()):
        if rec.is_dir():
            imgs = sorted(p for p in rec.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if imgs:
                records[rec.name] = imgs
    return records


def _variant(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def ecgimgdb_items(rng: np.random.Generator) -> list[Item]:
    items = []
    for part, quota in (("D0", QUOTAS["ecgimgdb_d0"]), ("D1", QUOTAS["ecgimgdb_d1"])):
        records = _ecgimgdb_records(part)
        chosen = rng.permutation(sorted(records))[:quota]
        for k, rec in enumerate(chosen):
            imgs = records[rec]
            # Round-robin over the variants this record has, so scans, phone photos and
            # generated sheets are all represented rather than whatever sorts first.
            img = imgs[k % len(imgs)]
            items.append(Item(f"ecgimgdb_{img.stem}", "ecg", f"ecgimgdb_{part.lower()}", _variant(img),
                              f"ecgimgdb_{rec}", str(img)))
    records = _ecgimgdb_records("D2")
    for rec in rng.permutation(sorted(records))[:QUOTAS["ecgimgdb_d2_records"]]:
        imgs = records[rec]
        for idx in sorted(rng.choice(len(imgs), size=min(2, len(imgs)), replace=False)):
            img = imgs[idx]
            items.append(Item(f"ecgimgdb_{img.stem}", "ecg", "ecgimgdb_d2", _variant(img)[0],
                              f"ecgimgdb_{rec}", str(img)))
    return items


def meeti_items(rng: np.random.Generator, cache: Path) -> list[Item]:
    if not cache.is_file():
        print(f"listing MEETI PNGs into {cache} (one-off, slow on this drive)", flush=True)
        found = []
        for dirpath, _dirs, files in os.walk(MEETI):
            found += [os.path.join(dirpath, f) for f in files if f.endswith(".png")]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(sorted(found)) + "\n")
    by_patient = defaultdict(list)
    for line in sorted(set(cache.read_text().split())):
        by_patient[Path(line).parts[-3]].append(line)
    items = []
    for patient in rng.permutation(sorted(by_patient))[:QUOTAS["meeti"]]:
        studies = by_patient[patient]
        path = Path(studies[int(rng.integers(len(studies)))])
        items.append(Item(f"meeti_{patient}_{path.stem}", "ecg", "meeti", "render", f"meeti_{patient}", str(path)))
    return items


# ---------------------------------------------------------------- negatives

def _image_bytes(cell) -> bytes:
    if isinstance(cell, dict):
        return cell["bytes"]
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unexpected image cell {type(cell)}")


def _iter_parquet(path: Path, columns: list[str]):
    pf = pq.ParquetFile(path)
    row = 0
    for batch in pf.iter_batches(batch_size=16, columns=columns):
        for rec in batch.to_pylist():
            yield row, rec
            row += 1


def _sniff_ext(data: bytes) -> str:
    fmt = Image.open(io.BytesIO(data)).format or "bin"
    return {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp"}.get(fmt, f".{fmt.lower()}")


def _write_raw(raw_dir: Path, name: str, data: bytes) -> Path:
    path = raw_dir / f"{name}{_sniff_ext(data)}"
    if not path.is_file() or path.stat().st_size != len(data):
        path.write_bytes(data)
    return path


def coco_items(rng: np.random.Generator, root: Path) -> list[Item]:
    raw = root / "data_raw" / "coco_val2017"
    raw.mkdir(parents=True, exist_ok=True)
    items = []
    with zipfile.ZipFile(root / "data_raw/_downloads/coco_val2017.zip") as zf:
        members = sorted(n for n in zf.namelist() if n.endswith(".jpg"))
        for name in rng.permutation(members)[:QUOTAS["coco_val2017"]]:
            path = _write_raw(raw, Path(name).stem, zf.read(name))
            items.append(Item(f"coco_{path.stem}", "not_ecg", "coco_val2017", "photo", f"coco_{path.stem}", str(path)))
    return items


def rvl_items(root: Path) -> list[Item]:
    raw = root / "data_raw" / "rvl_cdip"
    raw.mkdir(parents=True, exist_ok=True)
    items = []
    for row, rec in _iter_parquet(root / "data_raw/_downloads/rvl_cdip_100pc_train.parquet", ["image", "label"]):
        cls = RVL_CLASSES[int(rec["label"])]
        name = f"rvl_train_{row:04d}"
        path = _write_raw(raw, name, _image_bytes(rec["image"]))
        items.append(Item(name, "not_ecg", "rvl_cdip", cls.replace(" ", "_"), name, str(path)))
    counts = Counter(i.subsource for i in items)
    if len(counts) != 16 or set(counts.values()) != {50}:
        raise ValueError(f"RVL-CDIP subset is not 50 x 16: {counts}")
    return items


LINE_RE = re.compile(r"\blines?\b", re.IGNORECASE)
LINE_SOLID_MAX = 0.05


def _solid_ink_fraction(data: bytes) -> float:
    """Share of non-background ink surviving a 9x9 erosion: ~0 for line charts, 0.3-0.9 for bars/pies.

    Measured on the 2,563 unique ChartQA test+val images: bimodal, 346 below 0.05 (all line charts
    in a visual spot check), 25th percentile 0.49. Bars thinner than 9 px start at ~0.06.
    """
    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(np.int32)
    q = a >> 4
    key = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    k = int(np.bincount(key.ravel(), minlength=4096).argmax())
    background = np.array([(k >> 8) & 15, (k >> 4) & 15, k & 15]) * 16 + 8
    ink = (np.abs(a - background).sum(axis=2) > 60).astype(np.uint8)
    return float(cv2.erode(ink, np.ones((9, 9), np.uint8)).sum() / max(int(ink.sum()), 1))


def chartqa_items(rng: np.random.Generator, root: Path) -> list[Item]:
    raw = root / "data_raw" / "chartqa"
    raw.mkdir(parents=True, exist_ok=True)
    images: dict[str, dict] = {}
    for fname in DOWNLOADS["chartqa"]:
        path = root / "data_raw/_downloads" / fname
        cols = [c for c in pq.ParquetFile(path).schema_arrow.names]
        img_col = "image" if "image" in cols else next(c for c in cols if "im" in c.lower())
        query_col = next((c for c in ("query", "question") if c in cols), None)
        for _row, rec in _iter_parquet(path, [img_col] + ([query_col] if query_col else [])):
            data = _image_bytes(rec[img_col])
            sha = hashlib.sha256(data).hexdigest()
            entry = images.setdefault(sha, {"data": None, "line": False, "split": fname})
            if entry["data"] is None:
                entry["data"] = data
            if query_col and LINE_RE.search(str(rec[query_col] or "")):
                entry["line"] = True
    for entry in images.values():
        if not entry["line"] and _solid_ink_fraction(entry["data"]) < LINE_SOLID_MAX:
            entry["line"] = True
    line = [s for s in sorted(images) if images[s]["line"]]
    other = [s for s in sorted(images) if not images[s]["line"]]
    quota = QUOTAS["chartqa"]
    picked = list(rng.permutation(line)[:quota])
    picked += list(rng.permutation(other)[:quota - len(picked)])
    items = []
    for sha in picked:
        name = f"chartqa_{sha[:16]}"
        path = _write_raw(raw, name, images[sha]["data"])
        sub = "line_chart" if images[sha]["line"] else "other_chart"
        items.append(Item(name, "not_ecg", "chartqa", sub, name, str(path)))
    print(f"chartqa: {len(images)} unique images, {len(line)} line charts (query or pixel heuristic)", flush=True)
    return items


def cord_items(root: Path) -> list[Item]:
    raw = root / "data_raw" / "cord_v2"
    raw.mkdir(parents=True, exist_ok=True)
    items = []
    seen = set()
    for fname in DOWNLOADS["cord_v2"]:
        split = fname.removeprefix("cord_v2_").removesuffix(".parquet")
        for row, rec in _iter_parquet(root / "data_raw/_downloads" / fname, ["image"]):
            data = _image_bytes(rec["image"])
            sha = hashlib.sha256(data).hexdigest()
            if sha in seen:
                continue
            seen.add(sha)
            name = f"cord_{split}_{row:04d}"
            path = _write_raw(raw, name, data)
            items.append(Item(name, "not_ecg", "cord_v2", "receipt", name, str(path)))
    return items


def synthetic_items(seed: int) -> list[Item]:
    items = []
    for k, kind in enumerate(KINDS):
        for i in range(QUOTAS["synthetic_per_kind"]):
            img_seed = seed * 1000 + k * 100_000 + i
            photo = i % 2 == 1
            name = f"synth_{kind}_{i:03d}"
            items.append(Item(name, "not_ecg", "synthetic_lookalike", f"{kind}{'_photo' if photo else ''}",
                              name, f"synth:{kind}:{img_seed}:{int(photo)}"))
    return items


def self_collected_items(root: Path) -> list[Item]:
    """Every image under data_raw/self_collected; one subfolder = one scene/session = one split group."""
    base = root / "data_raw" / "self_collected"
    files = sorted(p for p in base.rglob("*") if p.is_file())
    if any(p.suffix.lower() in HEIF_EXTS for p in files):
        try:
            import pillow_heif  # noqa: F401
        except ImportError:
            sys.exit("self_collected has HEIC/HEIF photos: `pip install pillow-heif` or convert them to JPEG")
    items = []
    for path in files:
        if path.suffix.lower() not in IMAGE_EXTS | HEIF_EXTS:
            continue
        rel = path.relative_to(base)
        sub = rel.parts[0] if len(rel.parts) > 1 else "unsorted"
        digest = hashlib.sha1(str(rel).encode()).hexdigest()[:8]
        name = f"self_{re.sub(r'[^A-Za-z0-9_-]+', '_', path.stem)}_{digest}"
        group = name if sub == "unsorted" else f"self_{sub}"
        items.append(Item(name, "not_ecg", "self_collected", sub, group, str(path)))
    return items


# ---------------------------------------------------------------- processing

def _load(origin: str) -> tuple[Image.Image, tuple[int, int]]:
    """Return the EXIF-upright image and its full-resolution upright size."""
    if origin.startswith("synth:"):
        _, kind, seed, photo = origin.split(":")
        img = render(kind, int(seed), bool(int(photo)))
        return img, img.size
    if Path(origin).suffix.lower() in HEIF_EXTS:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    img = Image.open(origin)
    w, h = img.size
    if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):
        w, h = h, w
    if img.format == "JPEG":
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))
    return ImageOps.exif_transpose(img), (w, h)


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("I", "F") or img.mode.startswith("I;16"):
        a = np.asarray(img, dtype=np.float64)
        img = Image.fromarray((a * 255.0 / max(a.max(), 1.0)).clip(0, 255).astype(np.uint8), "L")
    if img.mode in ("RGBA", "LA", "La", "PA", "RGBa") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return img.convert("RGB")


def _dhash(img: Image.Image) -> str:
    """256-bit difference hash (16x16). The 64-bit version collides on every sheet sharing a layout."""
    g = np.asarray(img.convert("L").resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
    return np.packbits(g[:, 1:] > g[:, :-1]).tobytes().hex()


def process(args: tuple[dict, str]) -> dict:
    item, out_root = args
    img, (orig_w, orig_h) = _load(item["origin"])
    img = _to_rgb(img)
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    rel = Path("images") / item["label"] / item["source"] / f"{item['item_id']}.jpg"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    data = buf.getvalue()
    out = Path(out_root) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {**item, "path": str(rel), "width": img.size[0], "height": img.size[1],
            "orig_width": orig_w, "orig_height": orig_h,
            "sha256": hashlib.sha256(data).hexdigest(), "dhash": _dhash(img)}


# ---------------------------------------------------------------- splitting

def merge_near_duplicate_groups(rows: list[dict]) -> int:
    """Union groups of the same (label, source) that share a dHash, so re-saved copies (CORD has
    some) cannot straddle splits. Returns how many groups were absorbed."""
    parent: dict[str, str] = {}

    def find(g: str) -> str:
        while parent.setdefault(g, g) != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    first: dict[tuple, str] = {}
    for r in sorted(rows, key=lambda r: r["group_id"]):
        key = (r["label"], r["source"], r["dhash"])
        if key in first:
            a, b = find(first[key]), find(r["group_id"])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            first[key] = r["group_id"]
    merged = sum(1 for g in list(parent) if find(g) != g)
    for r in rows:
        r["group_id"] = find(r["group_id"])
    return merged


def assign_splits(rows: list[dict], rng: np.random.Generator) -> None:
    """Grouped, per-source stratified 70/15/15: each group goes to the split furthest below target."""
    strata = defaultdict(lambda: defaultdict(list))
    for r in rows:
        strata[(r["label"], r["source"])][r["group_id"]].append(r)
    for key in sorted(strata):
        groups = strata[key]
        total = sum(len(v) for v in groups.values())
        target = SPLIT_RATIOS * total
        have = np.zeros(3)
        for gid in rng.permutation(sorted(groups)):
            members = groups[gid]
            k = int(np.argmax(target - have))
            have[k] += len(members)
            for r in members:
                r["split"] = SPLITS[k]


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="delete an existing images/ tree and rebuild")
    args = ap.parse_args()
    root: Path = args.root

    missing = [f for files in DOWNLOADS.values() for f in files if not (root / "data_raw/_downloads" / f).is_file()]
    if missing:
        sys.exit(f"missing downloads in {root / 'data_raw/_downloads'}: {missing}")
    if (root / "images").exists():
        if not args.force:
            sys.exit(f"{root / 'images'} exists; pass --force to rebuild")
        shutil.rmtree(root / "images")
    (root / "data_raw" / "self_collected").mkdir(parents=True, exist_ok=True)

    # One generator per source keeps each source's sample independent of the others' sizes.
    seeds = np.random.SeedSequence(args.seed).spawn(5)
    rngs = [np.random.default_rng(s) for s in seeds]
    items: list[Item] = []
    items += clean12_items()
    items += ecgimgdb_items(rngs[0])
    items += meeti_items(rngs[1], root / "_cache" / "meeti_png_list.txt")
    items += coco_items(rngs[2], root)
    items += rvl_items(root)
    items += chartqa_items(rngs[3], root)
    items += cord_items(root)
    items += synthetic_items(args.seed)
    items += self_collected_items(root)

    ids = Counter(i.item_id for i in items)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        raise ValueError(f"duplicate item ids: {dupes[:10]}")
    print(f"processing {len(items)} images with {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, row in enumerate(pool.map(process, [(asdict(i), str(root)) for i in items], chunksize=8), 1):
            rows.append(row)
            if n % 500 == 0:
                print(f"  {n}/{len(items)}", flush=True)

    rows.sort(key=lambda r: (r["label"], r["source"], r["item_id"]))
    merged_groups = merge_near_duplicate_groups(rows)
    assign_splits(rows, rngs[4])
    fields = ["path", "label", "split", "source", "subsource", "group_id", "item_id", "width", "height",
              "orig_width", "orig_height", "sha256", "dhash", "origin"]
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    manifest_sha = hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest()
    (root / "manifest.csv.sha256").write_text(f"{manifest_sha}  manifest.csv\n")

    # Duplicate audit: identical bytes anywhere; identical 256-bit dHash across splits or labels
    # (same-source matches were merged into one group above, so anything left is cross-source).
    by_sha = defaultdict(list)
    by_dhash = defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r)
        by_dhash[r["dhash"]].append(r)
    exact = [[r["path"] for r in v] for v in by_sha.values() if len(v) > 1]
    near_cross = [[f"{r['split']}:{r['label']}:{r['path']}" for r in v] for v in by_dhash.values()
                  if len({r["split"] for r in v}) > 1 or len({r["label"] for r in v}) > 1]

    def table(key):
        out = defaultdict(lambda: dict.fromkeys(SPLITS, 0))
        for r in rows:
            out[r[key]][r["split"]] += 1
        return {k: {**v, "total": sum(v.values())} for k, v in sorted(out.items())}

    report = {
        "seed": args.seed,
        "manifest_sha256": manifest_sha,
        "n_images": len(rows),
        "processing": {"max_side": MAX_SIDE, "format": "JPEG", "quality": JPEG_QUALITY,
                       "steps": "EXIF transpose, alpha over white, RGB, LANCZOS thumbnail"},
        "split_ratios": dict(zip(SPLITS, SPLIT_RATIOS.tolist())),
        "split_unit": "group_id (patient / record / image / self_collected subfolder), same-source "
                      "identical-dHash groups merged, stratified by (label, source)",
        "near_duplicate_groups_merged": merged_groups,
        "by_label": table("label"),
        "by_source": table("source"),
        "by_subsource": {s: dict(Counter(r["subsource"] for r in rows if r["source"] == s))
                         for s in sorted({r["source"] for r in rows})},
        "groups_per_source": {s: len({r["group_id"] for r in rows if r["source"] == s})
                              for s in sorted({r["source"] for r in rows})},
        "exact_duplicate_sets": exact,
        "dhash_collisions_across_split_or_label": near_cross,
        "sources": SOURCES,
        "quotas": QUOTAS,
    }
    (root / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_images", "by_label", "by_source")}, indent=2))
    print(f"exact duplicate sets: {len(exact)}; dHash collisions across split/label: {len(near_cross)}")


if __name__ == "__main__":
    main()
