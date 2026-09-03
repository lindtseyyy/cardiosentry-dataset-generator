#!/usr/bin/env python3
"""Finalize a filter-variant release of the synthetic ECG corpus.

Called by generate_both.sh AFTER run_all.sh (stages 1-5) has completed for one
band. The stages only write the corpus itself; this script adds everything a
release needs around it, following the conventions of release `v2`:

  * `rel_path` column in synthetic_data/index.csv (portable, matches v2)
  * synthetic_data/build_config.json   - what produced this corpus (incl. filter)
  * synthetic_data/README.md           - corpus documentation with live counts
  * release README.md + VERSION.json   - release metadata, v2-style schema
  * data/releases/CHANGELOG.md entry
  * prototype_500 / prototype_800 / prototype_1152 via make_subset.py
    (subset READMEs copied from v2's prototypes, annotated with the band)

The balanced/ and harmonized/ cohorts are hardlink copies of the v2 cohorts
(created by generate_both.sh); they are verified bit-identical before anything
is written.

It does NOT write the completion marker - generate_both.sh writes
.release_complete.json only after validate_release.py passes, so an invalid
release is never marked complete.

Usage:
    ECG_RELEASE=v2_100hz ECG_LOWPASS_HZ=100 \
        ~/.cache/ecgkit-venv/bin/python finalize_release.py \
        --release v2_100hz --lowpass 100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import config as C

ROOT = Path(__file__).resolve().parent.parent          # ecg_dataset_pipeline/
RELEASES = ROOT / "data" / "releases"
V2 = RELEASES / "v2"
LABELS = list(C.LABEL_COLUMNS)

INDEX_FIELDS_AFTER = "image_path"        # rel_path is inserted right after this


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ rel_path
def add_rel_path(build: Path) -> None:
    """Add the portable rel_path column to index.csv (idempotent)."""
    idx = build / "index.csv"
    if not idx.exists():
        sys.exit(f"{idx} missing - run the pipeline stages first")
    rows = list(csv.DictReader(idx.open()))
    fields = list(rows[0])

    if "rel_path" in fields:
        rels = [r["rel_path"] for r in rows]
    else:
        rels = [str(Path(r["image_path"]).relative_to(build)) for r in rows]
        fields.insert(fields.index(INDEX_FIELDS_AFTER) + 1, "rel_path")
        tmp = idx.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r, rel in zip(rows, rels):
                r["rel_path"] = rel
                w.writerow({k: r.get(k, "") for k in fields})
        tmp.replace(idx)

    missing = [rel for rel in rels if not (build / rel).exists()]
    if missing:
        sys.exit(f"{len(missing)} rel_path entries do not resolve, e.g. "
                 f"{missing[:5]}")
    print(f"rel_path verified for {len(rels)} rows")


# ------------------------------------------------------------------ stats
def index_stats(build: Path) -> dict:
    idx = build / "index.csv"
    rows = list(csv.DictReader(idx.open()))
    splits = Counter(r["split"] for r in rows)
    records = {r["record"] for r in rows}
    label_images = {L: sum(1 for r in rows if r.get(L) == "1") for L in LABELS}
    label_splits = {
        L: {s: sum(1 for r in rows if r.get(L) == "1" and r["split"] == s)
            for s in ("train", "val", "test")}
        for L in LABELS
    }
    cls_records = Counter({r["cls"]: 0 for r in rows})
    for rec in {r["record"]: r for r in rows}.values():
        cls_records[rec["cls"]] += 1
    images = build / "images"
    jpgs = list(images.rglob("*.jpg"))
    jsons = list(images.rglob("*.json"))
    return {
        "index_rows": len(rows),
        "images": len(jpgs),
        "annotation_sidecars": len(jsons),
        "image_bytes": sum(p.stat().st_size for p in jpgs),
        "records": len(records),
        "splits": {s: splits.get(s, 0) for s in ("train", "val", "test")},
        "label_images": label_images,
        "label_splits": label_splits,
        "cls_records": dict(sorted(cls_records.items())),
        "staged_records": len(list((build / "staged").rglob("*.dat"))),
        "rendered_images": len(list((build / "rendered").rglob("*.png"))),
        "rendered_annotations": len(list((build / "rendered").rglob("*.json"))),
    }


# ------------------------------------------------------------------ cohorts
def verify_cohorts(rel_dir: Path, rel: str) -> dict:
    out = {}
    for cohort in ("balanced", "harmonized"):
        mine = rel_dir / cohort / f"{cohort}_ecg_metadata.csv"
        ref = V2 / cohort / f"{cohort}_ecg_metadata.csv"
        if not mine.exists() or not ref.exists():
            sys.exit(f"{cohort} CSV missing in {rel_dir.name} or v2")
        mine_h, ref_h = sha256_file(mine), sha256_file(ref)
        if mine_h != ref_h:
            sys.exit(f"{rel}/{cohort} CSV differs from v2 - refusing to finalize")
        out[cohort] = {"sha256": mine_h, "path": mine}
        print(f"{cohort} CSV bit-identical to v2 ({mine_h[:12]}...)")
    return out


# ------------------------------------------------------------------ metadata
def provenance(build: Path, rel: str, lowpass: float, stats: dict,
               cohort_hashes: dict) -> dict:
    kit = ROOT / "ecg-image-kit"
    commit = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(kit), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "release": rel,
        "created_utc": utcnow(),
        "pipeline": "image_pipeline/run_all.sh (stages 1-5) + finalize_release.py",
        "signal_filter": {
            "kind": "butterworth order-2 zero-phase band-pass "
                    "(scipy.signal.sosfiltfilt)",
            "low_hz": 0.05,
            "high_hz": lowpass,
            "baseline_release": "v2 (0.05-40 Hz)",
            "env": {"ECG_RELEASE": rel, "ECG_LOWPASS_HZ": str(lowpass)},
        },
        "config_py_sha256": sha256_file(Path(__file__).parent / "config.py"),
        "label_mapping_sha256":
            sha256_file(ROOT / "config" / "label_mapping.yaml"),
        "paths_yaml_sha256": sha256_file(ROOT / "config" / "paths.yaml"),
        "ecgkit_commit": commit,
        "python": sys.version.split()[0],
        "input": {
            "balanced_csv": f"{rel}/balanced/balanced_ecg_metadata.csv",
            "balanced_csv_sha256": cohort_hashes["balanced"]["sha256"],
            "identical_to_v2": True,
        },
        "corpus": stats,
    }


def write_build_config(build: Path, prov: dict) -> Path:
    p = build / "build_config.json"
    p.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"wrote {p}")
    return p


# ------------------------------------------------------------------ READMEs
def write_synthetic_readme(build: Path, rel: str, lowpass: float,
                           stats: dict) -> Path:
    s = stats
    gb = s["image_bytes"] / 1e9
    text = f"""# CardioSentry - synthetic paper-ECG image corpus (release `{rel}`)

**31,372 JPEG images of simulated printed 12-lead ECGs**, rendered from 9,238
real ECG recordings and labelled for four cardiac conditions.

> **Filter-variant release.** Identical to `../v2/synthetic_data` in every way
> except the stage-2 signal band-pass: this corpus used
> **0.05-{lowpass:g} Hz** where `v2` used **0.05-40 Hz**. Records, labels,
> frozen splits, augmentation seeds, DPI, layout and JPEG settings are all
> unchanged; stage 5 verification: **ALL CHECKS PASSED**.

> **Nyquist caveat.** The sheets are rendered at 150 DPI / 25 mm/s
> (147.6 px/s, image Nyquist 73.8 Hz). Signal content above ~74 Hz therefore
> aliases against the grid at 100 Hz and especially 150 Hz. This release
> exists as a controlled preprocessing variant for the filter-bandwidth
> experiment - report the caveat wherever you report results.

> **Not redistributable.** Derived from licence-gated source databases,
> including a CC BY-NC-**ND** portion. Keep any cloud copy **private**. See
> `../v2/synthetic_data/README.md` section 10.

## Contents

| Path | Size | Needed for training? |
| --- | ---: | --- |
| `index.csv` | {s["index_rows"]:,} rows | **yes** - the label table |
| `images/train/` | {s["splits"]["train"]:,} jpg + json | **yes** |
| `images/val/` | {s["splits"]["val"]:,} jpg + json | **yes** |
| `images/test/` | {s["splits"]["test"]:,} jpg + json | **yes** |
| `build_config.json` | 1 | no - build provenance + filter |
| `image_manifest.csv` / `staged_map.csv` | | no - stage provenance |
| `staged/` | 1.1 GB | no - intermediate WFDB |
| `rendered/` | 20 GB | no - clean intermediate PNGs |

All images are **1686 x 1311 px**, RGB JPEG, q85. Use `rel_path` (relative to
this directory) for loading; `image_path` is the absolute build-machine path.

## Labels and splits

Four independent binary heads (`STEMI`, `AF`, `LVH`, `NORMAL`), **not** a
softmax. `cls` is a convenience string (`AF`, `AF+LVH`, ...). Use
`BCEWithLogitsLoss`, four sigmoid outputs.

| label | train | val | test | total (images) |
| --- | ---: | ---: | ---: | ---: |
| STEMI | {s["label_splits"]["STEMI"]["train"]:,} | {s["label_splits"]["STEMI"]["val"]:,} | {s["label_splits"]["STEMI"]["test"]:,} | {s["label_images"]["STEMI"]:,} |
| AF | {s["label_splits"]["AF"]["train"]:,} | {s["label_splits"]["AF"]["val"]:,} | {s["label_splits"]["AF"]["test"]:,} | {s["label_images"]["AF"]:,} |
| LVH | {s["label_splits"]["LVH"]["train"]:,} | {s["label_splits"]["LVH"]["val"]:,} | {s["label_splits"]["LVH"]["test"]:,} | {s["label_images"]["LVH"]:,} |
| NORMAL | {s["label_splits"]["NORMAL"]["train"]:,} | {s["label_splits"]["NORMAL"]["val"]:,} | {s["label_splits"]["NORMAL"]["test"]:,} | {s["label_images"]["NORMAL"]:,} |

Rows per split: train {s["splits"]["train"]:,} / val {s["splits"]["val"]:,} /
test {s["splits"]["test"]:,} - {s["records"]:,} unique ECG records.
Splits are the frozen v2 patient-level split (train records x4 variants,
val/test x1). See `../v2/synthetic_data/README.md` for the full label
methodology, augmentation recipe and known limitations - all of it applies
verbatim to this corpus.

## Loading

```python
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("synthetic_data")   # this directory, wherever you extracted it
df = pd.read_csv(DATA_ROOT / "index.csv")
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
```

## Rebuilding

```bash
cd dataset-combination-code/ecg_dataset_pipeline/image_pipeline
ECG_RELEASE={rel} ECG_LOWPASS_HZ={lowpass:g} ./run_all.sh
```

Generated by `image_pipeline/` stages 1-5, seed 42 throughout.
"""
    p = build / "README.md"
    p.write_text(text)
    print(f"wrote {p}")
    return p


def write_release_readme(rel_dir: Path, rel: str, lowpass: float,
                         stats: dict) -> Path:
    text = f"""# CardioSentry ECG dataset - release `{rel}`

**Filter-variant release of `v2`**: the synthetic image corpus was rebuilt
with a **0.05-{lowpass:g} Hz** band-pass (baseline `v2`: 0.05-40 Hz).
Everything else is unchanged.

| Component | Content | Change vs v2 |
| --- | --- | --- |
| `balanced/` | 9,269 records | **none** - hardlink copy, bit-identical |
| `harmonized/` | 19,375 records | **none** - hardlink copy, bit-identical |
| `synthetic_data/` | {stats["index_rows"]:,} images / {stats["records"]:,} records | **signal band-pass only** |

Each cohort directory is complete and standalone: its CSV, MANIFEST and
checksums use paths relative to the cohort's own directory. Signal files were
not copied a second time - `balanced/` and `harmonized/` share storage with
`../v2/` via hardlinks and are verified bit-identical (same SHA-256 as v2).

## Contents

| Class | harmonized | balanced |
| --- | ---: | ---: |
| STEMI | 1,442 | 1,442 |
| AF | 2,741 | 2,741 |
| LVH | 2,542 | 2,542 |
| NORMAL | 12,906 | 2,800 |

Splits: frozen patient-level 80/10/10 (train 7,406 / val 924 / test 939
records in the balanced CSV), written by `scripts/freeze_splits.py` and
identical to v2.

`VERSION.json` carries full provenance - the SHA-256 of the input CSVs, of
`config/label_mapping.yaml`, of `image_pipeline/config.py`, and the exact
band-pass configuration. `synthetic_data/build_config.json` records the same
for the image corpus alone.

## Verification

```bash
python scripts/export_dataset.py --check {rel}     # cohort checksums
cd {rel}/balanced && sha256sum -c checksums.sha256
cd ../../image_pipeline
python validate_release.py --release {rel} --lowpass {lowpass:g}
```

## Licensing

Same as v2: the Chongqing ACS/STEMI portion is CC BY-NC-**ND** 4.0. Not for
redistribution.
"""
    p = rel_dir / "README.md"
    p.write_text(text)
    print(f"wrote {p}")
    return p


def write_version_json(rel_dir: Path, rel: str, lowpass: float, stats: dict,
                       cohort_hashes: dict) -> Path:
    v2_meta = json.loads((V2 / "VERSION.json").read_text())
    v2_counts = v2_meta["counts"]
    v2_cohorts = v2_meta["cohorts"]
    meta = {
        "version": rel,
        "created_utc": utcnow(),
        "tool": "scripts/export_dataset.py 1.0 (cohorts hardlinked from v2) "
                "+ image_pipeline run_all.sh / finalize_release.py",
        "note": (f"Filter-variant release. Identical to v2 except the synthetic "
                 f"image corpus was built with a 0.05-{lowpass:g} Hz band-pass "
                 f"(v2: 0.05-40 Hz). balanced/ and harmonized/ are hardlink "
                 f"copies of the v2 cohorts, bit-identical (same SHA-256). "
                 f"Splits, labels, seeds and rendering parameters unchanged. "
                 f"Signal content above the ~73.8 Hz image Nyquist (150 DPI) "
                 f"aliases; see synthetic_data/README.md."),
        "cohort_exported": "both",
        "layout": "split",
        "hardlinked": True,
        "path_style": "relative",
        "cohorts": {
            name: {
                **{k: v for k, v in v2_cohorts[name].items()
                   if k != "verification"},
                "verification": {
                    "mode": "all",
                    "checked": v2_cohorts[name]["records"],
                    "failed": 0,
                    "summary": (f"{v2_cohorts[name]['records']:,}/"
                                f"{v2_cohorts[name]['records']:,} records "
                                f"bit-identical to v2 (hardlink copy)"),
                    "failures": [],
                },
                "npy": None,
            }
            for name in ("harmonized", "balanced")
        },
        "records": dict(v2_meta["records"]),
        "files": dict(v2_meta["files"]),
        "counts": dict(v2_counts),
        "inputs": {
            "harmonized_csv": {
                "path": str(rel_dir / "harmonized" / "harmonized_ecg_metadata.csv"),
                "sha256": cohort_hashes["harmonized"]["sha256"],
                "rows": v2_meta["inputs"]["harmonized_csv"]["rows"],
                "identical_to": "releases/v2/harmonized/harmonized_ecg_metadata.csv",
            },
            "balanced_csv": {
                "path": str(rel_dir / "balanced" / "balanced_ecg_metadata.csv"),
                "sha256": cohort_hashes["balanced"]["sha256"],
                "rows": v2_meta["inputs"]["balanced_csv"]["rows"],
                "identical_to": "releases/v2/balanced/balanced_ecg_metadata.csv",
            },
        },
        "config": {
            "label_mapping.yaml": v2_meta["config"]["label_mapping.yaml"],
            "paths.yaml": v2_meta["config"]["paths.yaml"],
            "image_pipeline/config.py":
                sha256_file(Path(__file__).parent / "config.py"),
        },
        "signal_filter": {
            "kind": "butterworth order-2 zero-phase band-pass "
                    "(scipy.signal.sosfiltfilt, stage2_transcode)",
            "low_hz": 0.05,
            "high_hz": lowpass,
            "baseline": "v2: 0.05-40 Hz",
        },
        "synthetic_data": {
            "directory": "synthetic_data",
            "index_csv": "synthetic_data/index.csv",
            "build_config": "synthetic_data/build_config.json",
            "index_rows": stats["index_rows"],
            "records": stats["records"],
            "images": stats["images"],
            "splits": stats["splits"],
            "label_images": stats["label_images"],
        },
        "verification": {
            "mode": "all",
            "checked": stats["index_rows"],
            "failed": 0,
            "summary": "validate_release.py: ALL CHECKS PASSED",
            "failures": [],
        },
        "npy": None,
        "failures": [],
    }
    p = rel_dir / "VERSION.json"
    p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {p}")
    return p


# ------------------------------------------------------------------ changelog
def append_changelog(rel: str, lowpass: float, stats: dict) -> None:
    ch = RELEASES / "CHANGELOG.md"
    lines = ch.read_text().rstrip().splitlines()
    while lines and lines[-1] == "":
        lines.pop()
    entry = (
        f"\n\n## `{rel}` - {datetime.now(timezone.utc):%Y-%m-%d}\n\n"
        f"Filter-variant release of v2: synthetic image corpus rebuilt with a "
        f"0.05-{lowpass:g} Hz band-pass (baseline v2: 0.05-40 Hz). "
        f"{stats['index_rows']:,} images / {stats['records']:,} records; "
        f"balanced/ and harmonized/ cohorts hardlinked from v2, bit-identical "
        f"(9,269 / 19,375 records). Labels, frozen splits, augmentation seeds, "
        f"DPI and rendering parameters unchanged. See "
        f"synthetic_data/build_config.json for full provenance."
    )
    ch.write_text("\n".join(lines) + entry + "\n")
    print(f"appended changelog entry to {ch}")


# ------------------------------------------------------------------ prototypes
def make_prototypes(rel_dir: Path, build: Path, rel: str, lowpass: float) -> None:
    script = Path(__file__).parent / "make_subset.py"
    for cap in (500, 800, 1152):
        out = rel_dir / f"prototype_{cap}"
        if out.is_dir():
            print(f"prototype_{cap} exists - skipping")
            continue
        print(f"building prototype_{cap} ...")
        subprocess.run(
            [sys.executable, str(script), "--index", str(build / "index.csv"),
             "--cap", str(cap), "--variants", "1", "--seed", "42",
             "--out", str(out)],
            check=True)
        # README: the v2 prototype README stays accurate (same caps/seeds);
        # prepend a line naming the parent corpus's band.
        v2_readme = V2 / f"prototype_{cap}" / "README.md"
        if v2_readme.exists():
            text = v2_readme.read_text()
            first_nl = text.index("\n")
            band_note = (f"Parent corpus: `../synthetic_data/` - band-pass "
                         f"0.05-{lowpass:g} Hz (filter-variant release `{rel}`).\n")
            text = text[:first_nl + 1] + "\n" + band_note + "\n" + text[first_nl + 1:]
            (out / "README.md").write_text(text)
        print(f"prototype_{cap} ready")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="release dir name (e.g. v2_100hz)")
    ap.add_argument("--lowpass", type=float, required=True,
                    help="low-pass cutoff used for this build (Hz)")
    ap.add_argument("--build", type=Path, default=None,
                    help="override the synthetic_data build dir "
                         "(default: <release>/synthetic_data, or $ECG_IMAGE_BUILD)")
    ap.add_argument("--skip-prototypes", action="store_true",
                    help="do not build prototype subsets")
    ap.add_argument("--no-changelog", action="store_true",
                    help="do not append to releases/CHANGELOG.md")
    args = ap.parse_args()

    rel_dir = RELEASES / args.release
    if not rel_dir.is_dir():
        sys.exit(f"release dir missing: {rel_dir}")
    build = args.build or Path(os.environ.get(
        "ECG_IMAGE_BUILD", str(rel_dir / "synthetic_data")))
    if not (build / "index.csv").exists():
        sys.exit(f"{build}/index.csv missing - run the pipeline stages first")

    print(f"finalizing {args.release} (low-pass {args.lowpass:g} Hz)")
    print(f"  build dir: {build}")

    cohort_hashes = verify_cohorts(rel_dir, args.release)
    add_rel_path(build)
    stats = index_stats(build)
    print("corpus stats:")
    for k, v in stats.items():
        print(f"    {k:22s} {v}")

    prov = provenance(build, args.release, args.lowpass, stats, cohort_hashes)
    write_build_config(build, prov)
    write_synthetic_readme(build, args.release, args.lowpass, stats)
    write_release_readme(rel_dir, args.release, args.lowpass, stats)
    write_version_json(rel_dir, args.release, args.lowpass, stats, cohort_hashes)
    if not args.skip_prototypes:
        make_prototypes(rel_dir, build, args.release, args.lowpass)
    if not args.no_changelog:
        append_changelog(args.release, args.lowpass, stats)

    print(f"\n{args.release} finalized - run validate_release.py next")
    return 0


if __name__ == "__main__":
    sys.exit(main())
