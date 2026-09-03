#!/usr/bin/env python3
"""Validate a completed filter-variant release of the synthetic ECG corpus.

Runs after finalize_release.py and gates the start of the next release in
generate_both.sh. Checks, cheapest first:

  1. directory structure of the release (v2 conventions)
  2. balanced/ and harmonized/ cohorts bit-identical to v2
  3. the recorded band-pass configuration matches the requested one
  4. index.csv counts: rows, splits, records, labels, cls - identical to v2
  5. every rel_path (and its annotation sidecar) exists; JPEGs structurally
     valid; image size correct
  6. per-row agreement with the v2 corpus on every label/split/provenance
     field (the "only the filter changed" guarantee), with grid_palette /
     rotation_deg compared as a determinism check (WARN only)
  7. staged/rendered intermediate counts
  8. split integrity: no patient or record spans splits; eval splits have
     exactly one variant per record
  9. prototype subsets exist with the expected sizes

Exit 0 = ALL CHECKS PASSED, 1 = failure.

Usage:
    ECG_RELEASE=v2_100hz ~/.cache/ecgkit-venv/bin/python validate_release.py \
        --release v2_100hz --lowpass 100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import config as C

ROOT = Path(__file__).resolve().parent.parent          # ecg_dataset_pipeline/
RELEASES = ROOT / "data" / "releases"
V2 = RELEASES / "v2"
LABELS = list(C.LABEL_COLUMNS)

# Measured on the released v2 corpus - the new releases must reproduce these.
EXPECTED = {
    "index_rows": 31372,
    "split_rows": {"train": 29512, "val": 922, "test": 938},
    "records": 9238,
    "record_splits": {"train": 7378, "val": 922, "test": 938},
    "label_images": {"STEMI": 4898, "AF": 9251, "LVH": 8599, "NORMAL": 9486},
    "cls_records": {"AF": 2485, "AF+LVH": 256, "LVH": 2277,
                    "NORMAL": 2778, "STEMI": 1442},
    "staged": 9238,
    "rendered": 9238,
    "manifest_rows": 9238,
    "staged_map_rows": 9238,
    "prototype_rows": {500: 4062, 800: 5262, 1152: 6670},
    "image_size": (1686, 1311),
}

# Fields that must agree 100% with the v2 corpus row for row.
AGREE_FIELDS = ["record_id", "cls", "split", "source", "patient_id",
                "age", "sex", "note"] + LABELS
# Determinism checks - should agree too, but only WARN on mismatch.
DET_FIELDS = ["grid_palette", "rotation_deg"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jpeg_ok(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            head = fh.read(2)
            fh.seek(-2, os.SEEK_END)
            tail = fh.read(2)
        return head == b"\xff\xd8" and tail == b"\xff\xd9"
    except OSError:
        return False


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}"
          + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="release dir name (e.g. v2_100hz)")
    ap.add_argument("--lowpass", type=float, required=True,
                    help="expected low-pass cutoff in Hz (e.g. 100)")
    ap.add_argument("--build", type=Path, default=None,
                    help="override synthetic_data dir (default: <release>/synthetic_data)")
    ap.add_argument("--smoke", action="store_true",
                    help="relax the fixed corpus-size assertions (subset builds)")
    args = ap.parse_args()

    rel_dir = RELEASES / args.release
    build = args.build or rel_dir / "synthetic_data"
    v2_build = V2 / "synthetic_data"

    ok = True
    print(f"\n=== validate {args.release} "
          f"(expect low-pass {args.lowpass:g} Hz) ===")
    print(f"release dir: {rel_dir}\nbuild dir:   {build}")

    # ---- 1. structure ------------------------------------------------------
    print("[1] structure")
    ok &= check("release dir", rel_dir.is_dir())
    ok &= check("balanced/", (rel_dir / "balanced").is_dir())
    ok &= check("harmonized/", (rel_dir / "harmonized").is_dir())
    ok &= check("synthetic_data/", build.is_dir())
    ok &= check("VERSION.json", (rel_dir / "VERSION.json").exists())
    ok &= check("README.md", (rel_dir / "README.md").exists())
    ok &= check("index.csv", (build / "index.csv").exists())
    ok &= check("build_config.json", (build / "build_config.json").exists())
    ok &= check("synthetic README", (build / "README.md").exists())
    ok &= check("staged/", (build / "staged").is_dir())
    ok &= check("rendered/", (build / "rendered").is_dir())
    for sp in ("train", "val", "test"):
        ok &= check(f"images/{sp}/", (build / "images" / sp).is_dir())
    if not ok:
        print("\nSTRUCTURE INCOMPLETE")
        return 1

    # ---- 2. cohorts bit-identical to v2 -------------------------------------
    print("\n[2] cohort identity vs v2")
    for cohort in ("balanced", "harmonized"):
        mine = rel_dir / cohort / f"{cohort}_ecg_metadata.csv"
        ref = V2 / cohort / f"{cohort}_ecg_metadata.csv"
        same = mine.exists() and ref.exists() and \
            sha256_file(mine) == sha256_file(ref)
        ok &= check(f"{cohort} CSV bit-identical to v2", same)

    # ---- 3. recorded filter configuration -----------------------------------
    print("\n[3] filter configuration")
    bc = json.loads((build / "build_config.json").read_text())
    vj = json.loads((rel_dir / "VERSION.json").read_text())
    for name, blob in (("build_config.json", bc), ("VERSION.json", vj)):
        f = blob.get("signal_filter", {})
        ok &= check(f"{name}: high_hz == {args.lowpass:g}",
                    float(f.get("high_hz", -1)) == args.lowpass)
        ok &= check(f"{name}: low_hz == 0.05",
                    float(f.get("low_hz", -1)) == 0.05)
    ok &= check("completion marker absent", 
                not (rel_dir / ".release_complete.json").exists(),
                "(written by generate_both.sh after validation)")

    # ---- 4. index counts -----------------------------------------------------
    print("\n[4] index.csv counts")
    rows = list(csv.DictReader((build / "index.csv").open()))
    splits = Counter(r["split"] for r in rows)
    records = {r["record"] for r in rows}
    rec_split = Counter(r["split"] for r in {r["record"]: r for r in rows}.values())
    label_images = {L: sum(1 for r in rows if r.get(L) == "1") for L in LABELS}
    cls_records = Counter({r["cls"]: 0 for r in rows})
    for r in {r["record"]: r for r in rows}.values():
        cls_records[r["cls"]] += 1

    exp = dict(EXPECTED)
    if not args.smoke:
        ok &= check(f"rows == {exp['index_rows']:,}", len(rows) == exp["index_rows"])
        for sp, n in exp["split_rows"].items():
            ok &= check(f"{sp} rows == {n:,}", splits.get(sp, 0) == n)
        ok &= check(f"records == {exp['records']:,}", len(records) == exp["records"])
        for sp, n in exp["record_splits"].items():
            ok &= check(f"{sp} records == {n:,}", rec_split.get(sp, 0) == n)
        for L, n in exp["label_images"].items():
            ok &= check(f"{L} images == {n:,}", label_images.get(L, 0) == n)
        for cls, n in exp["cls_records"].items():
            ok &= check(f"cls {cls} records == {n:,}", cls_records.get(cls, 0) == n)
    else:
        ok &= check("rows > 0", len(rows) > 0, f"got {len(rows)}")
        ok &= check("splits valid", set(splits) <= {"train", "val", "test"},
                    f"got {dict(splits)}")
        print("  (smoke mode: fixed corpus-size assertions skipped)")

    # ---- 5. files resolve and are intact --------------------------------------
    print("\n[5] files on disk")
    rel_paths = [r["rel_path"] for r in rows]
    ok &= check("rel_path column present", all(r.get("rel_path") for r in rows))
    missing = [p for p in rel_paths if not (build / p).exists()]
    ok &= check(f"all {len(rel_paths):,} rel_path exist", not missing,
                f"{len(missing)} missing" if missing else "")
    side_missing = [r["rel_path"] for r in rows
                    if r.get("has_annotation") == "1"
                    and not (build / r["rel_path"]).with_suffix(".json").exists()]
    ok &= check("annotation sidecars present", not side_missing,
                f"{len(side_missing)} missing" if side_missing else "")
    n_jpg = len(list((build / "images").rglob("*.jpg")))
    ok &= check(f"jpg files == rows ({n_jpg:,} vs {len(rows):,})", n_jpg == len(rows))
    sample = rel_paths[:: max(1, len(rel_paths) // 1000)]
    bad = [p for p in sample if not jpeg_ok(build / p)]
    ok &= check(f"JPEG SOI/EOI on sample of {len(sample):,}", not bad,
                f"{len(bad)} corrupt" if bad else "")
    try:
        from PIL import Image
        sizes = {Image.open(build / p).size for p in sample[:50]}
        ok &= check(f"image size == {exp['image_size']}", sizes == {exp["image_size"]},
                    f"got {sizes}")
    except Exception as exc:                       # noqa: BLE001
        ok &= check("image size check", False, f"PIL failed: {exc}")

    # ---- 6. agreement with v2 (the controlled-variant guarantee) ---------------
    print("\n[6] per-row agreement with v2 corpus")
    if v2_build.is_dir() and (v2_build / "index.csv").exists():
        v2rows = list(csv.DictReader((v2_build / "index.csv").open()))
        v2key = {(r["record"], r["render_k"]): r for r in v2rows}
        key_mismatch = len(v2key) != len(rows) or \
            any((r["record"], r["render_k"]) not in v2key for r in rows)
        if not args.smoke:
            ok &= check(f"same (record, render_k) keys as v2 "
                        f"({len(v2key):,} vs {len(rows):,})", not key_mismatch)
        agree_bad = []
        det_bad = []
        for r in rows:
            v = v2key.get((r["record"], r["render_k"]))
            if v is None:
                agree_bad.append(f"{r['record']}_a{r['render_k']}: not in v2")
                continue
            for f in AGREE_FIELDS:
                if r.get(f) != v.get(f):
                    agree_bad.append(f"{r['record']}: {f} "
                                     f"{r.get(f)!r} != {v.get(f)!r}")
                    break
            for f in DET_FIELDS:
                if r.get(f) != v.get(f):
                    det_bad.append(f"{r['record']}: {f}")
        ok &= check(f"labels/split/provenance agree 100% ({len(rows):,} rows)",
                    not agree_bad, f"{len(agree_bad)} rows differ")
        if det_bad:
            print(f"  [WARN] grid_palette/rotation_deg differ on "
                  f"{len(det_bad)} rows - pixels may still match; "
                  f"check rendering determinism")
        else:
            print("  [OK] grid_palette/rotation_deg identical to v2 "
                  "(determinism held)")
    else:
        print("  skipped: v2 corpus index not found")

    # ---- 7. intermediates ------------------------------------------------------
    print("\n[7] staged / rendered intermediates")
    n_staged = len(list((build / "staged").rglob("*.dat")))
    n_rendered = len(list((build / "rendered").rglob("*.png")))
    n_rjson = len(list((build / "rendered").rglob("*.json")))
    if not args.smoke:
        ok &= check(f"staged .dat == {exp['staged']:,}", n_staged == exp["staged"],
                    f"got {n_staged:,}")
        ok &= check(f"rendered .png == {exp['rendered']:,}",
                    n_rendered == exp["rendered"], f"got {n_rendered:,}")
        ok &= check(f"rendered annotations == {exp['rendered']:,}",
                    n_rjson == exp["rendered"], f"got {n_rjson:,}")
    else:
        ok &= check(f"staged .dat == rendered .png ({n_staged} vs {n_rendered})",
                    n_staged == n_rendered)
    man = build / "image_manifest.csv"
    smap = build / "staged_map.csv"
    if man.exists() and not args.smoke:
        n = sum(1 for _ in man.open()) - 1
        ok &= check(f"image_manifest rows == {exp['manifest_rows']:,}",
                    n == exp["manifest_rows"], f"got {n:,}")
    if smap.exists() and not args.smoke:
        n = sum(1 for _ in smap.open()) - 1
        ok &= check(f"staged_map rows == {exp['staged_map_rows']:,}",
                    n == exp["staged_map_rows"], f"got {n:,}")

    # ---- 8. split integrity -----------------------------------------------------
    print("\n[8] split integrity")
    patient_splits = defaultdict(set)
    record_splits = defaultdict(set)
    variants = Counter()
    for r in rows:
        patient_splits[r["patient_id"]].add(r["split"])
        record_splits[r["record"]].add(r["split"])
        variants[(r["record"], r["split"])] += 1
    bad_p = {p: s for p, s in patient_splits.items() if len(s) > 1}
    bad_r = {t: s for t, s in record_splits.items() if len(s) > 1}
    ok &= check("0 patients spanning splits", not bad_p,
                f"{len(bad_p)} patients" if bad_p else "")
    ok &= check("0 records in >1 split", not bad_r,
                f"{len(bad_r)} records" if bad_r else "")
    bad_v = {k: v for k, v in variants.items()
             if k[1] in ("val", "test") and v != 1}
    ok &= check("val/test exactly 1 variant per record", not bad_v,
                f"{len(bad_v)} violations" if bad_v else "")
    if not args.smoke:
        bad_t = {k: v for k, v in variants.items()
                 if k[1] == "train" and v != 4}
        ok &= check("train exactly 4 variants per record", not bad_t,
                    f"{len(bad_t)} violations" if bad_t else "")

    # ---- 9. prototypes -----------------------------------------------------------
    print("\n[9] prototypes")
    if args.smoke:
        print("  skipped in smoke mode")
    else:
        for cap, want in exp["prototype_rows"].items():
            pdir = rel_dir / f"prototype_{cap}"
            have = None
            if (pdir / "index.csv").exists():
                have = sum(1 for _ in (pdir / "index.csv").open()) - 1
            ok &= check(f"prototype_{cap} index rows == {want:,}",
                        have == want, f"got {have}")
            if have is not None:
                prows = list(csv.DictReader((pdir / "index.csv").open()))
                miss = [r["rel_path"] for r in prows
                        if not (pdir / r["rel_path"]).exists()]
                ok &= check(f"prototype_{cap} images resolve", not miss,
                            f"{len(miss)} missing" if miss else "")
            ok &= check(f"prototype_{cap} README", (pdir / "README.md").exists())
            ok &= check(f"prototype_{cap} provenance",
                        (pdir / "SUBSET_PROVENANCE.json").exists())

    print("\n" + ("=== ALL CHECKS PASSED" if ok else "=== SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
