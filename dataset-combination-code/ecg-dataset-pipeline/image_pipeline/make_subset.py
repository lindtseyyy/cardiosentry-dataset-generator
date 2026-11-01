#!/usr/bin/env python3
"""
Build a small, self-contained prototype subset of the synthetic ECG corpus.

READ-ONLY with respect to the full corpus: it only ever copies files out of it.
Nothing in synthetic_data/ is moved, renamed, or deleted.

Design decisions, and why:

  * Records are sampled, never rows. All variants of one ECG belong to the
    same record; sampling rows could split them across a shuffle boundary.
  * Sampling is per `cls` group so co-occurrence structure (AF+LVH) survives.
  * VAL and TEST are copied WHOLE. They are ~1,860 images, there is nothing to
    gain by shrinking them, and shrinking them would make prototype numbers
    non-comparable with the full run.
  * Splits are inherited from the frozen `split` column. This script never
    reassigns anything, so patient-level integrity is preserved by construction.
  * Seeded and order-independent: records are sorted before shuffling, so the
    same seed gives the same subset regardless of CSV row order.
"""
from __future__ import annotations
import argparse, csv, collections, hashlib, json, random, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

LABELS = ["STEMI", "AF", "LVH", "NORMAL"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "data/releases/v2/synthetic_data/index.csv",
                    help="index.csv of the full corpus")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory (must not already exist)")
    ap.add_argument("--cap", type=int, default=500,
                    help="max TRAIN records per cls group (default 500)")
    ap.add_argument("--variants", type=int, default=1,
                    help="augmentation variants per train record (default 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-annotations", action="store_true",
                    help="also copy the .json sidecars (roughly +60%% size)")
    ap.add_argument("--symlink", action="store_true",
                    help="symlink instead of copy (local use only - symlinks "
                         "do NOT survive upload to cloud storage)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src_root = a.index.parent
    if a.out.exists() and not a.dry_run:
        sys.exit(f"refusing to write into existing directory: {a.out}\n"
                 f"remove it first, or pick another --out")

    rows = list(csv.DictReader(a.index.open()))
    if "rel_path" not in rows[0]:
        sys.exit("index.csv has no rel_path column")
    print(f"read {len(rows):,} rows from {a.index}")

    # ---- group rows by record, keep split/cls with it -----------------------
    by_record: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_record[r["record"]].append(r)
    for v in by_record.values():
        v.sort(key=lambda r: int(r["render_k"]))

    meta = {rec: v[0] for rec, v in by_record.items()}

    # ---- choose TRAIN records ----------------------------------------------
    rng = random.Random(a.seed)
    train_by_cls: dict[str, list[str]] = collections.defaultdict(list)
    for rec, m in meta.items():
        if m["split"] == "train":
            train_by_cls[m["cls"]].append(rec)

    chosen: list[str] = []
    sampling_report = []
    for cls in sorted(train_by_cls):
        pool = sorted(train_by_cls[cls])          # deterministic pre-shuffle order
        rng.shuffle(pool)
        take = pool[: a.cap]
        chosen += take
        sampling_report.append({
            "cls": cls, "available": len(pool), "cap": a.cap,
            "selected": len(take), "capped": len(take) < len(pool),
        })

    # ---- VAL / TEST are taken whole ----------------------------------------
    eval_recs = sorted(rec for rec, m in meta.items() if m["split"] in ("val", "test"))

    # ---- assemble the row list ---------------------------------------------
    out_rows: list[dict] = []
    for rec in sorted(chosen):
        out_rows += by_record[rec][: a.variants]
    for rec in eval_recs:
        out_rows += by_record[rec]                # already exactly 1 each

    print(f"\nselected {len(chosen):,} train records "
          f"x {a.variants} variant(s) = {sum(1 for r in out_rows if r['split']=='train'):,} images")
    for s in sampling_report:
        flag = "  <- capped" if s["capped"] else ""
        print(f"    {s['cls']:8} {s['selected']:5,} of {s['available']:5,}{flag}")
    print(f"val + test copied whole: "
          f"{sum(1 for r in out_rows if r['split']=='val'):,} + "
          f"{sum(1 for r in out_rows if r['split']=='test'):,}")
    print(f"TOTAL images: {len(out_rows):,}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # ---- copy ---------------------------------------------------------------
    for split in ("train", "val", "test"):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)

    n_img = n_json = 0
    missing: list[str] = []
    for i, r in enumerate(out_rows, 1):
        src = src_root / r["rel_path"]
        dst = a.out / r["rel_path"]
        if not src.exists():
            missing.append(r["rel_path"])
            continue
        if a.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        n_img += 1
        if a.with_annotations:
            sj, dj = src.with_suffix(".json"), dst.with_suffix(".json")
            if sj.exists():
                dj.symlink_to(sj.resolve()) if a.symlink else shutil.copy2(sj, dj)
                n_json += 1
        if i % 500 == 0:
            print(f"  {i:,}/{len(out_rows):,}")

    if missing:
        print(f"\nWARNING: {len(missing)} source images missing, e.g. {missing[:3]}")

    # ---- write the subset index --------------------------------------------
    fields = [f for f in out_rows[0] if f != "image_path"]   # drop the machine-specific path
    idx = a.out / "index.csv"
    with idx.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: v for k, v in r.items() if k != "image_path"})

    # ---- provenance ---------------------------------------------------------
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "image_pipeline/make_subset.py 1.0",
        "purpose": "prototype subset - NOT for final results",
        "source_index": str(a.index),
        "source_index_sha256": hashlib.sha256(a.index.read_bytes()).hexdigest(),
        "parameters": {"cap": a.cap, "variants": a.variants, "seed": a.seed,
                       "with_annotations": a.with_annotations, "symlink": a.symlink},
        "train_sampling": sampling_report,
        "counts": {
            "images": {s: sum(1 for r in out_rows if r["split"] == s)
                       for s in ("train", "val", "test")},
            "records": {s: len({r["record"] for r in out_rows if r["split"] == s})
                        for s in ("train", "val", "test")},
        },
        "files_copied": {"images": n_img, "annotations": n_json},
        "note": ("val/test are complete copies of the full corpus splits. "
                 "train is a seeded per-cls subsample. Splits were inherited "
                 "from the frozen split column; no reassignment occurred."),
    }
    (a.out / "SUBSET_PROVENANCE.json").write_text(json.dumps(prov, indent=2))

    print(f"\nwrote {n_img:,} images"
          + (f" + {n_json:,} annotations" if a.with_annotations else "")
          + f" -> {a.out}")
    print(f"index -> {idx}")
    print(f"provenance -> {a.out / 'SUBSET_PROVENANCE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
