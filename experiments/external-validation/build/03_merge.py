#!/usr/bin/env python3
"""
Stitch the two render passes into ONE evaluation corpus.

Produces `data/external_testset/` holding a flat image directory and an
index.csv in exactly the schema `prototype-code/.../manifest.py` requires:
rel_path, record, split, source, patient_id, STEMI, AF, LVH, NORMAL. That
matters - the evaluator reuses the training code's own loader rather than
reimplementing it, so anything the loader rejects fails here rather than
silently scoring something different.

THREE THINGS THIS DOES THAT ARE NOT BOOKKEEPING
-----------------------------------------------

1. LABELS ARE RE-DERIVED, AND THE SENTINEL IS EXPLICITLY UNDONE.
   Every label is read back from external_test_metadata*.csv - the file where
   the clinical decisions were made - and matched to the rendered sheet by
   (source, original_id). The label columns inside each pass's index.csv are
   ignored outright.

   That is not sufficient on its own, and the first version of this script got
   it wrong. The unlabelled pass writes its sentinel NORMAL=1 INTO its own
   metadata CSV - that is how stage 1 sees it - so re-deriving from that CSV
   reads the sentinel straight back, and 59 hard negatives silently became
   NORMAL positives. The `ambiguity_flag == "SENTINEL_LABEL"` column written
   beside it is the authority: where it is set, all four labels are forced to
   zero. The reconciliation at the end asserts the hard-negative count is
   non-zero, so this failure cannot recur silently.

2. RECORDS ARE RENUMBERED ACROSS PASSES.
   Both passes number from ECG000001, so their `record` values collide. Output
   names are EXT%06d assigned over the merged, already-shuffled ordering. The
   ID printed on the sheet itself keeps the per-pass name; that is recorded in
   `printed_id` so a sheet can still be traced back.

3. THE MERGE IS RECONCILED, NOT ASSUMED.
   Every metadata row must land exactly one image and every image exactly one
   metadata row. Records that stage 1 or 2 dropped (unreadable signal, missing
   age/sex) are reported by cohort and label, because a silent loss of STEMI
   positives would quietly shrink the only class this whole exercise is about.

    python 03_merge.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]

PASSES = [
    ("labelled", DATA / "external_test_metadata.csv", DATA / "image_build" / "labelled"),
    ("unlabelled", DATA / "external_test_metadata_unlabelled.csv",
     DATA / "image_build" / "unlabelled"),
]

# manifest.REQUIRED_COLUMNS plus the provenance the report needs. `source` holds
# the cohort (LUDB / PTBXL_MI / PTBXL_NORM) because that is the column the
# training code already groups per-source metrics on.
OUT_FIELDS = [
    "rel_path", "record", "split", "source", "patient_id", *LABELS,
    "cohort", "label_confidence", "seen_in_training", "printed_id",
    "original_id", "grid_palette", "rotation_deg", "age", "sex",
    "n_positive_labels", "original_labels", "accepted_label_mapping",
]


def _confidence(sentinel: bool, extra: dict, labels: dict[str, int]) -> str:
    """How much weight the STEMI label on this sheet can carry.

    Four values, and the distinction between the first two is the one that
    matters most when reading any result:

      gold_asserted_stemi   a cardiologist wrote "STEMI: <wall>" (LUDB)
      proxy_acute_mi        manual 'mi' + PTB-XL Stadium I. Acute, but acute MI
                            still admits NSTEMI - this is an inference
      hard_negative         abnormal sheet carrying none of the four classes
      gold_*                a negative on the STEMI head with a solid label
    """
    if sentinel:
        return "hard_negative"
    cohort = extra["cohort"]
    if cohort == "LUDB":
        return "gold_asserted_stemi" if labels["STEMI"] else "gold_ludb_nonstemi"
    if cohort == "PTBXL_MI":
        return "proxy_acute_mi"
    return "gold_norm"


def read_metadata(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        sys.exit(f"missing {path} - run 01_build_metadata.py")
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["source_dataset"], row["original_record_id"])] = row
    return out


def read_index(build: Path) -> list[dict]:
    idx = build / "index.csv"
    if not idx.exists():
        sys.exit(f"missing {idx} - run 02_render_images.sh")
    with open(idx, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(DATA / "external_testset"))
    ap.add_argument("--symlink", action="store_true",
                    help="link images instead of copying (saves ~300 MB, but the "
                         "folder is then not portable - do not use before tarring)")
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict] = []
    matched: set[tuple[str, str]] = set()
    all_meta: dict[tuple[str, str], dict] = {}

    for pass_name, meta_csv, build in PASSES:
        meta = read_metadata(meta_csv)
        all_meta.update(meta)
        for row in read_index(build):
            key = (row["source"], row["original_id"])
            src_meta = meta.get(key)
            if src_meta is None:
                sys.exit(f"[{pass_name}] rendered sheet {row['record']} has no "
                         f"metadata row for {key} - the two are out of sync; "
                         f"rerun 01 and 02 together")
            matched.add(key)
            extra = json.loads(src_meta["source_extra_json"])

            # The unlabelled pass wrote NORMAL=1 into its OWN metadata CSV so
            # stage 1 would carry the record, which means re-reading that CSV
            # reads the sentinel straight back. The flag written beside it is
            # the authority: when it is set, every label is false and the true
            # labels are all-zero.
            sentinel = src_meta.get("ambiguity_flag") == "SENTINEL_LABEL"
            true_labels = ({lab: 0 for lab in LABELS} if sentinel
                           else {lab: int(src_meta[lab]) for lab in LABELS})
            merged.append({
                "_src_image": build / "images" / row["split"] / f"{row['record']}_a{row['render_k']}.jpg",
                "_fallback_image": Path(row["image_path"]),
                "record": None,                       # assigned below
                "split": "test",
                "source": src_meta["source_dataset"],
                "patient_id": src_meta["patient_id"],
                # Re-derived from the metadata CSV. The pass's own index.csv
                # label columns are deliberately not consulted; see docstring.
                **true_labels,
                "cohort": extra["cohort"],
                # Re-derived rather than trusted, for the same reason the labels
                # are: an older metadata CSV tagged every LUDB row
                # "gold_asserted_stemi" regardless of whether it carried a STEMI
                # label, which makes the column useless for filtering. Deriving
                # it here keeps a stale CSV from reintroducing that.
                "label_confidence": _confidence(sentinel, extra, true_labels),
                "seen_in_training": extra["seen_in_training"],
                "printed_id": row["record"],
                "original_id": row["original_id"],
                "grid_palette": row.get("grid_palette", ""),
                "rotation_deg": row.get("rotation_deg", ""),
                "age": row.get("age", ""),
                "sex": row.get("sex", ""),
                "original_labels": src_meta["original_labels"],
                "accepted_label_mapping": src_meta["accepted_label_mapping"],
            })

    # Stable order: the shuffle that matters already happened in 01, before the
    # sheets were numbered. Sorting here only makes the build deterministic.
    merged.sort(key=lambda r: (r["source"], int(r["original_id"])
                               if r["original_id"].isdigit() else 0,
                               r["original_id"]))

    missing_files = []
    for i, row in enumerate(merged, start=1):
        row["record"] = f"EXT{i:06d}"
        src = row.pop("_src_image")
        fallback = row.pop("_fallback_image")
        if not src.exists():
            src = fallback                     # absolute path from the build host
        if not src.exists():
            missing_files.append(row["record"])
            continue
        dst = img_dir / f"{row['record']}.jpg"
        if args.symlink:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        row["rel_path"] = f"images/{row['record']}.jpg"
        row["n_positive_labels"] = sum(row[lab] for lab in LABELS)

    if missing_files:
        sys.exit(f"{len(missing_files)} rendered images not found on disk, "
                 f"first few: {missing_files[:5]}")

    merged = [r for r in merged if r.get("rel_path")]
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)

    # ---------------------------------------------------------------- reconcile
    lost = set(all_meta) - matched
    print(f"merged {len(merged)} sheets -> {out}")
    print()
    hdr = f"{'cohort':<12} {'n':>5}  " + "  ".join(f"{l:>6}" for l in LABELS)
    print(hdr)
    print("-" * len(hdr))
    for src in sorted({r["source"] for r in merged}):
        sub = [r for r in merged if r["source"] == src]
        counts = "  ".join(f"{sum(r[l] for r in sub):>6}" for l in LABELS)
        print(f"{src:<12} {len(sub):>5}  {counts}")
    print("-" * len(hdr))
    tot = "  ".join(f"{sum(r[l] for r in merged):>6}" for l in LABELS)
    print(f"{'TOTAL':<12} {len(merged):>5}  {tot}")

    neg = sum(1 for r in merged if r["n_positive_labels"] == 0)
    print(f"\nall-negative sheets (hard negatives, no target class): {neg}")
    # The unlabelled pass exists solely to supply these. Zero means the sentinel
    # survived and 59 of the sharpest negatives in the corpus are being scored
    # as NORMAL positives - which is how this script failed the first time.
    if neg == 0 and (DATA / "external_test_metadata_unlabelled.csv").exists():
        sys.exit("FATAL: zero hard negatives after merge, but an unlabelled pass "
                 "exists. The NORMAL=1 render sentinel was not undone.")
    seen = sum(1 for r in merged if str(r["seen_in_training"]) == "1")
    print(f"sheets whose patient appears in the training corpus:    {seen}")

    if lost:
        print(f"\n{len(lost)} metadata rows produced no sheet "
              f"(dropped by stage 1 or 2):")
        by = Counter()
        for key in lost:
            m = all_meta[key]
            pos = "+".join(l for l in LABELS if m[l] == "1") or "(none)"
            by[(m["source_dataset"], pos)] += 1
        for (src, pos), n in sorted(by.items()):
            print(f"    {n:>4}  {src:<12} {pos}")
        # Losing STEMI positives is the one loss that damages the experiment
        # rather than merely shrinking it - eight gold positives is already
        # the floor of what is usable.
        gold_lost = sum(n for (src, pos), n in by.items()
                        if src == "LUDB" and "STEMI" in pos)
        if gold_lost:
            print(f"\n  WARNING: {gold_lost} LUDB STEMI positive(s) lost. "
                  f"Investigate before trusting any STEMI number.")

    gold = sum(r["STEMI"] for r in merged if r["source"] == "LUDB")
    if gold == 0:
        print("\nFATAL: no gold STEMI positives survived.", file=sys.stderr)
        return 1
    print(f"\ngold (LUDB, asserted) STEMI positives: {gold}")
    print(f"proxy (PTB-XL acute MI) STEMI positives: "
          f"{sum(r['STEMI'] for r in merged if r['source'] == 'PTBXL_MI')}")
    print(f"\nindex: {out / 'index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
