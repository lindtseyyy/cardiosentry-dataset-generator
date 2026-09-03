#!/usr/bin/env python3
"""
Collect the annotation sidecars for an already-built prototype subset.

make_subset.py copies annotations only when it is given --with-annotations, and
prototype_1152 was built without it. Rebuilding the subset to get them would
draw a NEW sample and break comparability with every run that has already been
reported against it. This script instead reads the subset's frozen index.csv and
pulls the matching sidecar out of the full corpus, so the subset's own image
files and index are never touched.

Output goes to a SEPARATE directory that mirrors the subset's layout:

    prototype_1152_annotations/images/{train,val,test}/ECG######_a#.json

which can be shipped as its own tar, or merged in place later with

    cp -rn prototype_1152_annotations/images/. prototype_1152/images/

Two modes, and the difference is not cosmetic:

  --slim (default)  drops `plotted_pixels` and keeps everything else. That is
                    every lead's quad, its name, its sample range, the grid
                    calibration and the augmentation geometry - all an oracle
                    lead-CROP experiment can use. ~500x smaller.
  --full            keeps `plotted_pixels`, the 1250-point trace polyline per
                    lead. Only a digitisation model needs it, and it is 2.5 GB
                    for this subset.
"""
from __future__ import annotations
import argparse, csv, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data/releases/v2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=Path, default=HERE / "prototype_1152")
    ap.add_argument("--corpus", type=Path, default=HERE / "synthetic_data")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <subset>_annotations next to the subset")
    ap.add_argument("--full", action="store_true",
                    help="keep plotted_pixels (2.5 GB for prototype_1152)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = a.out or a.subset.parent / f"{a.subset.name}_annotations"
    index = a.subset / "index.csv"
    if not index.exists():
        sys.exit(f"no index.csv at {index}")

    rows = list(csv.DictReader(index.open()))
    print(f"{len(rows):,} rows from {index}")
    print(f"mode: {'full (with plotted_pixels)' if a.full else 'slim (boxes only)'}")
    print(f"out:  {out}")

    written = missing = 0
    total_in = total_out = 0
    per_split: dict[str, int] = {}
    for r in rows:
        rel = Path(r["rel_path"]).with_suffix(".json")
        src = a.corpus / rel
        if not src.exists():
            missing += 1
            continue
        total_in += src.stat().st_size
        if a.dry_run:
            written += 1
            per_split[rel.parts[1]] = per_split.get(rel.parts[1], 0) + 1
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if a.full:
            shutil.copy2(src, dst)
        else:
            d = json.load(src.open())
            d["leads"] = [{k: v for k, v in lead.items() if k != "plotted_pixels"}
                          for lead in d.get("leads", [])]
            d["_slim"] = "plotted_pixels dropped by collect_subset_annotations.py"
            json.dump(d, dst.open("w"), separators=(",", ":"))
        total_out += dst.stat().st_size
        written += 1
        per_split[rel.parts[1]] = per_split.get(rel.parts[1], 0) + 1

    print(f"\nwritten: {written:,}   missing upstream: {missing}")
    print("per split:", per_split)
    if not a.dry_run:
        print(f"source payload {total_in/1e6:,.1f} MB -> written {total_out/1e6:,.1f} MB")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
