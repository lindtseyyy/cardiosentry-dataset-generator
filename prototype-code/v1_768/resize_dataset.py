#!/usr/bin/env python3
"""
Pre-resize the prototype corpus to the training resolution, once.

    python resize_dataset.py --src /content/prototype_1152 --size 768

WHY THIS EXISTS

The corpus stores 1686x1311 sheets. Training resizes every one to 768x768 on the
CPU, every epoch, for every model - decoding 2.2 megapixels to produce 0.59.
On Colab's 2 cores that starves the GPU: ~21 img/s where the GPU could do more.
Doing the resize once removes ~4x of decode work and shrinks the folder ~4x.

IS THIS STILL A FAIR COMPARISON?

Yes, provided every model uses the same resized copy. IMG_SIZE, the interpolation
and the normalisation are unchanged; this is data preparation, not an
experimental variable. The one real cost is a second JPEG generation, negligible
at quality 95 and identical for all models.

DO NOT mix: a run on the original folder is not comparable with a run on the
resized folder. compare.py will catch it (different dataset root), but delete any
existing runs/ before switching so nothing resumes across the change.
"""
from __future__ import annotations
import argparse, csv, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def _one(job):
    src, dst, size, quality = job
    try:
        with Image.open(src) as im:
            im = im.convert("RGB").resize((size, size), Image.BICUBIC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return dst.stat().st_size
    except Exception as exc:
        return f"FAIL {src}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", type=Path, default=None,
                    help="default: <src>_<size>")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    a = ap.parse_args()

    src = a.src
    dst = a.dst or src.parent / f"{src.name}_{a.size}"
    if dst.exists():
        sys.exit(f"refusing to overwrite {dst} - remove it first")

    idx = src / "index.csv"
    if not idx.exists():
        sys.exit(f"no index.csv in {src}")
    rows = list(csv.DictReader(idx.open()))
    print(f"{len(rows):,} images   {src}  ->  {dst}   at {a.size}x{a.size}")

    jobs = [(src / r["rel_path"], dst / r["rel_path"], a.size, a.quality)
            for r in rows]
    missing = [j[0] for j in jobs if not j[0].exists()]
    if missing:
        sys.exit(f"{len(missing)} source image(s) missing, e.g. {missing[:3]}")

    workers = a.workers or None
    t0 = time.time()
    total_bytes, failures = 0, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_one, jobs, chunksize=32), 1):
            if isinstance(res, str):
                failures.append(res)
            else:
                total_bytes += res
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  {i:,}/{len(jobs):,}   {i/el:.0f} img/s   "
                      f"eta {(len(jobs)-i)/(i/el)/60:.1f} min", flush=True)

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:10]:
            print("   ", f)
        return 1

    # the index and sidecars travel with it; rel_path is unchanged by design
    for extra in ("index.csv", "README.md", "SUBSET_PROVENANCE.json"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)

    src_gb = sum(f.stat().st_size for f in src.rglob("*.jpg")) / 1024**3
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"  images   {src_gb:.2f} GB  ->  {total_bytes/1024**3:.2f} GB  "
          f"({total_bytes/1024**3/src_gb*100:.0f}%)")
    print(f"  output   {dst}")
    print(f"\nPoint DATA at it, and DELETE any existing runs/ first so nothing")
    print(f"resumes across the change:")
    print(f'    DATA = "{dst}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
