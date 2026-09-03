#!/usr/bin/env python3
"""
Pre-resize the prototype corpus to ONE geometry, once.

    CS_DIM=crop608 python resize_dataset.py --src /content/prototype_1152

WHY THIS EXISTS

The corpus stores 1686x1311 sheets (2.2 Mpx). Training resizes every one to the
target geometry on the CPU, every epoch, for every model. At crop608 that is
2.2 Mpx decoded to produce 0.62 - and on Colab's 2 cores it starves the GPU. The
v1 logs show the effect plainly: 21 img/s where the GPU could do 70-100. Doing
the resize once removes ~3.5x of decode work per epoch and shrinks the folder.

THE CROP IS BAKED IN. For a crop* geometry this writes the CROPPED sheet, so the
output folder is already header-free. data.HeaderCrop then re-crops the same
FRACTION of an already-cropped image, which would cut a second time and amputate
waveform - so the output folder gets a `.cs_geometry.json` marker and
manifest.load() reads it, turning the runtime crop off. Never point a crop*
resized folder at a different CS_DIM: the marker check will refuse it.

IS THIS STILL A FAIR COMPARISON?

Yes, provided every model uses the same resized copy. The geometry, the
interpolation and the normalisation are unchanged; this is data preparation, not
an experimental variable. The one real cost is a second JPEG generation,
negligible at quality 95 and identical for all models.

DO NOT MIX a run on the original folder with a run on a resized folder.
compare.py catches it (different dataset root), but delete any existing runs/
before switching so nothing resumes across the change.
"""
from __future__ import annotations
import argparse, csv, json, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config as C
import dimensions as DIM

MARKER = ".cs_geometry.json"


def _one(job):
    src, dst, size, crop, redact, quality = job
    h, w = size
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if crop:
                im = im.crop(DIM.crop_box(im.width, im.height))
            im = im.resize((w, h), Image.BICUBIC)      # PIL takes (w, h)
            if redact:
                # After the resize: the box is a fraction so it lands on the same
                # region either way, but the smaller box is 2.7x cheaper. Doing it
                # HERE rather than per-epoch is the whole point - ~3 ms x 6,670
                # images once, instead of on every image of every epoch of every
                # one of nine models.
                box = DIM.redact_box(im.width, im.height)
                patch = np.asarray(im.crop(box), dtype=np.float32)[::4, ::4]
                paper = np.percentile(patch.reshape(-1, patch.shape[-1]), 75, axis=0)
                ImageDraw.Draw(im).rectangle(list(box),
                                             fill=tuple(int(v) for v in paper))
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
                    help="default: <src>_<CS_DIM>")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    ap.add_argument("--force", action="store_true", help="overwrite an existing dst")
    a = ap.parse_args()

    src = a.src
    dst = a.dst or src.parent / f"{src.name}_{C.DIM_NAME}"
    if dst.exists() and not a.force:
        sys.exit(f"refusing to overwrite {dst} - remove it, or pass --force")

    idx = src / "index.csv"
    if not idx.exists():
        sys.exit(f"no index.csv in {src}")

    # Refuse to crop an already-cropped folder. This is the one mistake in this
    # script that produces a corpus which trains fine and is quietly amputated.
    src_marker = src / MARKER
    if src_marker.exists():
        prev = json.loads(src_marker.read_text())
        sys.exit(f"{src} is ALREADY a resized copy at geometry '{prev['dim_name']}' "
                 f"({prev['H']}x{prev['W']}, crop={prev['crop']}).\n"
                 f"Resize from the ORIGINAL corpus, not from another resized copy - "
                 f"cropping twice would cut into the waveform.")

    rows = list(csv.DictReader(idx.open()))
    print(f"{len(rows):,} images   {src}  ->  {dst}")
    print(f"geometry: {C.geometry_str()}")
    if C.CROP_HEADER:
        print(f"  header crop {DIM.CONTENT_CROP[0]:.3f}..{DIM.CONTENT_CROP[1]:.3f} "
              f"is BAKED IN; the runtime crop is disabled for this folder")
    if C.REDACT_HEADER:
        print(f"  ID redaction {DIM.REDACT_BOX} is BAKED IN; the runtime "
              f"redaction is disabled for this folder")

    jobs = [(src / r["rel_path"], dst / r["rel_path"], C.IMG_SIZE, C.CROP_HEADER,
             C.REDACT_HEADER, a.quality) for r in rows]
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

    (dst / MARKER).write_text(json.dumps({
        "dim_name": C.DIM_NAME, "H": C.IMG_H, "W": C.IMG_W,
        "crop": C.CROP_HEADER, "crop_box_frac": DIM.CONTENT_CROP,
        "redact": C.REDACT_HEADER, "redact_box_frac": DIM.REDACT_BOX,
        "source": str(src.resolve()),
        "fingerprint": C.fingerprint(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

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
