#!/usr/bin/env python3
"""
Pre-resize the prototype corpus to config.LOAD_SIZE, once, and verify it.

    python prepare_dataset.py --src /content/prototype_1152

WHY THIS EXISTS, AND WHY IT MUST BE 768 AND NOT 384

The corpus stores 1686x1311 sheets. Round 1 pre-resized to 384 and trained off
that copy — correct then, because 384 was the only view any model wanted.

This round has models that want 768. Resizing 1686x1311 -> 384 -> 768 would
UPSAMPLE a 384px image and invent detail that is not in it, which would make the
resolution comparison measure nothing at all. So the round-2 copy is built from
the ORIGINAL prototype_1152 tar at 768, and the 384 view is derived from it at
load time by a single bicubic step down.

The consequence to be aware of: you cannot reuse prototype_1152_384 for this
round, and data.py will refuse if you try (it checks LOAD_SIZE against the views
a model asks for).

IS THIS STILL A FAIR COMPARISON?

Yes, provided every model reads the same copy — which run_hybrids.sh enforces by
taking one --data path for the whole batch. Interpolation, normalisation and
augmentation are unchanged. The one real cost is a second JPEG generation at
quality 95, identical for every model and every view.
"""
from __future__ import annotations
import argparse, csv, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from PIL import Image

import config as C


def _one(job):
    src, dst, size, quality = job
    try:
        with Image.open(src) as im:
            w, h = im.size
            if min(w, h) < size:
                return (f"FAIL {src}: source is {w}x{h}, smaller than the requested "
                        f"{size}px. This folder has already been downscaled; "
                        f"start from the original prototype_1152.")
            im = im.convert("RGB").resize((size, size), Image.BICUBIC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return dst.stat().st_size
    except Exception as exc:
        return f"FAIL {src}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="the ORIGINAL prototype_1152 (1686x1311 sheets)")
    ap.add_argument("--dst", type=Path, default=None, help="default: <src>_<size>")
    ap.add_argument("--size", type=int, default=C.LOAD_SIZE)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = min(cores, 4). Capped because each worker holds a "
                         "decoded 1686x1311 sheet, and an over-subscribed pool "
                         "is what gets killed on a 12 GB Colab VM")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip images already written (default on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--force", action="store_true", help="overwrite an existing dst")
    a = ap.parse_args()

    src = a.src
    dst = a.dst or src.parent / f"{src.name}_{a.size}"
    if dst.exists():
        if a.force:
            print(f"--force: removing {dst}")
            shutil.rmtree(dst)
        elif (dst / "index.csv").exists():
            sys.exit(f"{dst} already exists and is complete - nothing to do.\n"
                     f"pass --force to rebuild it, or just point --data at it.")
        else:
            # index.csv is written LAST, so its absence means a previous attempt
            # died partway. Continue that attempt instead of refusing.
            n = sum(1 for _ in dst.rglob("*.jpg"))
            print(f"{dst} exists but has no index.csv - a previous run died "
                  f"partway ({n:,} images present). Resuming.")

    idx = src / "index.csv"
    if not idx.exists():
        sys.exit(f"no index.csv in {src}")
    rows = list(csv.DictReader(idx.open()))
    print(f"{len(rows):,} images   {src}  ->  {dst}   at {a.size}x{a.size}")

    # Fail before spending ten minutes, not after.
    probe = src / rows[0]["rel_path"]
    if probe.exists():
        with Image.open(probe) as im:
            w, h = im.size
        print(f"source sheets are {w}x{h}")
        if min(w, h) < a.size:
            sys.exit(
                f"\nSTOP: the source is {w}x{h}, smaller than the {a.size}px you asked\n"
                f"for. Upsampling would invent detail and silently invalidate the\n"
                f"whole resolution experiment. Point --src at the ORIGINAL\n"
                f"prototype_1152 tar contents, not at a resized copy.")

    jobs = [(src / r["rel_path"], dst / r["rel_path"], a.size, a.quality) for r in rows]
    missing = [j[0] for j in jobs if not j[0].exists()]
    if missing:
        sys.exit(f"{len(missing)} source image(s) missing, e.g. {missing[:3]}")

    # RESUME. A previous attempt that died partway leaves valid files behind;
    # redoing them is pure waste, and on Colab the second attempt is often
    # running against the clock of a session that already burned fifteen minutes.
    if a.resume:
        done = [j for j in jobs if j[1].exists()]
        if done:
            jobs = [j for j in jobs if not j[1].exists()]
            print(f"resuming: {len(done):,} already written, {len(jobs):,} to go")
    if not jobs:
        print("all images already present")

    # Free space, checked before rather than discovered at image 694. The output
    # runs about 250 KB per sheet at quality 95.
    est_gb = len(jobs) * 0.25 / 1024
    try:
        free_gb = shutil.disk_usage(dst.parent if dst.exists() else src.parent).free / 1024**3
        print(f"free space {free_gb:.1f} GB, need roughly {est_gb:.1f} GB")
        if free_gb < est_gb * 1.3:
            sys.exit(f"\nSTOP: not enough disk. {free_gb:.1f} GB free, need about "
                     f"{est_gb * 1.3:.1f} GB.\n"
                     f"  On Colab, delete the extracted source once this finishes,\n"
                     f"  or free space with:  !rm -rf ~/.cache /content/sample_data")
    except Exception:
        pass

    t0 = time.time()
    total_bytes, failures = 0, []
    workers = a.workers or min(os.cpu_count() or 2, 4)
    print(f"using {workers} worker process(es)")

    def _run(job_list, n_workers):
        """Returns (bytes, failures). Raises if the pool dies."""
        nonlocal total_bytes
        out_failures = []
        if n_workers <= 1:
            it = map(_one, job_list)
        else:
            ex = ProcessPoolExecutor(max_workers=n_workers)
            it = ex.map(_one, job_list, chunksize=32)
        try:
            for i, res in enumerate(it, 1):
                if isinstance(res, str):
                    out_failures.append(res)
                else:
                    total_bytes += res
                if i % 500 == 0:
                    el = time.time() - t0
                    print(f"  {i:,}/{len(job_list):,}   {i/el:.0f} img/s   "
                          f"eta {(len(job_list)-i)/(i/el)/60:.1f} min", flush=True)
        finally:
            if n_workers > 1:
                ex.shutdown(wait=False, cancel_futures=True)
        return out_failures

    try:
        failures = _run(jobs, workers)
    except BrokenProcessPool:
        # A worker was killed - almost always the OOM killer or a full disk.
        # This is the failure that silently produced a 10%-complete corpus,
        # because ex.map raises HERE, mid-iteration, and a `!python` that dies
        # does not stop a Colab cell. Retry serially rather than give up: it is
        # slower but it cannot be killed for using too many processes, and
        # --resume means the work already done is kept.
        done_now = sum(1 for j in jobs if j[1].exists())
        print(f"\n  WORKER POOL DIED after ~{done_now:,} of {len(jobs):,} images.")
        print(f"  That is the OOM killer or a full disk, not a bad image.")
        print(f"  Retrying the remainder with a single process (slower, safer) ...")
        remaining = [j for j in jobs if not j[1].exists()]
        try:
            failures = _run(remaining, 1)
        except Exception as exc:
            print(f"\n  serial retry also failed: {type(exc).__name__}: {exc}")
            print(f"  {sum(1 for j in jobs if j[1].exists()):,} images were written.")
            print(f"  Re-run with --resume to continue from here.")
            return 1

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:10]:
            print("   ", f)
        return 1

    # VERIFY BEFORE DECLARING VICTORY.
    #
    # index.csv is the completion sentinel: it is copied last, and every
    # downstream check keys off it. Writing it over an incomplete image set
    # turns a visible failure into an invisible one - a corpus that loads,
    # trains, and reports metrics computed on a tenth of the data.
    written = sum(1 for _ in dst.rglob("*.jpg"))
    if written != len(rows):
        missing = [r["rel_path"] for r in rows
                   if not (dst / r["rel_path"]).exists()][:5]
        print(f"\nINCOMPLETE: wrote {written:,} of {len(rows):,} images.")
        print(f"  first missing: {missing}")
        print(f"  index.csv has deliberately NOT been written, so nothing")
        print(f"  downstream will mistake this for a finished corpus.")
        print(f"  Re-run the same command to resume from here.")
        return 1
    print(f"verified {written:,} of {len(rows):,} images on disk")

    # rel_path is unchanged by design, so the index travels as-is. Written LAST,
    # and only now that the image count has been checked.
    for extra in ("index.csv", "README.md", "SUBSET_PROVENANCE.json"):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"  {total_bytes/1024**3:.2f} GB written to {dst}")
    ppm_v = (150 / 25.4) * (a.size / 1311)
    ppm_h = (150 / 25.4) * (a.size / 1686)
    print(f"  measurement precision at {a.size}px: {ppm_v:.2f} px/mm vertical "
          f"({ppm_v*10:.1f} px per mV), {ppm_h:.2f} px/mm horizontal")
    print(f'\n    DATA = "{dst}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
