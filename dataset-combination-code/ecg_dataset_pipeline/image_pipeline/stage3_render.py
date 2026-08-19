"""Stage 3 - render clean ECG sheets with ecg-image-kit, in parallel.

The kit emits exactly one image per record, so N variants of a record means N
invocations. Work is therefore a grid of (staged chunk) x (render index), which
also gives enough independent jobs to keep every core busy.

Be aware of what a second variant actually buys you at this layer: almost
nothing. `-se/--seed` is dead in batch mode -- gen_ecg_image_from_data.py only
calls random.seed() under `if hasattr(args, "st")`, an attribute the batch
driver never sets, and get_paper_ecg() accepts a `seed` parameter it never
reads. With the flags in KIT_RENDER_FLAGS every other choice is forced
(resolution, padding, calibration pulse, grid presence, the hardcoded "bw" grid
palette, header printing), so the only thing that differs between two renders
of one record is the typeface, drawn by an unseeded `random.choice` over the
kit's 10 fonts. The seed is still passed for the day upstream fixes this.

Real per-variant diversity comes from stage 4, which seeds its own RNG on
(record, render_k) and is genuinely reproducible.

Only the layout and calibration flags are passed. The kit's --augment and
--wrinkles are deliberately unused; stage 4 does that work. See stage4 for the
list of defects that decision rests on.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import config as C
import patch_kit


def _renders_per_chunk() -> dict[tuple[str, str], int]:
    """(class-slug, chunk) -> how many variants that chunk needs.

    Read from staged_map.csv rather than assumed from the class, so a per-record
    n_renders override in stage 1 is honoured here without touching this file.
    """
    # A PLAIN dict, deliberately. It used to be a defaultdict returning 1,
    # which silently made stage 3 render any chunk sitting in staged/ -
    # including leftovers from an earlier run with a different manifest. That
    # is how a one-record test turned into a multi-thousand-record render.
    out: dict[tuple[str, str], int] = {}
    if not C.STAGED_MAP.exists():
        return out
    for row in csv.DictReader(C.STAGED_MAP.open()):
        key = (C.slug(row["cls"]), row["chunk"])
        try:
            n = int(row.get("n_renders") or 1)
        except ValueError:
            n = 1
        out[key] = max(out.get(key, 1), n)
    return out


def _jobs() -> list[tuple[str, Path, int]]:
    """Chunks to render, taken from staged_map.csv - never from the directory.

    staged/ is not cleared between runs, so it can hold chunks that stage 2 did
    not just write. Rendering those wastes hours and pollutes the corpus with
    records the current manifest does not contain, so anything absent from
    staged_map.csv is skipped and reported.
    """
    per_chunk = _renders_per_chunk()
    out, orphans = [], []
    if not C.STAGED.is_dir():
        return out
    for cls_dir in sorted(p for p in C.STAGED.iterdir() if p.is_dir()):
        for chunk in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            key = (cls_dir.name, chunk.name)
            if key not in per_chunk:
                orphans.append(f"{cls_dir.name}/{chunk.name}")
                continue
            for k in range(per_chunk[key]):
                out.append((cls_dir.name, chunk, k))
    if orphans:
        print(f"skipping {len(orphans)} stale chunk(s) not in "
              f"{C.STAGED_MAP.name}: {', '.join(orphans[:6])}"
              + (" ..." if len(orphans) > 6 else ""))
        print("  (left on disk; delete staged/ to reclaim the space)")
    return out


def _run(job: tuple[str, Path, int]) -> tuple[str, int, str]:
    cls, chunk, k = job
    out_dir = C.RENDERED / cls / f"r{k}" / chunk.name
    tag = f"{cls}/{chunk.name}/r{k}"

    # Idempotent: a chunk already rendered is skipped, so an interrupted build
    # can simply be re-run.
    if out_dir.is_dir() and any(out_dir.glob("*.png")):
        n_in = len(list(chunk.glob("*.dat")))
        if len(list(out_dir.glob("*.png"))) >= n_in:
            return tag, 0, "skipped"

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(C.VENV_PY), "gen_ecg_images_from_data_batch.py",
        "-i", str(chunk), "-o", str(out_dir),
        "-se", str(C.SEED + k),
        *C.KIT_RENDER_FLAGS,
    ]
    proc = subprocess.run(cmd, cwd=str(C.KIT), capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        return tag, proc.returncode, tail[0]
    return tag, 0, f"{len(list(out_dir.glob('*.png')))} images"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--workers", type=int, default=C.WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="run only N jobs (smoke test)")
    args = ap.parse_args()

    if not C.KIT.is_dir():
        print(f"ecg-image-kit not found at {C.KIT}", file=sys.stderr)
        print("Clone it there, or set ECGKIT_DIR - see README.md", file=sys.stderr)
        return 1
    # The kit is a separate checkout, so a fresh clone is stock upstream and
    # would render the header flush against the paper corner.
    if not patch_kit.is_patched():
        print("ecg-image-kit is not patched - run: python patch_kit.py",
              file=sys.stderr)
        return 1
    if not C.VENV_PY.exists():
        print(f"interpreter not found: {C.VENV_PY}", file=sys.stderr)
        print("Run install.sh, or set ECGKIT_PY", file=sys.stderr)
        return 1

    jobs = _jobs()
    if not jobs:
        print("no staged chunks found - run stage2 first", file=sys.stderr)
        return 1
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(jobs)} render jobs on {args.workers} workers")
    started = time.time()
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            tag, code, msg = fut.result()
            if code:
                failures.append((tag, msg))
                print(f"  [{i}/{len(jobs)}] FAIL {tag}: {msg}")
            else:
                elapsed = time.time() - started
                rate = i / max(elapsed, 1e-6)
                eta = (len(jobs) - i) / max(rate, 1e-6)
                print(f"  [{i}/{len(jobs)}] {tag}: {msg}  (eta {eta/60:.0f} min)")

    total = sum(1 for _ in C.RENDERED.rglob("*.png"))
    print(f"\nrendered {total} images in {(time.time()-started)/60:.1f} min")
    if failures:
        print(f"{len(failures)} failed jobs:")
        for tag, msg in failures[:20]:
            print(f"    {tag}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
