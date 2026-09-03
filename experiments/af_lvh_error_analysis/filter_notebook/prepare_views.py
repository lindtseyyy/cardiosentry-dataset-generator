"""
Build the 768px stored views of the regenerated sheets - the EXACT transform
prepare_dataset.py used for the released 768px corpus.

The released prototype stores 1686x1311 sheets and the reference run read a
pre-resized 768x768 copy (RGB, BICUBIC to 768x768, JPEG quality 95). The
regenerated experiment sheets are also 1686x1311, so the same transform must
be applied BEFORE inference - and applied once, stored to disk, then decoded
by the same path the corpus loader used. Resizing inside the dataloader would
add a second interpolation and break the 'same bytes, same math' comparison.

The source sheets are never modified; only new view files are written.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

import config as C


def one_view(src: Path, dst: Path, size: int = C.LOAD_SIZE,
             quality: int = C.VIEW_QUALITY) -> int:
    """Single sheet -> stored 768px view. Returns bytes written."""
    with Image.open(src) as im:
        w, h = im.size
        if min(w, h) < size:
            raise SystemExit(
                f"source {src} is {w}x{h}, smaller than the requested "
                f"{size}px - it has already been downscaled. Point at the "
                f"original 1686x1311 sheets.")
        im = im.convert("RGB").resize((size, size), Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=quality, optimize=True)
    return dst.stat().st_size


def build_all(sources: dict[tuple[str, str], Path], views_root: Path,
              size: int = C.LOAD_SIZE) -> dict[tuple[str, str], Path]:
    """sources: {(case, variant): source sheet path} -> {(case, variant):
    stored 768px view path}. Resumable: existing views are reused."""
    out = {}
    for key, src in sorted(sources.items()):
        case, variant = key
        dst = views_root / case / f"{case}__{variant}__view768.jpg"
        if not dst.exists():
            one_view(src, dst, size)
        out[key] = dst
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-root", required=True, type=Path,
                    help="the filter_experiment folder (by_band/, by_case/)")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="where the 768px views are written")
    args = ap.parse_args()

    by_band = args.src_root / "by_band"
    by_case = args.src_root / "by_case"
    if not by_band.is_dir() or not by_case.is_dir():
        raise SystemExit(f"{args.src_root} has no by_band/ or by_case/ - "
                         f"upload the whole filter_experiment folder.")

    sources: dict[tuple[str, str], Path] = {}
    for case in C.CASES:
        orig = by_case / case / "original_0.05-40Hz.jpg"
        if not orig.exists():
            raise SystemExit(f"missing {orig} - upload filter_experiment/ "
                             f"complete (by_case/...).")
        sources[(case, "orig_40")] = orig
        for variant, band in (("regen_40", "0.05-40Hz"),
                              ("regen_100", "0.05-100Hz"),
                              ("regen_150", "0.05-150Hz")):
            src = by_band / band / f"{case}_a0.jpg"
            if not src.exists():
                raise SystemExit(f"missing {src} - upload "
                                 f"filter_experiment/ complete (by_band/).")
            sources[(case, variant)] = src

    views = build_all(sources, args.out_root)
    print(f"built/reused {len(views)} stored 768px views -> {args.out_root}")
    for (case, variant), p in sorted(views.items()):
        with Image.open(p) as im:
            w, h = im.size
        print(f"  {case:10s} {variant:10s} {w}x{h} {p.stat().st_size//1024} KB")
    return 0


if __name__ == "__main__":
    main()
