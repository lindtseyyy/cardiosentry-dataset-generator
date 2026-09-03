#!/usr/bin/env python3
"""
Render the candidate input geometries so you can LOOK at them before committing
GPU hours to one.

    python aspect_study.py --data <prototype_1152> --out dim_samples

Produces, under --out:

    geometry.txt              the arithmetic table (dimensions.py)
    profile.txt               where the ink actually is, measured on this corpus
    <record>/<dim>.png        one resized sample per candidate, per record
    <record>/_contact.png     all candidates side by side at a common display size
    <record>/_zoom.png        the SAME 2-second window of lead II, cropped from
                              each candidate at 1:1 - this is the one that decides it
    header_leak.png           the printed ID at full resolution - what the
                              redaction box covers

WHY THE ZOOM SHEET IS THE ONE THAT MATTERS
------------------------------------------
A contact sheet of whole ECGs at thumbnail size tells you nothing: every
candidate looks like a legible ECG, because your eye is reading the layout, not
the waveform. The diagnosis is not in the layout. STEMI is a 1 mm step in the ST
segment and 1 mm of voltage is 5.9 px on the source sheet, which is 2-4 px after
any of these resizes. `_zoom.png` crops the identical physical window - the same
millimetres of paper - out of every candidate and shows it at 1:1, so what you
are comparing is the thing the model has to measure.

Look for: is the ST segment still a distinguishable step, or has it become a
smear the width of the trace? Are the small deflections between QRS complexes
(the ones that separate AF from sinus rhythm) still individually visible?
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import dimensions as DIM

RESAMPLE = Image.BICUBIC          # same filter data.py uses, so this is honest


# --------------------------------------------------------------------- helpers
def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render(im: Image.Image, dim: str) -> Image.Image:
    """Exactly what data.build_transforms does, minus the tensor conversion.

    Including the redaction - a sample sheet that still shows the printed ID is
    not a picture of what the model receives.
    """
    import numpy as _np
    from PIL import ImageDraw as _ID
    d = DIM.resolve(dim)
    H, W = d["size"]
    out = im
    if d["crop"]:
        out = out.crop(DIM.crop_box(out.width, out.height))
    out = out.resize((W, H), RESAMPLE)           # PIL takes (w, h)
    if d.get("redact", True) and not d["crop"]:
        box = DIM.redact_box(out.width, out.height)
        patch = _np.asarray(out.crop(box), dtype=_np.float32)[::4, ::4]
        paper = _np.percentile(patch.reshape(-1, patch.shape[-1]), 75, axis=0)
        out = out.copy()
        _ID.Draw(out).rectangle(list(box), fill=tuple(int(v) for v in paper))
    return out


def pick_records(data: Path, per_class: int, seed: int) -> list[dict]:
    """One sheet per class, preferring the classes whose features are finest.

    STEMI first: it is the class defined by the smallest measurement on the page,
    so it is the class a resize destroys first. AF second: fibrillatory waves are
    low-amplitude and high-frequency, which is the other way to lose a diagnosis
    to interpolation.
    """
    import pandas as pd
    df = pd.read_csv(data / "index.csv")
    rng = np.random.default_rng(seed)
    out = []
    for cls in ("STEMI", "AF", "LVH", "NORMAL"):
        sub = df[df.cls == cls]
        if not len(sub):
            continue
        for i in rng.choice(len(sub), min(per_class, len(sub)), replace=False):
            r = sub.iloc[int(i)]
            out.append({"record": r["record"], "cls": cls,
                        "path": data / r["rel_path"], "source": r["source"]})
    return out


# ------------------------------------------------------------------- profiling
def ink_profile(data: Path, n: int, seed: int) -> str:
    """Measure where the waveform actually is, on THIS corpus, right now.

    dimensions.CONTENT_CROP is a number in a file. This is the measurement it
    came from, re-run against whatever corpus you point at, so a different
    release cannot silently invalidate the crop.
    """
    import pandas as pd
    df = pd.read_csv(data / "index.csv")
    rng = np.random.default_rng(seed)
    sel = df.iloc[rng.choice(len(df), min(n, len(df)), replace=False)]

    hdr_end, first, last, sizes = [], [], [], set()
    for _, r in sel.iterrows():
        with Image.open(data / r["rel_path"]) as im:
            sizes.add(im.size)
            a = np.asarray(im.convert("L"), dtype=np.float32)
        # Ink is the trace and the printed text. The grid sits at 102/191 grey
        # (image_pipeline/config.py), so a threshold of 90 catches ink only.
        ink = (a < 90).sum(1)
        H = len(ink)
        h = np.where(ink[:int(0.20 * H)] > 25)[0]
        hdr_end.append(h.max() / H if len(h) else 0.0)
        body = np.where(ink > 30)[0]
        body = body[body > int(0.22 * H)]
        if len(body):
            first.append(body.min() / H)
            b2 = body[body < int(0.95 * H)]
            if len(b2):
                last.append(b2.max() / H)

    L = [f"ink profile over {len(sel)} sheets from {data}",
         f"image sizes: {sorted(sizes)}", ""]
    for nm, v in (("header block ends", hdr_end),
                  ("first trace ink", first),
                  ("last trace ink", last)):
        v = np.asarray(v, dtype=float)
        L.append(f"  {nm:20} min={v.min():.4f}  p50={np.percentile(v,50):.4f}  "
                 f"max={v.max():.4f}   (as a fraction of image height)")
    L += ["",
          "  NOTE: these numbers are indicative only, and the min/max are bounded",
          "  by this function's own search windows (header searched below 0.20H,",
          "  traces above 0.22H). An earlier version of this file reported those",
          "  bounds as if they were data and made a correct crop look unsafe.",
          "",
          "  The AUTHORITATIVE bounds come from crop_audit.py, which measures the",
          "  clean stage-3 renders (where thresholding is exact) and maps them",
          "  through stage 4's full parameter range:",
          f"      lowest a header reaches   {DIM.MEASURED['header_bottom_max']} px "
          f"({DIM.MEASURED['header_bottom_max']/DIM.MEASURED['final_height']:.4f})",
          f"      highest a waveform reaches {DIM.MEASURED['trace_top_min']} px "
          f"({DIM.MEASURED['trace_top_min']/DIM.MEASURED['final_height']:.4f})",
          f"      lowest the rhythm strip reaches {DIM.MEASURED['rhythm_bottom_max']} px "
          f"of {DIM.MEASURED['final_height']}",
          f"      crop* geometries clip {DIM.MEASURED['clipped_by_crop']} extreme "
          f"LVH/STEMI sheets regardless of where the cut goes",
          "",
          f"  configured crop: y {DIM.CONTENT_CROP[0]:.4f} .. {DIM.CONTENT_CROP[1]:.4f}",
          f"  redaction box:   {DIM.REDACT_BOX}   <- the recommended fix; it removes",
          "                   the printed ID without touching geometry at all"]
    return "\n".join(L)


# ---------------------------------------------------------------- contact sheet
def contact_sheet(im: Image.Image, dims: list[str], cell_w: int = 560) -> Image.Image:
    """Each candidate scaled to a COMMON DISPLAY WIDTH.

    This is what the geometries look like as pictures - the aspect differences
    are visible, the resolution differences deliberately are NOT, because
    everything here has been rescaled to the same width. Resolution is what
    _zoom.png is for. Keeping the two questions in two files stops one from
    being mistaken for the other.
    """
    font, small = _font(20), _font(15)
    tiles = []
    for d in dims:
        r = render(im, d)
        h = max(1, round(cell_w * r.height / r.width))
        tiles.append((d, r, r.resize((cell_w, h), Image.LANCZOS)))

    pad, hdr = 16, 58
    cols = min(3, len(tiles))
    rows = -(-len(tiles) // cols)
    cell_h = max(t[2].height for t in tiles) + hdr
    W = cols * cell_w + (cols + 1) * pad
    H = rows * cell_h + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(sheet)

    for i, (name, real, disp) in enumerate(tiles):
        cx = pad + (i % cols) * (cell_w + pad)
        cy = pad + (i // cols) * (cell_h + pad)
        g = DIM.geometry(name)
        dr.text((cx, cy), f"{name}   {g['H']}x{g['W']}"
                          f"{'  cropped' if g['crop'] else ''}",
                fill=(15, 15, 15), font=font)
        dr.text((cx, cy + 24),
                f"aspect {g['aspect']:.2f}  distort {g['aspect_distortion']:.3f}  "
                f"{g['pixels']/1e6:.2f} Mpx ({g['px_rel_sq768']:.2f}x sq768)  "
                f"{g['px_per_mm_y']:.2f} px/mm V",
                fill=(90, 90, 90), font=small)
        sheet.paste(disp, (cx, cy + hdr))
        dr.rectangle([cx, cy + hdr, cx + disp.width - 1, cy + hdr + disp.height - 1],
                     outline=(200, 200, 200))
    return sheet


# -------------------------------------------------------------------- zoom sheet
# The window to magnify, in FRACTIONS OF THE SOURCE SHEET. Defaults to the start
# of the rhythm strip (lead II, bottom row), which is where rate and rhythm are
# read and where the trace is longest and least cluttered.
ZOOM_WINDOW = (0.06, 0.85, 0.30, 0.955)     # (x0, y0, x1, y1)


def zoom_sheet(im: Image.Image, dims: list[str], scale: int = 3) -> Image.Image:
    """The SAME millimetres of paper, cut out of every candidate, at 1:1 x scale.

    Every tile here shows an identical physical region of the ECG. They differ in
    pixel count, because that is the whole point: this is what the resolution
    difference between the candidates actually looks like on the waveform.
    """
    font, small = _font(20), _font(15)
    x0f, y0f, x1f, y1f = ZOOM_WINDOW
    tiles = []
    for d in dims:
        spec = DIM.resolve(d)
        r = render(im, d)
        # Map the source-sheet window into this candidate's own coordinates. On a
        # cropped geometry the vertical fractions are relative to the CROP, not
        # to the original sheet, so they have to be re-based or the window would
        # land somewhere else on the page.
        if spec["crop"]:
            c0, c1 = DIM.CONTENT_CROP
            yy0 = (y0f - c0) / (c1 - c0)
            yy1 = (y1f - c0) / (c1 - c0)
        else:
            yy0, yy1 = y0f, y1f
        box = (round(x0f * r.width), round(max(0.0, yy0) * r.height),
               round(x1f * r.width), round(min(1.0, yy1) * r.height))
        crop = r.crop(box)
        tiles.append((d, crop,
                      crop.resize((crop.width * scale, crop.height * scale),
                                  Image.NEAREST)))

    pad, hdr = 16, 56
    W = max(t[2].width for t in tiles) + 2 * pad
    H = sum(t[2].height + hdr + pad for t in tiles) + pad
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(sheet)
    y = pad
    for name, crop, disp in tiles:
        g = DIM.geometry(name)
        dr.text((pad, y), f"{name}   {crop.width}x{crop.height} px for this window"
                          f"   (shown at {scale}x, nearest-neighbour)",
                fill=(15, 15, 15), font=font)
        dr.text((pad, y + 24),
                f"1 mm ST step = {g['px_per_1mm_st']:.2f} px   "
                f"100 ms QRS = {g['px_per_qrs']:.1f} px   "
                f"time Nyquist {g['time_nyquist_hz']:.0f} Hz "
                f"(signal band goes to 40 Hz)",
                fill=(90, 90, 90), font=small)
        sheet.paste(disp, (pad, y + hdr))
        y += disp.height + hdr + pad
    return sheet


def header_leak(im: Image.Image, scale: int = 2) -> Image.Image:
    """The strip the crop removes, at source resolution.

    Printed there: `ID: ECG######`. In prototype_1152 that number is <= 007796
    for CHAPMAN and PTB-XL and >= 007797 for every STEMI record, without
    exception. It is a perfect STEMI label in ~25 px type.
    """
    y1 = int(DIM.CONTENT_CROP[0] * im.height)
    strip = im.crop((0, 0, im.width, y1))
    return strip.resize((strip.width * scale, strip.height * scale), Image.LANCZOS)


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="extracted prototype folder (needs index.csv)")
    ap.add_argument("--out", type=Path, default=Path("dim_samples"))
    ap.add_argument("--dims", nargs="*", default=None,
                    help=f"subset of {DIM.DIM_NAMES}")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--profile-n", type=int, default=150,
                    help="sheets to measure the ink profile over (0 to skip)")
    ap.add_argument("--zoom-scale", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dims = a.dims or DIM.DIM_NAMES
    bad = [d for d in dims if d not in DIM.DIMENSIONS]
    if bad:
        sys.exit(f"unknown dimension(s) {bad}; choose from {DIM.DIM_NAMES}")

    if not (a.data / "index.csv").exists():
        sys.exit(f"no index.csv in {a.data}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "geometry.txt").write_text(DIM.table(dims))
    print(DIM.table(dims))

    if a.profile_n:
        print(f"\nprofiling ink on {a.profile_n} sheets ...")
        prof = ink_profile(a.data, a.profile_n, a.seed)
        (a.out / "profile.txt").write_text(prof)
        print(prof)

    recs = pick_records(a.data, a.per_class, a.seed)
    print(f"\nrendering {len(dims)} geometries x {len(recs)} records ...")
    for r in recs:
        d = a.out / f"{r['cls']}_{r['record']}"
        d.mkdir(exist_ok=True)
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            for dim in dims:
                render(im, dim).save(d / f"{dim}.png")
            contact_sheet(im, dims).save(d / "_contact.png")
            zoom_sheet(im, dims, a.zoom_scale).save(d / "_zoom.png")
            if r is recs[0]:
                header_leak(im).save(a.out / "header_leak.png")
        print(f"  {d.name}  ({r['source']})")

    print(f"\nwrote {a.out}/")
    print("  read _zoom.png first - it is the only one that shows what a resize")
    print("  does to the ST segment. _contact.png shows aspect, not resolution.")
    print("  header_leak.png shows the printed ID that the redaction box covers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
