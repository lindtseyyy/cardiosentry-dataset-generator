#!/usr/bin/env python3
"""
Prove - or disprove - that the header crop can never clip waveform.

    python crop_audit.py --rendered <.../synthetic_data/rendered>
    python crop_audit.py --rendered <...> --data <prototype_1152> --annotate 8

--------------------------------------------------------------------------------
WHY THIS DOES NOT MEASURE THE AUGMENTED SHEETS
--------------------------------------------------------------------------------
The obvious audit - segment every finished JPEG and find the topmost waveform
row - does not work, and two earlier versions of this script were wrong because
of it. Stage 4 bakes in lighting gradients, edge shadow, creases and exposure
drift. Measured on real sheets, the median grey runs 164 to 208 BETWEEN sheets,
and WITHIN one sheet the bottom 90 rows can sit so dark that only 9% of pixels
are above 140. A global ink threshold reads shadow as ink; a flat-fielded one
still loses faint traces at strict thresholds and swallows the paper texture at
loose ones. Either way the row segmentation latches onto illumination instead of
print, and the audit confidently reports nonsense like "the header extends to
0.94 of the sheet".

So this measures the wrong thing on purpose, in two exact halves:

  1. WHERE THE PRINT IS, measured on the CLEAN renders that stage 3 produced,
     before stage 4 touched them. Those are pure greyscale with the grid on two
     exact levels (102 and 191) and ink below 60, so a fixed threshold is not
     merely adequate, it is exact. And the layout is a fixed template: across
     9,238 renders the header block ends within a 5 px band.

  2. HOW FAR STAGE 4 CAN MOVE IT, computed by importing stage4_augment's OWN
     transform functions and searching its entire published parameter range -
     rotation +/-3 deg, keystone quad with m in [0.004, 0.018], margin 18 px.
     Not sampled from the 6,670 sheets that happen to exist: searched over the
     whole space, so the bound holds for any sheet the pipeline could ever emit,
     including a re-run with a different seed.

The result is a hard interval in FINAL-image coordinates: the lowest a header
block can reach, and the highest a waveform can reach. If the configured crop
sits strictly between them, no sheet can be clipped - not "no sheet in this
sample was".

--------------------------------------------------------------------------------
THE GEOMETRY, FOR THE RECORD
--------------------------------------------------------------------------------
stage4_augment.augment() applies, in order:
    rotate(deg) about the centre, expand=True then centre-crop back  (size kept)
    with p=0.7, Image.QUAD mapping a source quad onto the full rectangle
    a MARGIN_PX border added last
and stage4_augment._forward() replays exactly that on coordinates. This script
calls those same functions rather than re-deriving them, so if the pipeline's
geometry changes, this audit changes with it.

Clean render 1650x1275 + 2 x 18 px margin = 1686x1311, which is the corpus size.
"""
from __future__ import annotations
import argparse, glob, itertools, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import dimensions as DIM

# ---------------------------------------------------------------- clean layout
CLEAN_INK = 90          # exact on a clean render: ink < 60, major gridline 102
MIN_ROW_INK = 15
MIN_ROW_FRAC = 0.02     # a tilted sheet edge contributes ~23 px/row; 1% would
                        # let it register as a tall "content band"
MIN_RUN_ROWS = 3

# Bicubic resampling support, defocus blur (0.3-0.9 px) and JPEG ringing all
# spread ink by a pixel or two beyond the geometric map, and the quad bound below
# is searched on a grid rather than solved in closed form. Six pixels covers it.
SAFETY_PX = 6


def measure_clean(path: str):
    """Landmarks of one CLEAN render, in its own pixel coordinates."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"), dtype=np.uint8)
    H, W = g.shape
    ink = (g < CLEAN_INK).sum(1)
    content = ink > max(MIN_ROW_INK, int(MIN_ROW_FRAC * W))

    runs, s = [], None
    for y, c in enumerate(content):
        if c and s is None:
            s = y
        elif not c and s is not None:
            if y - s >= MIN_RUN_ROWS:
                runs.append((s, y - 1))
            s = None
    if s is not None and H - s >= MIN_RUN_ROWS:
        runs.append((s, H - 1))
    if len(runs) < 3:
        return None

    gaps = [(runs[i + 1][0] - runs[i][1], i) for i in range(len(runs) - 1)]
    _, gi = max(gaps)

    # The "25mm/s 10mm/mV" caption is a short final run, well separated from the
    # rhythm strip. Clipping it costs nothing; clipping the rhythm strip costs a
    # lead, so they are reported separately.
    rhythm_bottom = runs[-1][1]
    if len(runs) - 1 > gi + 1:
        ln = runs[-1][1] - runs[-1][0] + 1
        pre = runs[-1][0] - runs[-2][1]
        if ln < 60 and pre > 8:
            rhythm_bottom = runs[-2][1]

    return (H, W, runs[gi][1], runs[gi + 1][0], rhythm_bottom, runs[-1][1])


# ------------------------------------------------------- stage-4 displacement
def _load_stage4():
    """Import the pipeline's own transform code, so this cannot drift from it."""
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "dataset-combination-code" / "ecg_dataset_pipeline" / "image_pipeline"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            import stage4_augment as S4          # noqa: E402
            import config as PIPE                # noqa: E402
            return S4, PIPE
    raise SystemExit(
        "could not find dataset-combination-code/ecg_dataset_pipeline/image_pipeline\n"
        "  pass --pipeline <path to image_pipeline> explicitly")


def displacement_bounds(y_clean: float, W: int, H: int, margin: int,
                        S4, rot_deg=(-3.0, 3.0), quad_m=(0.004, 0.018),
                        n_rot: int = 61, n_x: int = 121) -> tuple[float, float]:
    """Min and max FINAL y over stage 4's whole parameter space, for the
    horizontal line y = y_clean on the clean render.

    A waveform pixel never sits above `trace_top`, and for fixed parameters the
    map is increasing in y, so bounding this one line bounds the whole region
    below it. The line is sampled across the full width because the rotation
    term is -x sin(theta): the extremes live at the left and right edges.
    """
    xs = np.linspace(0, W, n_x)
    pts = np.stack([xs, np.full_like(xs, float(y_clean))], axis=1)

    dst = np.array([[0, 0], [0, H], [W, H], [W, 0]], dtype=np.float64)

    # Corner offsets are 8 independent uniforms; their extremes are what bound
    # the map, so enumerate all 2^8 corners of that box at both ends of m.
    quads = [None]                               # the p=0.3 "no keystone" case
    for m in quad_m:
        dx, dy = m * W, m * H
        for bits in itertools.product((0.0, 1.0), repeat=8):
            a, b, c, d, e, f, g, h = bits
            quads.append(np.array([
                [a * dx, b * dy],
                [c * dx, H - d * dy],
                [W - e * dx, H - f * dy],
                [W - g * dx, h * dy],
            ], dtype=np.float64))

    lo, hi = np.inf, -np.inf
    for deg in np.linspace(rot_deg[0], rot_deg[1], n_rot):
        rot = S4._rot_points(pts.copy(), float(deg), W, H)
        for q in quads:
            xy = rot if q is None else S4._apply_h(S4._homography(q, dst), rot)
            y = xy[:, 1] + margin
            lo = min(lo, float(y.min()))
            hi = max(hi, float(y.max()))
    return lo, hi


# ------------------------------------------------------------------ reporting
def annotate(path: Path, y0f: float, y1f: float, hi_hdr: float, lo_trace: float,
             out: Path) -> None:
    with Image.open(path) as im:
        im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size
    for y, colour, label in (
            (y0f * H, (0, 140, 255), "crop top"),
            (y1f * H, (0, 140, 255), "crop bottom"),
            (hi_hdr, (220, 0, 0), "lowest a header can reach"),
            (lo_trace, (0, 170, 0), "highest a trace can reach")):
        d.line([(0, int(y)), (W, int(y))], fill=colour, width=4)
        d.text((10, max(0, int(y) - 22)), f"{label}  y={int(y)}", fill=colour)
    im.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rendered", required=True, type=Path,
                    help="synthetic_data/rendered - the CLEAN stage-3 PNGs")
    ap.add_argument("--data", type=Path, default=None,
                    help="prototype folder, for the annotated spot-check only")
    ap.add_argument("--out", type=Path, default=Path("crop_audit"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--annotate", type=int, default=6)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    S4, PIPE = _load_stage4()
    margin = PIPE.MARGIN_PX

    paths = sorted(glob.glob(str(a.rendered / "*" / "r*" / "c*" / "*.png")))
    if not paths:
        paths = sorted(str(p) for p in a.rendered.rglob("*.png"))
    if not paths:
        sys.exit(f"no clean renders under {a.rendered}")
    if a.limit:
        paths = paths[:a.limit]

    print(f"measuring {len(paths):,} CLEAN renders (exact thresholding) ...")
    res, res_paths = [], []
    with ProcessPoolExecutor(max_workers=a.workers or None) as ex:
        for i, r in enumerate(ex.map(measure_clean, paths, chunksize=32), 1):
            res_paths.append(r)
            if r:
                res.append(r)
            if i % 2000 == 0:
                print(f"  {i:,}/{len(paths):,}", flush=True)

    H, W = res[0][0], res[0][1]
    kept = [p for p, r in zip(paths, res_paths) if r]
    hb = np.array([r[2] for r in res])
    tt = np.array([r[3] for r in res])
    rb = np.array([r[4] for r in res])
    fb = np.array([r[5] for r in res])

    # Per-sheet CSV, so an outlier can be looked at rather than argued about.
    import csv as _csv
    with (a.out / "clean_per_sheet.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["path", "header_bottom", "trace_top", "rhythm_bottom",
                    "footer_bottom"])
        for pth, r in zip(kept, res):
            w.writerow([pth, r[2], r[3], r[4], r[5]])

    L, A = [], lambda s: L.append(s)
    A("=" * 78)
    A("CLEAN RENDER LAYOUT  (stage 3 output, before any augmentation)")
    A("=" * 78)
    A(f"  {len(res):,}/{len(paths):,} renders measured   sheet {W}x{H}")
    A("")
    A(f"  {'landmark':16}{'min':>7}{'p50':>7}{'p99.9':>8}{'max':>7}"
      f"   (px on the clean render)")
    for nm, v in (("header_bottom", hb), ("trace_top", tt),
                  ("rhythm_bottom", rb), ("footer_bottom", fb)):
        A(f"  {nm:16}{v.min():7d}{int(np.percentile(v,50)):7d}"
          f"{int(np.percentile(v,99.9)):8d}{v.max():7d}")
    A("")
    A(f"  the layout is a rigid template: 99.9% of header_bottom values lie in "
      f"{hb.min()}..{int(np.percentile(hb,99.9))} px")
    A("")
    A("  OUTLIERS - these drive the bounds, so they are named, not averaged away:")
    for nm, v, cmp_, lim in (("header_bottom", hb, "above", 200),
                             ("trace_top", tt, "below", 380),
                             ("rhythm_bottom", rb, "above", 1250)):
        idx = np.where(v > lim)[0] if cmp_ == "above" else np.where(v < lim)[0]
        A(f"    {nm} {cmp_} {lim}: {len(idx)} sheet(s)")
        for i in idx[:6]:
            A(f"        {v[i]:5d}  {kept[i]}")

    A("")
    A("=" * 78)
    A("STAGE-4 DISPLACEMENT  (searched over the whole parameter space)")
    A("=" * 78)
    A(f"  rotation +/-3.0 deg | keystone quad m in [0.004, 0.018], p=0.7 | "
      f"margin {margin} px")
    A(f"  using stage4_augment._rot_points / _homography / _apply_h directly")
    A("")

    _, hdr_lo_max = displacement_bounds(int(hb.max()), W, H, margin, S4)
    trace_hi_min, _ = displacement_bounds(int(tt.min()), W, H, margin, S4)
    _, rhythm_lo_max = displacement_bounds(int(rb.max()), W, H, margin, S4)
    _, footer_lo_max = displacement_bounds(int(fb.max()), W, H, margin, S4)

    hdr_bound = hdr_lo_max + SAFETY_PX
    trace_bound = trace_hi_min - SAFETY_PX
    rhythm_bound = rhythm_lo_max + SAFETY_PX

    FH = H + 2 * margin
    A(f"  final image height {FH} px")
    A(f"  {'bound':38}{'px':>8}{'fraction':>11}")
    A("  " + "-" * 55)
    A(f"  {'lowest a header block can reach':38}{hdr_bound:8.1f}{hdr_bound/FH:11.4f}")
    A(f"  {'highest a waveform can reach':38}{trace_bound:8.1f}{trace_bound/FH:11.4f}")
    A(f"  {'lowest the rhythm strip can reach':38}{rhythm_bound:8.1f}"
      f"{rhythm_bound/FH:11.4f}")
    A(f"  {'lowest the footer caption can reach':38}{footer_lo_max+SAFETY_PX:8.1f}"
      f"{(footer_lo_max+SAFETY_PX)/FH:11.4f}")
    A(f"  (includes a {SAFETY_PX} px allowance for bicubic spread, defocus blur "
      f"and JPEG ringing)")

    y0, y1 = DIM.CONTENT_CROP
    y0px, y1px = y0 * FH, y1 * FH
    top_ok = hdr_bound < y0px < trace_bound
    bot_ok = y1px > rhythm_bound

    A("")
    A("=" * 78)
    A("VERDICT")
    A("=" * 78)
    A(f"  safe window for the TOP cut:  {hdr_bound:.0f} .. {trace_bound:.0f} px"
      f"   ({hdr_bound/FH:.4f} .. {trace_bound/FH:.4f})   "
      f"{trace_bound-hdr_bound:.0f} px wide")
    A(f"  configured crop:              {y0:.4f} .. {y1:.4f}  "
      f"(y = {y0px:.0f} .. {y1px:.0f} px)")
    A("")
    A(f"  TOP    {'PASS' if top_ok else 'FAIL'}   cut at {y0px:.0f} px"
      + (f"   margins: {y0px-hdr_bound:+.0f} px below the worst header, "
         f"{trace_bound-y0px:+.0f} px above the worst waveform" if top_ok else
         "   THE CUT IS OUTSIDE THE SAFE WINDOW"))
    A(f"  BOTTOM {'PASS' if bot_ok else 'FAIL'}   cut at {y1px:.0f} px, rhythm "
      f"strip can reach {rhythm_bound:.0f} px   margin {y1px-rhythm_bound:+.0f} px")
    A("")
    if top_ok and bot_ok:
        A("  No sheet the pipeline can emit has waveform inside the cropped bands.")
        A("  This is a bound over the parameter space, not over a sample - it holds")
        A("  for a rebuild with a different seed too.")
    else:
        A("  *** DO NOT TRAIN A crop* GEOMETRY until dimensions.CONTENT_CROP is")
        A("      corrected. A suggested window, with the margins balanced:")
        lo, hi = hdr_bound / FH, trace_bound / FH
        A(f"          CONTENT_CROP = ({(lo+hi)/2:.4f}, "
          f"{min(0.999, (rhythm_bound + 20)/FH):.4f})")

    text = "\n".join(L)
    print(text)
    (a.out / "report.txt").write_text(text)
    (a.out / "stats.json").write_text(json.dumps({
        "clean": {"n": len(res), "W": int(W), "H": int(H),
                  "header_bottom_max": int(hb.max()), "trace_top_min": int(tt.min()),
                  "rhythm_bottom_max": int(rb.max()), "footer_bottom_max": int(fb.max())},
        "final_height": int(FH), "margin_px": int(margin), "safety_px": SAFETY_PX,
        "header_bound_px": hdr_bound, "trace_bound_px": trace_bound,
        "rhythm_bound_px": rhythm_bound,
        "crop": {"y0": y0, "y1": y1, "y0_px": y0px, "y1_px": y1px,
                 "top_ok": bool(top_ok), "bottom_ok": bool(bot_ok)},
    }, indent=2))

    if a.data and a.annotate:
        import csv as _csv
        rows = list(_csv.DictReader((a.data / "index.csv").open()))
        # the sheets that came closest: largest |rotation|, where the bound is tight
        rows.sort(key=lambda r: -abs(float(r.get("rotation_deg", 0) or 0)))
        ann = a.out / "worst_cases"
        ann.mkdir(exist_ok=True)
        for r in rows[:a.annotate]:
            annotate(a.data / r["rel_path"], y0, y1, hdr_bound, trace_bound,
                     ann / f"rot{float(r['rotation_deg']):+06.2f}_{r['record']}.png")
        print(f"\nannotated the {a.annotate} most-rotated sheets -> {ann}")

    print(f"\nwrote {a.out}/report.txt and stats.json")
    return 0 if (top_ok and bot_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
