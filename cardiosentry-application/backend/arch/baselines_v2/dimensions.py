"""
The candidate INPUT GEOMETRIES for the resolution/aspect study.

Pure data + arithmetic. No torch, no PIL beyond the crop helper, so it can be
imported by config.py, by aspect_study.py and by a notebook cell that just wants
the table.

--------------------------------------------------------------------------------
WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
cardiosentry_runs_v1 trained at 384x384 and v1_768 at 768x768, and 768 won on
every head. The obvious reading is "more pixels is better". It is the wrong
reading, or at least an incomplete one, because BOTH runs threw pixels away in
the same two places:

  1. ASPECT. The sheet is 1686x1311 (1.286:1). Squaring it compresses the TIME
     axis by 1.286x more than the VOLTAGE axis. At 768x768 the horizontal scale
     is 768/1686 = 0.456 and the vertical is 768/1311 = 0.586: every sheet is
     squeezed 22% along time before any model sees it.

  2. BLANK PAPER. Measured over 150 sheets (see aspect_study.py --profile), the
     four trace rows occupy y = 289..1244 out of 1311. The top 22% of every
     sheet is the printed header plus empty paper, and the bottom 5% is the
     "25mm/s 10mm/mV" footer. A square resize spends 27% of its vertical pixel
     budget on paper that carries no waveform.

So "768x768 beat 384x384" is really "4x the pixels, of which ~27% were still
spent on blank paper and the time axis was still squeezed 22%". This file
enumerates the geometries that spend the same budget differently.

--------------------------------------------------------------------------------
THE HEADER IS ALSO A LABEL LEAK  -- read this before choosing a full-sheet size
--------------------------------------------------------------------------------
The kit prints `ID: ECG######` at the top of every sheet, and stage 1 assigned
those numbers by iterating the sources in order. In prototype_1152:

      CHAPMAN   ECG000002 .. ECG003241
      PTBXL     ECG003242 .. ECG007796
      STEMI     ECG007797 .. ECG009238      <- every STEMI record, and only those

STEMI is perfectly separable from the FOURTH CHARACTER of a string printed in
~25 px type on the sheet. Any model that learns to read one digit gets the STEMI
head for free, and it also gets `source` for free, which is exactly the confound
stage 5's leakage probe flagged (0.515 vs 0.372 chance).

The fix is REDACT_BOX below: a fixed rectangle painted over the printed block,
which removes the ID without touching geometry, resolution or aspect. Cropping
the whole band was the first answer and it was the wrong one - crop_audit.py
measured that a full-width cut also clips waveform on the sheets whose R waves
run tallest, which are exactly the LVH and STEMI records the study cares about.

Redaction also removes the printed Age and Sex. Those are genuine clinical
covariates rather than leakage, so losing them is a real cost - though it does
buy the stronger claim that classification came from the waveform alone.
"""
from __future__ import annotations

# ---------------------------------------------------------------- source sheet
SOURCE_W, SOURCE_H = 1686, 1311           # every image in the v2 corpus
SOURCE_ASPECT = SOURCE_W / SOURCE_H       # 1.2861

# Render geometry, from image_pipeline/config.py. Used to convert a pixel size
# into physical resolution, which is the only unit in which "is this enough
# detail" has an answer.
DPI = 150
PX_PER_MM = DPI / 25.4                    # 5.906
PAPER_SPEED_MM_S = 25                     # 1 s  = 25 mm = 147.6 px
PAPER_GAIN_MM_MV = 10                     # 1 mV = 10 mm =  59.1 px

# ---------------------------------------------------------------- content crop
# Fractions of the SOURCE height. Measured, not guessed: over 150 random sheets
# the printed header block never extended past y = 261 (0.199 H) and the first
# trace ink never started before y = 289 (0.220 H); the last trace ink never
# went past y = 1244 (0.949 H).
#
# CORRECTED 2026-08-25 by crop_audit.py, which measured the clean renders and
# mapped them through stage 4's full parameter range. The previous window
# (0.207, 0.964) was set from a sampled measurement whose search bounds made the
# header/trace gap look wider than it is, and it was wrong at BOTH ends:
#
#   top     0.207 cleared the worst-case waveform by only ~8 px, not the ~60 the
#           sample suggested. The binding sheet is an LVH record whose R waves
#           reach y=328 on the clean render.
#   bottom  0.964 clipped rhythm-strip ink on roughly 1-3% of sheets. The strip
#           can reach y=1215 clean, which lands at ~1304 of 1311 after rotation,
#           so NO bottom cut is safe. The footer caption is 40 px of constant
#           text and costs nothing to keep.
#
# Even corrected, a full-width cut still clips ~7 sheets whose waveform overlaps
# the header band after rotation. That is why REDACT_BOX exists and why the
# crop* geometries are no longer the default - see the module docstring.
CONTENT_CROP = (0.19, 1.0)                # (y0, y1) as fractions of height
CROP_H = (CONTENT_CROP[1] - CONTENT_CROP[0]) * SOURCE_H     # 992.4 px
CROP_ASPECT = SOURCE_W / CROP_H                             # 1.699


# ---------------------------------------------------------------- redaction
# A fixed rectangle over the printed ID / Age / Sex block, in fractions of the
# FINAL image. Everything about it is measured, not guessed:
#
#   * On the clean stage-3 renders the block is rigid - ID at x 238-407 y 83-102,
#     Age at x 236-364 y 112-133, Sex at x 238-373 y 141-158, on a 1650x1275
#     sheet, identical across 9,238 renders.
#   * Mapping those corners through stage 4's whole parameter range (rotation
#     +/-3 deg, keystone m in [0.004, 0.018], +18 px margin) and adding ~10 px of
#     slack gives the box below.
#   * crop_audit.py then verified the region against all 9,238 renders: it holds
#     printed text and the sheet's 1 px border, and nothing else, on 9,230 of
#     them. Eight STEMI sheets have a sliver of lead I/II/III waveform intruding
#     (worst case 454 ink pixels - the tip of one deflection, not a lead).
#
# WHY A PLAIN RECTANGLE AND NOT A SMARTER ERASE. Removing only the ink leaves an
# anti-aliased ghost that is still perfectly legible. Dilating the mask to catch
# the halo means erasing a TEXT-SHAPED region - and a glyph-shaped hole encodes
# the glyphs just as well as the glyphs do. Any shape-following erasure leaks, so
# the redaction is a fixed rectangle identical on every sheet.
REDACT_BOX = (0.10, 0.02, 0.30, 0.18)      # (x0, y0, x1, y1), fractions

# Records whose RENDER is unusable, not merely unusual. Both carry a saturation
# artifact in V4-V6 that spans the full height of the sheet, so the trace is
# clipped by the renderer itself and no crop or redaction can repair it.
# ECG009191 sits in the TEST split, so leaving it in scores a model on a
# corrupted sheet. manifest.load() drops these and says so.
EXCLUDED_RECORDS = ("ECG009191", "ECG009005")


# Bounds established by crop_audit.py: measured on all 9,238 CLEAN stage-3
# renders (where thresholding is exact), then mapped through stage 4's whole
# parameter range. In FINAL-image pixels on the 1311-px sheet.
#
# These are stored rather than re-derived, because measuring them on the
# AUGMENTED sheets does not work - lighting gradients and edge shadow defeat any
# ink threshold, and two earlier attempts produced confident nonsense. Re-run
# crop_audit.py to refresh them; do not re-implement the measurement elsewhere.
MEASURED = {
    "final_height": 1311,
    "header_bottom_max": 217,     # lowest any printed header block can reach
    "trace_top_min": 275,         # highest any waveform reaches, EXCLUDING the
                                  # 7 extreme LVH/STEMI sheets a full-width cut
                                  # cannot protect anyway
    "rhythm_bottom_max": 1304,    # of 1311 - which is why there is no bottom cut
    "clipped_by_crop": 7,         # sheets a crop* geometry damages, of 9,238
}


def redact_box(w: int, h: int) -> tuple[int, int, int, int]:
    """(left, upper, right, lower) in pixels for an image of size w x h.

    Fractions, so it is valid before OR after the resize - which matters,
    because applying it after is 2.7x cheaper (smaller box) and avoids bicubic
    smearing the rectangle's edges.
    """
    x0, y0, x1, y1 = REDACT_BOX
    return (int(round(x0 * w)), int(round(y0 * h)),
            int(round(x1 * w)), int(round(y1 * h)))


def crop_box(w: int = SOURCE_W, h: int = SOURCE_H) -> tuple[int, int, int, int]:
    """(left, upper, right, lower) for PIL, in pixels, for an image of size w x h.

    Expressed in fractions so it survives a pre-resized copy of the corpus.
    """
    y0 = int(round(CONTENT_CROP[0] * h))
    y1 = int(round(CONTENT_CROP[1] * h))
    return (0, y0, w, y1)


# ---------------------------------------------------------------- candidates
#
# size is (H, W) - torch order, NOT PIL order. Every H and W is a multiple of
# 32 so no CNN loses a row to an odd stride, and of 16 so ViT-B/16 tiles exactly.
#
# `px` relative to 768x768 = 589,824 is the honest cost axis: these geometries
# are meant to be compared at roughly EQUAL COMPUTE, so that a win is a win from
# spending the budget better rather than from spending more of it.
DIMENSIONS: dict[str, dict] = {
    # ---- controls: reproduce the two runs that already exist -----------------
    "sq384": {
        "size": (384, 384), "crop": False,
        "note": "control - identical to cardiosentry_runs_v1",
    },
    "sq768": {
        "size": (768, 768), "crop": False,
        "note": "control - identical to v1_768. The one to beat.",
    },

    # ---- full sheet, aspect corrected ---------------------------------------
    "nat672": {
        "size": (672, 864), "crop": False,
        "note": "native aspect (864/672 = 1.2857 vs sheet 1.2861), ~equal px to sq768",
    },
    "nat768": {
        "size": (768, 1024), "crop": False,
        "note": "native aspect, +33% px - isolates 'more pixels' from 'better aspect'",
    },
    "wide576": {
        "size": (576, 1024), "crop": False,
        "note": "EXACTLY sq768's pixel count, spent on the time axis instead",
    },

    # ---- header cropped ------------------------------------------------------
    # Kept for the ablation, NOT recommended: crop_audit.py showed a full-width
    # cut clips waveform on the tallest-R-wave sheets even at the corrected
    # window. REDACT_BOX removes the ID leak without that cost.
    "crop640": {
        "size": (640, 1024), "crop": True,
        "note": "header cropped at the CORRECTED window; aspect preserved "
                "(1.600 vs 1.587). Clips ~7 extreme LVH/STEMI sheets.",
    },
    "crop544": {
        "size": (544, 864), "crop": True,
        "note": "header cropped, 20% cheaper than sq768 - the 'can we go smaller' probe",
    },
}

# nat768: the full sheet at its own aspect. Same voltage resolution as sq768,
# +33% on time, distortion 1.04, and the first candidate whose time-Nyquist
# (44.8 Hz) clears the 40 Hz band the corpus was rendered in. Both dimensions are
# multiples of 256, so Swin-V2 needs no padding wrapper and the nine-model cost
# table stays directly comparable. The ID leak is handled by REDACT_BOX instead
# of by cropping.
DEFAULT_DIM = "nat768"
DIM_NAMES = list(DIMENSIONS)


def resolve(name: str) -> dict:
    if name not in DIMENSIONS:
        raise KeyError(f"unknown dimension '{name}'. choose from: {DIM_NAMES}")
    d = dict(DIMENSIONS[name])
    d["name"] = name
    return d


# ---------------------------------------------------------------- arithmetic
def geometry(name: str) -> dict:
    """Everything derivable about one candidate, in physical units.

    The two numbers that decide whether a geometry can carry a diagnosis:

      px_per_mm_x   how many output pixels one millimetre of TIME survives as.
                    QRS duration is 2-3 mm; a normal-vs-wide QRS is a 1 mm call.
      px_per_mm_y   ...one millimetre of VOLTAGE. ST elevation is diagnosed at
                    1 mm (0.1 mV). This is the finest measurement on the sheet
                    and it lives on the axis a square resize protects LEAST
                    relative to how much detail it carries.
    """
    d = resolve(name)
    H, W = d["size"]
    src_h = CROP_H if d["crop"] else float(SOURCE_H)
    src_w = float(SOURCE_W)

    sx, sy = W / src_w, H / src_h
    return {
        "name": name,
        "H": H, "W": W,
        "crop": d["crop"],
        "note": d["note"],
        "aspect": W / H,
        "source_aspect": src_w / src_h,
        # >1 stretches time relative to voltage, <1 squeezes it
        "aspect_distortion": (W / H) / (src_w / src_h),
        "pixels": H * W,
        "px_rel_sq768": H * W / (768 * 768),
        "scale_x": sx,
        "scale_y": sy,
        "px_per_mm_x": PX_PER_MM * sx,
        "px_per_mm_y": PX_PER_MM * sy,
        "px_per_second": PX_PER_MM * PAPER_SPEED_MM_S * sx,
        "px_per_mv": PX_PER_MM * PAPER_GAIN_MM_MV * sy,
        # the diagnostic thresholds, in output pixels
        "px_per_1mm_st": PX_PER_MM * sy,            # 0.1 mV ST step
        "px_per_qrs": PX_PER_MM * PAPER_SPEED_MM_S * 0.10 * sx,   # 100 ms QRS
        # image Nyquist along time, in Hz: half the sampling rate the paper
        # carries after the resize. The signal band is 0.05-40 Hz.
        "time_nyquist_hz": PX_PER_MM * PAPER_SPEED_MM_S * sx / 2.0,
    }


def table(names: list[str] | None = None) -> str:
    names = names or DIM_NAMES
    L = []
    A = L.append
    A(f"source sheet {SOURCE_W}x{SOURCE_H} ({SOURCE_ASPECT:.3f}:1) at {DPI} DPI"
      f"  ->  {PX_PER_MM:.3f} px/mm, {PX_PER_MM*PAPER_SPEED_MM_S:.1f} px/s,"
      f" {PX_PER_MM*PAPER_GAIN_MM_MV:.1f} px/mV")
    A(f"content crop  y {CONTENT_CROP[0]:.3f}..{CONTENT_CROP[1]:.3f}"
      f"  ->  {SOURCE_W}x{CROP_H:.0f} ({CROP_ASPECT:.3f}:1), header removed")
    A("")
    hdr = (f"{'dim':10}{'HxW':>11}{'crop':>6}{'aspect':>8}{'distort':>9}"
           f"{'Mpx':>7}{'x sq768':>9}{'px/mm x':>9}{'px/mm y':>9}"
           f"{'px/1mm ST':>11}{'px/QRS':>8}{'Nyq Hz':>8}")
    A(hdr)
    A("-" * len(hdr))
    for n in names:
        g = geometry(n)
        A(f"{n:10}{g['H']}x{g['W']:<7}{'yes' if g['crop'] else 'no':>6}"
          f"{g['aspect']:8.3f}{g['aspect_distortion']:9.3f}"
          f"{g['pixels']/1e6:7.3f}{g['px_rel_sq768']:9.2f}"
          f"{g['px_per_mm_x']:9.3f}{g['px_per_mm_y']:9.3f}"
          f"{g['px_per_1mm_st']:11.2f}{g['px_per_qrs']:8.1f}{g['time_nyquist_hz']:8.1f}")
    A("")
    A("  distort   1.000 = the sheet's own aspect is preserved. <1 squeezes TIME,")
    A("            >1 stretches it. sq768 sits at 0.78: 22% of the time axis is")
    A("            thrown away before any model sees the sheet.")
    A("  px/1mm ST 1 mm of voltage = 0.1 mV = the ST-elevation decision threshold.")
    A("            Under ~3 px the step a STEMI is defined by is 2-3 pixels tall.")
    A("  px/QRS    a 100 ms QRS complex. Under ~8 px, normal vs wide is a marginal")
    A("            call. No candidate at this pixel budget reaches 10; 9.0 is the")
    A("            ceiling, and only the three widest geometries get there.")
    A("  Nyq Hz    highest temporal frequency the resized image can still represent.")
    A("            The rendered band is 0.05-40 Hz, so anything under 40 is lossy.")
    return "\n".join(L)


if __name__ == "__main__":
    print(table())
