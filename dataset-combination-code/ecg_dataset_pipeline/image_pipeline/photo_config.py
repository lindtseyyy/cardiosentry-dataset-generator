"""Parameters for the print -> paper -> camera model in stage4_photo.py.

Separate from config.py on purpose: config.py is what the published v2 corpus
was built with and must keep producing that corpus byte-for-byte. Everything
here is new and applies only to the photo-realistic build.

Every number below was set by measuring the reference photograph
`experiments/actual_ecg_image sample/ecg_sample.jpg` (Xiaomi POCO X6 5G,
4624x2600) against a clean 150 dpi render, not by eye. The measurements:

    property                 real photo      old synthetic
    px per mm                    15.7             5.91
    1 mm grid visible?            yes    no - destroyed by stage 4
    trace stroke width          0.19 mm          0.34 mm
    paper luminance (p98)          207              250
    background visible?           yes    no - 18 px paper-coloured border

The model is deliberately ordered as the physical chain, because that is what
makes the result look photographed rather than noised:

    clean vector render (kit)
      -> PRESS   ink laid on paper: density variation, diffusion, grid wear
      -> SHEET   paper stock: texture, creases, fibre
      -> CAMERA  pose, illumination, optical integration, sensor, JPEG
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------- print stage
# The kit renders at this DPI; the camera model then downsamples. Rendering
# above the output resolution is the whole point: area-averaging a 300 dpi
# sheet down to the frame is optical integration, and it is what removes the
# hard vector edges that no amount of post-hoc blurring convincingly fakes.
RENDER_DPI = int(os.environ.get("ECG_DPI", 300))
RENDER_PX_PER_MM = RENDER_DPI / 25.4

# Matplotlib points, passed to the kit through the environment (see patch_kit).
# The call site multiplies the trace by 1.5, so 0.40 -> 0.60 pt -> 0.21 mm,
# against 0.19 mm measured on the reference. The kit's stock 0.75 gives 0.40 mm.
TRACE_LINE_WIDTH = float(os.environ.get("ECG_LINE_WIDTH", 0.40))
GRID_LINE_WIDTH = float(os.environ.get("ECG_GRID_LINE_WIDTH", 0.40))

# --- ink separation ------------------------------------------------------
# The clean render is greyscale with the grid on two exact levels (major 102,
# minor 191), the trace below 60 and bare paper above 250. Splitting on those
# levels recovers a trace layer and a grid layer, which can then be given
# different physics - the trace is a hot stylus, the grid is offset printing.
INK_TRACE_CUT = 92        # grey below this is trace/text, not grid
INK_PAPER_LEVEL = 252     # grey at or above this is bare paper

# MEASURED, not assumed. The reference photo's trace core reaches grey 1-3 -
# a thermal stylus lays down an essentially black line. An earlier guess of
# "printed ink is never pure black" (ink 32, density 0.97) gave a core of 39 and
# a visibly washed-out trace. What is NOT uniform is the density ALONG the
# stroke, which the mottle and grain fields below supply.
# Slightly WARM, not neutral. Measured: the reference's trace runs R-B = +8,
# and a neutral ink came back at R-B = -2.8 because 4:2:0 chroma subsampling
# on a 1.5 px dark stroke pulls it blue. Biasing the ink warm cancels that.
TRACE_INK_RGB = (17, 13, 10)
TRACE_INK_JITTER = 6              # +- per channel, per image
TRACE_DENSITY = (0.945, 0.995)    # peak alpha at one unit of ink
# Low-frequency density variation along the stroke (stylus heat, paper contact).
TRACE_MOTTLE_SCALE_MM = (3.0, 14.0)
TRACE_MOTTLE_AMP = (0.10, 0.24)
# Fine grain: paper absorbency at the fibre scale.
TRACE_GRAIN_AMP = (0.04, 0.10)
# Ink diffusion: the dot spreads into the paper. Sigma in MILLIMETRES, so it is
# resolution-independent and stays physical if RENDER_DPI changes.
TRACE_DIFFUSE_MM = (0.028, 0.055)
TRACE_BLEED = (0.05, 0.13)        # fraction of ink pushed into the halo

# --- grid ----------------------------------------------------------------
GRID_DENSITY = (0.30, 0.52)       # the grid is printed MUCH lighter than the trace
GRID_MOTTLE_SCALE_MM = (8.0, 30.0)
GRID_MOTTLE_AMP = (0.08, 0.20)    # roller pressure varies across the sheet
GRID_DIFFUSE_MM = (0.02, 0.05)
# Real ECG paper grids are printed in one ink; the major line is the same ink
# laid heavier, not a different colour.
GRID_INK_RGB = {
    "red":    (206, 122, 116),
    "orange": (208, 146, 104),
    "pink":   (208, 146, 164),
    "grey":   (140, 137, 132),      # neutral-warm, not blue: a blue grid reads as graph paper
}
GRID_PALETTE_WEIGHTS = {"red": 0.44, "orange": 0.22, "pink": 0.14, "grey": 0.20}
# No major/minor emphasis knob: the kit renders major at grey 102 and minor at
# 191, so measured against bare paper the major line already carries ~2.5x the
# ink. separate_ink() expresses the grid in units of "one minor line", and
# Beer-Lambert then darkens the major line by exactly that ratio for free.

# --- paper stock ---------------------------------------------------------
PAPER_RGB = (250, 248, 245)
PAPER_TINT_JITTER = 4
PAPER_FIBRE_AMP = (0.006, 0.016)  # fine multiplicative grain

# Wrinkle field. The kit's 19 crease photographs carry large dark blobs as well
# as fine texture; composited whole they read as a stained, crumpled sheet
# rather than a flat printout on a desk. They are therefore HIGH-PASSED - the
# blob scale is subtracted and only the fine relief is kept - and applied at a
# fraction of the amplitude stage4_augment.py used.
WRINKLE_PROB = 0.70
WRINKLE_AMP = (0.014, 0.036)
WRINKLE_HIGHPASS_MM = 7.0         # remove structure coarser than this
WRINKLE_CLIP = (0.974, 1.026)
CREASE_PROB = 0.40
CREASE_AMP = (0.025, 0.065)

# ---------------------------------------------------------------- camera
# Output frame. The reference is 4624x2600 (12 MP) at 15.7 px/mm; matching that
# with the whole sheet in frame needs ~5000 px and ~7 MB per image, which is
# 250 GB for a 37 k corpus. 2600 px is the default compromise and gives about
# 8.2 px/mm on the sheet. Set ECG_PHOTO_LONG_EDGE=5000 to reproduce the
# reference's sampling exactly.
OUT_LONG_EDGE = int(os.environ.get("ECG_PHOTO_LONG_EDGE", 2600))
OUT_ASPECT = (1.45, 1.62)         # frame aspect; phone held to fit a wide sheet

# How much of the frame's short dimension the sheet spans. Below 1.0 the whole
# sheet is inside the frame with background around it; the top of the range
# lets an edge run out of frame, as the reference photo does.
SHEET_FILL = (0.84, 0.99)

# Camera pose. Yaw/pitch are rotations of the SHEET about axes in its own
# plane, which is what produces true keystone; roll is the handheld tilt.
CAM_ROLL_DEG = (-2.8, 2.8)
CAM_YAW_DEG = (-7.0, 7.0)
CAM_PITCH_DEG = (-5.5, 5.5)
CAM_DISTANCE_W = (2.6, 4.5)       # viewing distance in sheet-widths (perspective strength)
CAM_OFFSET = (-0.05, 0.05)        # principal-point offset, fraction of frame

# --- scene ---------------------------------------------------------------
# Background the sheet lies on. Procedural, so no assets to ship.
BG_RGB_CHOICES = (
    (74, 66, 58),      # dark wood desk, as in the reference
    (58, 58, 62),      # grey desktop
    (96, 84, 70),      # lighter wood
    (44, 46, 50),      # near-black surface
)
BG_NOISE = (3.0, 9.0)
BG_BLUR_MM = (1.2, 3.5)           # background is off the focal plane
BG_SHADE_AMP = (0.06, 0.13)       # broad shading; above this it reads as smoke

# Drop shadow cast by the sheet onto the surface.
SHADOW_OFFSET_MM = (1.0, 4.0)
SHADOW_BLUR_MM = (1.5, 5.0)
SHADOW_STRENGTH = (0.25, 0.55)

# --- lighting ------------------------------------------------------------
# Multiplicative field over the sheet. The reference's paper runs 207 at p98
# with a visible gradient, so the target below is well under pure white.
LIGHT_RAMP = (0.06, 0.17)         # directional gradient amplitude
LIGHT_FALLOFF = (0.04, 0.13)      # radial vignette on the sheet
LIGHT_SHADOW_PROB = 0.55          # a soft shadow crossing one edge
LIGHT_SHADOW_AMP = (0.08, 0.22)
# Set on the SHEET, before the camera. The lens vignette and the JPEG pass
# then pull the in-frame level down by roughly 20 levels, landing near the
# reference's measured 205-207.
PAPER_TARGET_P98 = (222, 248)

# --- optics and sensor ---------------------------------------------------
# Applied AFTER the downsample, at output resolution, so they describe the
# camera and not the print.
LENS_BLUR_PX = (0.35, 0.95)
LENS_VIGNETTE = (0.03, 0.10)
CHROMA_SHIFT_PX = (0.0, 0.28)     # lateral chromatic aberration at the corners
SENSOR_READ_NOISE = (1.2, 3.0)    # constant sigma, in 8-bit levels
SENSOR_SHOT_NOISE = (0.55, 1.35)  # sigma scales with sqrt(signal)
WB_JITTER = 0.018                 # per-channel white-balance drift
JPEG_QUALITY = (78, 90)
JPEG_SUBSAMPLING = 2              # 4:2:0, as every phone camera writes
