"""Empirical local grid-spacing validation (PLAN 10.7).

Measures the major-grid period directly on output pixels in 9 windows and
compares against nominal (17.79 px final / 39.37 px master) and against the
clean sibling. No analytic distortion claim is made anywhere; this module
is the instrument that replaces it.

Method per window: palette-aware grid-ink mask (pixels whose chroma
direction from paper white matches the grid colour), horizontal/vertical
projection profiles, autocorrelation peak search in [0.8x, 1.25x] nominal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import (GRID_PALETTES, GRID_WINDOW_SIZE, GRID_SEARCH_BAND,
                     MAX_NOMINAL_DEV, MAX_CLEAN_DERIVED_MISMATCH,
                     MAX_FAILED_WINDOWS, NOMINAL_MINOR_FINAL_PX,
                     NOMINAL_MINOR_MASTER_PX)


def window_boxes(w: int, h: int, size: int = GRID_WINDOW_SIZE):
    """9 windows: 4 quadrants + 4 edge-centres + centre (x0, y0, x1, y1)."""
    xs = [0, (w - size) // 2, w - size]
    ys = [0, (h - size) // 2, h - size]
    return [(x, y, x + size, y + size) for y in ys for x in xs]


def grid_mask(arr: np.ndarray, palette_name: str | None) -> np.ndarray:
    """Boolean mask of MAJOR-grid-ink pixels (cores only). Diagnostic helper
    (mask overlays, debugging); the estimator itself uses the spectrum, not
    this mask.

    Major-only is deliberate: the minor grid (3.56 px period) would inject a
    competing 4th harmonic at ~14.2 px inside the [0.8x, 1.25x] search band
    and corrupt the major-period estimate (observed on real renders).

    Coloured palettes: chroma-direction test from paper white + strength cut
    (major |d|~242, shaded ~190; minor ~102, shaded ~80; halos ~14).
    Grey/None: luminance band (major 102, shaded ~80; minor 191 excluded).
    Trace ink is excluded by the luminance floor in both branches.
    """
    f = arr.astype(np.float32)
    lum = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if palette_name is None or palette_name == "grey":
        return (lum > 65) & (lum < 130)
    pal = np.asarray(GRID_PALETTES[palette_name][0], dtype=np.float32)  # MAJOR
    paper = np.asarray(GRID_PALETTES[palette_name][2], dtype=np.float32)
    grid_dir = pal - paper
    grid_dir /= (np.linalg.norm(grid_dir) + 1e-9)
    d = f - paper[None, None, :]
    cos = (d * grid_dir[None, None, :]).sum(axis=2) / (
        np.linalg.norm(d, axis=2) + 1e-9)
    strength = np.linalg.norm(d, axis=2)
    return (cos > 0.95) & (strength > 120) & (lum < 235) & (lum > 80)


PEAK_CONTRAST_MIN = 4.0   # spectral peak / annulus median (else illegible)
WEDGE_HALF_DEG = 15.0   # exclusion wedge around axis 1 when seeking axis 2


def _refine_2d(power: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Separable parabolic sub-bin refinement of a spectral peak."""
    h, w = power.shape
    fx = float(ix)
    if 0 < ix < w - 1:
        y0, y1, y2 = power[iy, ix - 1], power[iy, ix], power[iy, ix + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            fx += max(-0.5, min(0.5, 0.5 * (y0 - y2) / den))
    fy = float(iy)
    if 0 < iy < h - 1:
        y0, y1, y2 = power[iy - 1, ix], power[iy, ix], power[iy + 1, ix]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            fy += max(-0.5, min(0.5, 0.5 * (y0 - y2) / den))
    return fx, fy


def _axis_periods(power: np.ndarray, nominal: float) -> tuple | None:
    """Two grid-axis periods (+ angles) from a window power spectrum.

    Searches the annulus |k| in [N/(1.25*nominal), N/(0.8*nominal)] for the
    strongest peak (axis 1), masks a wedge around its direction, and takes
    the strongest remaining peak (axis 2, the perpendicular grid family).
    Returns ((p1, a1), (p2, a2)) or None when the grid is illegible.
    """
    n = power.shape[0]
    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_lo = n / (GRID_SEARCH_BAND[1] * nominal)
    r_hi = n / (GRID_SEARCH_BAND[0] * nominal)
    ring = (rr >= r_lo) & (rr <= r_hi)
    if not ring.any():
        return None
    med = float(np.median(power[ring]))
    if med <= 0:
        return None

    masked = np.where(ring, power, -1.0)
    i1 = int(np.argmax(masked))
    iy1, ix1 = divmod(i1, n)
    if power[iy1, ix1] / med < PEAK_CONTRAST_MIN:
        return None
    fx1, fy1 = _refine_2d(power, iy1, ix1)
    r1 = float(np.hypot(fx1 - cx, fy1 - cy))
    a1 = float(np.degrees(np.arctan2(fy1 - cy, fx1 - cx)))

    ang = np.degrees(np.arctan2(yy - cy, xx - cx))
    sep = np.abs((ang - a1 + 180.0) % 360.0 - 180.0)
    ring2 = ring & (sep > WEDGE_HALF_DEG) & (sep < 180.0 - WEDGE_HALF_DEG)
    if not ring2.any():
        return None
    masked2 = np.where(ring2, power, -1.0)
    i2 = int(np.argmax(masked2))
    iy2, ix2 = divmod(i2, n)
    if power[iy2, ix2] / med < PEAK_CONTRAST_MIN:
        return None
    fx2, fy2 = _refine_2d(power, iy2, ix2)
    r2 = float(np.hypot(fx2 - cx, fy2 - cy))
    a2 = float(np.degrees(np.arctan2(fy2 - cy, fx2 - cx)))
    if r1 <= 0 or r2 <= 0:
        return None
    return ((n / r1, a1), (n / r2, a2))


def measure_window(arr: np.ndarray, box, palette: str | None,
                   nominal: float) -> dict:
    """Local grid periods in one window via 2D-FFT spectral peaks.

    Operates on grayscale (no colour classification — immune to trace-ink
    contamination and palette shading), tolerates tilt (peaks rotate but
    their radius, the scale measurement, is preserved), and ignores the
    minor grid (5x frequency, outside the search annulus). `palette` is
    accepted for API compatibility and ignored.
    """
    del palette
    x0, y0, x1, y1 = box
    win = arr[y0:y1, x0:x1].astype(np.float64)
    if win.ndim == 3:
        win = win @ np.array([0.299, 0.587, 0.114])
    win = win - win.mean()
    if win.std() < 1e-9:
        return {"px": None, "py": None, "ok": False, "reason": "flat"}
    hann = np.hanning(win.shape[0])[:, None] * np.hanning(win.shape[1])[None, :]
    spec = np.fft.fftshift(np.fft.fft2(win * hann))
    power = (spec.real ** 2 + spec.imag ** 2)
    found = _axis_periods(power, nominal)
    if found is None:
        return {"px": None, "py": None, "ok": False, "reason": "no_peak"}
    (p1, a1), (p2, a2) = found
    # sorted assignment keeps clean-vs-derived correspondence under tilt
    (plo, alo), (phi, ahi) = sorted([(p1, a1), (p2, a2)])
    return {"px": plo, "py": phi, "ok": True,
            "angle_lo": alo, "angle_hi": ahi,
            "dev_x": (plo - nominal) / nominal,
            "dev_y": (phi - nominal) / nominal}


def measure_image(path: str | Path, palette: str | None,
                  nominal: float) -> dict:
    """Measure local grid periods (9 windows). No derotation is needed: the
    spectral estimator is rotation-tolerant by construction (tilt rotates
    the peaks; their radius is the scale measurement)."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    wins = []
    for i, box in enumerate(window_boxes(w, h)):
        m = measure_window(arr, box, palette, nominal)
        m["window"] = i
        wins.append(m)
    ok = [m for m in wins if m["ok"]]
    max_dev = max([max(abs(m["dev_x"]), abs(m["dev_y"])) for m in ok],
                  default=None)
    return {"path": str(path), "palette": palette, "nominal": nominal,
            "windows": wins, "n_ok": len(ok),
            "n_failed": len(wins) - len(ok), "max_dev": max_dev}


def nominal_for_size(w: int, h: int) -> float:
    """Minor-grid nominal period (the spectral validator measures the 1 mm
    grid; see config note). Major-equivalent is 5x this value."""
    if (w, h) == (1024, 768):
        return NOMINAL_MINOR_FINAL_PX
    if (w, h) == (2200, 1700):
        return NOMINAL_MINOR_MASTER_PX
    raise ValueError(f"no nominal grid period for size {w}x{h}")


def check_pair(clean_meas: dict, derived_meas: dict) -> dict:
    """Compare a derived view against its clean sibling, window by window."""
    mism = []
    for a, b in zip(clean_meas["windows"], derived_meas["windows"]):
        if not (a["ok"] and b["ok"]):
            continue
        mism.append(max(abs(a["px"] - b["px"]) / a["px"],
                        abs(a["py"] - b["py"]) / a["py"]))
    return {"max_mismatch": max(mism) if mism else None,
            "n_compared": len(mism)}


def gate_report(meas_list: list[dict], pairs: list[tuple[dict, dict]]) -> dict:
    """Apply PLAN 10.7 gates. Returns report with pass/fail + worst cases."""
    fails: list[str] = []
    worst_nom, worst_mm = 0.0, 0.0
    for m in meas_list:
        if m["n_failed"] > MAX_FAILED_WINDOWS:
            fails.append(f"{m['path']}: {m['n_failed']} failed windows")
        if m["max_dev"] is not None:
            worst_nom = max(worst_nom, m["max_dev"])
            if m["max_dev"] > MAX_NOMINAL_DEV:
                fails.append(f"{m['path']}: nominal deviation "
                             f"{m['max_dev']:.1%} > {MAX_NOMINAL_DEV:.0%}")
    for c, d in pairs:
        r = check_pair(c, d)
        if r["max_mismatch"] is not None:
            worst_mm = max(worst_mm, r["max_mismatch"])
            if r["max_mismatch"] > MAX_CLEAN_DERIVED_MISMATCH:
                fails.append(f"{d['path']}: clean-vs-derived mismatch "
                             f"{r['max_mismatch']:.1%} "
                             f"> {MAX_CLEAN_DERIVED_MISMATCH:.0%}")
    return {"pass": not fails, "failures": fails,
            "n_images": len(meas_list), "n_pairs": len(pairs),
            "worst_nominal_dev": worst_nom, "worst_mismatch": worst_mm}
