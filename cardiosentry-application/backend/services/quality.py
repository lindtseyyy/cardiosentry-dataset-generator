"""Image quality checks (plan §14).

POLICY: warn, never block. The purpose of this app is to observe how the model
behaves on imperfect real images; a rejected photo is a missing data point.
`settings.quality_block_on_poor` exists and defaults to false.

Metric notes:
- Sharpness (variance of Laplacian) is ALWAYS computed on `corrected.png`
  (1686x1311). The metric is resolution-dependent; the fixed canvas makes it
  comparable between a 12 MP and a 48 MP phone.
- `est_px_per_mm` = quad width / physical sheet width (279.4 mm Letter
  landscape) — directly comparable to training's 5.91 px/mm at 150 DPI, and
  the most physically meaningful metric in the list.
- The grid FFT check is an independent px/mm estimate that does not assume the
  paper size; it empirically confirms the 1 mm grid survived the resize to the
  model input, whatever geometry that is. The search band (1.5–6.0 px) covers
  every geometry the app serves: the round-2 square resolves the grid at
  2.71 px/mm across and 3.46 down, the round-3 768x1024 at 3.59 and 3.46.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.schemas.capture import QualityMetrics
from backend.settings import settings

_POOR = dict(blur_varlap=50.0, brightness_lo=0.10, brightness_hi=0.95,
             contrast=0.15, clipped=0.15, glare=0.05, px_per_mm=4.0,
             skew_deg=40.0, area=0.20)
_GRID_MIN_PERIOD_PX = 1.5
_GRID_MAX_PERIOD_PX = 6.0


def _gray_small(rgb: np.ndarray) -> np.ndarray:
    """Downscale to ≤1024 for the photometric metrics (they are scale-robust)."""
    h, w = rgb.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def blur_varlap(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def glare_blob_frac(gray: np.ndarray) -> float:
    """Fraction of near-saturated pixels inside connected components larger
    than 0.5% of the image — distinguishes specular glare from an evenly
    bright sheet."""
    mask = (gray >= 250).astype(np.uint8)
    total = gray.size
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0.0
    frac = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.005 * total:
            frac += int(stats[i, cv2.CC_STAT_AREA])
    return frac / total


def _quad_metrics(quad_px, img_w: int, img_h: int):
    """area_fraction, skew_deg, est_px_per_mm for a quad (TL,TR,BR,BL)."""
    pts = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    area_frac = float(cv2.contourArea(pts.astype(np.float32)) / (img_w * img_h))
    max_dev = 0.0
    for i in range(4):
        a = pts[i] - pts[(i - 1) % 4]
        b = pts[(i + 1) % 4] - pts[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la == 0 or lb == 0:
            continue
        cosang = float(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0))
        max_dev = max(max_dev, abs(90.0 - np.degrees(np.arccos(cosang))))
    top_px = float(np.linalg.norm(pts[1] - pts[0]))
    px_per_mm = top_px / settings.paper_width_mm if settings.paper_width_mm else None
    return area_frac, max_dev, px_per_mm


def _level_from_flags(flags: list[str]) -> str:
    if not flags:
        return "ok"
    return "poor" if any(f.startswith("poor_") for f in flags) else "warn"


def quality_original(rgb_oriented: np.ndarray, quad_px=None,
                     img_w: int | None = None, img_h: int | None = None) -> QualityMetrics:
    """Metrics available right after ingest. Blur is deliberately absent —
    it is computed on the fixed 1686x1311 canvas after rectification (§14)."""
    gray = _gray_small(rgb_oriented)
    small = cv2.resize(rgb_oriented, (gray.shape[1], gray.shape[0]),
                       interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0
    brightness = float(L.mean())
    contrast = float(np.percentile(L, 95) - np.percentile(L, 5))
    clipped_high = float((gray >= 250).mean())
    clipped_low = float((gray <= 5).mean())
    glare = glare_blob_frac(gray)

    flags: list[str] = []
    s = settings

    def check(cond, flag):
        if cond:
            flags.append(flag)

    check(brightness < s.warn_brightness_lo, "low_brightness")
    check(brightness > s.warn_brightness_hi, "high_brightness")
    check(contrast < s.warn_contrast_p5p95, "low_contrast")
    check(clipped_high > s.warn_clipped_high_frac, "clipped_highlights")
    check(clipped_low > s.warn_clipped_low_frac, "clipped_shadows")
    check(glare > s.warn_glare_blob_frac, "glare")
    check(brightness < _POOR["brightness_lo"] or brightness > _POOR["brightness_hi"],
          "poor_exposure")
    check(contrast < _POOR["contrast"], "poor_contrast")
    check(clipped_high > _POOR["clipped"] or clipped_low > _POOR["clipped"],
          "poor_clipping")
    check(glare > _POOR["glare"], "poor_glare")

    area_frac = skew = px_per_mm = None
    if quad_px is not None:
        area_frac, skew, px_per_mm = _quad_metrics(quad_px, img_w, img_h)
        check(area_frac < s.warn_area_fraction, "small_sheet_area")
        check(skew > s.warn_skew_deg, "extreme_perspective")
        check(px_per_mm is not None and px_per_mm < s.warn_px_per_mm, "low_resolution")
        check(area_frac < _POOR["area"], "poor_area")
        check(skew > _POOR["skew_deg"], "poor_perspective")
        check(px_per_mm is not None and px_per_mm < _POOR["px_per_mm"],
              "poor_resolution")

    return QualityMetrics(
        level=_level_from_flags(flags),
        brightness=round(brightness, 4), contrast_p5p95=round(contrast, 4),
        clipped_high_frac=round(clipped_high, 4), clipped_low_frac=round(clipped_low, 4),
        glare_blob_frac=round(glare, 4),
        est_px_per_mm=round(px_per_mm, 2) if px_per_mm else None,
        area_fraction=round(area_frac, 4) if area_frac else None,
        skew_deg=round(skew, 2) if skew is not None else None,
        flags=flags)


def grid_period_fft(gray_model_input: np.ndarray) -> tuple[float, bool]:
    """1-D FFT of mid-sheet row+column of model_input.png; look for the 1 mm
    grid peak. Returns (period_px, detected). The geometric estimate assumes
    the paper size; this one does not (§18.2).

    Reads the shape from the array rather than assuming a square, so a
    768x1024 input is measured on its own mid row and mid column — which is
    also why the returned period is the MEAN of the two: on any non-square
    geometry the horizontal and vertical px/mm genuinely differ."""
    h, w = gray_model_input.shape
    periods, prominences = [], []

    for signal in (gray_model_input[h // 2].astype(np.float64),
                   gray_model_input[:, w // 2].astype(np.float64)):
        signal = signal - signal.mean()
        # Detrend with a light high-pass so the illumination ramp does not
        # swamp the grid peak.
        kernel = np.ones(max(3, len(signal) // 32)) / max(3, len(signal) // 32)
        trend = np.convolve(signal, kernel, mode="same")
        signal = signal - trend
        n = len(signal)
        mag = np.abs(np.fft.rfft(signal))[1:]
        freqs = np.fft.rfftfreq(n)[1:]
        band = (freqs >= 1 / _GRID_MAX_PERIOD_PX) & (freqs <= 1 / _GRID_MIN_PERIOD_PX)
        if not band.any():
            continue
        band_mag = mag[band]
        if band_mag.max() <= 0:
            continue
        period = float(1.0 / freqs[band][int(np.argmax(band_mag))])
        med = float(np.median(band_mag))
        prominences.append(band_mag.max() / med if med > 0 else 1.0)
        periods.append(period)

    if not periods:
        return 0.0, False
    period = float(np.mean(periods))
    detected = float(np.mean(prominences)) > 2.5
    return round(period, 3), bool(detected)


def quality_model_input(corrected_rgb: np.ndarray,
                        gray_model_input: np.ndarray) -> QualityMetrics:
    """Post-rectification metrics: sharpness on the fixed canvas, grid
    resolvability on the model input."""
    blur = blur_varlap(cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2GRAY))
    period, grid_ok = grid_period_fft(gray_model_input)

    flags: list[str] = []
    if blur < settings.warn_blur_varlap:
        flags.append("low_sharpness")
    if blur < _POOR["blur_varlap"]:
        flags.append("poor_sharpness")
    if not grid_ok:
        flags.append("grid_not_resolvable")

    return QualityMetrics(
        level=_level_from_flags(flags),
        blur_varlap=round(blur, 1),
        grid_period_px=period,
        grid_detected=grid_ok,
        flags=flags)
