"""ECG paper quad detection (plan §8).

Classical OpenCV only, with a fallback ladder, because a *perfect* detector is
not required: manual corner adjustment is a hard requirement anyway, so
automatic detection only needs to save time. Each result records the method
that produced it, so failure modes are countable later (§8.3).

Pipeline (§8.2), all at a ≤1024 px working scale:
    gray -> bilateral (kills the 1mm grid, the dominant inner edge source)
         -> morph close -> Canny (median-derived auto thresholds) -> dilate
         -> findContours -> area filter -> approxPolyDP quads -> score -> best
         -> order corners TL,TR,BR,BL -> cornerSubPix refine -> scale back

Candidate scoring (§8.4): area 0.30, convexity 0.20, corner-angle sanity 0.20,
aspect proximity 0.15 (weak prior, configurable/nullable), edge support 0.15.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.settings import settings

POINTS = tuple[tuple[float, float], tuple[float, float],
               tuple[float, float], tuple[float, float]]


@dataclass
class DetectionResult:
    success: bool
    method: str                      # canny_contour | hough_lines | threshold_bbox | none
    confidence: float
    quad_norm: POINTS | None         # TL,TR,BR,BL in [0,1] of the oriented grid
    quad_px: POINTS | None           # same, in oriented-grid pixels
    area_fraction: float | None = None
    aspect_ratio: float | None = None
    params: dict = field(default_factory=dict)
    score_breakdown: dict = field(default_factory=dict)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL,TR,BR,BL. Input/output shape (4,2)."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    order = [int(np.argmin(s)), int(np.argmax(d)), int(np.argmax(s)), int(np.argmin(d))]
    out = pts[order].astype(np.float64)
    if len({tuple(map(round, p)) for p in out}) < 4:
        raise ValueError("degenerate corner ordering")
    return out


def _quad_aspect(pts: np.ndarray) -> float:
    tl, tr, br, bl = pts
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if h <= 0:
        return 0.0
    return float(w / h)


def _corner_angles_ok(pts: np.ndarray, tol_deg: float = 30.0) -> tuple[float, float]:
    """Fraction of corners within 90°±tol, and the max deviation in degrees."""
    pts = pts.astype(np.float64)
    n = len(pts)
    ok, max_dev = 0, 0.0
    for i in range(n):
        a = pts[i] - pts[(i - 1) % n]
        b = pts[(i + 1) % n] - pts[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la == 0 or lb == 0:
            continue
        cosang = float(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0))
        dev = abs(90.0 - np.degrees(np.arccos(cosang)))
        max_dev = max(max_dev, dev)
        if dev <= tol_deg:
            ok += 1
    return ok / n, max_dev


def _edge_support(pts: np.ndarray, edge_mask: np.ndarray) -> float:
    """Fraction of sampled perimeter points with an edge pixel within 3 px."""
    dil = cv2.dilate(edge_mask, np.ones((7, 7), np.uint8), iterations=1)
    h, w = dil.shape
    pts = np.vstack([pts, pts[:1]])
    total = hits = 0
    for a, b in zip(pts[:-1], pts[1:]):
        n = 48
        for t in np.linspace(0, 1, n):
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t))
            if 0 <= x < w and 0 <= y < h:
                total += 1
                hits += int(dil[y, x] > 0)
    return hits / total if total else 0.0


def _aspect_score(aspect: float, expected: float | None) -> float | None:
    if expected is None:
        return None
    if aspect <= 0:
        return 0.0
    return float(np.exp(-0.5 * ((np.log(aspect / expected)) / 0.35) ** 2))


def _score_quad(pts: np.ndarray, frame_area: int, area: float,
                edge_mask: np.ndarray, expected_aspect: float | None) -> dict:
    aspect = _quad_aspect(pts)
    corners_ok, _ = _corner_angles_ok(pts)
    area_frac = area / frame_area
    terms = {
        "area": min(area_frac / 0.35, 1.0),
        "convexity": 1.0,             # by construction (approxPolyDP output is convex)
        "corners": corners_ok,
        "aspect": _aspect_score(aspect, expected_aspect),
        "edge_support": _edge_support(pts, edge_mask),
    }
    weights = {"area": 0.30, "convexity": 0.20, "corners": 0.20,
               "aspect": 0.15, "edge_support": 0.15}
    active = {k: v for k, v in terms.items() if v is not None}
    wsum = sum(weights[k] for k in active)
    score = sum(weights[k] * v for k, v in active.items()) / wsum if wsum else 0.0
    return {"confidence": float(score), "aspect_ratio": aspect,
            "area_fraction": area_frac, "terms": {k: float(v) for k, v in active.items()}}


def _canny_contour_quad(gray: np.ndarray, frame_area: int,
                        expected_aspect: float | None) -> dict:
    """Primary method (§8.2). Returns best quad + scoring, or None."""
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    med = float(np.median(closed))
    lo, hi = int(0.66 * med), int(1.33 * med)
    edges = cv2.Canny(closed, lo, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    min_area = settings.detect_min_area_frac * frame_area
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(np.float64)
        if not (0 <= pts[:, 0].max() < w and 0 <= pts[:, 1].max() < h):
            continue
        try:
            pts = _order_corners(pts)
        except ValueError:
            continue
        scored = _score_quad(pts, frame_area, area, edges, expected_aspect)
        if best is None or scored["confidence"] > best["confidence"]:
            best = {"pts": pts, **scored}
    if best is None:
        return None
    return {"quad": best["pts"], "confidence": best["confidence"],
            "aspect_ratio": best["aspect_ratio"], "area_fraction": best["area_fraction"],
            "terms": best["terms"], "edges": edges, "canny": (lo, hi),
            "gray_for_subpix": blurred}


def _normalize_hough_lines(lines) -> np.ndarray | None:
    """HoughLinesP output shape changed across OpenCV versions:
    cv2 4.x -> (N, 1, 4), cv2 5.x -> (N, 4). Normalize to (N, 4)."""
    if lines is None:
        return None
    arr = np.asarray(lines)
    if arr.ndim == 3 and arr.shape[1:] == (1, 4):
        arr = arr[:, 0, :]
    elif arr.ndim == 2 and arr.shape[1] == 4:
        pass
    elif arr.ndim == 1 and arr.shape[0] == 4:
        arr = arr.reshape(1, 4)
    else:
        return None
    return arr.astype(np.float64)


def _hough_quad(gray: np.ndarray, frame_area: int,
                expected_aspect: float | None) -> dict | None:
    """Fallback 2 (§8.3): cluster Hough lines into 2 horizontal + 2 vertical."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = _normalize_hough_lines(cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=60,
        minLineLength=int(0.25 * min(h, w)), maxLineGap=40))
    if lines is None or len(lines) < 4:
        return None

    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang < 20 or ang > 160:
            horizontals.append((y1 + y2) / 2)          # near-horizontal: line y
        elif 70 < ang < 110:
            verticals.append((x1 + x2) / 2)            # near-vertical: line x
    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    top, bottom = min(horizontals), max(horizontals)
    left, right = min(verticals), max(verticals)
    if bottom - top < 0.1 * h or right - left < 0.1 * w:
        return None
    pts = _order_corners(np.array([[left, top], [right, top],
                                   [right, bottom], [left, bottom]], dtype=np.float64))
    area = (right - left) * (bottom - top)
    scored = _score_quad(pts, frame_area, area, edges, expected_aspect)
    return {"quad": pts, "confidence": scored["confidence"],
            "aspect_ratio": scored["aspect_ratio"], "area_fraction": scored["area_fraction"],
            "terms": scored["terms"], "edges": edges, "canny": None, "gray_for_subpix": gray}


def _threshold_bbox_quad(rgb: np.ndarray, frame_area: int,
                         expected_aspect: float | None) -> dict | None:
    """Fallback 3 (§8.3): Otsu on HSV value -> largest component -> minAreaRect."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, otsu = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Bright sheet on dark desk could be either polarity; take the larger of the two.
    n_light = int((otsu > 0).sum())
    if n_light < frame_area / 2:
        otsu = cv2.bitwise_not(otsu)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 0.15 * frame_area:
        return None
    mask = (labels == idx).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    pts = _order_corners(cv2.boxPoints(rect))
    area = cv2.contourArea(c)
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 50, 150)
    scored = _score_quad(pts, frame_area, area, edges, expected_aspect)
    return {"quad": pts, "confidence": scored["confidence"],
            "aspect_ratio": scored["aspect_ratio"], "area_fraction": scored["area_fraction"],
            "terms": scored["terms"], "edges": edges, "canny": None,
            "gray_for_subpix": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)}


def _default_inset(w: int, h: int) -> np.ndarray:
    """Fallback 4 (§8.3): 5% inset rectangle, presented for manual dragging."""
    m = 0.05
    return _order_corners(np.array(
        [[m * w, m * h], [(1 - m) * w, m * h],
         [(1 - m) * w, (1 - m) * h], [m * w, (1 - m) * h]], dtype=np.float64))


def detect_quad(rgb_oriented: np.ndarray) -> DetectionResult:
    """Detect the ECG sheet quad in the oriented RGB grid.

    All returned pixel coordinates are in the ORIENTED grid (the same space the
    rectification stage consumes).
    """
    h, w = rgb_oriented.shape[:2]
    frame_area = h * w
    scale = settings.detect_downscale_to / max(h, w)
    if scale < 1.0:
        work = cv2.resize(rgb_oriented, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    else:
        work, scale = rgb_oriented, 1.0
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    wh, ww = gray.shape

    expected_aspect = settings.detect_expected_aspect
    params = {"downscale_to": settings.detect_downscale_to, "expected_aspect": expected_aspect}

    import logging
    log = logging.getLogger("cardiosentry.detect")
    for method, fn, needs_rgb in (
        ("canny_contour", _canny_contour_quad, False),
        ("hough_lines", _hough_quad, False),
        ("threshold_bbox", _threshold_bbox_quad, True),
    ):
        try:
            args = (work,) if needs_rgb else (gray,)
            found = fn(*args, frame_area=wh * ww, expected_aspect=expected_aspect)
            if found is None:
                continue
            # Acceptance gate for the weaker methods: a quad with essentially
            # no edge support or a sub-warn composite score is worse than an
            # honest "not found" — the user drags corners either way, but a
            # hallucinated quad wastes their time twice (plan §8.3, §8.4).
            if method in ("hough_lines", "threshold_bbox"):
                edge_support = found["terms"].get("edge_support", 0.0)
                if edge_support < 0.05 or found["confidence"] < 0.45:
                    continue
            pts = found["quad"]
            # Sub-pixel refinement on the working-scale grayscale (§8.2 step 12).
            try:
                refined = cv2.cornerSubPix(
                    found["gray_for_subpix"], pts.astype(np.float32).reshape(-1, 1, 2),
                    (7, 7), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
                refined = np.clip(refined.reshape(4, 2), 0, [ww - 1, wh - 1])
                if cv2.contourArea(refined.astype(np.float32)) > 0.01 * wh * ww:
                    pts = refined
            except Exception:
                pass                        # refinement is a nicety, never a gate
            pts = _order_corners(np.asarray(pts, dtype=np.float64))
            # Back to the oriented grid's pixel space.
            px = np.clip(pts / scale, 0, [w - 1, h - 1])
            if found["canny"] is not None:
                params["canny_lo"], params["canny_hi"] = found["canny"]
            params["approx_eps_frac"] = 0.02
            quad_px = tuple(tuple(map(float, p)) for p in px)
            quad_norm = tuple((x / w, y / h) for x, y in quad_px)
            return DetectionResult(
                success=True, method=method, confidence=float(found["confidence"]),
                quad_px=quad_px, quad_norm=quad_norm,
                area_fraction=float(found["area_fraction"]),
                aspect_ratio=float(found["aspect_ratio"]),
                params=params, score_breakdown=found["terms"])
        except Exception as exc:
            # A broken fallback must never fail the whole upload (plan §16 #4:
            # detection failure is not an error — the user corrects by hand).
            # Log it and drop to the next rung.
            log.warning("detection method %s raised: %s", method, exc)
            continue

    # Fallback 4: default inset rectangle. Not an error (§16 case 4).
    pts = _default_inset(w, h)
    quad_px = tuple(tuple(map(float, p)) for p in pts)
    quad_norm = tuple((x / w, y / h) for x, y in quad_px)
    return DetectionResult(
        success=False, method="none", confidence=0.0,
        quad_px=quad_px, quad_norm=quad_norm,
        area_fraction=None, aspect_ratio=None,
        params=params, score_breakdown={})
