"""Detection tests on the synthetic photo fixtures (plan §8)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.services.detect import detect_quad


def _load_rgb(fixtures, name: str) -> np.ndarray:
    img = cv2.imread(str(fixtures / name))
    assert img is not None, f"fixture {name} missing — run tests/make_fixtures.py"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _quad_area_frac(quad_px, img_w, img_h):
    return cv2.contourArea(np.asarray(quad_px, dtype=np.float32)) / (img_w * img_h)


def test_detects_clean_photo(fixtures):
    rgb = _load_rgb(fixtures, "photo_clean.jpg")
    h, w = rgb.shape[:2]
    det = detect_quad(rgb)
    assert det.success, f"clean photo should be detected, got method={det.method}"
    assert det.method in ("canny_contour", "hough_lines", "threshold_bbox")
    assert det.quad_px is not None
    # The sheet occupies ~80% of the frame in the clean fixture.
    assert _quad_area_frac(det.quad_px, w, h) > 0.5
    assert det.area_fraction is not None and det.area_fraction > 0.5
    assert det.confidence > 0.5


def test_corner_order_is_tl_tr_br_bl(fixtures):
    rgb = _load_rgb(fixtures, "photo_clean.jpg")
    det = detect_quad(rgb)
    tl, tr, br, bl = det.quad_px
    assert tl[0] < tr[0] and bl[0] < br[0]     # left corners left of right corners
    assert tl[1] < bl[1] and tr[1] < br[1]     # top corners above bottom corners
    assert tl[0] + tl[1] <= br[0] + br[1] + 1e-6


def test_tilted_photo_still_detected(fixtures):
    rgb = _load_rgb(fixtures, "photo_tilted.jpg")
    h, w = rgb.shape[:2]
    det = detect_quad(rgb)
    # Tilt may or may not be detected automatically — but the result must
    # always be usable: corners inside the frame, quad convex.
    pts = np.asarray(det.quad_px, dtype=np.float64)
    assert pts.min() >= 0 and pts[:, 0].max() < w and pts[:, 1].max() < h
    if det.success:
        assert cv2.isContourConvex(pts.astype(np.float32))
        assert _quad_area_frac(det.quad_px, w, h) > 0.3


def test_dark_photo_warned_not_blocked(fixtures):
    from backend.services import quality as qual
    rgb = _load_rgb(fixtures, "photo_dark.jpg")
    det = detect_quad(rgb)
    q = qual.quality_original(rgb, det.quad_px, rgb.shape[1], rgb.shape[0])
    # The dark fixture is ~45% brightness: it must be flagged, and it must NOT
    # be an error (warn, never block — §14).
    assert q.level in ("warn", "poor")
    assert any("brightness" in f for f in q.flags)
    assert isinstance(det.method, str)


def test_no_paper_falls_back_honestly(fixtures):
    rgb = _load_rgb(fixtures, "no_paper.jpg")
    h, w = rgb.shape[:2]
    det = detect_quad(rgb)
    assert det.success is False, "a desk with no paper must not 'find' the sheet"
    assert det.method == "none"
    # The default inset quad is still usable for manual dragging.
    pts = np.asarray(det.quad_px, dtype=np.float64)
    assert pts.min() >= 0 and pts[:, 0].max() < w and pts[:, 1].max() < h
    assert det.confidence == 0.0


def test_hough_lines_shape_normalization():
    """Regression: OpenCV 5 returns HoughLinesP as (N,4); OpenCV 4 as (N,1,4).
    The first shape crashed the hough fallback with a TypeError on real phone
    photos whose borders broke the primary contour method."""
    from backend.services.detect import _normalize_hough_lines
    n4 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)          # cv2 5.x
    n14 = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.int32)     # cv2 4.x
    flat = np.array([1, 2, 3, 4], dtype=np.int32)
    assert _normalize_hough_lines(n4).shape == (2, 4)
    assert _normalize_hough_lines(n14).shape == (2, 4)
    assert _normalize_hough_lines(flat).shape == (1, 4)
    assert _normalize_hough_lines(None) is None
    assert _normalize_hough_lines(np.zeros((2, 3), np.int32)) is None
    np.testing.assert_array_equal(_normalize_hough_lines(n4), n4.astype(float))


def test_hough_quad_runs_with_real_opencv5_output(fixtures):
    """Regression for the field crash: `_hough_quad` itself must consume
    cv2.HoughLinesP's OpenCV-5 (N,4) output without raising — whether or not it
    accepts the quad (the acceptance gate may pass the torch to the next rung)."""
    from backend.services import detect as detmod
    rgb = _load_rgb(fixtures, "photo_clean.jpg")
    h, w = rgb.shape[:2]
    scale = 1024 / max(h, w)
    work = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    found = detmod._hough_quad(gray, frame_area=gray.size, expected_aspect=1.294)
    assert found is None or (
        "quad" in found and np.asarray(found["quad"]).shape == (4, 2))


def test_hough_fallback_reached_without_crashing(fixtures, monkeypatch):
    """With the primary contour method disabled, the ladder must run the hough
    rung (real cv2 output) and land on a usable quad from whichever rung
    succeeds — never an exception."""
    from backend.services import detect as detmod
    monkeypatch.setattr(detmod, "_canny_contour_quad", lambda *a, **k: None)
    rgb = _load_rgb(fixtures, "photo_clean.jpg")
    h, w = rgb.shape[:2]
    det = detect_quad(rgb)
    assert det.success, "some fallback should find the clean sheet"
    assert det.method in ("hough_lines", "threshold_bbox")
    assert _quad_area_frac(det.quad_px, w, h) > 0.5


def test_detect_degrades_gracefully_when_all_methods_broken(fixtures, monkeypatch):
    """A bug in every detection rung must yield the manual-correction quad,
    never an exception (§16 #4)."""
    from backend.services import detect as detmod

    def boom(*a, **k):
        raise RuntimeError("simulated bug")

    for name in ("_canny_contour_quad", "_hough_quad", "_threshold_bbox_quad"):
        monkeypatch.setattr(detmod, name, boom)
    rgb = _load_rgb(fixtures, "photo_clean.jpg")
    h, w = rgb.shape[:2]
    det = detect_quad(rgb)                     # must not raise
    assert det.success is False and det.method == "none"
    pts = np.asarray(det.quad_px, dtype=np.float64)
    assert pts.min() >= 0 and pts[:, 0].max() < w and pts[:, 1].max() < h
