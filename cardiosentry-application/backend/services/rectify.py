"""Rectification: homography warp of the full-resolution original to a fixed
canonical canvas (plan §8.5).

THE GEOMETRY DECISION THAT MATTERS MOST. Training resampled the 1686x1311 sheet
to a fixed model input — a *specific anisotropic scaling*, and a different one
per geometry: the round-2 square was 768x768 (x: 0.456, y: 0.586), the round-3
`nat768` rectangle is 768x1024 (x: 1024/1686 = 0.607, y: 768/1311 = 0.586). The
model learned ECG features under exactly its own anisotropy, and LVH criteria
are amplitude-in-millimetres criteria. Warping every capture to the fixed
1650x1275 canvas (+18 px margin) keeps that anisotropy identical for every
capture and identical to training, whichever model is active.
`geometry_mode: native_aspect` exists to make this assumption falsifiable, and
the mode is recorded per capture (§17.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.errors import ApiError
from backend.services import preprocess as prep
from backend.settings import settings


@dataclass
class RectifyResult:
    corrected: np.ndarray          # RGB, (1311, 1686, 3) for training_canvas
    corrected_px: tuple[int, int]
    homography: list[list[float]]
    paper_fill_rgb: list[int]
    warp_interpolation: str
    canvas_px: tuple[int, int]
    margin_px: int
    est_px_per_mm_original: float | None
    scale_factor_to_input: float        # horizontal: model-input width / corrected width


def _paper_colour(sheet: np.ndarray) -> np.ndarray:
    """Median of the sheet's four corner patches — the same rule
    `image_pipeline/stage4_augment.py::_paper_colour` used, so the margin fill
    matches the training pipeline's."""
    h, w = sheet.shape[:2]
    k = max(8, min(h, w) // 40)
    patches = np.concatenate([
        sheet[:k, :k].reshape(-1, 3), sheet[:k, -k:].reshape(-1, 3),
        sheet[-k:, :k].reshape(-1, 3), sheet[-k:, -k:].reshape(-1, 3),
    ])
    return np.median(patches, axis=0)


def estimate_aspect_from_quad(pts: np.ndarray) -> float:
    """Recover width/height of the photographed rectangle from its quad.

    Zhang-style homography decomposition: with H mapping the unit square to the
    quad, the aspect ratio is |H^-1 · col1| / |H^-1 · col0| (the world
    rectangle [0,w]x[0,h] is mapped to the unit square by S∘H^-1 with
    S = diag(1/w, 1/h, 1)).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    H = cv2.getPerspectiveTransform(unit, pts)
    Hi = np.linalg.inv(H)
    w = float(np.linalg.norm(Hi @ np.array([1.0, 0.0, 0.0])))
    h = float(np.linalg.norm(Hi @ np.array([0.0, 1.0, 0.0])))
    if w <= 0:
        return 1.0
    return h / w


def validate_quad(quad_px: np.ndarray, img_w: int, img_h: int) -> None:
    """Reject degenerate user quads (§16 case 6)."""
    quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    if not (np.all(quad >= 0) and np.all(quad[:, 0] <= img_w - 1)
            and np.all(quad[:, 1] <= img_h - 1)):
        raise ApiError(422, "INVALID_QUAD",
                       "Corners must lie inside the image.")
    if not cv2.isContourConvex(quad.astype(np.float32)):
        raise ApiError(422, "INVALID_QUAD",
                       "Those corners cross over — reset and try again.")
    if cv2.contourArea(quad.astype(np.float32)) < 0.02 * img_w * img_h:
        raise ApiError(422, "INVALID_QUAD",
                       "The selected quadrilateral is too small.")


def rectify(rgb_oriented: np.ndarray, quad_px: np.ndarray,
            geometry_mode: str = "training_canvas",
            input_size=None) -> RectifyResult:
    """Warp the oriented original to the canonical canvas + paper margin.

    quad_px: (4,2) float, order TL,TR,BR,BL, in oriented-grid pixels.
    """
    img_h, img_w = rgb_oriented.shape[:2]
    validate_quad(quad_px, img_w, img_h)
    quad = np.asarray(quad_px, dtype=np.float32).reshape(4, 2)
    # Width only: the reported scale factor is the horizontal one, and the
    # canvas is what fixes the vertical (see the module docstring).
    input_w = prep.as_hw(input_size or settings.input_size)[1]

    if geometry_mode == "training_canvas":
        cw, ch = int(settings.canvas_px[0]), int(settings.canvas_px[1])
    elif geometry_mode == "native_aspect":
        r = float(estimate_aspect_from_quad(quad.astype(np.float64)))
        cw = int(settings.canvas_px[0])
        ch = max(1, int(round(cw / r)))
    else:
        raise ApiError(422, "INVALID_GEOMETRY_MODE",
                       f"unknown geometry_mode {geometry_mode!r}")

    dst = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)

    flags = cv2.INTER_CUBIC
    warped = cv2.warpPerspective(rgb_oriented, H, (cw, ch), flags=flags,
                                 borderMode=cv2.BORDER_REPLICATE)

    margin = int(settings.margin_px)
    fill = _paper_colour(warped)
    corrected = cv2.copyMakeBorder(
        warped, margin, margin, margin, margin,
        borderType=cv2.BORDER_CONSTANT,
        value=tuple(int(round(c)) for c in fill))

    # Top edge of the quad in original pixels / physical sheet width.
    top_edge_px = float(np.linalg.norm(quad_px[1] - quad_px[0]))
    est_px_per_mm = (top_edge_px / settings.paper_width_mm
                     if settings.paper_width_mm > 0 else None)

    ch_f, cw_f = corrected.shape[:2]
    return RectifyResult(
        corrected=corrected,
        corrected_px=(cw_f, ch_f),
        homography=H.tolist(),
        paper_fill_rgb=[int(round(c)) for c in fill],
        warp_interpolation="INTER_CUBIC",
        canvas_px=(cw, ch),
        margin_px=margin,
        est_px_per_mm_original=float(est_px_per_mm) if est_px_per_mm else None,
        scale_factor_to_input=input_w / cw_f,
    )
