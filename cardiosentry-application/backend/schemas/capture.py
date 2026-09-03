"""Request/response schemas for captures, and the capture.json record schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CaptureIdPattern = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}_[0-9a-f]{6}$"

Point = tuple[float, float]   # normalized [0,1] in API requests


# ------------------------------------------------------------------ requests
class RectifyRequest(BaseModel):
    quad_norm: list[Point] = Field(min_length=4, max_length=4)
    source: Literal["auto", "manual"] = "auto"
    geometry_mode: Literal["training_canvas", "native_aspect"] = "training_canvas"


class PredictRequest(BaseModel):
    model_id: str | None = None                       # None = active model
    thresholds: dict[str, float] | None = None        # per-label override


class AnnotationRequest(BaseModel):
    reference_labels: dict[str, bool] | None = None
    labels_entered_before_prediction: bool | None = None
    truth_source: str | None = None
    notes: str | None = None
    sheet_id: str | None = None
    device_vendor: str | None = None
    paper_size: str | None = None


# --------------------------------------------------------------- sub-records
class QualityMetrics(BaseModel):
    level: Literal["ok", "warn", "poor"]
    blur_varlap: float | None = None
    brightness: float | None = None
    contrast_p5p95: float | None = None
    clipped_high_frac: float | None = None
    clipped_low_frac: float | None = None
    glare_blob_frac: float | None = None
    est_px_per_mm: float | None = None
    area_fraction: float | None = None
    skew_deg: float | None = None
    grid_period_px: float | None = None
    grid_detected: bool | None = None
    flags: list[str] = Field(default_factory=list)


class DetectionInfo(BaseModel):
    attempted: bool = True
    success: bool = False
    method: str = "none"
    confidence: float = 0.0
    quad_auto_px: list[Point] | None = None
    quad_norm: list[Point] | None = None
    area_fraction: float | None = None
    aspect_ratio: float | None = None
    params: dict = Field(default_factory=dict)
    score_breakdown: dict = Field(default_factory=dict)


class CorrectionInfo(BaseModel):
    quad_final_px: list[Point] | None = None
    quad_auto_px: list[Point] | None = None
    corners_manually_adjusted: bool = False
    mean_corner_shift_px: float | None = None
    source: str = "auto"
    geometry_mode: str = "training_canvas"
    homography: list[list[float]] | None = None
    warp_interpolation: str = "INTER_CUBIC"
    canvas_px: list[int] | None = None
    margin_px: int | None = None
    paper_fill_rgb: list[int] | None = None
    corrected_px: list[int] | None = None
    est_px_per_mm_original: float | None = None
    # Horizontal model-input width / corrected width. Named for a fixed 768
    # until the model input stopped being square; rectify.py has always
    # written `scale_factor_to_input`, which the old name silently dropped.
    scale_factor_to_input: float | None = None


class Annotation(BaseModel):
    reference_labels: dict[str, bool] | None = None
    labels_entered_utc: str | None = None
    labels_entered_before_prediction: bool | None = None
    truth_source: str | None = None
    notes: str = ""
    device_vendor: str = ""
    paper_size: str = ""


class CaptureRecord(BaseModel):
    """The persisted capture.json — everything about one capture, except runs."""
    schema_version: int = 1
    capture_id: str
    created_utc: str
    app: dict = Field(default_factory=dict)
    session: dict = Field(default_factory=dict)     # sheet_id, retake_of, blind_mode
    source: dict = Field(default_factory=dict)
    original: dict = Field(default_factory=dict)
    quality_original: QualityMetrics | None = None
    detection: DetectionInfo | None = None
    correction: CorrectionInfo | None = None
    preprocess: dict | None = None
    quality_model_input: QualityMetrics | None = None
    annotation: Annotation = Field(default_factory=Annotation)
    predictions: list[str] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)   # explain run ids
