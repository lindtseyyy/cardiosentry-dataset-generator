"""Explainability schemas — the request, the response, and the persisted record.

An explain run is provenance-carrying in exactly the way a prediction run is
(§13.3): it names the model, the checkpoint hash and the sha256 of the pixels it
attributed, so a saved heat map can always be traced to the photograph and the
weights that produced it. A map whose `model_input_sha256` no longer matches the
capture's current model_input.png was made from pixels that have since been
re-rectified, and the record says so rather than the picture quietly lying.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .prediction import ModelRunInfo

Method = Literal["gradcam", "integrated_gradients", "jacobian"]


# ------------------------------------------------------------------ requests
class ExplainRequest(BaseModel):
    method: Method
    model_id: str | None = None            # None = active model
    # None = every label for gradcam/jacobian (both are cheap per extra label),
    # and the POSITIVE labels for integrated_gradients (which is not).
    labels: list[str] | None = None
    target_layer: str | None = None        # gradcam only; None = descriptor/auto
    steps: int | None = None               # IG only
    baseline: Literal["white", "black", "grey", "blur"] | None = None   # IG only
    colormap: str | None = None
    overlay_alpha: float | None = None
    run_id: str | None = None              # prediction run this explains, if any

    @model_validator(mode="after")
    def _check(self):
        if self.labels is not None and not self.labels:
            raise ValueError("labels, when given, must not be empty")
        if self.overlay_alpha is not None and not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be within [0, 1]")
        if self.steps is not None and self.steps < 2:
            raise ValueError("steps must be >= 2")
        return self


# --------------------------------------------------------------- sub-records
class HeaderCheck(BaseModel):
    """Attribution mass inside the printed ID/Age/Sex block (explain.py)."""
    box: list[float]
    box_area_frac: float
    frac: float
    lift: float | None = None
    peak_inside: bool = False


class MapSummary(BaseModel):
    """The map's shape as numbers, so a record is readable without the PNG."""
    empty: bool = False
    centroid_xy: list[float] | None = None      # fractions of (width, height)
    peak_xy: list[float] | None = None
    peak_value: float | None = None
    mass_half_area_frac: float | None = None    # smallest area holding half the mass
    positive_frac: float | None = None          # signed maps only


class LabelAttribution(BaseModel):
    label: str
    display_name: str
    signed: bool                                # IG is signed; the others are not
    score: float | None = None                  # this label's sigmoid score, if known
    threshold: float | None = None
    positive: bool | None = None
    stats: dict = Field(default_factory=dict)   # method-specific (completeness…)
    summary: MapSummary
    header: HeaderCheck | None = None
    normalization: dict = Field(default_factory=dict)
    per_view: dict[str, list[float]] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)   # heatmap | overlay URLs


# ------------------------------------------------------------------ response
class ExplainResponse(BaseModel):
    explain_id: str
    capture_id: str
    method: Method
    model: ModelRunInfo
    run_id: str | None = None
    labels: list[str]
    attributions: list[LabelAttribution]
    meta: dict = Field(default_factory=dict)         # target_layer, steps, baseline…
    extras: dict = Field(default_factory=dict)       # jacobian coupling matrix
    timing_ms: dict[str, float] = Field(default_factory=dict)
    created_utc: str
    model_input_sha256: str
    caveats: list[str] = Field(default_factory=list)


class ExplainRecord(ExplainResponse):
    """Persisted explain/<explain_id>/record.json — the response plus versions."""
    schema_version: int = 1
    env: dict = Field(default_factory=dict)


class ExplainSummary(BaseModel):
    """One row of GET /api/captures/{id}/explain."""
    explain_id: str
    method: Method
    model_id: str
    created_utc: str
    labels: list[str]
    run_id: str | None = None
    stale: bool = False              # attributed pixels != the capture's current ones
    max_header_frac: float | None = None
    thumbnail_url: str | None = None
