"""Prediction run schemas — API response and the persisted predictions/*.json."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .model_descriptor import ModelDescriptor


class ModelRunInfo(BaseModel):
    id: str
    display_name: str
    version: str
    checkpoint_sha256: str
    checkpoint_sha256_short: str
    protocol_fingerprint: str | None = None
    epoch: int | None = None
    # (height, width) per view, and for the model input as a whole. Records
    # written before the round-3 rectangular geometry spelled both as bare
    # ints; the validators below promote those so old predictions/*.json still
    # load (§16 #18 — provenance must survive a model swap).
    views: list[list[int]] = Field(default_factory=list)
    input_size: list[int]
    device: str = "cpu"
    dtype: str = "float32"
    amp: bool = False
    torch_num_threads: int | None = None

    @field_validator("input_size", mode="before")
    @classmethod
    def _square_input(cls, v):
        return [v, v] if isinstance(v, int) else v

    @field_validator("views", mode="before")
    @classmethod
    def _square_views(cls, v):
        if isinstance(v, list):
            return [[e, e] if isinstance(e, int) else e for e in v]
        return v


class PredictionResponse(BaseModel):
    run_id: str
    capture_id: str
    model: ModelRunInfo
    labels: list[str]                 # the scoring head's label order
    display_names: dict[str, str]
    scores: dict[str, float]
    logits: dict[str, float]
    thresholds: dict[str, float]
    threshold_source: str            # checkpoint | descriptor | user_override
    positive: list[str]
    timing_ms: dict[str, float]      # preprocess | forward | total
    created_utc: str
    model_input_sha256: str


class PredictionRecord(BaseModel):
    """Persisted predictions/<run_id>.json — the response plus provenance."""
    schema_version: int = 1
    run_id: str
    capture_id: str
    model: ModelRunInfo
    descriptor_snapshot: ModelDescriptor
    scores: dict[str, float]
    logits: dict[str, float]
    thresholds: dict[str, float]
    threshold_source: str
    positive: list[str]
    timing_ms: dict[str, float]
    created_utc: str
    model_input_sha256: str
    env: dict = Field(default_factory=dict)   # torch/torchvision/timm/opencv/pillow versions


class CaptureSummary(BaseModel):
    capture_id: str
    created_utc: str
    sheet_id: str | None = None
    thumbnail_url: str | None = None
    has_correction: bool = False
    latest_model_id: str | None = None
    latest_positive: list[str] = Field(default_factory=list)
    quality_level: str = "ok"
