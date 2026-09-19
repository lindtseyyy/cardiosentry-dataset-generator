"""Model descriptor schema (models/*.yaml).

A model is data, not code: one `.pt` + one YAML descriptor. These pydantic
models validate the YAML at startup so a malformed descriptor fails loudly with
a readable error rather than at first inference.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class NormalizeSpec(BaseModel):
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]


class PreprocessSpec(BaseModel):
    canvas: list[int] = [1650, 1275]          # warp destination, pre-margin
    margin_px: int = 18
    # Stored model_input.png size. `768` (an int) means the square 768x768 the
    # round-1/round-2 checkpoints were trained at; `[768, 1024]` means a
    # rectangle written in TORCH order — [height, width], the same order as the
    # research code's config.IMG_SIZE and the OPPOSITE of PIL's (w, h). Read
    # `input_hw` rather than this field: it normalizes both spellings.
    input_size: int | list[int] = 768
    resample: Literal["pil_bicubic"] = "pil_bicubic"
    match_training_jpeg: bool = True          # in-memory JPEG q95 round-trip
    jpeg_quality: int = 95
    normalize: NormalizeSpec = Field(default_factory=NormalizeSpec)

    @property
    def input_hw(self) -> tuple[int, int]:
        """(height, width) of the model input, whichever spelling the YAML used."""
        if isinstance(self.input_size, int):
            return (self.input_size, self.input_size)
        return (int(self.input_size[0]), int(self.input_size[1]))

    @model_validator(mode="after")
    def _check(self):
        if len(self.canvas) != 2 or min(self.canvas) <= 0:
            raise ValueError("preprocess.canvas must be [width, height] > 0")
        if isinstance(self.input_size, list):
            if len(self.input_size) != 2 or min(self.input_size) <= 0:
                raise ValueError(
                    "preprocess.input_size must be an int (square) or "
                    "[height, width] > 0 — torch order, not PIL's (w, h)")
        elif self.input_size <= 0:
            raise ValueError("preprocess.input_size must be > 0")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("invalid jpeg_quality")
        if len(self.normalize.mean) != 3 or len(self.normalize.std) != 3:
            raise ValueError("normalize mean/std must have 3 channels")
        return self


class ArchitectureSpec(BaseModel):
    module: str                                # e.g. arch.hybrids_v1
    registry_key: str                          # key in that module's REGISTRY
    pretrained: bool = False                   # weights overwritten anyway
    call_style: Literal["view_dict", "tensor"] = "view_dict"   # F3 adapter
    # Override for the per-view input sizes; default comes from model.views.
    # Each entry is either an int (square view, the round-1/round-2 spelling)
    # or [height, width]. Read `view_sizes` rather than this field.
    views: list[int | list[int]] | None = None

    @property
    def view_sizes(self) -> list[tuple[int, int]] | None:
        """Each declared view as (height, width), or None if not overridden."""
        if self.views is None:
            return None
        return [(v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))
                for v in self.views]

    @model_validator(mode="after")
    def _check_views(self):
        for v in self.views or []:
            if isinstance(v, list) and (len(v) != 2 or min(v) <= 0):
                raise ValueError(
                    "architecture.views entries must be an int (square) or "
                    "[height, width] > 0")
            if isinstance(v, int) and v <= 0:
                raise ValueError("architecture.views entries must be > 0")
        if self.call_style == "view_dict":
            for h, w in self.view_sizes or []:
                if h != w:
                    raise ValueError(
                        "call_style=view_dict keys its input dict by a single "
                        f"integer size, so it cannot carry a {h}x{w} view — "
                        "a non-square model must use call_style=tensor")
        return self


class ThresholdSpec(BaseModel):
    source: Literal["checkpoint", "explicit"] = "checkpoint"
    explicit: list[float] | None = None        # only used when source == explicit


class OutputSpec(BaseModel):
    """Serving policy for a model whose checkpoint head must stay intact.

    A suppressed label is still present in ``ModelDescriptor.labels`` because
    that list is the checkpoint's immutable logit order.  It is omitted from
    API/UI scores, thresholds, positives and explainability until the label is
    removed from this list.  This makes a temporary product decision reversible
    without editing weights or lying about the head that was loaded.
    """
    suppressed_labels: list[str] = Field(default_factory=list)
    reason: str = ""


class ExplainSpec(BaseModel):
    """Explainability defaults for this checkpoint (backend/services/explain.py).

    `target_layer` names the Grad-CAM feature map as a dotted module path
    (e.g. `attn`, or `backbone.layer4`). Leaving it null lets the service pick
    the last spatial feature map the model actually produced, which is the
    right answer for every checkpoint vendored here — set it only when you
    want a DIFFERENT depth than the one the head consumes.

    `header_box` is the printed ID/Age/Sex block as (x0, y0, x1, y1) fractions
    of the model input. It is not used to alter any pixels; it is the region
    every attribution map is measured against, because the record ID printed
    there separates STEMI exactly in the synthetic corpus. A map that puts its
    mass in that box is reporting a shortcut, and `header_redacted_in_training`
    says whether this checkpoint ever saw printed text there — a nat768
    hybrids_v2 run did not (the corpus was redacted), so header attribution on
    a real photograph is doubly suspect for those.
    """
    enabled: bool = True
    target_layer: str | None = None
    header_box: list[float] | None = [0.10, 0.02, 0.30, 0.18]
    header_redacted_in_training: bool = False

    @model_validator(mode="after")
    def _check(self):
        b = self.header_box
        if b is not None:
            if len(b) != 4:
                raise ValueError("explain.header_box must be [x0, y0, x1, y1]")
            if not all(0.0 <= v <= 1.0 for v in b):
                raise ValueError("explain.header_box values are fractions in [0, 1]")
            if b[0] >= b[2] or b[1] >= b[3]:
                raise ValueError("explain.header_box needs x0 < x1 and y0 < y1")
        return self


class VerifySpec(BaseModel):
    """Startup self-check inputs for THIS checkpoint (§10.6).

    Both fields exist because the self-check is a property of the checkpoint,
    not of the application:

    `expected_scores` is the run's own stored score vector for the fixture
    sheet, as a path relative to the app root. Pinning it here is what stops a
    model swap from leaving the check comparing the newly active model against
    another run's numbers — the failure mode is a boot-time "self-check FAILED"
    that means nothing, which is worse than no check at all.

    `score_tolerance` overrides `settings.verify_score_tolerance`. The default
    2e-3 is quoted as the CPU-fp32 vs training-GPU-AMP gap, but it is a bound in
    PROBABILITY space, so what it actually permits depends on where the run's
    scores sit on the sigmoid: at p=0.999 it allows a logit shift of ~2.0, at
    p=0.93 only ~0.03. A checkpoint whose fixture scores are unsaturated
    therefore needs a larger number to express the SAME tolerance in logits.
    Raise it only against a measurement, and record the measurement in `notes`.
    """
    expected_scores: str | None = None         # path relative to the app root
    score_tolerance: float | None = None       # None = settings.verify_score_tolerance

    @model_validator(mode="after")
    def _check(self):
        if self.score_tolerance is not None and not 0.0 < self.score_tolerance < 1.0:
            raise ValueError("verify.score_tolerance is a score delta in (0, 1)")
        return self


class RuntimeSpec(BaseModel):
    device: str = "cpu"
    dtype: str = "float32"
    num_threads: int | None = None             # None = settings.num_threads


class ModelDescriptor(BaseModel):
    schema_version: int = 1
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    display_name: str
    version: str = ""
    notes: str = ""
    weights: str                               # path relative to models/
    expected_sha256: str | None = None         # filled on first load; mismatch = refuse
    expected_protocol_fingerprint: str | None = None
    architecture: ArchitectureSpec
    labels: list[str]
    display_names: dict[str, str] = Field(default_factory=dict)
    thresholds: ThresholdSpec = Field(default_factory=ThresholdSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    preprocess: PreprocessSpec = Field(default_factory=PreprocessSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    explain: ExplainSpec = Field(default_factory=ExplainSpec)
    verify: VerifySpec = Field(default_factory=VerifySpec)

    @property
    def served_labels(self) -> list[str]:
        """Labels exposed by the API, preserving checkpoint head order."""
        suppressed = set(self.output.suppressed_labels)
        return [label for label in self.labels if label not in suppressed]

    @property
    def served_display_names(self) -> dict[str, str]:
        return {label: self.display_names[label] for label in self.served_labels}

    @model_validator(mode="after")
    def _check(self):
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("labels must be unique")
        for l in self.labels:
            if l not in self.display_names:
                self.display_names[l] = l
        if len(self.output.suppressed_labels) != len(set(self.output.suppressed_labels)):
            raise ValueError("output.suppressed_labels must be unique")
        unknown_suppressed = [
            label for label in self.output.suppressed_labels if label not in self.labels
        ]
        if unknown_suppressed:
            raise ValueError(
                "output.suppressed_labels must name checkpoint labels; unknown: "
                + ", ".join(unknown_suppressed))
        if not self.served_labels:
            raise ValueError("output.suppressed_labels cannot suppress every label")
        if self.thresholds.source == "explicit":
            if not self.thresholds.explicit:
                raise ValueError("thresholds.source=explicit requires thresholds.explicit")
            if len(self.thresholds.explicit) != len(self.labels):
                raise ValueError(
                    "thresholds.explicit length must match labels "
                    f"({len(self.labels)})")
            if not all(0.0 <= t <= 1.0 for t in self.thresholds.explicit):
                raise ValueError("thresholds must be in [0, 1]")
        return self
