"""GET /api/config — everything the frontend needs, from the active model.

The frontend holds NO model knowledge: labels, display names, thresholds and
input size all come from here, so swapping the .pt changes the UI with no
rebuild (plan §6.3).
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.api.state import get_state
from backend.services.explain import (IG_BASELINES, METHODS as EXPLAIN_METHODS,
                                      estimate_passes)
from backend.settings import settings

router = APIRouter(tags=["config"])

APP_VERSION = "0.1.0"


@router.get("/config")
def app_config(request: Request):
    st = get_state(request)
    d = st.active_model.descriptor
    return {
        "app": {
            "version": APP_VERSION,
            "git_sha": getattr(request.app.state, "git_sha", None),
        },
        "active_model": {
            "id": d.id,
            "display_name": d.display_name,
            "version": d.version,
            "labels": d.labels,
            "display_names": d.display_names,
            "input_size": list(d.preprocess.input_hw),      # [height, width]
            "views": [list(v) for v in st.active_model.views],
            "checkpoint_sha256_short": st.active_model.checkpoint_sha256[:8],
            "thresholds": dict(zip(d.labels, st.active_model.thresholds)),
            "threshold_source": d.thresholds.source,
        },
        "quality": {
            "block_on_poor": settings.quality_block_on_poor,
            "warn_blur_varlap": settings.warn_blur_varlap,
            "warn_brightness_lo": settings.warn_brightness_lo,
            "warn_brightness_hi": settings.warn_brightness_hi,
            "warn_contrast_p5p95": settings.warn_contrast_p5p95,
            "warn_px_per_mm": settings.warn_px_per_mm,
            "warn_area_fraction": settings.warn_area_fraction,
            "warn_skew_deg": settings.warn_skew_deg,
        },
        "detection": {
            "confidence_warn": settings.detect_confidence_warn,
            "expected_aspect": settings.detect_expected_aspect,
        },
        "rectify": {
            "canvas_px": settings.canvas_px,
            "margin_px": settings.margin_px,
            "geometry_modes": ["training_canvas", "native_aspect"],
            "default_geometry_mode": settings.geometry_mode,
        },
        "explain": {
            "enabled": settings.explain_enabled and d.explain.enabled,
            "methods": list(EXPLAIN_METHODS),
            "default_ig_steps": settings.explain_ig_steps,
            "max_ig_steps": settings.explain_ig_max_steps,
            "default_ig_baseline": settings.explain_ig_baseline,
            "ig_baselines": list(IG_BASELINES),
            "default_colormap": settings.explain_colormap,
            "default_overlay_alpha": settings.explain_overlay_alpha,
            "header_box": d.explain.header_box,
            "header_redacted_in_training": d.explain.header_redacted_in_training,
            # Whole model passes per method, so the UI can warn about the wait
            # BEFORE the user starts a minutes-long Integrated Gradients run
            # rather than after (§XAI cost).
            "cost": {
                m: estimate_passes(m, len(d.labels), settings.explain_ig_steps)
                for m in EXPLAIN_METHODS
            },
        },
        "capture": {
            "max_upload_bytes": settings.max_upload_bytes,
            "blind_mode_available": True,
        },
        "notice": (
            "Research output — not a medical diagnosis. These are scores from "
            "an experimental model evaluated on photographs of ECG paper. They "
            "have not been clinically validated or calibrated and must not be "
            "used for patient care."
        ),
    }
