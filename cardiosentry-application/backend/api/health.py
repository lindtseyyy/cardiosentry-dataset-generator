"""GET /api/health — liveness + model status (§7.1)."""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.api.state import get_state
from backend.services.storage import free_space_gb
from backend.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    st = get_state(request)
    import torch
    lm = st.active_model
    return {
        "status": "ok",
        "model_loaded": lm is not None,
        "active_model": lm.id if lm else None,
        "device": settings.device,
        "torch_version": torch.__version__,
        "uptime_s": round(st.uptime_s, 1),
        "free_space_gb": round(free_space_gb(), 1),
        "free_space_warn": free_space_gb() < settings.min_free_gb_warn,
    }
