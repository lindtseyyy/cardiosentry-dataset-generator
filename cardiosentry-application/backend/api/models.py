"""GET /api/models — discovered descriptors + which is active (§7.1)."""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.api.state import get_state

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models(request: Request):
    st = get_state(request)
    out = []
    for mid, ld in sorted(st.descriptors.items()):
        d = ld.descriptor
        out.append({
            "id": d.id,
            "display_name": d.display_name,
            "version": d.version,
            "notes": d.notes,
            "labels": d.served_labels,
            "model_labels": d.labels,
            "display_names": d.served_display_names,
            "suppressed_labels": d.output.suppressed_labels,
            "views": [list(v) for v in (d.architecture.view_sizes or [])],
            "call_style": d.architecture.call_style,
            "input_size": list(d.preprocess.input_hw),      # [height, width]
            "checkpoint_sha256_short": ld.checkpoint_sha256[:8],
            "weights": d.weights,
            "active": mid == st.active_model.id,
            "loaded": mid in st._models_cache,
        })
    return {"models": out, "active_id": st.active_model.id}


@router.get("/models/active")
def active_model(request: Request):
    st = get_state(request)
    return st.active_model.run_info().model_dump(mode="json")
