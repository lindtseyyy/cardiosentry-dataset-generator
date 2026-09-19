"""Explainability endpoints — where the model looked (backend/services/explain.py).

    POST   /api/captures/{id}/explain                 run one method -> maps
    GET    /api/captures/{id}/explain                 this capture's runs
    GET    /api/captures/{id}/explain/{xid}           one run's full record
    GET    /api/captures/{id}/explain/{xid}/files/{n} the heat maps and overlays
    DELETE /api/captures/{id}/explain/{xid}           remove one run
    GET    /api/models/{model_id}/layers              Grad-CAM target candidates

A separate call from `predict`, for the same reason rectify and predict are
separate (§7.1): attribution costs whole model passes — minutes for Integrated
Gradients on CPU — and making every prediction pay that would turn a two-second
capture into a coffee break. It also means a map is attached to STORED pixels,
so the same photograph can be explained by a second method, or by a second
model, and the two are directly comparable.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from backend.api.state import get_state
from backend.errors import ApiError, NotFound
from backend.schemas.explain import (ExplainRecord, ExplainRequest,
                                     ExplainResponse, ExplainSummary,
                                     LabelAttribution)
from backend.schemas.capture import CaptureRecord
from backend.services import explain as xai
from backend.services import storage
from backend.settings import settings

router = APIRouter(prefix="/captures", tags=["explain"])
models_router = APIRouter(tags=["explain"])


def _capture_dir(capture_id: str) -> Path:
    try:
        return storage.capture_dir(capture_id)
    except KeyError:
        raise NotFound("CAPTURE_NOT_FOUND", f"no capture {capture_id}")


def _latest_prediction(d: Path, record: dict, model_id: str) -> dict | None:
    """The newest prediction run on this capture from this model, if any.

    Used only to decorate the maps with the score they explain: an attribution
    means something different next to 0.97 than next to 0.11, and a heat map
    shown without its score invites reading confidence into colour intensity.
    """
    for run_id in reversed(record.get("predictions", [])):
        p = d / "predictions" / f"{run_id}.json"
        if not p.is_file():
            continue
        run = json.loads(p.read_text(encoding="utf-8"))
        if run.get("model", {}).get("id") == model_id:
            return run
    return None


def _slug(label: str, index: int) -> str:
    """A label -> a filename-safe token.

    Labels are descriptor data and nothing constrains their characters, but the
    files they name are served through `storage.EXPLAIN_FILE_RE`, which accepts
    `[A-Za-z0-9_.-]` only. Writing `ST elevation_heatmap.png` would succeed and
    then 404 forever — a broken image with no error anywhere. Slugging keeps the
    real label in the record and the safe one on disk; the index disambiguates
    two labels that slug the same.
    """
    safe = "".join(c if (c.isascii() and (c.isalnum() or c in "._-")) else "-"
                   for c in label).strip("-.") or "label"
    return f"{safe}" if safe == label else f"{safe}{index}"


def _prediction_by_id(d: Path, run_id: str) -> dict:
    p = d / "predictions" / f"{run_id}.json"
    if not p.is_file():
        raise NotFound("RUN_NOT_FOUND", f"no prediction run {run_id} on this capture")
    return json.loads(p.read_text(encoding="utf-8"))


def _caveats(lm, method: str, attributions: list[LabelAttribution],
             meta: dict) -> list[str]:
    """What a reader must know before believing the picture.

    These are generated from the run's OWN numbers rather than being static
    boilerplate, so a caveat appearing is information: it means this map, not
    attribution in general, has the problem named.
    """
    out = [
        "An attribution map shows where the input mattered to THIS model's "
        "score. It is not a clinical finding and it is not evidence the model "
        "is right — a confidently wrong score has a heat map too.",
    ]
    ex = lm.descriptor.explain
    if method == "gradcam":
        cell = meta.get("cell_px")
        if cell:
            out.append(
                f"Grad-CAM resolves to the {meta['feature_grid'][0]}x"
                f"{meta['feature_grid'][1]} feature grid — one cell is about "
                f"{cell[0]:.0f}x{cell[1]:.0f} input pixels. It can point at a "
                "lead; it cannot point at a segment within a beat.")
    if method == "integrated_gradients":
        worst = max((a.stats.get("completeness_delta") or 0.0)
                    for a in attributions) if attributions else 0.0
        if worst > 0.2:
            out.append(
                f"Completeness residual is {worst:.0%} at {meta.get('steps')} "
                "steps: the attributions do NOT yet sum to the logit "
                "difference they are meant to explain. Re-run with more steps "
                "before reading this map closely.")
        out.append(
            f"Attribution is measured against a {meta.get('baseline')} baseline. "
            "The map answers 'what did this photograph add relative to that "
            "reference', and a different baseline gives a different answer.")
    if method == "jacobian":
        sat = [a.label for a in attributions if a.stats.get("saturated")]
        if sat:
            out.append(
                "Near-zero gradient for " + ", ".join(sat) + ": the logit is "
                "saturated, so its row of the Jacobian is numerical noise and "
                "its map should not be read.")
        out.append(
            "The Jacobian is a LOCAL, first-order picture — it describes the "
            "model in a small neighbourhood of this one photograph only.")

    hot = [a for a in attributions
           if a.header and a.header.lift is not None and a.header.lift >= 2.0]
    if hot:
        msg = ("Attribution concentrates on the printed ID/Age/Sex header for "
               + ", ".join(f"{a.label} (x{a.header.lift:.1f} its area share)"
                           for a in hot) + ". ")
        if ex.header_redacted_in_training:
            msg += ("This checkpoint trained on a corpus with that block "
                    "painted out, so it never saw printed text there — and the "
                    "record ID separates STEMI exactly in that corpus. Treat "
                    "this score as contaminated until the crop is tested.")
        else:
            msg += ("The record ID printed there separates STEMI exactly in "
                    "the training corpus, so header attribution is the "
                    "shortcut, not the finding.")
        out.append(msg)
    return out


# ------------------------------------------------------------------- explain
@router.post("/{capture_id}/explain", status_code=201)
def create_explanation(capture_id: str, body: ExplainRequest, request: Request):
    if not settings.explain_enabled:
        raise ApiError(503, "EXPLAIN_DISABLED",
                       "Explainability is switched off on this server "
                       "(CARDIOSENTRY_EXPLAIN_ENABLED=false).")
    st = get_state(request)
    d = _capture_dir(capture_id)
    record = storage.load_capture_json(d)

    model_input = d / "model_input.png"
    if not model_input.is_file():
        raise NotFound("MISSING_MODEL_INPUT",
                       "model_input.png not found — rectify this capture first.")

    lm = st.model(body.model_id)
    desc = lm.descriptor
    if not desc.explain.enabled:
        raise ApiError(422, "EXPLAIN_UNSUPPORTED",
                       f"{lm.id}: the descriptor disables explainability for "
                       "this model.")

    if len(storage.iter_explain_runs(capture_id)) >= settings.explain_max_runs_per_capture:
        raise ApiError(422, "TOO_MANY_EXPLAIN_RUNS",
                       f"this capture already has "
                       f"{settings.explain_max_runs_per_capture} explain runs — "
                       "delete some before adding more.")

    pred = (_prediction_by_id(d, body.run_id) if body.run_id
            else _latest_prediction(d, record, lm.id))
    if body.run_id and pred.get("model", {}).get("id") != lm.id:
        raise ApiError(422, "RUN_MODEL_MISMATCH",
                       f"run {body.run_id} was scored by "
                       f"{pred.get('model', {}).get('id')!r}, not {lm.id!r} — "
                       "explain the model that produced the score.")

    labels = _resolve_labels(lm, body, pred)
    steps = _resolve_steps(body)

    out, ms = xai.run(lm, model_input, body.method, labels,
                      target_layer=body.target_layer or desc.explain.target_layer,
                      steps=steps, baseline=body.baseline)

    # ---- render + persist ------------------------------------------------
    colormap = body.colormap or settings.explain_colormap
    alpha = (body.overlay_alpha if body.overlay_alpha is not None
             else settings.explain_overlay_alpha)
    base_rgb = cv2.cvtColor(cv2.imread(str(model_input)), cv2.COLOR_BGR2RGB)

    xdir, explain_id = storage.new_explain_dir(capture_id, body.method, lm.id)
    base_url = f"/api/captures/{capture_id}/explain/{explain_id}/files"
    attributions: list[LabelAttribution] = []
    raw: dict[str, np.ndarray] = {}
    try:
        for i, m in enumerate(out.maps):
            norm, norm_info = xai.normalize_map(
                m.data, m.signed, settings.explain_percentile_clip)
            heat = xai.colorize(norm, m.signed, colormap)
            over = xai.overlay(base_rgb, heat, norm, m.signed, alpha)
            slug = _slug(m.label, i)
            hname = f"{body.method}_{slug}_heatmap.png"
            oname = f"{body.method}_{slug}_overlay.png"
            (xdir / hname).write_bytes(xai.png_bytes(heat))
            (xdir / oname).write_bytes(xai.png_bytes(over))
            if settings.explain_save_raw:
                raw[m.label] = m.data.astype(np.float16)

            scores = (pred or {}).get("scores", {})
            thr = (pred or {}).get("thresholds", {})
            attributions.append(LabelAttribution(
                label=m.label,
                display_name=desc.display_names.get(m.label, m.label),
                signed=m.signed,
                score=scores.get(m.label),
                threshold=thr.get(m.label),
                positive=(m.label in (pred or {}).get("positive", [])
                          if pred else None),
                stats=m.stats,
                summary=xai.map_summary(m.data, m.signed),
                header=xai.box_fraction(m.data, desc.explain.header_box),
                normalization={**norm_info, "colormap":
                               ("diverging_bwr" if m.signed else colormap),
                               "overlay_alpha": alpha},
                per_view=m.per_view,
                files={"heatmap": f"{base_url}/{hname}",
                       "overlay": f"{base_url}/{oname}"}))
        if raw:
            np.savez_compressed(xdir / "raw.npz", **raw)

        created = storage.utcnow()
        resp = ExplainResponse(
            explain_id=explain_id, capture_id=capture_id, method=body.method,
            model=lm.run_info(), run_id=body.run_id or (pred or {}).get("run_id"),
            labels=labels, attributions=attributions,
            meta={**out.meta,
                  "estimated_passes": xai.estimate_passes(
                      body.method, len(labels), steps or 0),
                  "header_box": desc.explain.header_box,
                  "header_redacted_in_training":
                      desc.explain.header_redacted_in_training,
                  "raw_saved": bool(raw)},
            extras=out.extras,
            timing_ms={"attribute": round(ms, 1)},
            created_utc=created,
            model_input_sha256=_sha256(model_input),
            caveats=_caveats(lm, body.method, attributions, out.meta))
        rec = ExplainRecord(**resp.model_dump(), env=storage.app_env_record())
        storage.atomic_write_json(xdir / "record.json", rec.model_dump(mode="json"))
    except Exception:
        shutil.rmtree(xdir, ignore_errors=True)     # never leave a half-run behind
        raise

    record["explanations"] = list(dict.fromkeys(
        record.get("explanations", []) + [explain_id]))
    storage.save_capture_record(d, CaptureRecord.model_validate(record))
    return resp.model_dump(mode="json")


def _resolve_labels(lm, body: ExplainRequest, pred: dict | None) -> list[str]:
    """Which labels to attribute, and why the default differs by method.

    Grad-CAM and the Jacobian pay almost nothing per extra label — a partial
    backward and a full backward respectively — so both default to the whole
    head, which is what makes the four maps comparable side by side. Integrated
    Gradients pays `steps` full backwards PER label, so defaulting it to four
    labels would quadruple a run that is already minutes long: it defaults to
    what the model actually flagged, and to the top-scoring label when the model
    flagged nothing.
    """
    all_labels = lm.descriptor.served_labels
    if body.labels is not None:
        unknown = [l for l in body.labels if l not in all_labels]
        if unknown:
            raise ApiError(422, "UNKNOWN_LABEL",
                           f"unknown label(s) {', '.join(unknown)} — this model "
                           f"scores {', '.join(all_labels)}")
        return [l for l in all_labels if l in set(body.labels)]
    if body.method != "integrated_gradients":
        return list(all_labels)
    positive = [l for l in all_labels if l in (pred or {}).get("positive", [])]
    if positive:
        return positive
    scores = (pred or {}).get("scores")
    if scores:
        return [max(all_labels, key=lambda l: scores.get(l, 0.0))]
    return [all_labels[0]]


def _resolve_steps(body: ExplainRequest) -> int | None:
    if body.method != "integrated_gradients":
        return None
    steps = body.steps or settings.explain_ig_steps
    if steps > settings.explain_ig_max_steps:
        raise ApiError(422, "STEPS_OUT_OF_RANGE",
                       f"steps must be <= {settings.explain_ig_max_steps} "
                       "(each one is a full forward and backward on CPU)")
    return steps


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- reads
@router.get("/{capture_id}/explain")
def list_explanations(capture_id: str, request: Request):
    d = _capture_dir(capture_id)
    current = _sha256(d / "model_input.png") if (d / "model_input.png").is_file() else None
    out: list[dict] = []
    for xd in storage.iter_explain_runs(capture_id):
        rp = xd / "record.json"
        if not rp.is_file():
            continue
        rec = json.loads(rp.read_text(encoding="utf-8"))
        fracs = [a.get("header", {}).get("frac") for a in rec.get("attributions", [])
                 if a.get("header")]
        first = (rec.get("attributions") or [{}])[0].get("files", {})
        out.append(ExplainSummary(
            explain_id=rec["explain_id"], method=rec["method"],
            model_id=rec["model"]["id"], created_utc=rec["created_utc"],
            labels=rec.get("labels", []), run_id=rec.get("run_id"),
            # The pixels moved under the map: this run explains a model_input
            # this capture no longer has (§16 — a re-rectify invalidates it).
            stale=bool(current and rec.get("model_input_sha256") != current),
            max_header_frac=max(fracs) if fracs else None,
            thumbnail_url=first.get("overlay"),
        ).model_dump(mode="json"))
    return {"capture_id": capture_id, "explanations": out, "total": len(out)}


@router.get("/{capture_id}/explain/{explain_id}")
def get_explanation(capture_id: str, explain_id: str, request: Request):
    _capture_dir(capture_id)
    try:
        xd = storage.explain_dir(capture_id, explain_id)
    except KeyError:
        raise NotFound("EXPLAIN_NOT_FOUND", f"no explain run {explain_id}")
    rp = xd / "record.json"
    if not rp.is_file():
        raise NotFound("EXPLAIN_NOT_FOUND", "explain run has no record.json")
    return json.loads(rp.read_text(encoding="utf-8"))


@router.get("/{capture_id}/explain/{explain_id}/files/{name}")
def get_explain_file(capture_id: str, explain_id: str, name: str, request: Request):
    _capture_dir(capture_id)
    if not storage.EXPLAIN_FILE_RE.fullmatch(name):
        raise NotFound("UNKNOWN_FILE",
                       "explain files are <method>_<LABEL>_heatmap.png, "
                       "<method>_<LABEL>_overlay.png, record.json or raw.npz")
    try:
        xd = storage.explain_dir(capture_id, explain_id)
    except KeyError:
        raise NotFound("EXPLAIN_NOT_FOUND", f"no explain run {explain_id}")
    p = (xd / name).resolve()
    if not p.is_file() or xd.resolve() != p.parent:
        raise NotFound("UNKNOWN_FILE", "no such file")
    media = {"png": "image/png", "json": "application/json",
             "npz": "application/octet-stream"}[name.rsplit(".", 1)[1]]
    return FileResponse(p, media_type=media)


@router.delete("/{capture_id}/explain/{explain_id}")
def delete_explanation(capture_id: str, explain_id: str, request: Request):
    d = _capture_dir(capture_id)
    try:
        xd = storage.explain_dir(capture_id, explain_id)
    except KeyError:
        raise NotFound("EXPLAIN_NOT_FOUND", f"no explain run {explain_id}")
    # Attributions are derived data — every one of them is reproducible from
    # model_input.png plus the record's parameters — so unlike a capture these
    # are removed outright rather than moved to _trash (§17.8).
    shutil.rmtree(xd)
    record = storage.load_capture_json(d)
    record["explanations"] = [e for e in record.get("explanations", [])
                              if e != explain_id]
    storage.save_capture_record(d, CaptureRecord.model_validate(record))
    return {"capture_id": capture_id, "explain_id": explain_id, "deleted": True}


# -------------------------------------------------------------------- layers
@models_router.get("/models/{model_id}/layers")
def list_layers(model_id: str, request: Request):
    """Grad-CAM target candidates for one model, discovered by probe forward.

    Loads the model if it is not already resident, which is the same cost a
    first prediction against it pays.
    """
    st = get_state(request)
    lm = st.model(model_id if model_id != "active" else None)
    cands = xai.probe_layers(lm)
    auto = xai.auto_target_layer(lm)
    h, w = lm.descriptor.preprocess.input_hw
    return {
        "model_id": lm.id,
        "auto_target_layer": auto.name,
        "descriptor_target_layer": lm.descriptor.explain.target_layer,
        "input_size": [h, w],
        "layers": [
            {"name": c.name, "type": c.module_type, "channels": c.channels,
             "grid": list(c.grid), "order": c.order,
             "cell_px": [round(h / c.grid[0], 1), round(w / c.grid[1], 1)],
             "auto": c.name == auto.name}
            for c in cands
        ],
    }
