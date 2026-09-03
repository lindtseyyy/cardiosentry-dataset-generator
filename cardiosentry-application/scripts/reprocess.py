#!/usr/bin/env python3
"""Re-score stored captures with any model — no re-photographing.

    python scripts/reprocess.py --all                       # active model, every capture
    python scripts/reprocess.py --capture 2026-08-23_014215_a3f9c1 --model resnet50_dual
    python scripts/reprocess.py --all --model resnet50_dual --model resnet50_v1

Guarantees:
- predict operates on the PERSISTED model_input.png, so every model scores the
  same pixels (bit-identical input across models at the same size).
- If model_input.png is missing, it is regenerated from original.jpg using the
  RECORDED final quad and geometry mode — never re-detected — and the
  regeneration is verified byte-identical against the stored one when it
  exists.
- Each run writes predictions/<run_id>.json and updates capture.json.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from backend.schemas.capture import CaptureRecord  # noqa: E402
from backend.schemas.prediction import PredictionRecord  # noqa: E402
from backend.services import ingest as ing  # noqa: E402
from backend.services import preprocess as prep  # noqa: E402
from backend.services import rectify as rect  # noqa: E402
from backend.services import runner  # noqa: E402
from backend.services import storage  # noqa: E402
from backend.services.registry import discover, active  # noqa: E402
from backend.services.runner import LoadedModel, load_model  # noqa: E402


def regenerate_model_input(d: Path, record: dict, desc) -> tuple[bytes, bytes]:
    """Rebuild corrected.png + model_input.png from the recorded geometry.

    Returns (model_input_png_bytes, corrected_png_bytes). Raises on failure.
    """
    rgb = ing.decode_record_original(d, record)
    corr = record.get("correction")
    if not corr or not corr.get("quad_final_px"):
        raise RuntimeError(f"{d.name}: no recorded correction — nothing to re-run")
    quad_px = np.asarray(corr["quad_final_px"], dtype=np.float64)
    pre_spec = desc.preprocess
    r = rect.rectify(rgb, quad_px,
                     geometry_mode=corr.get("geometry_mode", "training_canvas"),
                     input_size=pre_spec.input_hw)
    corrected_png = cv2.imencode(".png", cv2.cvtColor(r.corrected, cv2.COLOR_RGB2BGR))[1]
    res = prep.preprocess_corrected(
        r.corrected, input_size=pre_spec.input_hw,
        match_training_jpeg=pre_spec.match_training_jpeg,
        jpeg_quality=pre_spec.jpeg_quality,
        mean=pre_spec.normalize.mean, std=pre_spec.normalize.std)
    return res.model_input_png, corrected_png.tobytes()


def _ensure_model_input(d: Path, record: dict, lm: LoadedModel,
                        force: bool, dry_run: bool) -> Path:
    model_input = d / "model_input.png"
    want_hw = lm.descriptor.preprocess.input_hw
    stored_hw = None
    if model_input.is_file():
        with Image.open(model_input) as im:
            stored_hw = (im.height, im.width)
    if not model_input.is_file():
        print(f"  {d.name}: model_input.png missing — regenerating from recorded geometry")
        mi_bytes, corr_bytes = regenerate_model_input(d, record, lm.descriptor)
        if not dry_run:
            model_input.write_bytes(mi_bytes)
            (d / "corrected.png").write_bytes(corr_bytes)
    elif stored_hw != want_hw:
        # The stored PNG was written for a model of a different geometry.
        # Resampling it to this model's shape (which is what derive_views would
        # do) stacks a second resize on top of the first. Go back to
        # corrected.png and run the chain once, properly.
        print(f"  {d.name}: stored model_input.png is {stored_hw[1]}x{stored_hw[0]}, "
              f"this model wants {want_hw[1]}x{want_hw[0]} — regenerating")
        mi_bytes, corr_bytes = regenerate_model_input(d, record, lm.descriptor)
        if not dry_run:
            model_input.write_bytes(mi_bytes)
            (d / "corrected.png").write_bytes(corr_bytes)
    elif not force:
        # Verify the stored model_input.png still matches the recorded chain
        # (catches silent drift in preprocess code between versions).
        try:
            mi_bytes, corr_bytes = regenerate_model_input(d, record, lm.descriptor)
            if mi_bytes != model_input.read_bytes():
                print(f"  WARN {d.name}: regenerated model_input.png differs from "
                      "the stored one (preprocess code changed?). Keeping the "
                      "stored pixels; pass --force to regenerate.")
            if ((d / "corrected.png").is_file()
                    and corr_bytes != (d / "corrected.png").read_bytes()):
                print(f"  WARN {d.name}: regenerated corrected.png differs from "
                      "the stored one.")
        except Exception as exc:
            print(f"  WARN {d.name}: regeneration check failed ({exc}); "
                  "using stored model_input.png as-is.")
    elif force:
        mi_bytes, corr_bytes = regenerate_model_input(d, record, lm.descriptor)
        if not dry_run:
            model_input.write_bytes(mi_bytes)
            (d / "corrected.png").write_bytes(corr_bytes)
    return model_input


def run_capture(d: Path, record: dict, lm: LoadedModel,
                force: bool, dry_run: bool) -> None:
    model_input = _ensure_model_input(d, record, lm, force, dry_run)
    if dry_run:
        print(f"  [dry-run] would run {lm.id} on {d.name}")
        return

    pr = runner.predict(lm, model_input)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_id = f"run_{stamp}_{lm.id}"
    rec = PredictionRecord(
        run_id=run_id, capture_id=record["capture_id"], model=lm.run_info(),
        descriptor_snapshot=lm.descriptor,
        scores=pr.scores, logits=pr.logits, thresholds=pr.thresholds,
        threshold_source=pr.threshold_source, positive=pr.positive,
        timing_ms=pr.timing_ms,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        model_input_sha256=pr.model_input_sha256,
        env=storage.app_env_record())
    storage.atomic_write_json(d / "predictions" / f"{run_id}.json",
                              rec.model_dump(mode="json"))
    record.setdefault("predictions", []).append(run_id)
    record["predictions"] = list(dict.fromkeys(record["predictions"]))
    storage.save_capture_record(d, CaptureRecord.model_validate(record))
    pretty = {k: round(v, 4) for k, v in pr.scores.items()}
    print(f"  {d.name}: {lm.id} run {run_id} positive={pr.positive} scores={pretty}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="process every capture")
    ap.add_argument("--capture", action="append", default=[], help="capture id (repeatable)")
    ap.add_argument("--model", action="append", default=[],
                    help="model id (repeatable; default: active model)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate model_input.png instead of reusing the stored one")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    loaded = discover()
    model_ids = a.model or [active(loaded).id]
    for mid in model_ids:
        if mid not in loaded:
            sys.exit(f"unknown model {mid!r}; known: {', '.join(sorted(loaded))}")

    if a.capture:
        dirs = [storage.capture_dir(cid) for cid in a.capture]
    elif a.all:
        dirs = storage.iter_captures()
    else:
        sys.exit("choose --all or --capture ID")

    for mid in model_ids:
        lm = load_model(loaded[mid])
        print(f"model {mid}: views={lm.views} "
              f"thresholds={[round(t, 3) for t in lm.thresholds]}")
        for d in dirs:
            cj = d / "capture.json"
            if not cj.is_file():
                continue
            record = storage.load_capture_json(d)
            try:
                run_capture(d, record, lm, a.force, a.dry_run)
            except Exception as exc:
                print(f"  FAIL {d.name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
