#!/usr/bin/env python3
"""Export all stored captures to a CSV for analysis.

    python scripts/export_dataset.py --out captures_export.csv

One row per prediction run (multi-model captures produce several rows; a
capture with no run yet produces one row with empty model columns). The CSV is
shaped to feed the existing bootstrap/metrics machinery (external-validation /
prototype-results tooling) rather than inventing new metrics (§17.10):
per-label scores AND logits are stored at full precision, plus thresholds and
the derived positives, so post-hoc threshold sweeps need no re-inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.services import storage  # noqa: E402

BASE_COLS = ["capture_id", "created_utc", "sheet_id", "retake_of", "blind_mode",
             "source_mode", "geometry_mode", "quality_level",
             "blur_varlap", "brightness", "est_px_per_mm_original",
             "scale_factor_to_input", "quality_flags",
             "model_id", "model_version", "run_id", "checkpoint_sha256",
             "threshold_source",
             "ref_STEMI", "ref_AF", "ref_LVH", "ref_NORMAL", "truth_source",
             "labels_entered_before_prediction", "notes"]


def _row_capture(rec: dict) -> dict:
    session = rec.get("session", {})
    ann = rec.get("annotation", {})
    q = rec.get("quality_model_input", {}) or rec.get("quality_original", {}) or {}
    corr = rec.get("correction", {}) or {}
    refs = ann.get("reference_labels", {}) or {}
    return {
        "capture_id": rec.get("capture_id"),
        "created_utc": rec.get("created_utc"),
        "sheet_id": session.get("sheet_id"),
        "retake_of": session.get("retake_of"),
        "blind_mode": session.get("blind_mode"),
        "source_mode": (rec.get("source") or {}).get("mode"),
        "geometry_mode": corr.get("geometry_mode"),
        "quality_level": q.get("level"),
        "blur_varlap": q.get("blur_varlap"),
        "brightness": (rec.get("quality_original") or {}).get("brightness"),
        "est_px_per_mm_original": corr.get("est_px_per_mm_original"),
        "scale_factor_to_input": corr.get("scale_factor_to_input"),
        "quality_flags": ";".join(q.get("flags", [])),
        "model_id": "", "model_version": "", "run_id": "", "checkpoint_sha256": "",
        "threshold_source": "",
        "ref_STEMI": refs.get("STEMI", ""),
        "ref_AF": refs.get("AF", ""),
        "ref_LVH": refs.get("LVH", ""),
        "ref_NORMAL": refs.get("NORMAL", ""),
        "truth_source": ann.get("truth_source"),
        "labels_entered_before_prediction": ann.get("labels_entered_before_prediction"),
        "notes": ann.get("notes"),
    }


def export(out_path: Path) -> int:
    runs_cols = []
    for label in ("STEMI", "AF", "LVH", "NORMAL"):
        runs_cols += [f"score_{label}", f"logit_{label}", f"thr_{label}",
                      f"positive_{label}"]
    cols = BASE_COLS + runs_cols + ["forward_ms", "preprocess_ms", "run_created_utc"]

    n_captures = n_runs = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in storage.iter_captures():
            cj = d / "capture.json"
            if not cj.is_file():
                continue
            rec = storage.load_capture_json(d)
            n_captures += 1
            runs = []
            for run_id in rec.get("predictions", []):
                rp = d / "predictions" / f"{run_id}.json"
                if rp.is_file():
                    runs.append(json.loads(rp.read_text(encoding="utf-8")))
            if not runs:
                w.writerow(_row_capture(rec))
                continue
            for run in runs:
                row = _row_capture(rec)
                model = run.get("model", {})
                row.update({
                    "model_id": model.get("id"),
                    "model_version": model.get("version"),
                    "run_id": run.get("run_id"),
                    "checkpoint_sha256": model.get("checkpoint_sha256"),
                    "threshold_source": run.get("threshold_source"),
                    "forward_ms": run.get("timing_ms", {}).get("forward"),
                    "preprocess_ms": run.get("timing_ms", {}).get("preprocess"),
                    "run_created_utc": run.get("created_utc"),
                })
                for label in ("STEMI", "AF", "LVH", "NORMAL"):
                    row[f"score_{label}"] = run.get("scores", {}).get(label)
                    row[f"logit_{label}"] = run.get("logits", {}).get(label)
                    row[f"thr_{label}"] = run.get("thresholds", {}).get(label)
                    row[f"positive_{label}"] = 1 if label in run.get("positive", []) else 0
                w.writerow(row)
                n_runs += 1
    print(f"exported {n_captures} captures / {n_runs} prediction runs -> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("captures_export.csv"))
    a = ap.parse_args()
    return export(a.out)


if __name__ == "__main__":
    sys.exit(main())
