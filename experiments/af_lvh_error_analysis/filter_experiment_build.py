#!/usr/bin/env python3
"""
Filter-bandwidth A/B experiment for the 10 AF+LVH LVH-false-negative test cases.

Reuses the existing synthetic-ECG pipeline VERBATIM (image_pipeline stages 2-4:
stage2_transcode.transcode -> ecg-image-kit render -> stage4_augment.augment).
Only the band-pass frequency variable (config.BANDPASS) is changed AT RUNTIME,
per band; image_pipeline/config.py on disk is NEVER modified.

Bands regenerated for the same 10 records:
  (0.05, 40)   control - must reproduce the released images (validates the run)
  (0.05, 100)
  (0.05, 150)

All other generation parameters (DPI, layout, grid, augmentation RNG seeds,
JPEG settings, staged record names, printed Age/Sex) are identical to the
released corpus build.

Run with the same interpreter the pipeline uses:
  ~/.cache/ecgkit-venv/bin/python filter_experiment_build.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/lindtseyxvii/Codes/cardiosentry/dataset-combination-code"
                            "/ecg_dataset_pipeline/image_pipeline")))

import config as C                                  # noqa: E402
import stage2_transcode as s2                       # noqa: E402
import stage4_augment as s4                         # noqa: E402

ROOT = Path("/home/lindtseyxvii/Codes/cardiosentry")
PIPE = ROOT / "dataset-combination-code/ecg_dataset_pipeline"
CORPUS = PIPE / "data/releases/v2/synthetic_data"
PROTO = PIPE / "data/releases/v2/prototype_1152"
OUT = ROOT / "af_lvh_error_analysis/filter_experiment"

RECORDS = ["ECG003330", "ECG003966", "ECG004025", "ECG004242", "ECG005462",
           "ECG005573", "ECG005876", "ECG005955", "ECG006463", "ECG006595"]

BANDS = [(0.05, 40.0), (0.05, 100.0), (0.05, 150.0)]

CONFIG_BEFORE = hashlib.sha256(Path(PIPE / "image_pipeline/config.py").read_bytes()).hexdigest()


def band_slug(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}Hz"


def load_case_meta():
    """Rows from the earlier trace-back table (labels + model probabilities)."""
    rows = list(csv.DictReader(open(ROOT / "af_lvh_error_analysis"
                                    "/test_af_lvh_case_table.csv")))
    return {r["sample_id"]: r for r in rows}


def load_stage_rows():
    """staged_map + manifest rows for the 10 records (exact source signal path)."""
    smap = list(csv.DictReader(open(CORPUS / "staged_map.csv")))
    man = {r["record_id"]: r for r in csv.DictReader(open(CORPUS / "image_manifest.csv"))}
    out = {}
    for s in smap:
        if s["record"] in RECORDS:
            m = man[s["record_id"]]
            out[s["record"]] = {
                "signal_path": m["signal_path"],  # balanced/signals/PTBXL/PTBXL_000452
                "age": s["age"],                  # as printed on the sheet
                "sex": s["sex"],
            }
    return out


def stage_band(rows: dict, band, staged_dir: Path):
    """stage2 transcode verbatim, with BANDPASS overridden for this band only."""
    C.BANDPASS = band                      # in-memory only; config.py untouched
    sos = s2._bandpass_sos(C.FS)
    staged_dir.mkdir(parents=True, exist_ok=True)
    for rec in RECORDS:
        err = s2.transcode(rows[rec], rec, staged_dir, sos)
        if err:
            raise RuntimeError(f"transcode failed for {rec}: {err}")


def render_band(staged_dir: Path, rendered_dir: Path):
    """stage3 render verbatim: one kit invocation per band chunk, r0, same seed."""
    rendered_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(C.VENV_PY), "gen_ecg_images_from_data_batch.py",
           "-i", str(staged_dir), "-o", str(rendered_dir),
           "-se", str(C.SEED + 0),        # render index 0, exactly like the corpus
           *C.KIT_RENDER_FLAGS]
    proc = subprocess.run(cmd, cwd=str(C.KIT), capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(f"kit render failed rc={proc.returncode}: {tail}")
    pngs = sorted(rendered_dir.rglob("*.png"))
    if len(pngs) != len(RECORDS):
        raise RuntimeError(f"expected {len(RECORDS)} PNGs, got {len(pngs)}")
    return pngs


def augment_band(pngs, out_dir: Path):
    """stage4 augment verbatim: seed parts (record, 0) == released corpus build."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = {}
    for png in pngs:
        record = png.stem.rsplit("-", 1)[0]
        dst = out_dir / f"{record}_a0.jpg"
        meta = s4.augment(png, dst, (record, 0))
        metas[record] = meta
    return metas


def signal_metrics(staged_dir: Path):
    """Peak-based Sokolow-Lyon proxy + amplitudes from the staged WFDB signals."""
    import numpy as np
    import wfdb
    out = {}
    for rec in RECORDS:
        sig = wfdb.rdrecord(str(staged_dir / rec)).p_signal.astype(float)
        names = [n.strip().upper() for n in wfdb.rdrecord(str(staged_dir / rec)).sig_name]
        idx = {n: i for i, n in enumerate(names)}
        v1 = sig[:, idx["V1"]]
        v5 = sig[:, idx["V5"]]
        v6 = sig[:, idx["V6"]]
        sv1 = -v1.min()
        rv5 = v5.max()
        rv6 = v6.max()
        sokolow = sv1 + max(rv5, rv6)
        out[rec] = {
            "sokolow_mV": round(float(sokolow), 3),
            "max_V5_mV": round(float(rv5), 3),
            "max_V6_mV": round(float(rv6), 3),
            "rms_mV": round(float(np.sqrt((sig ** 2).mean())), 3),
        }
    return out


def main() -> int:
    if OUT.exists():                       # scratch dir: rebuild from scratch
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    case_meta = load_case_meta()
    stage_rows = load_stage_rows()
    assert len(stage_rows) == 10

    proto_idx = {r["record"]: r for r in csv.DictReader(open(PROTO / "index.csv"))}
    all_metrics = {}          # band -> rec -> metrics
    image_map = {}            # (rec, slug) -> image path

    for band in BANDS:
        slug = band_slug(*band)
        staged = OUT / "staged" / slug
        rendered = OUT / "rendered" / slug
        band_images = OUT / "by_band" / slug
        print(f"\n=== band {slug}: staging -> rendering -> augmenting ===")
        stage_band(stage_rows, band, staged)
        pngs = render_band(staged, rendered)
        metas = augment_band(pngs, band_images)
        mets = signal_metrics(staged)
        all_metrics[slug] = mets
        for rec in RECORDS:
            image_map[(rec, slug)] = band_images / f"{rec}_a0.jpg"
            print(f"  {rec}: palette={metas[rec]['grid_palette']:6s} "
                  f"rot={metas[rec]['rotation_deg']:>6} sokolow={mets[rec]['sokolow_mV']} mV")

        if band == BANDS[0]:
            mism = [rec for rec in RECORDS
                    if metas[rec]["grid_palette"] != proto_idx[rec]["grid_palette"]
                    or float(metas[rec]["rotation_deg"]) != float(proto_idx[rec]["rotation_deg"])]
            print(f"  40 Hz control vs released index.csv: "
                  f"{'ALL MATCH (palette+rotation)' if not mism else f'MISMATCH {mism}'}")

    # ---- per-case comparison folders ----------------------------------------
    print("\n=== organising by_case/ ===")
    for rec in RECORDS:
        case_dir = OUT / "by_case" / rec
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROTO / proto_idx[rec]["rel_path"],
                     case_dir / "original_0.05-40Hz.jpg")
        for band in BANDS:
            slug = band_slug(*band)
            shutil.copy2(image_map[(rec, slug)], case_dir / f"{slug}.jpg")

    # ---- metadata.csv --------------------------------------------------------
    print("=== writing metadata.csv ===")
    fields = ["case", "band", "prototype_cls", "LVH_prob", "LVH_pred",
              "original_labels", "additional_conditions",
              "grid_palette", "rotation_deg",
              "sokolow_mV", "max_V5_mV", "max_V6_mV", "rms_mV",
              "image_path", "staged_path"]
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for band in BANDS:
            slug = band_slug(*band)
            for rec in RECORDS:
                m = case_meta[rec]
                mm = all_metrics[slug][rec]
                w.writerow({
                    "case": rec,
                    "band": slug,
                    "prototype_cls": m["prototype_labels"],
                    "LVH_prob": m["LVH_prob"],
                    "LVH_pred": m["LVH_pred"],
                    "original_labels": m["original_labels"],
                    "additional_conditions": m["additional_conditions"],
                    "grid_palette": proto_idx[rec]["grid_palette"],
                    "rotation_deg": proto_idx[rec]["rotation_deg"],
                    "sokolow_mV": mm["sokolow_mV"],
                    "max_V5_mV": mm["max_V5_mV"],
                    "max_V6_mV": mm["max_V6_mV"],
                    "rms_mV": mm["rms_mV"],
                    "image_path": str(image_map[(rec, slug)]),
                    "staged_path": str(OUT / "staged" / slug / rec),
                })

    # ---- config restoration proof -------------------------------------------
    after = hashlib.sha256(Path(PIPE / "image_pipeline/config.py").read_bytes()).hexdigest()
    print(f"\nconfig.py checksum before={CONFIG_BEFORE[:12]}... "
          f"after={after[:12]}...  {'UNCHANGED' if after == CONFIG_BEFORE else 'CHANGED!'}")
    print(f"\ndone -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
