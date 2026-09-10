"""Batch test runner for ConvNeXt Grad-CAM on available ECG fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root and local experiment path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "experiments" / "convnext_gradcam"))

from gradcam_convnext import (
    LABELS,
    build_model,
    preprocess_ecg,
    compute_gradcam,
    save_per_condition_overlays,
    render_composite_figure,
)

FIXTURES_DIR = ROOT_DIR / "cardiosentry-application" / "tests" / "fixtures"
WEIGHTS_PATH = ROOT_DIR / "cardiosentry-application" / "models" / "weights" / "convnextv2_tiny@nat768.pt"
OUTPUT_BASE = ROOT_DIR / "experiments" / "convnext_gradcam" / "results"

TEST_IMAGES = [
    ("corpus_sheet.jpg", "Corpus Sheet (ECG000002_a0, True AF Positive)"),
    ("photo_clean.jpg", "Real Smartphone Photograph (Clean illumination)"),
    ("photo_dark.jpg", "Real Photograph (Dim / Dark illumination)"),
    ("photo_tilted.jpg", "Real Photograph (Oblique angle / Tilted perspective)"),
]


def run_batch():
    device = "cpu"
    print(f"Loading ConvNeXt-V2 Tiny model on {device}...")
    model, thresholds, meta = build_model(
        timm_tag="convnextv2_tiny.fcmae_ft_in1k",
        weights_path=WEIGHTS_PATH,
        device=device,
    )

    all_results = {}

    for filename, description in TEST_IMAGES:
        img_path = FIXTURES_DIR / filename
        if not img_path.is_file():
            print(f"Skipping {filename}: not found.")
            continue

        print(f"\n=======================================================")
        print(f"Processing: {filename} ({description})")
        print(f"=======================================================")

        tensor, rgb_img = preprocess_ecg(img_path, input_hw=(768, 1024))
        probs, cams, cam_meta = compute_gradcam(
            model=model,
            input_tensor=tensor,
            target_layer_name="stages.3",
            device=device,
        )

        stem = img_path.stem
        out_dir = OUTPUT_BASE / stem
        cond_paths = save_per_condition_overlays(
            rgb_img=rgb_img,
            probs=probs,
            cams=cams,
            thresholds=thresholds,
            meta=cam_meta,
            out_dir=out_dir,
            image_stem=stem,
        )

        comp_path = out_dir / f"{stem}_gradcam_composite.png"
        render_composite_figure(
            rgb_img=rgb_img,
            probs=probs,
            cams=cams,
            thresholds=thresholds,
            meta=cam_meta,
            out_path=comp_path,
        )

        print("\nPredictions:")
        for i, label in enumerate(LABELS):
            p = float(probs[i])
            th = float(thresholds[i])
            flag = " [POSITIVE]" if p >= th else " [NEGATIVE]"
            print(f"  {label:<8}: prob = {p:.4f} vs threshold = {th:.2f} {flag}")

        print("\nSaved Result Images:")
        for label, path in cond_paths.items():
            print(f"  {label:<8}: {path.resolve()}")
        print(f"  Composite: {comp_path.resolve()}")

        all_results[filename] = {
            "probs": {label: float(probs[i]) for i, label in enumerate(LABELS)},
            "thresholds": {label: float(thresholds[i]) for i, label in enumerate(LABELS)},
            "paths": {label: str(p.resolve()) for label, p in cond_paths.items()},
            "composite": str(comp_path.resolve()),
        }

    print("\n\nAll test runs finished successfully!")
    return all_results


if __name__ == "__main__":
    run_batch()
