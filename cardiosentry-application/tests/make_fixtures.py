#!/usr/bin/env python3
"""Generate the checked-in test fixtures.

`corpus_sheet.jpg` is a real 1686x1311 stage-4 corpus sheet (the exact image
the prototype's evaluation run scored — see expected_scores.json). The four
"photo" fixtures are SYNTHETIC phone-style photographs: the corpus sheet
warped onto a dark desk-like background with per-fixture perspective, scale,
and lighting. Real phone photos should be added by hand as they are collected
(collecting them is the point of the app); these stand-ins keep the detection
and API tests deterministic.

    python tests/make_fixtures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
REPO_ROOT = HERE.parent.parent
CORPUS_SHEET = (REPO_ROOT / "dataset-combination-code/ecg_dataset_pipeline"
                / "data/releases/v2/prototype_1152/images/test/ECG000002_a0.jpg")

# Prototype evaluation runs' stored scores for ECG000002 (GPU+AMP), one entry
# per round-3 checkpoint:
# prototype-results/<run>/predictions/test_predictions.csv
# Each is written to expected_scores_<id>.json, which is the file each round-3
# descriptor's `verify.expected_scores` points at. expected_scores.json — the
# fallback for a descriptor that pins nothing — is a copy of the ACTIVE model's,
# so regenerating fixtures can never leave the self-check comparing the served
# model against another run's numbers.
ACTIVE_MODEL = "convnextv2_tiny_v1img_v1"
EXPECTED_SCORES = {
    "convnextv2_tiny_v1img_v1": {
        "model": "convnextv2_tiny_v1img_v1",
        "image": "corpus_sheet.jpg",
        "source": ("CardioSentry serving path, CPU float32; no matching V1 "
                   "evaluation image is available locally"),
        "scores": {
            "NORM": 0.0004257865366525948,
            "AF": 0.9984742999076843,
            "IAVB": 0.00034827488707378507,
            "LBBB": 0.002829588484019041,
            "RBBB": 0.000134890346089378,
            "PAC": 0.0012787270825356245,
            "PVC": 0.00010621726687531918,
            "LAFB": 0.00169347261544317,
            "LAE": 0.00009923530888045207,
            "TInv": 0.11236025393009186,
            "LQT": 0.000049815145757747814,
            "PRWP": 0.00027405432774685323,
        },
    },
    "efficientnetv2_s_nat768": {
        "model": "efficientnetv2_s_nat768",
        "image": "ECG000002_a0.jpg",
        "source": "test_predictions.csv (prototype GPU+AMP)",
        "scores": {"STEMI": 3.3737226e-05, "AF": 0.9320833,
                   "LVH": 0.0006851712, "NORMAL": 0.003324437}},
    "efficientnetv2_s_cbam_nat768": {
        "model": "efficientnetv2_s_cbam_nat768",
        "image": "ECG000002_a0.jpg",
        "source": "test_predictions.csv (prototype GPU+AMP)",
        "scores": {"STEMI": 0.0013511209, "AF": 0.99879813,
                   "LVH": 0.005957154, "NORMAL": 0.0035518846}},
    "resnet50_cbam_nat768": {
        "model": "resnet50_cbam_nat768",
        "image": "ECG000002_a0.jpg",
        "source": "test_predictions.csv (prototype GPU+AMP)",
        "scores": {"STEMI": 0.005819959, "AF": 0.9944666,
                   "LVH": 0.009376237, "NORMAL": 0.0005274784}},
}


def _desk_bg(w: int, h: int, base: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.full((h, w, 3), base, dtype=np.float32)
    bg *= (1.0 + 0.08 * (xx / w - 0.5))[..., None]
    bg += rng.normal(0, 2.5, (h, w, 1))
    return np.clip(bg, 0, 255).astype(np.uint8)


def _photo(sheet: np.ndarray, quad_frac: tuple[float, ...],
           bright: float, rng: np.random.Generator, out: Path,
           desk_base: int = 34) -> None:
    W, H = 2400, 1800
    bg = _desk_bg(W, H, desk_base, rng)
    sh, sw = sheet.shape[:2]

    fx0, fy0, fx1, fy1, fx2, fy2, fx3, fy3 = quad_frac
    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
    dst = np.array([[fx0 * W, fy0 * H], [fx1 * W, fy1 * H],
                    [fx2 * W, fy2 * H], [fx3 * W, fy3 * H]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, Hm, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT)
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), Hm, (W, H),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0).astype(np.float32) / 255.0
    photo = (bg.astype(np.float32) * (1 - mask[..., None])
             + warped.astype(np.float32) * mask[..., None] * bright)
    photo = np.clip(photo + rng.normal(0, 3.0, photo.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(out), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out.name}: {photo.shape[1]}x{photo.shape[0]}")


def main() -> int:
    if not CORPUS_SHEET.is_file():
        raise SystemExit(f"corpus sheet not found: {CORPUS_SHEET}")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CORPUS_SHEET, FIXTURES / "corpus_sheet.jpg")
    for model_id, expected in EXPECTED_SCORES.items():
        (FIXTURES / f"expected_scores_{model_id}.json").write_text(
            json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    (FIXTURES / "expected_scores.json").write_text(
        json.dumps(EXPECTED_SCORES[ACTIVE_MODEL], indent=2) + "\n",
        encoding="utf-8")

    sheet = cv2.cvtColor(cv2.imread(str(FIXTURES / "corpus_sheet.jpg")),
                         cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(42)

    _photo(sheet, (0.08, 0.06, 0.94, 0.05, 0.97, 0.93, 0.05, 0.95),
           1.0, rng, FIXTURES / "photo_clean.jpg")
    _photo(sheet, (0.16, 0.12, 0.88, 0.02, 0.99, 0.86, 0.03, 0.99),
           0.95, rng, FIXTURES / "photo_tilted.jpg")
    _photo(sheet, (0.08, 0.06, 0.94, 0.05, 0.97, 0.93, 0.05, 0.95),
           0.30, rng, FIXTURES / "photo_dark.jpg", desk_base=18)
    _photo(sheet, (0.36, 0.34, 0.66, 0.32, 0.69, 0.64, 0.33, 0.67),
           1.0, rng, FIXTURES / "photo_small.jpg")

    # A desk with no paper — the fallback path must report failure honestly.
    rng2 = np.random.default_rng(7)
    noise = _desk_bg(2400, 1800, 40, rng2)
    cv2.imwrite(str(FIXTURES / "no_paper.jpg"),
                cv2.cvtColor(noise, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote no_paper.jpg: 2400x1800")
    print("\nfixtures ready in", FIXTURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
