#!/usr/bin/env python3
"""Show each paper/photo augmentation effect on its own, on one ECG sheet.

For one ECG of a sample set written by ``export_correct_predictions.py`` (its
grey kit master is kept in ``_work/masters`` by ``export_augmented_preview.py``),
writes one 1024x768 image per effect of ``stage4_augment.augment``, the stage
the training corpus was augmented with. Each image is the clean red sheet with
exactly one effect applied and nothing else (no other effect, no added margin),
so a figure can show what every effect does:

    00 clean (reference)          08 wrinkled-paper texture
    01-04 grid red/orange/pink/grey 09 fold creases
    05 faded ink                  10 uneven lighting + edge shadow
    06a/06b tilt +3 / -3 deg      11 colour tint      12 brightness
    07 perspective                13 contrast         14 sensor noise
                                  15 blur             16 JPEG compression

Effects use the strongest setting in the corpus range so they are visible; the
setting is listed in ``effects.csv``. Texture, creases and lighting call the
pipeline's own functions; the others are one-line operations taken from
``augment`` with the parameter fixed. Effects are applied at the kit's
2200x1700 master resolution and then fitted to 1024x768, as in the corpus, so
noise, blur and JPEG are faint at final size by design; ``zoom/`` repeats one
lead I patch of every image at 4x so they can be seen.

Usage::

    ~/.cache/ecgkit-venv/bin/python export_augmentation_effects.py \\
        --samples ../../prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42/correct_single_label_samples \\
        --record STACH_Sinus_tachycardia/04
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

import export_cardiologist_samples as base

ZOOM_BOX = (40, 110, 296, 302)  # (left, top, right, bottom) on 1024x768: start of lead I + calibration pulse
WORST_SEARCH = 500  # draws tried for the worst-case texture / crease images
WRINKLE_HIGHPASS_SIGMA = 20  # px at master resolution; keeps wrinkle lines, drops facet shading
MANY_FOLDS = 6  # fold count for the many-creases preview image (corpus: 1-3)


class _FoldCount:
    """Generator proxy for stage4_augment._creases: fixes the fold count
    (`integers(1, 4)`), drops the bowing waves (`integers(1, 3)` -> 0) and
    records each fold's orientation draw (`random()` < 0.5 is horizontal)."""

    def __init__(self, rng, folds: int):
        self._rng, self._folds, self.orientation = rng, folds, []

    def integers(self, low, high=None, *args, **kwargs):
        if (low, high) == (1, 4):
            return self._folds
        if (low, high) == (1, 3):
            return 0
        return self._rng.integers(low, high, *args, **kwargs)

    def random(self, *args, **kwargs):
        value = self._rng.random(*args, **kwargs)
        self.orientation.append(value)
        return value

    def __getattr__(self, name):
        return getattr(self._rng, name)


def worst_draw(effect, seed: int) -> tuple[int, float]:
    """(seed, shading sd) of the draw whose multiplicative field varies most."""
    blank = np.full((340, 440, 3), 200.0, dtype=np.float32)
    best = (seed, -1.0)
    for k in range(seed, seed + WORST_SEARCH):
        spread = float((effect(blank, np.random.default_rng(k))[..., 0] / 200.0).std())
        if spread > best[1]:
            best = (k, spread)
    return best


def grid_lut(sa, palette: tuple, saturation: float = 1.0, ink: float = 20.0) -> np.ndarray:
    """stage4_augment._grid_lut with its per-image jitter fixed (ink, paper, saturation)."""
    major, minor, paper = (np.asarray(c, dtype=np.float32) for c in palette)

    def _sat(c):
        lum = float(np.dot(c, (0.299, 0.587, 0.114)))
        return np.clip(lum + (c - lum) * saturation, 0, 255)

    major, minor = _sat(major), _sat(minor)
    xs = np.array([0, sa.C.GRID_TRACE_MAX, sa.C.GRID_MAJOR_LEVEL,
                   sa.C.GRID_MINOR_LEVEL, sa.C.GRID_PAPER_LEVEL, 255], dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    for ch in range(3):
        ys = np.array([ink * 0.35, ink, major[ch], minor[ch], paper[ch], min(255.0, paper[ch] + 3)],
                      dtype=np.float32)
        lut[:, ch] = np.interp(np.arange(256, dtype=np.float32), xs, ys)
    return np.clip(lut, 0, 255)


def recolour(sa, master: Image.Image, palette: str, saturation: float = 1.0) -> Image.Image:
    lut = grid_lut(sa, sa.C.GRID_PALETTES[palette], saturation)
    return Image.fromarray(lut[np.asarray(master.convert("L"))].astype(np.uint8), "RGB")


def as_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


class _FirstRandom:
    """Generator proxy that records the first random() value."""

    def __init__(self, rng):
        self._rng, self.first = rng, None

    def random(self, *args, **kwargs):
        value = self._rng.random(*args, **kwargs)
        if self.first is None:
            self.first = value
        return value

    def __getattr__(self, name):
        return getattr(self._rng, name)


def effects(sa, master: Image.Image, seed: int) -> list[tuple[str, str, str, Image.Image]]:
    """(file stem, effect, setting, image at master resolution) for every effect."""
    clean = recolour(sa, master, "red")
    arr = np.asarray(clean, dtype=np.float32)
    fill = tuple(int(v) for v in sa._paper_colour(arr))
    w, h = clean.size
    out = [("00_clean", "Clean sheet (reference)", "red grid, full ink, no effect", clean)]

    for i, name in enumerate(("red", "orange", "pink", "grey"), 1):
        major = sa.C.GRID_PALETTES[name][0]
        out.append((f"{i:02d}_grid_{name}", f"Grid colour: {name}",
                    f"major line RGB {major}, full ink", recolour(sa, master, name)))
    low = sa.C.GRID_SATURATION[0]
    out.append(("05_ink_faded", "Faded ink (grid strength)",
                f"red grid at saturation {low} (corpus range {sa.C.GRID_SATURATION[0]}-{sa.C.GRID_SATURATION[1]})",
                recolour(sa, master, "red", low)))

    for suffix, deg in (("a", 3.0), ("b", -3.0)):
        out.append((f"06{suffix}_tilt_{deg:+.0f}deg", "Tilt", f"{deg:+.1f} deg (corpus range -3 to +3)",
                    sa._rotate_on_paper(clean, deg, fill)))

    m = 0.018  # keystone strength, top of augment's uniform(0.004, 0.018)
    dx, dy = m * w, m * h
    quad = (dx, dy, 0, h, w, h, w - dx, dy)  # top edge pulled in: sheet photographed from below
    out.append(("07_perspective", "Perspective (keystone)",
                f"corner offset {m:.3f} x sheet size, top edge narrowed",
                clean.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=fill)))

    strength, amp = sa.C.TEXTURE_STRENGTH, sa.C.CREASE_AMP
    sa.C.TEXTURE_STRENGTH, sa.C.CREASE_AMP = (strength[1], strength[1]), (amp[1], amp[1])
    try:
        textured = sa._paper_texture(arr, np.random.default_rng(seed))
        creased = sa._creases(arr, np.random.default_rng(seed))
        # Worst case within the corpus distribution: at maximum strength, the
        # draw (texture photo, crop, flips / fold count, position, width, bowing)
        # whose shading field varies most. Draw counts do not depend on image
        # size, so the search runs on a small blank sheet.
        texture_k, texture_sd = worst_draw(sa._paper_texture, seed)
        crease_k, crease_sd = worst_draw(sa._creases, seed)
        texture_worst = sa._paper_texture(arr, np.random.default_rng(texture_k))
        crease_worst = sa._creases(arr, np.random.default_rng(crease_k))
        both_worst = sa._creases(texture_worst, np.random.default_rng(crease_k))
        # Same draw as 08b, as the shading field alone (texture of a blank sheet).
        field = sa._paper_texture(np.ones_like(arr), np.random.default_rng(texture_k))[..., 0]
    finally:
        sa.C.TEXTURE_STRENGTH, sa.C.CREASE_AMP = strength, amp
    out.append(("08_paper_texture", "Wrinkled-paper texture",
                f"strength {strength[1]} (corpus range {strength[0]}-{strength[1]})", as_image(textured)))
    out.append(("08b_paper_texture_worst", "Wrinkled-paper texture, worst case",
                f"strength {strength[1]}; most uneven of {WORST_SEARCH} texture draws "
                f"(shading sd {texture_sd:.3f})", as_image(texture_worst)))
    # NOT a corpus effect: the wrinkle photographs also carry broad light/shadow
    # patches (the crumpled facets' shading); dividing the field by its own
    # blur keeps only the fine wrinkle lines, so the paper keeps its colour.
    fine = np.clip(field / gaussian_filter(field, WRINKLE_HIGHPASS_SIGMA), *sa.C.TEXTURE_CLIP)
    out.append(("08c_paper_texture_worst_wrinkles_only", "Wrinkle lines only (preview variant, not in corpus)",
                f"08b's texture high-passed (Gaussian sigma {WRINKLE_HIGHPASS_SIGMA} px at 2200x1700): "
                "broad shading removed", as_image(arr * fine[..., None])))
    out.append(("09_fold_creases", "Fold creases and paper bowing",
                f"crease depth {amp[1]} (corpus range {amp[0]}-{amp[1]})", as_image(creased)))
    out.append(("09b_fold_creases_worst", "Fold creases and paper bowing, worst case",
                f"crease depth {amp[1]}; most uneven of {WORST_SEARCH} crease draws "
                f"(shading sd {crease_sd:.3f})", as_image(crease_worst)))
    out.append(("09c_texture_and_creases_worst", "Worst wrinkled texture + worst creases",
                "08b then 09b, in augment's order (texture, then creases)", as_image(both_worst)))

    # NOT a corpus effect: the pipeline folds a sheet 1-3 times. Same crease
    # code at maximum depth, with the fold count set to MANY_FOLDS (half
    # horizontal, half vertical) and the broad paper bowing switched off.
    sa.C.CREASE_AMP = (amp[1], amp[1])
    try:
        for k in range(seed, seed + 1000):
            folds = _FoldCount(np.random.default_rng(k), MANY_FOLDS)
            many = sa._creases(arr, folds)
            if sum(v < 0.5 for v in folds.orientation) == MANY_FOLDS // 2:
                break
    finally:
        sa.C.CREASE_AMP = amp
    out.append((f"09d_fold_creases_{MANY_FOLDS}_folds", f"{MANY_FOLDS} fold creases (preview variant, not in corpus)",
                f"{MANY_FOLDS} folds at depth {amp[1]} ({MANY_FOLDS // 2} horizontal, {MANY_FOLDS - MANY_FOLDS // 2} "
                "vertical), no paper bowing; corpus uses 1-3 folds", as_image(many)))

    # First seed whose lighting draw includes the edge shadow (augment: 65% of sheets).
    for k in range(seed, seed + 1000):
        probe = _FirstRandom(np.random.default_rng(k))
        lit = sa._illumination(arr, probe)
        if probe.first < 0.65:
            break
    out.append(("10_uneven_lighting", "Uneven lighting, corner falloff, edge shadow",
                "one corpus lighting draw with an edge shadow", as_image(lit)))

    gains = np.array([1.024, 1.0, 0.976], dtype=np.float32)  # 2 sd of augment's normal(0, 0.012)
    out.append(("11_colour_tint", "Colour tint (white balance)",
                "R x1.024, B x0.976 (2 sd warm drift)", as_image(arr * gains)))
    out.append(("12_brightness", "Brightness (exposure)", "x0.92 (corpus range 0.92-1.06)", as_image(arr * 0.92)))
    mean = arr.mean()
    out.append(("13_contrast", "Contrast", "x1.08 about the mean (corpus range 0.94-1.08)",
                as_image((arr - mean) * 1.08 + mean)))
    noise = np.random.default_rng(seed).normal(0.0, 4.5, size=arr.shape)
    out.append(("14_sensor_noise", "Sensor noise", "Gaussian sd 4.5 (corpus range 1.5-4.5)", as_image(arr + noise)))
    radius = sa.C.DEFOCUS_RADIUS[1]
    out.append(("15_blur", "Blur (defocus)",
                f"Gaussian radius {radius} px at 2200x1700 (corpus range {sa.C.DEFOCUS_RADIUS[0]}-{radius})",
                clean.filter(ImageFilter.GaussianBlur(radius=radius))))
    buffer = io.BytesIO()
    clean.save(buffer, "JPEG", quality=sa.C.JPEG_QUALITY, subsampling=sa.C.JPEG_SUBSAMPLING, optimize=True)
    out.append(("16_jpeg", "JPEG compression",
                f"quality {sa.C.JPEG_QUALITY}, 4:2:0 chroma (corpus setting)", Image.open(buffer).convert("RGB")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=Path, required=True,
                        help="folder written by export_correct_predictions.py (holds index.csv)")
    parser.add_argument("--record", default="STACH_Sinus_tachycardia/04",
                        help="<condition folder>/<number> of the ECG to use")
    parser.add_argument("--out", type=Path, help="default: <samples>/augmentation_effects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notebook", type=Path,
                        default=base.RENDER_CODE_DIR / "Clean12_ECG_Signal_Images_EC2.ipynb")
    parser.add_argument("--kit-bundle", type=Path, default=base.RENDER_CODE_DIR / "ecg_image_v1_code.tar.gz")
    parser.add_argument("--augment-code", type=Path, default=base.AUGMENT_CODE_DIR)
    args = parser.parse_args()

    samples = args.samples.expanduser().resolve()
    out = (args.out or samples / "augmentation_effects").expanduser().resolve()
    folder, number = args.record.rsplit("/", 1)
    with (samples / "index.csv").open(newline="", encoding="utf-8") as handle:
        row = next((r for r in csv.DictReader(handle)
                    if r["folder"] == folder and Path(r["clean_image"]).name.startswith(f"{number}_")), None)
    if row is None:
        raise SystemExit(f"{args.record}: no such ECG in {samples / 'index.csv'}")
    master_png = samples / "_work" / "masters" / f"{row['file_id']}.png"
    if not master_png.is_file():
        raise SystemExit(f"{master_png} missing: run export_augmented_preview.py first")

    config = base.prepare_render(samples / "_work", args.notebook.resolve(), args.kit_bundle.resolve(),
                                 args.augment_code.resolve())
    import ecg_clean_worker as worker
    import stage4_augment as sa
    sa.C.KIT = Path(config["kit_dir"])  # wrinkled-paper photographs ship with the kit bundle

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with Image.open(master_png) as master:
        master.load()
    rows = []
    (out / "zoom").mkdir()
    for stem, effect, setting, image in effects(sa, master, args.seed):
        final = worker.to_final(image)
        final.save(out / f"{stem}.png", compress_level=config["png_compress_level"])
        # Same lead I patch of every image at 4x (nearest), where the faint
        # effects (noise, blur, JPEG, contrast) can be seen pixel by pixel.
        patch = final.crop(ZOOM_BOX)
        patch.resize((patch.width * 4, patch.height * 4), Image.NEAREST).save(
            out / "zoom" / f"{stem}_zoom4x.png", compress_level=config["png_compress_level"])
        rows.append({"image": f"{stem}.png", "zoom": f"zoom/{stem}_zoom4x.png", "effect": effect,
                     "setting": setting})
        print(f"  {stem}.png  {effect}: {setting}")
    with (out / "effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "zoom", "effect", "setting"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "README.txt").write_text(
        "One augmentation effect per image, on the same ECG\n"
        "==================================================\n\n"
        f"ECG: {row['clean_image']} ({row['true_labels']}, {row['record_path']})\n\n"
        "Each image is the clean red-grid sheet with exactly one effect of the\n"
        "corpus augmentation (stage4_augment.augment) applied and nothing else.\n"
        "Effects are at the strongest setting of their corpus range so they are\n"
        "visible; in the corpus each is drawn at random within that range and\n"
        "they are combined. Applied at 2200x1700 and fitted to 1024x768, so noise,\n"
        "blur and JPEG are faint at final size; zoom/ holds the same lead I patch\n"
        f"{ZOOM_BOX} of every image at 4x (nearest neighbour). Settings: effects.csv.\n")
    print(f"Done: {len(rows)} images in {out}")


if __name__ == "__main__":
    main()
