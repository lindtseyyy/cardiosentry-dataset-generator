#!/usr/bin/env python3
"""Show every EC2 augmentation step on its own, on one ECG.

``Clean12_Augmented_ECG_Images_EC2.ipynb`` / ``augment_worker.py`` apply grid
recolour, tilt, perspective, wrinkled-paper texture, fold creases, uneven
lighting, colour tint, exposure, contrast, paper-level normalisation, sensor
noise, blur, margin and JPEG. Wrinkles/creases are stronger than stage 4.
``replay`` mirrors that sequence with the balanced per-record plan and the
same RNG draws, also applying each step alone to the clean sheet. Its combined
result is checked pixel for pixel against ``augment_worker.augment_master``
and the sample set's augmented image.

By default the single-condition ECG whose plan enables every optional step
(perspective, creases, edge shadow, blur) and whose settings are furthest from
neutral is selected; if none enables all steps, the most-enabled plan wins,
then distance from neutral. Grey kit masters are read from ``_work/masters``.

Output (``<samples>/augmentation_effects``)::

    00_clean_reference.png       the sample set's clean image of that ECG
    01a-01c_grid_colour_<name>   orange/pink/grey (the plan's ink/paper jitter)
    02_tilt ... 14_jpeg          one step each, on the clean sheet
    15_all_steps_combined.png    the sample set's augmented image of that ECG
    comparisons/<image>_vs_clean.png   clean | step side by side, + 4x zoom
    effects.csv, README.txt

Usage::

    .venv/bin/python augmentation_breakdown.py --samples <run>/correct_prediction_samples \\
        [--record single/STACH_Sinus_tachycardia/03]

Use the same interpreter as the sample set (the repository ``.venv``): the
pixel checks depend on the image-library versions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import json
import re
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import export_cardiologist_samples as base

ZOOM_BOX = (40, 110, 296, 302)  # on 1024x768: start of lead I + calibration pulse
ZOOM = 4
AUGMENT_DIR = base.HERE.parent / "clean_convnext_codes" / "clean12_augmented_ecg_images_ec2"
OPTIONAL_STEPS = ("perspective", "fold_creases", "edge_shadow", "blur")


def master_path(work: Path, site: str, file_id: str) -> Path:
    return work / "masters" / f"{site}_{file_id}.png"


def _as_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _spread(field: np.ndarray) -> str:
    lo, hi = np.percentile(field, [1, 99])
    return f"{(lo - 1) * 100:+.0f}% to {(hi - 1) * 100:+.0f}%"


def replay(AW, master: Image.Image, clean: Image.Image, seed_parts: tuple,
           plan: dict) -> tuple[list[dict], dict, Image.Image]:
    """Mirror AW.augment draw for draw, forking each isolated effect's RNG."""
    rng = AW.stage4_rng(seed_parts)
    meta: dict = {}
    steps: list[dict] = []

    def fork():
        return copy.deepcopy(rng)

    def check(probe, effect):
        if probe.bit_generator.state != rng.bit_generator.state:
            raise RuntimeError(f"{effect} draw mirror is out of step with augment_worker")

    def add(key, effect, setting, image, applied=True):
        steps.append({"key": key, "effect": effect, "setting": setting, "applied": applied, "image": image})

    h, w = clean.height, clean.width
    clean_arr = np.asarray(clean, dtype=np.float32)
    clean_fill = tuple(int(v) for v in AW._paper_colour(clean_arr))
    ones = np.ones((h, w, 1), dtype=np.float32)

    before = fork()
    img, palette, sat, ink = AW._recolour_grid(master, rng, plan["grid_palette"], plan["ink_bin"])
    meta.update(grid_palette=palette, grid_saturation=round(sat, 4), ink_level=round(ink, 2))
    probe = copy.deepcopy(before)
    names = list(AW.GRID_PALETTES)
    weights = np.array([AW.GRID_PALETTE_WEIGHTS.get(n, 0.0) for n in names], dtype=np.float64)
    probe.choice(len(names), p=weights / weights.sum())
    probe.uniform(8, 34), probe.normal(0, 3, size=3), probe.uniform(0.985, 1.005), probe.normal(0, 1.2, size=3)
    probe.uniform(*AW.GRID_SATURATION)
    check(probe, "grid-colour")
    lo, hi = AW.INK_BINS[plan["ink_bin"]]
    for name in AW.AUG_PALETTES:
        add(f"grid_colour_{name}", f"Grid colour: {name}",
            f"{name} grid, ink strength {sat:.2f} (planned bin {lo}-{hi}; lower = faded), "
            f"ink/paper jitter as drawn; {'used by plan' if name == palette else 'alternative palette'}",
            AW._recolour_grid(master, copy.deepcopy(before), name, plan["ink_bin"])[0], name == palette)

    arr0 = np.asarray(img, dtype=np.float32)
    fill = tuple(int(v) for v in AW._paper_colour(arr0))
    deg = float(rng.uniform(*AW.TILT_RANGE))
    lo, hi = AW.TILT_BINS[plan["tilt_bin"]]
    deg = lo + (deg - AW.TILT_RANGE[0]) / (AW.TILT_RANGE[1] - AW.TILT_RANGE[0]) * (hi - lo)
    meta["tilt_deg"] = round(deg, 3)
    img = AW._rotate_on_paper(img, deg, fill)
    add(f"tilt_{deg:+.1f}deg", "Tilt", f"{deg:+.2f} deg (planned bin {lo} to {hi})",
        AW._rotate_on_paper(clean, deg, clean_fill))

    keystone = rng.random() < AW.KEYSTONE_PROB
    keystone = bool(plan["perspective"])
    meta.update(perspective=int(keystone), perspective_mag="")
    if keystone:
        pw, ph = img.size
        m = rng.uniform(*AW.KEYSTONE_MAG)
        dx, dy = m * pw, m * ph
        quad = (rng.uniform(0, dx), rng.uniform(0, dy),
                rng.uniform(0, dx), ph - rng.uniform(0, dy),
                pw - rng.uniform(0, dx), ph - rng.uniform(0, dy),
                pw - rng.uniform(0, dx), rng.uniform(0, dy))
        meta["perspective_mag"] = round(float(m), 5)
        img = img.transform((pw, ph), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=fill)
        add("perspective", "Perspective (keystone)",
            f"corners pulled in up to {m:.2%} (range {AW.KEYSTONE_MAG[0]:.1%}-{AW.KEYSTONE_MAG[1]:.1%})",
            clean.transform((pw, ph), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=clean_fill))
    else:
        add("perspective", "Perspective (keystone)", "off in plan", None, False)

    arr = np.asarray(img, dtype=np.float32)
    alone, shade = fork(), fork()
    arr = AW._paper_texture(arr, rng, meta)
    isolated = AW._paper_texture(clean_arr, alone, {})
    check(alone, "paper texture")
    add("paper_texture", "Wrinkled-paper texture",
        f"{meta['wrinkle_texture']}, strength {meta['wrinkle_strength']} (range {AW.TEXTURE_STRENGTH}); "
        f"clip {AW.TEXTURE_CLIP}, shading {_spread(AW._paper_texture(ones, shade, {}))}",
        _as_image(isolated))

    creases = rng.random() < AW.CREASE_PROB
    creases = bool(plan["fold_creases"])
    meta.update(fold_creases=int(creases), fold_lines="", paper_bows="")
    if creases:
        alone, shade = fork(), fork()
        arr = AW._creases(arr, rng, meta)
        isolated = AW._creases(clean_arr, alone, {})
        check(alone, "creases")
        add("fold_creases", "Fold creases and paper bowing",
            f"{len(meta['fold_lines'])} folds + {meta['paper_bows']} bows; amplitude {AW.CREASE_AMP}, "
            f"width {AW.CREASE_WIDTH}, bow {AW.BOW_AMP}, clip {AW.CREASE_CLIP}; "
            f"shading {_spread(AW._creases(ones, shade, {}))}", _as_image(isolated))
    else:
        add("fold_creases", "Fold creases and paper bowing", "off in plan", None, False)

    alone, shade = fork(), fork()
    arr = AW._illumination(arr, rng, meta, plan["edge_shadow"])
    isolated = AW._illumination(clean_arr, alone, {}, plan["edge_shadow"])
    check(alone, "illumination")
    edge = f"shadow on {meta['shadow_edge']} edge" if meta["edge_shadow"] else "no edge shadow (plan)"
    add("uneven_lighting", "Uneven lighting",
        f"directional falloff, darker corners; {edge}; "
        f"shading {_spread(AW._illumination(ones, shade, {}, plan['edge_shadow']))}", _as_image(isolated))

    alone = fork()
    arr = AW._colour_cast(arr, rng, meta)
    isolated = AW._colour_cast(clean_arr, alone, {})
    check(alone, "colour tint")
    add("colour_tint", "Colour tint (white balance)",
        f"RGB gains {meta['tint_rgb_gain']} (sd 0.012 each)", _as_image(isolated))

    exposure = rng.uniform(0.92, 1.06)
    arr *= exposure
    add(f"exposure_x{exposure:.2f}", "Exposure (brightness)", f"x{exposure:.3f} (range 0.92-1.06)",
        _as_image(clean_arr * exposure))
    mean = arr.mean()
    contrast = rng.uniform(0.94, 1.08)
    arr = (arr - mean) * contrast + mean
    meta.update(brightness=round(float(exposure), 4), contrast=round(float(contrast), 4))
    add(f"contrast_x{contrast:.2f}", "Contrast", f"x{contrast:.3f} about the mean (range 0.94-1.08)",
        _as_image((clean_arr - clean_arr.mean()) * contrast + clean_arr.mean()))

    alone = fork()
    arr = AW._normalise_paper(arr, rng, meta)
    isolated = AW._normalise_paper(clean_arr, alone, {})
    check(alone, "paper normalisation")
    add("paper_level", "Paper-level normalisation",
        f"paper (95th-percentile brightness) at {meta.get('paper_level', 'unchanged')}/255 "
        f"(range {AW.PAPER_LEVEL_TARGET})", _as_image(isolated))

    noise = rng.uniform(1.5, 4.5)
    meta["sensor_noise_sigma"] = round(float(noise), 3)
    noise_field = rng.normal(0.0, noise, size=arr.shape)
    arr += noise_field
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    add("sensor_noise", "Sensor noise", f"Gaussian, sd {noise:.2f} grey levels (range 1.5-4.5)",
        _as_image(clean_arr + noise_field))

    alone = fork()
    out = AW._defocus(Image.fromarray(arr), rng, meta, plan["blur"])
    isolated = AW._defocus(clean, alone, {}, plan["blur"])
    check(alone, "defocus")
    add("blur", "Blur (lens defocus)",
        f"Gaussian radius {meta['blur_radius_px']} px at {w}x{h} (range {AW.DEFOCUS_RADIUS})"
        if meta["blur"] else "off in plan", isolated if meta["blur"] else None, bool(meta["blur"]))

    m = AW.MARGIN_PX
    canvas = Image.new("RGB", (out.width + 2 * m, out.height + 2 * m), fill)
    canvas.paste(out, (m, m))
    framed = Image.new("RGB", (w + 2 * m, h + 2 * m), clean_fill)
    framed.paste(clean, (m, m))
    add("margin", "Margin", f"{m} px paper-coloured border at {w}x{h}", framed)
    add("jpeg_compression", "JPEG compression",
        f"quality {AW.JPEG_QUALITY}, chroma subsampling "
        f"{'4:2:0' if AW.JPEG_SUBSAMPLING == 2 else AW.JPEG_SUBSAMPLING}", AW.jpeg_roundtrip(clean))
    return steps, meta, AW.jpeg_roundtrip(canvas)


def _visibility(AW, meta: dict, plan: dict) -> float:
    """Distance from neutral, including defocus only when the plan enables it."""
    scores = [abs(meta["tilt_deg"]) / max(abs(v) for v in AW.TILT_RANGE),
              abs(meta["brightness"] - 1) / 0.08,
              abs(meta["contrast"] - 1) / 0.08,
              (meta["sensor_noise_sigma"] - 1.5) / 3.0]
    if plan["blur"]:
        scores.append((meta["blur_radius_px"] - AW.DEFOCUS_RADIUS[0]) /
                      (AW.DEFOCUS_RADIUS[1] - AW.DEFOCUS_RADIUS[0]))
    return float(np.mean(scores))


def _load(master_png: Path) -> tuple[Image.Image, Image.Image]:
    """(master as augment reads it, clean sheet as the sample set's worker draws it)."""
    import ecg_clean_worker as worker
    with Image.open(master_png) as image:
        image.load()
        return image.convert("RGB"), worker.recolour_clean(image)


def _scan(task: tuple) -> tuple[int, float]:
    position, master_png, seed, rel, plan, kit_dir = task
    import augment_worker as AW
    AW.set_texture_dir(Path(kit_dir) / "CreasesWrinkles" / "wrinkles-dataset")
    with Image.open(master_png) as master:
        _, meta = AW.augment_master(master, seed, rel, plan)
    AW.check_plan(meta, plan)
    return position, _visibility(AW, meta, plan)


def _panel(reference: Image.Image, image: Image.Image, title: str, setting: str) -> Image.Image:
    """Clean | step side by side, with the same lead I patch of both at 4x below."""
    pad, head, label = 20, 96, 40
    big, normal = ImageFont.load_default(size=34), ImageFont.load_default(size=26)
    zw, zh = (ZOOM_BOX[2] - ZOOM_BOX[0]) * ZOOM, (ZOOM_BOX[3] - ZOOM_BOX[1]) * ZOOM
    width = 2 * max(reference.width, zw) + 3 * pad
    canvas = Image.new("RGB", (width, head + 2 * label + reference.height + zh + pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), title, fill="black", font=big)
    draw.text((pad, 54), setting, fill=(70, 70, 70), font=normal)
    for column, (name, sheet) in enumerate((("Clean reference", reference), (title, image))):
        x = pad + column * (reference.width + pad)
        draw.text((x, head + 4), name, fill="black", font=normal)
        canvas.paste(sheet, (x, head + label))
        y = head + label + reference.height
        draw.text((x, y + 6), f"{ZOOM}x zoom, start of lead I", fill=(70, 70, 70), font=normal)
        canvas.paste(sheet.crop(ZOOM_BOX).resize((zw, zh), Image.NEAREST), (x, y + label))
    return canvas


README = """Each augmentation step on its own, on the same ECG
==================================================

ECG: {clean} (HEEDB record {record}; {labels})
Augmented image of this ECG in the sample set: {augmented}

The source is Clean12_Augmented_ECG_Images_EC2.ipynb / augment_worker.py:
the EC2 paper/photo pipeline, with stronger wrinkles/creases than stage 4.
It applies these steps in order, using a balanced per-record plan:
{order}

Here every step is applied on its own to the clean sheet (00), with all other
steps off, using exactly the setting this ECG's augmented image received.
Applying them sequentially produces 15: the replay was checked to reproduce
augment_worker.augment_master pixel for pixel, and 00 and 15 are
pixel-identical to the clean and augmented images in the sample folder.

By default the ECG is chosen because its plan enables every optional step
(perspective, creases, edge shadow and blur) and its settings are furthest
from neutral among eligible single-condition ECGs. If none enables all four,
the plan with the most optional effects wins, then distance from neutral;
skipped steps are listed below. An explicit --record overrides this choice.
Several steps are deliberately mild (noise, blur, JPEG, tint); the zoomed
lead I patch in comparisons/ shows them.
Skipped steps in this ECG: {skipped}.

Grid colour: 01a-01c show AW.AUG_PALETTES (orange, pink, grey) with the plan's
ink strength; the augmented image used {palette}. Red is the clean view's grid.

comparisons/: clean reference and the step side by side, with a 4x zoom of the
same lead I patch underneath. effects.csv: setting and mean pixel change
(0-255) relative to the clean reference for every image.
"""


def export(samples: Path, seed: int, record: str | None, notebook: Path, kit_bundle: Path,
           augment_dir: Path, workers: int) -> None:
    work = samples / "_work"
    with (samples / "index.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    config = base.prepare_render(work, notebook, kit_bundle, base.AUGMENT_CODE_DIR)
    augment_path = str(augment_dir)
    if augment_path not in sys.path:
        sys.path.insert(0, augment_path)
    pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if augment_path not in pythonpath:
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [augment_path, *pythonpath]))
    import ecg_clean_worker as worker
    import augment_worker as AW
    AW.set_texture_dir(Path(config["kit_dir"]) / "CreasesWrinkles" / "wrinkles-dataset")

    def plan_for(row):
        return {key: row[key] if key == "grid_palette" else int(row[key]) for key in AW.PLAN_FACTORS}

    if record:
        folder, number = record.rstrip("/").rsplit("/", 1)
        row = next((r for r in index if r["folder"] == folder
                    and Path(r["clean_image"]).name.startswith(f"{number}_")), None)
        if row is None:
            raise SystemExit(f"{record}: no such ECG in {samples / 'index.csv'}")
    else:
        candidates = [r for r in index if r["folder"].startswith("single/")]
        if not candidates:
            raise SystemExit("No single-condition ECGs in index.csv")
        most_enabled = max(sum(plan_for(r)[k] for k in OPTIONAL_STEPS) for r in candidates)
        candidates = [r for r in candidates if sum(plan_for(r)[k] for k in OPTIONAL_STEPS) == most_enabled]
        tasks = [(i, str(master_path(work, r["site"], r["file_id"])), seed, r["record_id"],
                  plan_for(r), config["kit_dir"]) for i, r in enumerate(candidates)]
        print(f"Choosing the augmentation-effects ECG among {len(tasks)} single-condition ECGs ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scores = [(score, i) for i, score in pool.map(_scan, tasks)]
        row = candidates[max(scores)[1]]

    master, clean = _load(master_path(work, row["site"], row["file_id"]))
    plan = plan_for(row)
    steps, params, combined = replay(AW, master, clean, (seed, row["record_id"], "clean12-aug"), plan)
    AW.check_plan(params, plan)
    missing = [key for key in OPTIONAL_STEPS if not plan[key]]
    if missing:
        print(f"WARNING: this ECG's plan skips {', '.join(missing)}; those steps are off")

    # Check the replay against the EC2 worker and both sample files before replacing output.
    expected, meta = AW.augment_master(master, seed, row["record_id"], plan)
    AW.check_plan(meta, plan)
    reference = worker.to_final(clean)
    final_combined = worker.to_final(combined)
    if not np.array_equal(np.asarray(final_combined), np.asarray(expected)):
        raise RuntimeError("step-by-step replay differs from augment_worker.augment_master")
    for mine, theirs in ((reference, row["clean_image"]), (final_combined, row["augmented_image"])):
        with Image.open(samples / theirs) as image:
            if not np.array_equal(np.asarray(image.convert("RGB")), np.asarray(mine)):
                raise RuntimeError(f"{theirs} differs from the replayed image")

    out = samples / "augmentation_effects"
    shutil.rmtree(out, ignore_errors=True)
    (out / "comparisons").mkdir(parents=True)
    level = config["png_compress_level"]
    images = [("00_clean_reference", "Clean reference", "the sample set's clean image", True, reference)]
    number = 0
    for step in steps:
        if step["key"].startswith("grid_colour_"):
            letter = "abc"[[s["key"] for s in steps].index(step["key"])]
            stem = f"01{letter}_{step['key']}"
        else:
            number += 1
            stem = f"{number + 1:02d}_{step['key']}"
        if step["image"] is not None:
            images.append((stem, step["effect"], step["setting"], step["applied"], worker.to_final(step["image"])))
    images.append((f"{number + 2:02d}_all_steps_combined", "All steps combined",
                   "the sample set's augmented image: every step above, in order", True, final_combined))

    ref = np.asarray(reference, dtype=np.int16)
    rows = []
    for stem, effect, setting, applied, image in images:
        image.save(out / f"{stem}.png", compress_level=level)
        comparison = f"comparisons/{stem}_vs_clean.png"
        if stem != "00_clean_reference":
            _panel(reference, image, effect, setting).save(out / comparison, compress_level=level)
        rows.append({"image": f"{stem}.png", "comparison": comparison if stem != "00_clean_reference" else "",
                     "effect": effect, "setting": setting,
                     "in_augmented_image": "yes" if applied else "no",
                     "mean_pixel_change": f"{np.abs(np.asarray(image, dtype=np.int16) - ref).mean():.2f}"})
        print(f"  {stem}.png  {effect}: {setting}")
    with (out / "effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    order = [s for s in steps if not s["key"].startswith("grid_colour_")]
    (out / "README.txt").write_text(README.format(
        clean=row["clean_image"], augmented=row["augmented_image"], record=row["record_id"],
        labels=row["label_names"], palette=params["grid_palette"], skipped=", ".join(missing) or "none",
        order="\n".join(f"  {i:2d}. {s['effect']}" for i, s in enumerate(
            [{"effect": "Grid colour (with ink strength and paper shade)"}, *order], 1))))
    print(f"Augmentation effects: {len(rows)} images of {row['clean_image']} in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=Path, required=True,
                        help="folder written by export_correct_combinations.py (holds index.csv)")
    parser.add_argument("--record", help="<folder>/<NN>, e.g. single/STACH_Sinus_tachycardia/03")
    parser.add_argument("--seed", type=int, help="must match the sample set; defaults to EC2 notebook AUGMENT_SEED")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--notebook", type=Path,
                        default=base.RENDER_CODE_DIR / "Clean12_ECG_Signal_Images_EC2.ipynb")
    parser.add_argument("--kit-bundle", type=Path, default=base.RENDER_CODE_DIR / "ecg_image_v1_code.tar.gz")
    parser.add_argument("--augment-dir", type=Path, default=AUGMENT_DIR)
    args = parser.parse_args()
    if args.seed is None:
        nb = json.loads((args.augment_dir / "Clean12_Augmented_ECG_Images_EC2.ipynb").read_text())
        source = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        args.seed = int(re.search(r"^AUGMENT_SEED = (\d+)", source, re.M).group(1))
    export(args.samples.expanduser().resolve(), args.seed, args.record, args.notebook.resolve(),
           args.kit_bundle.resolve(), args.augment_dir.resolve(), args.workers)


if __name__ == "__main__":
    main()
