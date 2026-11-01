#!/usr/bin/env python3
"""Add one augmented sheet per ECG to a correct-prediction sample set, one
augmentation combination per condition, for the paper's dataset preview.

Takes the folder ``export_correct_predictions.py`` wrote and, for every ECG in
it, writes ``NN_<site>_<file_id>_augmented_<grid>_tilt<deg>.png`` next to its
``_clean.png``. All ECGs of a condition get that condition's combination from
COMBINATIONS (grid colour, tilt, ink strength, keystone, creases, edge shadow,
blur), so whichever clean sheet is picked for the preview, its augmented
counterpart shows the combination planned for that condition.

The augmentation is the image pipeline's own ``stage4_augment.augment`` (the
paper/photo stage the training corpus used), applied to the grey kit master
of the same render and fitted to 1024x768 exactly as the clean view. No effect
is hand-set except the grid palette, which ``augment(force_palette=...)``
overrides after drawing it, so every other random value is unchanged. The other
properties are found by trying augmentation draws ``(seed, record_path, k)``,
k = 0, 1, 2, ..., and keeping the first draw that produces the combination;
every effect therefore keeps its corpus distribution (texture strength, crease
depth, shadow width, blur radius, noise, JPEG).

The grey masters are not kept by the first export, so the 60 records are
rendered once more (into ``_work/masters``) and each re-render is checked to be
pixel-identical to the clean sheet already in the folder.

Usage::

    ~/.cache/ecgkit-venv/bin/python export_augmented_preview.py \\
        --samples ../../prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42/correct_single_label_samples
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import export_cardiologist_samples as base


@dataclass(frozen=True)
class Combination:
    grid: str        # red / orange / pink / grey
    tilt: float      # degrees; +-3 means the 2.8-3.0 end of the range
    ink: str | None  # faded (grid saturation <= 0.80) / fresh (>= 0.95); None for grey (no hue)
    keystone: bool   # slight perspective distortion
    creases: bool    # fold lines + paper bowing
    shadow: bool     # edge shadow (uneven lighting and corner falloff are always on)
    blur: bool       # mild defocus

    def describe(self) -> str:
        parts = [f"{self.grid} grid", *([f"{self.ink} ink"] if self.ink else []), f"tilt {self.tilt:+.0f} deg"]
        parts += [name for name, on in (("perspective", self.keystone), ("fold creases", self.creases),
                                        ("edge shadow", self.shadow), ("blur", self.blur)) if on]
        return ", ".join(parts)


# Every row differs; each grid colour appears 3 times, tilt spans -3..+3, and the
# optional effects appear at roughly their corpus rates (perspective 0.70,
# creases 0.62, edge shadow 0.65, blur 0.55, faded ink the common case).
COMBINATIONS = {
    "NORM":  Combination("red",    0.0, "fresh", False, False, False, False),
    "SB":    Combination("orange", 1.0, "faded", True,  False, True,  False),
    "STACH": Combination("pink",  -1.0, "fresh", True,  True,  False, False),
    "LBBB":  Combination("grey",   2.0, None,    True,  False, True,  True),
    "LQT":   Combination("red",   -2.0, "faded", False, True,  True,  False),
    "AF":    Combination("red",   -3.0, "fresh", True,  True,  True,  False),
    "AFL":   Combination("orange", 3.0, "faded", True,  True,  False, True),
    "3AVB":  Combination("pink",   2.0, "faded", False, False, True,  True),
    "2AVB":  Combination("grey",  -2.0, None,    True,  True,  False, True),
    "1AVB":  Combination("orange",-1.0, "fresh", False, True,  True,  True),
    "STEMI": Combination("grey",   1.0, None,    True,  False, True,  True),
    "STE":   Combination("pink",  -3.0, "faded", True,  True,  False, False),
}
FADED_MAX, FRESH_MIN = 0.80, 0.95
KEYSTONE_PROB = 0.7  # stage4_augment.augment: `if rng.random() < 0.7` right after the tilt


def ink_matches(target: Combination, saturation: float | None) -> bool:
    if target.ink is None:
        return True
    return saturation <= FADED_MAX if target.ink == "faded" else saturation >= FRESH_MIN


def tilt_matches(target: float, deg: float) -> bool:
    if abs(target) >= 3.0:
        return deg * np.sign(target) >= 2.8
    if target == 0.0:
        return abs(deg) < 0.15
    return abs(deg - target) <= 0.2


# --------------------------------------------------------------------------
# Pool children: stage4_augment with probes that reject a draw as soon as it
# departs from the target combination (before the expensive steps run).
# --------------------------------------------------------------------------
class _Reject(Exception):
    pass


_STATE: dict = {}


class _RecordingRng:
    """Forwards to the real generator and records random() results."""

    def __init__(self, rng: np.random.Generator):
        self._rng, self.randoms = rng, []

    def random(self, *args, **kwargs):
        value = self._rng.random(*args, **kwargs)
        self.randoms.append(value)
        return value

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _init_preview(kit_dir: str) -> None:
    import stage4_augment as sa
    sa.C.KIT = Path(kit_dir)  # wrinkled-paper photographs ship with the kit bundle
    recolour, grid_lut, rotate, creases, illumination, defocus = (
        sa._recolour_grid, sa._grid_lut, sa._rotate_on_paper, sa._creases, sa._illumination, sa._defocus)

    def recolour_probe(img, rng, force=None):
        _STATE["rng"] = rng  # for the keystone draw in rotate_probe
        return recolour(img, rng, force)

    def grid_lut_probe(palette, rng):
        lut = grid_lut(palette, rng)
        # The LUT passes through the major-line colour scaled about its own
        # luminance by the drawn saturation (ink strength); read it back.
        nominal = np.asarray(palette[0], dtype=np.float64)
        drawn = np.asarray(lut[sa.C.GRID_MAJOR_LEVEL], dtype=np.float64)
        luma = np.array([0.299, 0.587, 0.114])
        chroma = np.linalg.norm(nominal - nominal @ luma)
        # A neutral (grey) grid has no hue for the saturation to act on.
        _STATE["saturation"] = (float(np.linalg.norm(drawn - drawn @ luma) / chroma)
                                if chroma > 1.0 else None)
        return lut

    def rotate_probe(img, deg, fill):
        target = _STATE["target"]
        keystone = _peek(_STATE["rng"]).random() < KEYSTONE_PROB
        if not (ink_matches(target, _STATE["saturation"]) and tilt_matches(target.tilt, deg)
                and keystone == target.keystone):
            raise _Reject
        _STATE.update(keystone=keystone, creases=False)
        return rotate(img, deg, fill)

    def creases_probe(arr, rng):
        _STATE["creases"] = True
        return creases(arr, rng)

    def illumination_probe(arr, rng):
        if _STATE["creases"] != _STATE["target"].creases:
            raise _Reject
        recording = _RecordingRng(rng)
        out = illumination(arr, recording)
        _STATE["shadow"] = bool(recording.randoms[0] < 0.65)  # its only random(): the edge shadow
        if _STATE["shadow"] != _STATE["target"].shadow:
            raise _Reject
        return out

    def defocus_probe(img, rng):
        out = defocus(img, rng)
        _STATE["blur"] = out is not img
        if _STATE["blur"] != _STATE["target"].blur:
            raise _Reject
        return out

    sa._recolour_grid, sa._grid_lut, sa._rotate_on_paper = recolour_probe, grid_lut_probe, rotate_probe
    sa._creases, sa._illumination, sa._defocus = creases_probe, illumination_probe, defocus_probe


def _peek(rng: np.random.Generator) -> np.random.Generator:
    """An independent generator at rng's current state (look at the next draw)."""
    copy = np.random.Generator(type(rng.bit_generator)())
    copy.bit_generator.state = rng.bit_generator.state
    return copy


def _prefix_matches(sa, seed_parts: tuple, target: Combination) -> bool:
    """Cheap screen of a draw's grid ink, tilt and keystone on a 1x1 sheet.

    Replays augment()'s first draws: the grid recolour (real code; draw count
    does not depend on image size), then its tilt and keystone draws. Only a
    speed-up: draws that pass still go through the full augment(), whose
    probes decide.
    """
    rng = sa._rng(seed_parts)
    sa._recolour_grid(Image.new("L", (1, 1), 255), rng, target.grid)
    deg = rng.uniform(-3.0, 3.0)
    keystone = rng.random() < KEYSTONE_PROB
    return (ink_matches(target, _STATE["saturation"]) and tilt_matches(target.tilt, deg)
            and keystone == target.keystone)


def augment_one(job: dict, seed: int, max_draws: int, compress_level: int) -> dict:
    import ecg_clean_worker as worker
    import stage4_augment as sa
    target = Combination(**job["combination"])
    _STATE["target"] = target
    with tempfile.TemporaryDirectory() as tmp:
        photo = Path(tmp) / "augmented.jpg"
        for draw in range(max_draws):
            if not _prefix_matches(sa, (seed, job["rel"], draw), target):
                continue
            try:
                meta = sa.augment(Path(job["master"]), photo, (seed, job["rel"], draw),
                                  force_palette=target.grid)
            except _Reject:
                continue
            name = f"{job['stem']}_augmented_{meta['grid_palette']}_tilt{meta['rotation_deg']:+.1f}.png"
            with Image.open(photo) as image:
                worker.to_final(image).save(Path(job["dir"]) / name, compress_level=compress_level)
            return {"image": name, "draw": draw, "grid": meta["grid_palette"],
                    "tilt_deg": meta["rotation_deg"],
                    "grid_saturation": "" if _STATE["saturation"] is None else round(_STATE["saturation"], 3),
                    **{k: _STATE[k] for k in ("keystone", "creases", "shadow", "blur")}}
    raise RuntimeError(f"{job['rel']}: no draw in {max_draws} gives {target.describe()}")


# --------------------------------------------------------------------------
# Parent
# --------------------------------------------------------------------------
def ensure_masters(rows: list[dict], samples: Path, work: Path, args) -> None:
    """Re-render records whose grey master is missing; check them against the clean sheets."""
    missing = [r for r in rows if not (work / "masters" / f"{r['file_id']}.png").is_file()]
    if not missing:
        return
    scratch = work / "rerender"
    shutil.rmtree(scratch, ignore_errors=True)
    jobs = [{"rel": r["record_path"], "dir": str(scratch / r["folder"]),
             "stem": Path(r["clean_image"]).name.removesuffix("_clean.png"),
             "master_copy": str(work / "masters" / f"{r['file_id']}.png")} for r in missing]
    results = base.render(jobs, work / "signals", work, args.notebook.resolve(), args.kit_bundle.resolve(),
                          args.augment_code.resolve(), args.workers, args.seed, 0)
    shutil.rmtree(work / "kit_scratch", ignore_errors=True)
    errors = [f"{rel}: {res['error']}" for rel, res in results.items() if "error" in res]
    if errors:
        raise SystemExit("re-render failed:\n  " + "\n  ".join(errors))
    for row in missing:
        with Image.open(scratch / row["clean_image"]) as new, Image.open(samples / row["clean_image"]) as old:
            if not np.array_equal(np.asarray(new), np.asarray(old)):
                (work / "masters" / f"{row['file_id']}.png").unlink()
                raise SystemExit(f"{row['clean_image']}: re-render differs from the existing clean sheet")
    shutil.rmtree(scratch)
    print(f"Re-rendered {len(missing)} masters; every one matches its clean sheet pixel for pixel")


PREVIEW_README = """Augmented preview: one augmentation combination per condition
==============================================================

Each NN_<site>_<id>_augmented_<grid>_tilt<deg>.png is the clean sheet with
the same number prefix after the image pipeline's paper/photo augmentation
(stage4_augment.augment, as used for the training corpus): grid recoloured to
a paper style with varying ink strength, rotation within +-3 deg, usually a
slight perspective distortion, wrinkled-paper texture, occasional fold
creases, uneven lighting with corner and edge shadows, small tint, brightness
and contrast changes, mild sensor noise, occasional slight blur, and JPEG
compression. All ECGs of a condition share that condition's combination:

{table}

Always on: wrinkled-paper texture, uneven lighting / corner falloff, tint,
brightness, contrast, sensor noise, JPEG. Faded ink: grid saturation <= {faded};
fresh: >= {fresh} (corpus range 0.72-1.02); a grey grid has no hue, so ink
strength is not set for it. Tilt +-3 means 2.8-3.0 deg.

The grid colour is forced; everything else is the first augmentation draw
(seed {seed}, record path, draw number k) that produced the combination, so
effect strengths follow the corpus distribution. augmented_preview.csv lists
the measured values per image.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=Path, required=True,
                        help="folder written by export_correct_predictions.py (holds index.csv)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-draws", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--notebook", type=Path,
                        default=base.RENDER_CODE_DIR / "Clean12_ECG_Signal_Images_EC2.ipynb")
    parser.add_argument("--kit-bundle", type=Path, default=base.RENDER_CODE_DIR / "ecg_image_v1_code.tar.gz")
    parser.add_argument("--augment-code", type=Path, default=base.AUGMENT_CODE_DIR)
    args = parser.parse_args()

    samples = args.samples.expanduser().resolve()
    work = samples / "_work"
    with (samples / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    ensure_masters(rows, samples, work, args)
    config = base.prepare_render(work, args.notebook.resolve(), args.kit_bundle.resolve(),
                                 args.augment_code.resolve())

    jobs = []
    for row in rows:
        folder = samples / row["folder"]
        for stale in folder.glob("*_augmented*.png"):
            stale.unlink()
        code = row["folder"].split("_", 1)[0]
        jobs.append({"rel": row["record_path"], "dir": str(folder),
                     "master": str(work / "masters" / f"{row['file_id']}.png"),
                     "stem": Path(row["clean_image"]).name.removesuffix("_clean.png"),
                     "code": code, "combination": asdict(COMBINATIONS[code])})

    print(f"Augmenting {len(jobs)} ECGs, {len(COMBINATIONS)} combinations, {args.workers} workers ...", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_preview,
                             initargs=(config["kit_dir"],)) as pool:
        futures = {pool.submit(augment_one, job, args.seed, args.max_draws, config["png_compress_level"]):
                   job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            results[job["rel"]] = future.result()
            print(f"  {len(results)}/{len(jobs)} {job['code']:<6} {results[job['rel']]['image']} "
                  f"(draw {results[job['rel']]['draw']})", flush=True)

    preview_rows = []
    for row, job in zip(rows, jobs):
        result = results[job["rel"]]
        row["augmented_images"] = f"{row['folder']}/{result['image']}"
        preview_rows.append({"folder": row["folder"], "clean_image": row["clean_image"],
                             "augmented_image": row["augmented_images"],
                             "combination": COMBINATIONS[job["code"]].describe(),
                             "augmentation_draw": result["draw"], "grid": result["grid"],
                             "tilt_deg": result["tilt_deg"], "grid_saturation": result["grid_saturation"],
                             "perspective": int(result["keystone"]), "fold_creases": int(result["creases"]),
                             "edge_shadow": int(result["shadow"]), "blur": int(result["blur"])})
    for name, table in (("index.csv", rows), ("augmented_preview.csv", preview_rows)):
        with (samples / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (samples / "AUGMENTED_PREVIEW.txt").write_text(PREVIEW_README.format(
        table="\n".join(f"  {code:<6} {combo.describe()}" for code, combo in COMBINATIONS.items()),
        faded=FADED_MAX, fresh=FRESH_MIN, seed=args.seed))
    print(f"Done: {len(results)} augmented sheets under {samples}")


if __name__ == "__main__":
    main()
