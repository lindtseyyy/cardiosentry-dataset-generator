"""Augmented clean12 worker: one paper/photo sheet per clean12 ECG record.

Each record is rendered by the clean12 render worker (ecg_clean_worker, the
module embedded in Clean12_ECG_Signal_Images_EC2.ipynb: canonical WFDB
staging + the patched ECG-image-kit with the clean12 flags) to its 2200x1700
grey kit master, exactly as for the clean12 sheets. The master then goes
through the image pipeline's stage-4 paper/photo augmentation
(dataset-combination-code/ecg-dataset-pipeline/image_pipeline/stage4_augment.py,
the code behind the correct_prediction_samples previews), a JPEG q85 round
trip, and the clean view's own 1024x768 fit (ecg_clean_worker.to_final).

The stage-4 functions below are a line-for-line port (checked pixel for
pixel against stage4_augment.augment with the stage-4 settings). Two changes:
wrinkled paper and fold creases are stronger than stage 4 (WRINKLE/CREASE
settings below, raised on request after comparing previews), and with a plan
the balanced factors (grid palette, ink strength, tilt, perspective, fold
creases, edge shadow, blur) are forced to the planned level. Every other
value keeps its stage-4 distribution.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import ecg_clean_worker as R

# ---------------------------------------------------------------- stage-4 config
# Values copied from image_pipeline/config.py (the corpus stage-4 settings),
# except the wrinkle/crease block marked STRONGER.
STAGE4_SEED = 20260818
GRID_PALETTES = {
    # major gridline        minor gridline        paper
    "red":    ((178, 86, 84), (228, 180, 176), (253, 249, 247)),
    "pink":   ((188, 124, 148), (236, 200, 212), (254, 251, 252)),
    "orange": ((186, 126, 86), (236, 202, 176), (254, 250, 245)),
    "grey":   ((124, 124, 124), (198, 198, 198), (252, 252, 252)),
}
GRID_PALETTE_WEIGHTS = {"red": 0.38, "orange": 0.22, "pink": 0.20, "grey": 0.20}
GRID_SATURATION = (0.72, 1.02)
GRID_TRACE_MAX = 55
GRID_MAJOR_LEVEL = 102
GRID_MINOR_LEVEL = 191
GRID_PAPER_LEVEL = 250
TILT_RANGE = (-3.0, 3.0)
KEYSTONE_PROB = 0.7
KEYSTONE_MAG = (0.004, 0.018)
# STRONGER than stage 4 (stage 4: texture 0.30-0.62 clip 0.86-1.10; crease
# amp 0.06-0.13 clip 0.88-1.14 width 0.006-0.020; bow 0.02-0.06). Wrinkles
# read clearly and folds show as sharp ridges; the darkest fold still sits far
# above the trace ink, so traces stay legible.
TEXTURE_STRENGTH = (0.65, 1.00)
TEXTURE_CLIP = (0.76, 1.14)
CREASE_PROB = 0.62               # stage-4 draw; the balanced plan sets creases on/off
CREASE_AMP = (0.16, 0.26)
CREASE_CLIP = (0.76, 1.22)
CREASE_WIDTH = (0.004, 0.012)   # fraction of the long side
BOW_AMP = (0.03, 0.08)
EDGE_SHADOW_PROB = 0.65
PAPER_LEVEL_TARGET = (216, 248)
DEFOCUS_PROB = 0.55
DEFOCUS_RADIUS = (0.3, 0.9)
MARGIN_PX = 18
JPEG_QUALITY = 85
JPEG_SUBSAMPLING = 2  # 4:2:0

# ---------------------------------------------------------------- balanced plan
AUG_PALETTES = ("orange", "pink", "grey")  # red is the clean view's grid
INK_BINS = ((0.72, 0.82), (0.82, 0.92), (0.92, 1.02))
TILT_BINS = ((-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
PLAN_FACTORS = {
    "grid_palette": AUG_PALETTES,
    "ink_bin": tuple(range(len(INK_BINS))),
    "tilt_bin": tuple(range(len(TILT_BINS))),
    "perspective": (0, 1),
    "fold_creases": (0, 1),
    "edge_shadow": (0, 1),
    "blur": (0, 1),
}

MANIFEST_FIELDS = [
    "record_path", "image_path", "status", "error", "image_bytes",
    "grid_palette", "ink_bin", "grid_saturation", "ink_level", "tilt_bin", "tilt_deg",
    "perspective", "perspective_mag", "wrinkle_texture", "wrinkle_strength",
    "fold_creases", "fold_lines", "paper_bows", "light_angle_deg", "light_ramp",
    "corner_falloff", "edge_shadow", "shadow_edge", "shadow_depth", "shadow_width",
    "tint_rgb_gain", "brightness", "contrast", "paper_level", "sensor_noise_sigma",
    "blur", "blur_radius_px",
]

TEXTURE_DIR: Path | None = None
_TEXTURES: tuple[Path, ...] | None = None


def set_texture_dir(path) -> None:
    global TEXTURE_DIR, _TEXTURES
    TEXTURE_DIR, _TEXTURES = Path(path), None


def _texture_files() -> tuple[Path, ...]:
    """The kit's wrinkled-paper photographs (CreasesWrinkles/wrinkles-dataset)."""
    global _TEXTURES
    if _TEXTURES is None:
        if TEXTURE_DIR is None or not TEXTURE_DIR.is_dir():
            raise RuntimeError(f"wrinkle texture folder missing: {TEXTURE_DIR}")
        _TEXTURES = tuple(sorted(p for p in TEXTURE_DIR.iterdir()
                                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
        if not _TEXTURES:
            raise RuntimeError(f"no wrinkle textures in {TEXTURE_DIR}")
    return _TEXTURES


def blake_int(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def stage4_rng(seed_parts: tuple) -> np.random.Generator:
    """stage4_augment._rng: per-image generator keyed by output identity only."""
    return np.random.default_rng(blake_int(*seed_parts) ^ STAGE4_SEED)


# ---------------------------------------------------------------- stage-4 port
def _paper_colour(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    k = max(8, min(h, w) // 40)
    patches = np.concatenate([
        img[:k, :k].reshape(-1, 3), img[:k, -k:].reshape(-1, 3),
        img[-k:, :k].reshape(-1, 3), img[-k:, -k:].reshape(-1, 3)])
    return np.median(patches, axis=0)


def _rotate_on_paper(img: Image.Image, deg: float, fill) -> Image.Image:
    w, h = img.size
    big = img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    bw, bh = big.size
    left, top = (bw - w) // 2, (bh - h) // 2
    return big.crop((left, top, left + w, top + h))


def _illumination(arr, rng, meta, shadow=None):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
    ramp_amp = rng.uniform(0.10, 0.22)
    field = 1.0 + ramp_amp * ramp * 2.0
    r = np.sqrt((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    falloff = rng.uniform(0.06, 0.18)
    field *= 1.0 - falloff * (r / r.max()) ** 2
    drawn = rng.random() < EDGE_SHADOW_PROB
    on = drawn if shadow is None else bool(shadow)
    meta.update(light_angle_deg=round(float(np.degrees(ang)), 1),
                light_ramp=round(float(ramp_amp), 4), corner_falloff=round(float(falloff), 4),
                edge_shadow=int(on), shadow_edge="", shadow_depth="", shadow_width="")
    if on:
        edge = int(rng.integers(0, 4))
        d = {0: xx / w, 1: 1 - xx / w, 2: yy / h, 3: 1 - yy / h}[edge]
        depth, width = rng.uniform(0.10, 0.26), rng.uniform(0.04, 0.12)
        field *= 1.0 - depth * np.exp(-d / width)
        meta.update(shadow_edge=("left", "right", "top", "bottom")[edge],
                    shadow_depth=round(float(depth), 4), shadow_width=round(float(width), 4))
    return arr * field[..., None]


def _paper_texture(arr, rng, meta):
    files = _texture_files()
    h, w = arr.shape[:2]
    pick = files[int(rng.integers(len(files)))]
    tex = Image.open(pick).convert("L")
    tw, th = tex.size
    scale = rng.uniform(0.55, 1.0)
    cw, ch = max(16, int(tw * scale)), max(16, int(th * scale))
    x0 = int(rng.integers(0, max(1, tw - cw + 1)))
    y0 = int(rng.integers(0, max(1, th - ch + 1)))
    tex = tex.crop((x0, y0, x0 + cw, y0 + ch))
    if rng.random() < 0.5:
        tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        tex = tex.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.5:
        tex = tex.transpose(Image.ROTATE_90)
    field = np.asarray(tex.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    mean = float(field.mean())
    if mean <= 1e-3:
        raise RuntimeError(f"blank wrinkle texture {pick.name}")
    strength = rng.uniform(*TEXTURE_STRENGTH)
    meta.update(wrinkle_texture=pick.name, wrinkle_strength=round(float(strength), 4))
    field = 1.0 + (field / mean - 1.0) * strength
    return arr * np.clip(field, *TEXTURE_CLIP)[..., None]


def _creases(arr, rng, meta):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.ones((h, w), dtype=np.float32)
    folds = []
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:
            pos = rng.uniform(0.12, 0.88) * h
            tilt = rng.uniform(-0.06, 0.06)
            dist = yy - (pos + tilt * (xx - w / 2))
            folds.append("h")
        else:
            pos = rng.uniform(0.12, 0.88) * w
            tilt = rng.uniform(-0.06, 0.06)
            dist = xx - (pos + tilt * (yy - h / 2))
            folds.append("v")
        width = rng.uniform(*CREASE_WIDTH) * max(h, w)
        amp = rng.uniform(*CREASE_AMP)
        g = np.exp(-(dist ** 2) / (2 * width ** 2))
        field *= 1.0 + amp * g * (dist / width) / 1.6
    bows = int(rng.integers(1, 3))
    for _ in range(bows):
        k = rng.uniform(0.6, 1.8)
        phase = rng.uniform(0, 2 * np.pi)
        axis = yy / h if rng.random() < 0.5 else xx / w
        field *= 1.0 + rng.uniform(*BOW_AMP) * np.sin(2 * np.pi * k * axis + phase)
    meta.update(fold_lines="".join(folds), paper_bows=bows)
    return arr * np.clip(field, *CREASE_CLIP)[..., None]


def _normalise_paper(arr, rng, meta):
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    paper = float(np.percentile(lum, 95))
    if paper <= 1.0:
        return arr
    target = rng.uniform(*PAPER_LEVEL_TARGET)
    meta["paper_level"] = round(float(target), 2)
    return arr * (target / paper)


def _colour_cast(arr, rng, meta):
    gains = 1.0 + rng.normal(0.0, 0.012, size=3)
    meta["tint_rgb_gain"] = "/".join(f"{g:.4f}" for g in gains)
    return arr * gains[None, None, :]


def _grid_lut(palette, rng, sat_bin=None):
    major, minor, paper = (np.asarray(c, dtype=np.float32) for c in palette)
    ink = np.full(3, rng.uniform(8, 34), dtype=np.float32)
    ink += rng.normal(0, 3, size=3)
    paper = paper * rng.uniform(0.985, 1.005) + rng.normal(0, 1.2, size=3)
    sat = rng.uniform(*GRID_SATURATION)
    if sat_bin is not None:  # same draw, mapped into the planned bin
        lo, hi = INK_BINS[sat_bin]
        u = (sat - GRID_SATURATION[0]) / (GRID_SATURATION[1] - GRID_SATURATION[0])
        sat = lo + u * (hi - lo)

    def _sat(c):
        lum = float(np.dot(c, (0.299, 0.587, 0.114)))
        return np.clip(lum + (c - lum) * sat, 0, 255)

    major, minor = _sat(major), _sat(minor)
    xs = np.array([0, GRID_TRACE_MAX, GRID_MAJOR_LEVEL, GRID_MINOR_LEVEL,
                   GRID_PAPER_LEVEL, 255], dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    grid = np.arange(256, dtype=np.float32)
    for ch in range(3):
        ys = np.array([ink[ch] * 0.35, ink[ch], major[ch], minor[ch], paper[ch],
                       min(255.0, paper[ch] + 3)], dtype=np.float32)
        lut[:, ch] = np.interp(grid, xs, ys)
    return np.clip(lut, 0, 255), float(sat), float(ink.mean())


def _recolour_grid(master: Image.Image, rng, force=None, sat_bin=None):
    names = list(GRID_PALETTES)
    weights = np.array([GRID_PALETTE_WEIGHTS.get(n, 0.0) for n in names], dtype=np.float64)
    weights = weights / weights.sum()
    drawn = names[int(rng.choice(len(names), p=weights))]  # draw, THEN override (stage 4)
    name = force if force in GRID_PALETTES else drawn
    lut, sat, ink = _grid_lut(GRID_PALETTES[name], rng, sat_bin)
    out = Image.fromarray(lut[np.asarray(master.convert("L"))].astype(np.uint8), "RGB")
    return out, name, sat, ink


def _defocus(img, rng, meta, blur=None):
    drawn = rng.random() < DEFOCUS_PROB
    on = drawn if blur is None else bool(blur)
    meta.update(blur=int(on), blur_radius_px="")
    if not on:
        return img
    radius = float(rng.uniform(*DEFOCUS_RADIUS))
    meta["blur_radius_px"] = round(radius, 4)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def augment(master: Image.Image, rng: np.random.Generator, plan: dict | None = None):
    """stage4_augment.augment on a grey kit master (RGB), in memory.

    Returns (RGB sheet with the stage-4 margin, effect metadata). plan keys are
    PLAN_FACTORS; plan=None reproduces stage 4 draw for draw.
    """
    plan = plan or {}
    meta: dict = {}
    img, palette, sat, ink = _recolour_grid(master, rng, plan.get("grid_palette"), plan.get("ink_bin"))
    meta.update(grid_palette=palette, grid_saturation=round(sat, 4), ink_level=round(ink, 2))
    arr0 = np.asarray(img, dtype=np.float32)
    fill = tuple(int(v) for v in _paper_colour(arr0))

    deg = float(rng.uniform(*TILT_RANGE))
    if "tilt_bin" in plan:
        lo, hi = TILT_BINS[plan["tilt_bin"]]
        deg = lo + (deg - TILT_RANGE[0]) / (TILT_RANGE[1] - TILT_RANGE[0]) * (hi - lo)
    meta["tilt_deg"] = round(deg, 3)
    img = _rotate_on_paper(img, deg, fill)

    keystone = rng.random() < KEYSTONE_PROB
    if "perspective" in plan:
        keystone = bool(plan["perspective"])
    meta.update(perspective=int(keystone), perspective_mag="")
    if keystone:
        w, h = img.size
        m = rng.uniform(*KEYSTONE_MAG)
        dx, dy = m * w, m * h
        quad = (rng.uniform(0, dx), rng.uniform(0, dy),
                rng.uniform(0, dx), h - rng.uniform(0, dy),
                w - rng.uniform(0, dx), h - rng.uniform(0, dy),
                w - rng.uniform(0, dx), rng.uniform(0, dy))
        meta["perspective_mag"] = round(float(m), 5)
        img = img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=fill)

    arr = np.asarray(img, dtype=np.float32)
    arr = _paper_texture(arr, rng, meta)
    creases = rng.random() < CREASE_PROB
    if "fold_creases" in plan:
        creases = bool(plan["fold_creases"])
    meta.update(fold_creases=int(creases), fold_lines="", paper_bows="")
    if creases:
        arr = _creases(arr, rng, meta)
    arr = _illumination(arr, rng, meta, plan.get("edge_shadow"))
    arr = _colour_cast(arr, rng, meta)
    exposure = rng.uniform(0.92, 1.06)
    arr *= exposure
    mean = arr.mean()
    contrast = rng.uniform(0.94, 1.08)
    arr = (arr - mean) * contrast + mean
    meta.update(brightness=round(float(exposure), 4), contrast=round(float(contrast), 4))
    arr = _normalise_paper(arr, rng, meta)
    noise = rng.uniform(1.5, 4.5)
    meta["sensor_noise_sigma"] = round(float(noise), 3)
    arr += rng.normal(0.0, noise, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = _defocus(Image.fromarray(arr), rng, meta, plan.get("blur"))

    m = MARGIN_PX
    canvas = Image.new("RGB", (out.width + 2 * m, out.height + 2 * m), fill)
    canvas.paste(out, (m, m))
    return canvas, meta


def jpeg_roundtrip(img: Image.Image) -> Image.Image:
    """Stage 4 writes its photograph as JPEG q85 4:2:0; keep those artifacts."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def augment_master(master: Image.Image, seed: int, record_path: str, plan: dict):
    """Grey kit master -> final 1024x768 augmented sheet + metadata."""
    sheet, meta = augment(master.convert("RGB"), stage4_rng((seed, record_path, "clean12-aug")), plan)
    return R.to_final(jpeg_roundtrip(sheet)), meta


def check_plan(meta: dict, plan: dict) -> None:
    """Raise unless the rendered effects are exactly the planned levels."""
    for factor in ("grid_palette", "perspective", "fold_creases", "edge_shadow", "blur"):
        if meta[factor] != plan[factor]:
            raise RuntimeError(f"{factor}: planned {plan[factor]!r}, rendered {meta[factor]!r}")
    lo, hi = TILT_BINS[plan["tilt_bin"]]
    if not lo - 1e-3 <= meta["tilt_deg"] <= hi + 1e-3:
        raise RuntimeError(f"tilt {meta['tilt_deg']} outside planned bin {lo}..{hi}")
    lo, hi = INK_BINS[plan["ink_bin"]]
    if not lo - 1e-3 <= meta["grid_saturation"] <= hi + 1e-3:
        raise RuntimeError(f"ink strength {meta['grid_saturation']} outside planned bin {lo}..{hi}")


# ---------------------------------------------------------------- balanced plan
def balanced_plan(records: list[dict], seed: int) -> dict[str, dict]:
    """Assign every factor level equally, independently of the labels.

    records: dicts with record_path, site, labels. For each factor the records
    are ordered by stratum (site, label set) and, inside a stratum, by a
    per-factor seeded hash; levels are dealt round-robin along that order with
    one counter that runs across strata. So every factor is exactly balanced
    overall (counts differ by at most 1), balanced to +-1 inside every label
    combination, and paired with the other factors by independent shuffles.
    """
    keyed = sorted(records, key=lambda r: (r["site"], r["labels"], r["record_path"]))
    plan = {r["record_path"]: {} for r in keyed}
    for factor, levels in PLAN_FACTORS.items():
        order = sorted(keyed, key=lambda r: (r["site"], r["labels"],
                                            blake_int(seed, factor, r["record_path"])))
        offset = blake_int(seed, factor, "offset") % len(levels)
        for i, r in enumerate(order):
            plan[r["record_path"]][factor] = levels[(i + offset) % len(levels)]
    return plan


# ---------------------------------------------------------------- pool worker
_SEED: int | None = None


def init_worker(config: dict) -> None:
    """config: the clean12 render_config plus seed, texture_dir, input_dir, output_dir."""
    global _SEED
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    R.init_worker(config, config["input_dir"], config["output_dir"])
    set_texture_dir(config["texture_dir"])
    _texture_files()
    _SEED = int(config["seed"])


def process_batch(jobs: list[dict]) -> list[dict]:
    """jobs: record_path, image_path, plan. One kit invocation renders the whole
    batch to grey masters (scratch, deleted afterwards); each master is then
    augmented and written to OUTPUT/image_path (temp file + atomic rename)."""
    scratch = Path(R.CFG["kit_scratch_dir"]) / f"a{os.getpid()}-{int(time.time() * 1000) % 100000000}"
    staged, render = scratch / "staged", scratch / "render"
    rows, todo = {}, {}
    for job in jobs:
        rel = job["record_path"]
        rows[rel] = {"record_path": rel, "image_path": job["image_path"], "status": "", "error": ""}
        try:
            header = R.INPUT / rel
            prows, names, fs, nsamp, _, comments = R.parse_header(header)
            mv = R.load_mv(header, prows, names, nsamp)
            rid = R.record_id_for(rel)
            if rid in todo:
                raise RuntimeError(f"duplicate staged id {rid}")
            R.stage_record(rid, mv, fs, comments, staged)
            todo[rid] = job
        except Exception as exc:  # noqa: BLE001 - one bad record never stops the batch
            rows[rel].update(status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
    try:
        if todo:
            try:
                R.run_kit(staged, render)
                masters = R.collect_pngs(render, set(todo))
            except Exception as exc:  # noqa: BLE001
                masters = {}
                for job in todo.values():
                    rows[job["record_path"]].update(status="failed",
                                                    error=f"{type(exc).__name__}: {exc}"[:1000])
            for rid, png in masters.items():
                job = todo[rid]
                row = rows[job["record_path"]]
                try:
                    with Image.open(png) as master:
                        if master.size != (R.MASTER_W, R.MASTER_H):
                            raise RuntimeError(f"master size {master.size}")
                        final, meta = augment_master(master, _SEED, job["record_path"], job["plan"])
                    check_plan(meta, job["plan"])
                    out = R.OUTPUT / job["image_path"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    temp = out.with_name(out.name + f".{os.getpid()}.part")
                    final.save(temp, "PNG", compress_level=R.CFG["png_compress_level"])
                    os.replace(temp, out)
                    row.update(meta, status="success", image_bytes=out.stat().st_size,
                               ink_bin=job["plan"]["ink_bin"], tilt_bin=job["plan"]["tilt_bin"])
                except Exception as exc:  # noqa: BLE001
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return [rows[job["record_path"]] for job in jobs]
