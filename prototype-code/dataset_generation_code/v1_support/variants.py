"""Variant engine (PLAN 10.x): clean / seeded-augmented / fixed-stress views.

Faithful port of the validated image_pipeline/stage4 math (bounds unchanged)
restructured so every effect takes explicit parameters: seeded draws produce
v01-v03, fixed constants produce clean and stress. Operates at master
resolution; downsampling happens afterwards (PLAN 7.3).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .config import (GRID_PALETTES, GRID_PALETTE_WEIGHTS, GRID_SATURATION,
                     GRID_TRACE_MAX, GRID_MAJOR_LEVEL, GRID_MINOR_LEVEL,
                     GRID_PAPER_LEVEL, STRESS_RECIPE, SEED)
from .util import rng_from_parts


# ---------------------------------------------------------------- textures

def texture_files(texture_dir: str | Path) -> list[Path]:
    root = Path(texture_dir)
    files = sorted(p for p in root.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise ValueError(f"no texture photos in {texture_dir}")
    return files


# ---------------------------------------------------------------- recolour

def _lut(palette: tuple, ink: np.ndarray, paper: np.ndarray, sat: float) -> np.ndarray:
    major, minor, paper_c = (np.asarray(c, dtype=np.float32) for c in palette)
    paper_c = paper.astype(np.float32)

    def _sat(c):
        lum = float(np.dot(c, (0.299, 0.587, 0.114)))
        return np.clip(lum + (c - lum) * sat, 0, 255)

    major, minor = _sat(major), _sat(minor)
    xs = np.array([0, GRID_TRACE_MAX, GRID_MAJOR_LEVEL,
                   GRID_MINOR_LEVEL, GRID_PAPER_LEVEL, 255], dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    grid = np.arange(256, dtype=np.float32)
    for ch in range(3):
        ys = np.array([ink[ch] * 0.35, ink[ch], major[ch],
                       minor[ch], paper_c[ch], min(255.0, paper_c[ch] + 3)],
                      dtype=np.float32)
        lut[:, ch] = np.interp(grid, xs, ys)
    return np.clip(lut, 0, 255)


def recolour_fixed(grey: Image.Image, palette_name: str) -> Image.Image:
    """Deterministic recolour for clean images: fixed ink/paper/saturation."""
    palette = GRID_PALETTES[palette_name]
    ink = np.full(3, 20.0, dtype=np.float32)
    paper = np.asarray(palette[2], dtype=np.float32)
    lut = _lut(palette, ink, paper, 1.0)
    arr = lut[np.asarray(grey.convert("L"))].astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def recolour_seeded(grey: Image.Image, rng: np.random.Generator,
                    force_palette: str | None = None,
                    sat_override: float | None = None,
                    ink_override: np.ndarray | None = None,
                    paper_override: np.ndarray | None = None,
                    ) -> tuple[Image.Image, str]:
    names = list(GRID_PALETTES)
    weights = np.array([GRID_PALETTE_WEIGHTS[n] for n in names], dtype=np.float64)
    weights /= weights.sum()
    drawn = names[int(rng.choice(len(names), p=weights))]
    name = force_palette if force_palette in GRID_PALETTES else drawn
    if ink_override is None:
        ink = (np.full(3, rng.uniform(8, 34), dtype=np.float32)
               + rng.normal(0, 3, size=3))
    else:
        ink = np.asarray(ink_override, dtype=np.float32)
    if paper_override is None:
        paper = (np.asarray(GRID_PALETTES[name][2], dtype=np.float32)
                 * rng.uniform(0.985, 1.005) + rng.normal(0, 1.2, size=3))
    else:
        paper = np.asarray(paper_override, dtype=np.float32)
    sat = float(rng.uniform(*GRID_SATURATION)) if sat_override is None \
        else float(sat_override)
    lut = _lut(GRID_PALETTES[name], ink, paper, sat)
    arr = lut[np.asarray(grey.convert("L"))].astype(np.uint8)
    return Image.fromarray(arr, "RGB"), name


# ---------------------------------------------------------------- geometry

def paper_colour(arr: np.ndarray) -> tuple[int, int, int]:
    # 90th percentile (not median): 200-DPI renders carry ~40% antialiased
    # grid-halo pixels, which drag a median toward tint; p90 recovers the
    # actual paper white even in grid-heavy corners.
    h, w = arr.shape[:2]
    k = max(8, min(h, w) // 40)
    patches = np.concatenate([
        arr[:k, :k].reshape(-1, 3), arr[:k, -k:].reshape(-1, 3),
        arr[-k:, :k].reshape(-1, 3), arr[-k:, -k:].reshape(-1, 3)])
    return tuple(int(v) for v in np.percentile(patches, 90, axis=0))


def protective_shift(img: Image.Image, fill: tuple[int, int, int],
                     px: int = 20) -> Image.Image:
    """Fixed content shift up-left by `px` (master scale) before any tilt /
    keystone. Worst-case geometry (3 deg tilt + 1.8% keystone aligned) can
    otherwise bite <=11 px off a rhythm-strip end or the footer legend in
    ~0.015% of augmented views (margin analysis, PLAN §20.7); the shift
    moves the tight bottom/right margins from 71/91 px to 91/111 px so the
    pathological stack stays >=4 px inside the frame. Deterministic,
    translation changes no scale, unveiled strips take the paper colour."""
    w, h = img.size
    return img.transform((w, h), Image.AFFINE, (1, 0, -px, 0, 1, -px),
                         resample=Image.BICUBIC, fillcolor=fill)


def rotate_on_paper(img: Image.Image, deg: float,
                    fill: tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    big = img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    bw, bh = big.size
    left, top = (bw - w) // 2, (bh - h) // 2
    return big.crop((left, top, left + w, top + h))


def apply_quad(img: Image.Image, quad: tuple[float, ...],
               fill: tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    return img.transform((w, h), Image.QUAD, quad,
                         resample=Image.BICUBIC, fillcolor=fill)


def draw_quad(rng: np.random.Generator, w: int, h: int,
              mag_lo: float = 0.004, mag_hi: float = 0.018):
    m = float(rng.uniform(mag_lo, mag_hi))
    dx, dy = m * w, m * h
    return (float(rng.uniform(0, dx)), float(rng.uniform(0, dy)),
            float(rng.uniform(0, dx)), h - float(rng.uniform(0, dy)),
            w - float(rng.uniform(0, dx)), h - float(rng.uniform(0, dy)),
            w - float(rng.uniform(0, dx)), float(rng.uniform(0, dy)))


# ---------------------------------------------------------------- shading

def illumination_field(h: int, w: int, angle: float, ramp: float,
                       falloff: float, edge: int | None = None,
                       edge_depth: float = 0.0, edge_width: float = 0.08):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = np.cos(angle) * (xx / w - 0.5) + np.sin(angle) * (yy / h - 0.5)
    field = 1.0 + ramp * proj * 2.0
    r = np.sqrt((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    field *= 1.0 - falloff * (r / r.max()) ** 2
    if edge is not None:
        d = {0: xx / w, 1: 1 - xx / w, 2: yy / h, 3: 1 - yy / h}[edge]
        field *= 1.0 - edge_depth * np.exp(-d / edge_width)
    return field


def paper_texture(arr: np.ndarray, tex_img: Image.Image,
                  strength: float, clip: tuple[float, float]) -> np.ndarray:
    h, w = arr.shape[:2]
    field = np.asarray(tex_img.resize((w, h), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    mean = float(field.mean())
    if mean <= 1e-3:
        return arr
    field = 1.0 + (field / mean - 1.0) * strength
    return arr * np.clip(field, *clip)[..., None]


def draw_texture_view(rng: np.random.Generator,
                      files: list[Path]) -> Image.Image:
    tex = Image.open(files[int(rng.integers(len(files)))]).convert("L")
    tw, th = tex.size
    scale = float(rng.uniform(0.55, 1.0))
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
    return tex


def creases_field(h: int, w: int, folds: list[tuple], bows: list[tuple]):
    """folds: (axis, pos_frac, tilt, width_frac, amp); bows: (k, phase, amp, vert)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.ones((h, w), dtype=np.float32)
    for axis, pos, tilt, width_frac, amp in folds:
        width = width_frac * max(h, w)
        if axis == "h":
            dist = yy - (pos * h + tilt * (xx - w / 2))
        else:
            dist = xx - (pos * w + tilt * (yy - h / 2))
        g = np.exp(-(dist ** 2) / (2 * width ** 2))
        field *= 1.0 + amp * g * (dist / width) / 1.6
    for k, phase, amp, vert in bows:
        ax = yy / h if vert else xx / w
        field *= 1.0 + amp * np.sin(2 * np.pi * k * ax + phase)
    return field


def normalise_paper(arr: np.ndarray, target: float) -> np.ndarray:
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    paper = float(np.percentile(lum, 95))
    if paper <= 1.0:
        return arr
    return arr * (target / paper)


# ---------------------------------------------------------------- pipelines

def _add_noise_defocus(arr: np.ndarray, rng: np.random.Generator,
                       noise_sigma: float, defocus_prob: float,
                       defocus_radius: float) -> Image.Image:
    arr = arr + rng.normal(0.0, noise_sigma, size=arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < defocus_prob:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(defocus_radius)))
    return out


def make_clean(master_grey: Image.Image) -> tuple[Image.Image, dict]:
    """Canonical clean view: fixed red grid, no other effects."""
    img = recolour_fixed(master_grey, "red")
    return img, {"variant_type": "clean", "grid_palette": "red",
                 "rotation_deg": 0.0, "render_recipe": "clean-fixed"}


def make_augmented(master_grey: Image.Image, record_id: str, variant_id: str,
                   texture_dir: str | Path, seed: int = SEED):
    """Seeded augmented view (train v01-v03)."""
    from .config import (ROT_DEG, PERSPECTIVE_PROB, TEXTURE_STRENGTH,
                         TEXTURE_CLIP, CREASE_PROB, CREASE_AMP, CREASE_CLIP,
                         RAMP_RANGE, FALLOFF_RANGE, EDGE_SHADOW_PROB,
                         EDGE_SHADOW_DEPTH, WB_SIGMA, EXPOSURE, CONTRAST,
                         PAPER_LEVEL_TARGET, SENSOR_NOISE_SIGMA, DEFOCUS_PROB,
                         DEFOCUS_RADIUS)
    rng = rng_from_parts(seed, record_id, variant_id)
    files = texture_files(texture_dir)
    img, palette = recolour_seeded(master_grey.convert("RGB"), rng)
    fill = paper_colour(np.asarray(img, dtype=np.float32))
    img = protective_shift(img, fill)
    rot = float(rng.uniform(*ROT_DEG))
    img = rotate_on_paper(img, rot, fill)
    quad = None
    if rng.random() < PERSPECTIVE_PROB:
        w, h = img.size
        quad = draw_quad(rng, w, h)
        img = apply_quad(img, quad, fill)
    arr = np.asarray(img, dtype=np.float32)
    tex = draw_texture_view(rng, files)
    arr = paper_texture(arr, tex, float(rng.uniform(*TEXTURE_STRENGTH)), TEXTURE_CLIP)
    if rng.random() < CREASE_PROB:
        folds = []
        for _ in range(int(rng.integers(1, 4))):
            axis = "h" if rng.random() < 0.5 else "v"
            folds.append((axis, float(rng.uniform(0.12, 0.88)),
                          float(rng.uniform(-0.06, 0.06)),
                          float(rng.uniform(0.006, 0.020)),
                          float(rng.uniform(*CREASE_AMP))))
        bows = [(float(rng.uniform(0.6, 1.8)), float(rng.uniform(0, 2 * np.pi)),
                 float(rng.uniform(0.02, 0.06)), bool(rng.random() < 0.5))
                for _ in range(int(rng.integers(1, 3)))]
        h, w = arr.shape[:2]
        arr = arr * np.clip(creases_field(h, w, folds, bows), *CREASE_CLIP)[..., None]
    h, w = arr.shape[:2]
    edge = int(rng.integers(0, 4)) if rng.random() < EDGE_SHADOW_PROB else None
    arr = arr * illumination_field(
        h, w, float(rng.uniform(0, 2 * np.pi)),
        float(rng.uniform(*RAMP_RANGE)), float(rng.uniform(*FALLOFF_RANGE)),
        edge, float(rng.uniform(*EDGE_SHADOW_DEPTH)) if edge is not None else 0.0,
        float(rng.uniform(0.04, 0.12)))[..., None]
    arr = arr * (1.0 + rng.normal(0.0, WB_SIGMA, size=3))[None, None, :]
    arr *= float(rng.uniform(*EXPOSURE))
    mean = arr.mean()
    arr = (arr - mean) * float(rng.uniform(*CONTRAST)) + mean
    arr = normalise_paper(arr, float(rng.uniform(*PAPER_LEVEL_TARGET)))
    out = _add_noise_defocus(arr, rng, float(rng.uniform(*SENSOR_NOISE_SIGMA)),
                             DEFOCUS_PROB, float(rng.uniform(*DEFOCUS_RADIUS)))
    return out, {"variant_type": "augmented", "grid_palette": palette,
                 "rotation_deg": round(rot, 3)}


def make_stress(master_grey: Image.Image, texture_dir: str | Path):
    """Fixed-recipe artifact-stress view (val/test). Bit-deterministic."""
    from .config import TEXTURE_CLIP, CREASE_CLIP
    R = STRESS_RECIPE
    rng = rng_from_parts(SEED, "stress-fixed")
    files = texture_files(texture_dir)
    pal = GRID_PALETTES[R["palette"]]
    img, _ = recolour_seeded(
        master_grey.convert("RGB"), rng, force_palette=R["palette"],
        sat_override=R["saturation"], ink_override=np.full(3, 20.0),
        paper_override=np.asarray(pal[2], dtype=np.float32))
    fill = paper_colour(np.asarray(img, dtype=np.float32))
    img = protective_shift(img, fill)
    img = rotate_on_paper(img, R["rot_deg"], fill)
    w, h = img.size
    dx, dy = R["perspective_mag"] * w, R["perspective_mag"] * h
    quad = (dx, dy * 0.5, dx * 0.5, h - dy, w - dx * 0.5, h - dy * 0.5,
            w - dx, dy)
    img = apply_quad(img, tuple(float(v) for v in quad), fill)
    arr = np.asarray(img, dtype=np.float32)
    tex = Image.open(files[R["texture_index"] % len(files)]).convert("L")
    arr = paper_texture(arr, tex, R["texture_strength"], TEXTURE_CLIP)
    h, w = arr.shape[:2]
    folds = [("h", 0.35, 0.02, 0.012, R["crease_amp"]),
             ("v", 0.62, -0.015, 0.012, R["crease_amp"])]
    bows = [(1.2, 0.5, 0.04, True)]
    arr = arr * np.clip(creases_field(h, w, folds, bows), *CREASE_CLIP)[..., None]
    arr = arr * illumination_field(h, w, R["illum_angle"], R["illum_ramp"],
                                   R["illum_falloff"], R["edge_shadow"],
                                   R["edge_depth"], 0.08)[..., None]
    arr *= R["exposure"]
    mean = arr.mean()
    arr = (arr - mean) * R["contrast"] + mean
    arr = normalise_paper(arr, R["paper_target"])
    arr = arr + rng.normal(0.0, R["noise_sigma"], size=arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(radius=R["defocus_radius"]))
    return out, {"variant_type": "stress", "grid_palette": R["palette"],
                 "rotation_deg": R["rot_deg"], "render_recipe": "stress-fixed"}
