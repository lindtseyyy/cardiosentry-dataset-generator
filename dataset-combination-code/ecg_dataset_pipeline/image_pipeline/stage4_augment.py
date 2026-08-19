"""Stage 4 - paper/scan augmentation, replacing the kit's --augment and --wrinkles.

The kit's distortion layer was tested and rejected. Confirmed defects:

  * -t/--temperature is dead code. gen_ecg_image_from_data.py hardcodes
        blue_temp = random.choice((True, False))
        temp = random.choice(range(2000,4000)) if blue_temp
               else random.choice(range(10000,20000))
    so every image gets a heavy orange or blue cast with no neutral option.
  * -noise 0 crashes (random.choice on an empty range); noise is additive in
    0-255 units, so even 5 balloons a PNG from 0.4 MB to 4.4 MB.
  * iaa.Affine fills rotation corners with black, and with no margin the
    rotation clips the printed ID/Age/Sex header off the top of the sheet.
  * --augment silently requires --store_config, and --lead_bbox silently
    disables cropping.
  * --wrinkles at its lightest setting (-nv 1 -nh 1) still lays a heavy
    grey texture over the whole sheet and buries the trace.

Everything here is therefore done directly, so each effect is bounded, neutral
centred, and drawn from one distribution shared by all four classes. Style must
never correlate with class, or the model learns the renderer instead of the ECG.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import config as C


@lru_cache(maxsize=1)
def _texture_files() -> tuple[Path, ...]:
    """The kit's wrinkled-paper photographs, reused here as shading fields."""
    root = C.KIT / "CreasesWrinkles" / "wrinkles-dataset"
    if not root.is_dir():
        return ()
    return tuple(sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ))


def _rng(seed_parts: tuple) -> np.random.Generator:
    """Deterministic per-image RNG keyed only by output identity, never by class.

    Uses blake2b rather than hash(): CPython randomises string hashing per
    process unless PYTHONHASHSEED is pinned, which would make the corpus
    irreproducible across runs.
    """
    key = "|".join(str(p) for p in seed_parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big") ^ C.SEED)


def _paper_colour(img: np.ndarray) -> np.ndarray:
    """Median of the sheet's corners - the fill used wherever geometry exposes canvas."""
    h, w = img.shape[:2]
    k = max(8, min(h, w) // 40)
    patches = np.concatenate([
        img[:k, :k].reshape(-1, 3), img[:k, -k:].reshape(-1, 3),
        img[-k:, :k].reshape(-1, 3), img[-k:, -k:].reshape(-1, 3),
    ])
    return np.median(patches, axis=0)


def _rotate_on_paper(img: Image.Image, deg: float, fill: tuple[int, int, int]) -> Image.Image:
    """Rotate about the centre with a paper-coloured background.

    expand=True then centre-crop back, so the printed header at the top of the
    sheet survives instead of being clipped the way the kit's Affine clips it.
    """
    w, h = img.size
    big = img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    bw, bh = big.size
    left, top = (bw - w) // 2, (bh - h) // 2
    return big.crop((left, top, left + w, top + h))


def _illumination(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uneven ambient light: a directional ramp plus corner falloff.

    Multiplicative and bounded well above zero, so the darkest corner of the
    sheet still reads as paper and the trace stays legible.
    """
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
    field = 1.0 + rng.uniform(0.10, 0.22) * ramp * 2.0

    r = np.sqrt((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    field *= 1.0 - rng.uniform(0.06, 0.18) * (r / r.max()) ** 2

    # A soft off-sheet shadow along one edge, as when paper is photographed
    # on a desk rather than scanned flat.
    if rng.random() < 0.65:
        edge = rng.integers(0, 4)
        d = {0: xx / w, 1: 1 - xx / w, 2: yy / h, 3: 1 - yy / h}[int(edge)]
        field *= 1.0 - rng.uniform(0.10, 0.26) * np.exp(-d / rng.uniform(0.04, 0.12))
    return arr * field[..., None]


def _paper_texture(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shade the sheet with a photograph of real wrinkled paper.

    The kit ships 19 such photographs and composites one with a full-strength
    overlay blend, which is why its --wrinkles output buries the trace. The
    texture itself is excellent though - it is the blend that is wrong.

    Here the photograph is turned into a multiplicative shading field centred
    on 1.0, which is what crumpled paper physically does to reflected light,
    and its dynamic range is compressed so the darkest fold still sits far
    above the trace.
    """
    files = _texture_files()
    if not files:
        return arr

    h, w = arr.shape[:2]
    tex = Image.open(files[int(rng.integers(len(files)))]).convert("L")

    # Random crop, flip and rotate so 19 photographs yield far more than 19 looks.
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
        return arr

    strength = rng.uniform(*C.TEXTURE_STRENGTH)
    field = 1.0 + (field / mean - 1.0) * strength
    return arr * np.clip(field, *C.TEXTURE_CLIP)[..., None]


def _creases(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fold lines: a soft bright ridge with a darker shadow on one side.

    Real creases on a printout are a lighting effect, not ink, so this is a
    multiplicative field with a narrow profile. Amplitude is capped so a fold
    never crosses a trace hard enough to break it.
    """
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.ones((h, w), dtype=np.float32)

    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:          # horizontal-ish fold
            pos = rng.uniform(0.12, 0.88) * h
            tilt = rng.uniform(-0.06, 0.06)
            dist = yy - (pos + tilt * (xx - w / 2))
        else:                           # vertical-ish fold
            pos = rng.uniform(0.12, 0.88) * w
            tilt = rng.uniform(-0.06, 0.06)
            dist = xx - (pos + tilt * (yy - h / 2))

        width = rng.uniform(0.006, 0.020) * max(h, w)
        amp = rng.uniform(*C.CREASE_AMP)
        # Odd (derivative-of-Gaussian) profile: highlight one side, shadow the
        # other, which is what a physical fold does to reflected light.
        g = np.exp(-(dist ** 2) / (2 * width ** 2))
        field *= 1.0 + amp * g * (dist / width) / 1.6

    # Broad, low-frequency bowing: sheets photographed on a desk are never
    # perfectly flat, and this is the cue that reads as "paper" rather than
    # "scan" at a glance.
    for _ in range(int(rng.integers(1, 3))):
        k = rng.uniform(0.6, 1.8)
        phase = rng.uniform(0, 2 * np.pi)
        axis = yy / h if rng.random() < 0.5 else xx / w
        field *= 1.0 + rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * k * axis + phase)

    return arr * np.clip(field, *C.CREASE_CLIP)[..., None]


def _normalise_paper(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rescale so the sheet's bright paper sits in a plausible window.

    The shading effects above are each individually mild, but they multiply, so
    an unlucky draw compounds them into a sheet that reads as dark card. This
    measures what the paper actually came out at and scales it back into
    PAPER_LEVEL_TARGET, preserving every effect's *relative* shading while
    bounding their product. Scaling only -- no per-pixel clipping -- so the
    trace keeps its contrast against the paper.
    """
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    paper = float(np.percentile(lum, 95))
    if paper <= 1.0:
        return arr
    target = rng.uniform(*C.PAPER_LEVEL_TARGET)
    return arr * (target / paper)


def _colour_cast(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild, neutral-centred white-balance drift - never the kit's bimodal cast."""
    gains = 1.0 + rng.normal(0.0, 0.012, size=3)
    return arr * gains[None, None, :]


# ---------------------------------------------------------------------------
# Geometry bookkeeping, so the kit's annotations survive augmentation.
#
# The kit writes lead boxes, label boxes and the traced pixel path against the
# CLEAN render. Stage 4 then rotates the sheet, warps it and adds a margin, so
# those coordinates stop describing the image the model actually sees. Every
# geometric step below therefore also records its parameters, and
# `transform_annotation` replays the same maths on the coordinates.
#
# Kit convention, verified against a real render: every coordinate is a
# [y, x] pair with y measured DOWN from the top-left. Internally we work in
# (x, y) and swap at the boundary.
# ---------------------------------------------------------------------------

def _rot_points(pts: np.ndarray, deg: float, w: int, h: int) -> np.ndarray:
    """Forward-map (x, y) points through the same rotation PIL applied.

    PIL's rotate(deg) turns the image counter-clockwise on screen; in a y-down
    coordinate system that is x' = x cos + y sin, y' = -x sin + y cos. Checked
    against deg=90: a point right of centre lands above it.
    """
    a = np.radians(deg)
    ca, sa = np.cos(a), np.sin(a)
    cx, cy = w / 2.0, h / 2.0
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([ca * x + sa * y + cx, -sa * x + ca * y + cy], axis=1)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Projective map sending the four src points to the four dst points."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    q = p @ H.T
    return q[:, :2] / q[:, 2:3]


def _forward(pts_yx, geom: dict) -> list:
    """Map kit [y, x] coordinates onto the finished image."""
    pts = np.asarray(pts_yx, dtype=np.float64)
    if pts.size == 0:
        return []
    xy = pts[:, ::-1].copy()                       # [y,x] -> (x,y)
    xy = _rot_points(xy, geom["rot_deg"], geom["w"], geom["h"])
    if geom.get("quad") is not None:
        w, h = geom["w"], geom["h"]
        q = np.asarray(geom["quad"], dtype=np.float64).reshape(4, 2)
        # PIL QUAD lists the SOURCE corners that become the destination
        # rectangle's UL, LL, LR, UR - so the forward map is quad -> rect.
        dst = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float64)
        xy = _apply_h(_homography(q, dst), xy)
    xy += geom["margin"]
    return [[round(float(b), 2), round(float(a), 2)] for a, b in xy]   # back to [y,x]


def transform_annotation(ann: dict, geom: dict) -> dict:
    """Rewrite a kit annotation JSON so it matches the augmented image."""
    out = dict(ann)
    out["width"] = geom["w"] + 2 * geom["margin"]
    out["height"] = geom["h"] + 2 * geom["margin"]
    out["augmentation_geometry"] = {
        "rotation_deg": round(geom["rot_deg"], 4),
        "perspective_quad": (None if geom.get("quad") is None
                             else [round(float(v), 2) for v in geom["quad"]]),
        "margin_px": geom["margin"],
        "source_size": [geom["w"], geom["h"]],
        "note": "coordinates are [y, x] from the top-left of THIS image",
    }
    leads = []
    for e in ann.get("leads", []):
        e = dict(e)
        for key in ("lead_bounding_box", "text_bounding_box"):
            box = e.get(key)
            if isinstance(box, dict) and box:
                order = sorted(box, key=lambda k: int(k))
                moved = _forward([box[k] for k in order], geom)
                e[key] = {k: v for k, v in zip(order, moved)}
        if e.get("plotted_pixels"):
            e["plotted_pixels"] = _forward(e["plotted_pixels"], geom)
        leads.append(e)
    out["leads"] = leads
    return out


def _grid_lut(palette: tuple, rng: np.random.Generator) -> np.ndarray:
    """256x3 lookup table mapping the kit's grey sheet onto a coloured one.

    The kit renders the grid at two exact greys (major 102, minor 191) with the
    trace below 60 and paper above 250, so a piecewise-linear map over those
    control points recolours the grid cleanly while leaving the ink neutral.
    Because it is a LUT, antialiased pixels interpolate smoothly instead of
    banding, and the whole thing costs one fancy-index over the image.
    """
    major, minor, paper = (np.asarray(c, dtype=np.float32) for c in palette)

    # Per-image ink and paper jitter: printers and cameras never agree exactly.
    ink = np.full(3, rng.uniform(8, 34), dtype=np.float32)
    ink += rng.normal(0, 3, size=3)
    paper = paper * rng.uniform(0.985, 1.005) + rng.normal(0, 1.2, size=3)
    sat = rng.uniform(*getattr(C, "GRID_SATURATION", (0.82, 1.12)))

    # Saturation scales the grid colours toward/away from their own luminance,
    # so a faded printout and a fresh one both occur.
    def _sat(c):
        lum = float(np.dot(c, (0.299, 0.587, 0.114)))
        return np.clip(lum + (c - lum) * sat, 0, 255)

    major, minor = _sat(major), _sat(minor)

    xs = np.array([0, C.GRID_TRACE_MAX, C.GRID_MAJOR_LEVEL,
                   C.GRID_MINOR_LEVEL, C.GRID_PAPER_LEVEL, 255], dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    grid = np.arange(256, dtype=np.float32)
    for ch in range(3):
        ys = np.array([ink[ch] * 0.35, ink[ch], major[ch],
                       minor[ch], paper[ch], min(255.0, paper[ch] + 3)],
                      dtype=np.float32)
        lut[:, ch] = np.interp(grid, xs, ys)
    return np.clip(lut, 0, 255)


def _recolour_grid(img: Image.Image, rng: np.random.Generator,
                   force: str | None = None) -> Image.Image:
    """Recolour the grey grid to a realistic ECG-paper palette.

    Palette choice is drawn from this image's own RNG, which is seeded on the
    output identity only -- never on the class -- so grid colour carries no
    label information. That matters: if red sheets were mostly STEMI the model
    would learn the paper, not the pathology.
    """
    names = list(C.GRID_PALETTES)
    weights = np.array([C.GRID_PALETTE_WEIGHTS.get(n, 0.0) for n in names],
                       dtype=np.float64)
    weights = weights / weights.sum()
    # Draw unconditionally, THEN override. Skipping the draw when a palette is
    # forced would shift every later effect's random values, so a --palette
    # preview would show lighting and geometry the corpus never produces for
    # that (record, variant). Verified: skipping it changed the rotation of the
    # very next draw from -1.51 to +2.65 degrees.
    drawn = names[int(rng.choice(len(names), p=weights))]
    name = force if force in C.GRID_PALETTES else drawn

    grey = np.asarray(img.convert("L"))
    lut = _grid_lut(C.GRID_PALETTES[name], rng)
    return Image.fromarray(lut[grey].astype(np.uint8), "RGB"), name


def _defocus(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Mild lens defocus. Bounded hard -- see config.DEFOCUS_RADIUS."""
    if rng.random() >= C.DEFOCUS_PROB:
        return img
    lo, hi = C.DEFOCUS_RADIUS
    return img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(lo, hi))))


def augment(src: Path, dst: Path, seed_parts: tuple,
            force_palette: str | None = None) -> dict:
    """Render one augmented photograph. Returns the effect metadata applied,
    so index.csv can record it and you can audit that no effect correlates
    with the class."""
    rng = _rng(seed_parts)
    img = Image.open(src).convert("RGB")

    # --- paper stock: recolour the grid before any geometry or lighting, so
    # the tint sits under the shading exactly as printed ink does -------------
    img, palette = _recolour_grid(img, rng, force_palette)

    arr0 = np.asarray(img, dtype=np.float32)
    fill = tuple(int(v) for v in _paper_colour(arr0))

    # --- geometry: tilt, then perspective -------------------------------
    # Every parameter is recorded so transform_annotation() can replay it.
    src_w, src_h = img.size
    geom = {"w": src_w, "h": src_h, "margin": C.MARGIN_PX,
            "rot_deg": float(rng.uniform(-3.0, 3.0)), "quad": None}

    img = _rotate_on_paper(img, geom["rot_deg"], fill)

    if rng.random() < 0.7:                       # keystone, as if photographed
        w, h = img.size
        m = rng.uniform(0.004, 0.018)
        dx, dy = m * w, m * h
        quad = (
            rng.uniform(0, dx), rng.uniform(0, dy),
            rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), rng.uniform(0, dy),
        )
        geom["quad"] = tuple(float(v) for v in quad)
        img = img.transform((w, h), Image.QUAD, quad,
                            resample=Image.BICUBIC, fillcolor=fill)

    # --- paper: texture, then sharp folds, then lighting -----------------
    # Real crumpled paper shows both: a diffuse wrinkle field over the whole
    # sheet, and a few hard fold lines where it was actually creased. Texture
    # is unconditional - a photographed sheet is never perfectly flat, and a
    # skip probability here is what let some renders come out looking clean.
    arr = np.asarray(img, dtype=np.float32)
    arr = _paper_texture(arr, rng)
    if rng.random() < C.CREASE_PROB:
        arr = _creases(arr, rng)
    arr = _illumination(arr, rng)
    arr = _colour_cast(arr, rng)
    arr *= rng.uniform(0.92, 1.06)                              # exposure
    mean = arr.mean()
    arr = (arr - mean) * rng.uniform(0.94, 1.08) + mean         # contrast
    # Bound the product of everything above; see _normalise_paper.
    arr = _normalise_paper(arr, rng)

    # Sensor noise. Kept low deliberately: at 150 DPI the ST segment is only a
    # few pixels tall, and heavy noise buries the very feature STEMI depends on.
    arr += rng.normal(0.0, rng.uniform(1.5, 4.5), size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Defocus last among the optical effects: a real lens blurs the scene it
    # sees, including the shading and the noise floor above.
    out = _defocus(Image.fromarray(arr), rng)

    # --- thin margin, added last ---------------------------------------
    # The kit's --pad_inches is int-valued (0 or a full inch), so the thin
    # border the corpus spec asks for is applied here instead.
    m = C.MARGIN_PX
    canvas = Image.new("RGB", (out.width + 2 * m, out.height + 2 * m), fill)
    canvas.paste(out, (m, m))

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, "JPEG", quality=C.JPEG_QUALITY,
                subsampling=C.JPEG_SUBSAMPLING, optimize=True)

    # Carry the kit's annotations forward onto THIS image, if it wrote any.
    ann_src = src.with_suffix(".json")
    if not ann_src.exists():                       # kit names it <record>-0.json
        ann_src = src.parent / (src.stem + ".json")
    wrote_ann = False
    if ann_src.exists():
        try:
            with open(ann_src) as fh:
                ann = json.load(fh)
            tr = transform_annotation(ann, geom)
            tr["augmentation_geometry"]["grid_palette"] = palette
            with open(dst.with_suffix(".json"), "w") as fh:
                json.dump(tr, fh)
            wrote_ann = True
        except Exception:                          # noqa: BLE001 - never abort a build
            wrote_ann = False

    return {"grid_palette": palette,
            "rotation_deg": round(geom["rot_deg"], 3),
            "has_annotation": int(wrote_ann)}


# The four binary label columns are carried straight through from the input
# CSV: index.csv, not the directory name, is what a training script should read.
INDEX_FIELDS = (["image_path", "record", "record_id", "original_id", "cls"]
                + C.LABEL_COLUMNS
                + ["split", "source", "render_k", "grid_palette", "rotation_deg",
                   "has_annotation", "patient_id",
                   "age", "sex", "note"])

CARRY = ["record_id", "original_id", "cls", "split", "source",
         "patient_id", "age", "sex", "note"] + C.LABEL_COLUMNS


def _variants_for(split: str) -> int:
    """How many augmented variants one record of this split contributes.

    val/test are pinned to exactly 1 regardless of what any config or manifest
    says. Extra variants of an evaluation record silently reweight that record
    in every metric, so this is enforced here as well as in stage 1 -- the
    guarantee should not depend on one file being edited correctly.
    """
    if split in C.EVAL_SPLITS:
        return 1
    return max(1, int(C.VARIANTS_PER_SPLIT.get(split, 1)))


def _plan() -> list[dict]:
    """Pair every rendered PNG with its staged-map row, expanding TRAIN records
    into their augmented variants. One rendered sheet becomes N images: the
    render is the same paper, stage 4 makes each photograph of it different."""
    if not C.STAGED_MAP.exists():
        return []
    meta = {r["record"]: r for r in csv.DictReader(C.STAGED_MAP.open())}
    jobs = []
    for png in sorted(C.RENDERED.rglob("*.png")):
        record = png.stem.rsplit("-", 1)[0]          # kit writes <record>-0.png
        row = meta.get(record)
        if row is None:
            continue
        # .../rendered/<cls-slug>/r<k>/<chunk>/<record>-0.png
        k = int(png.parent.parent.name[1:])
        # A per-record n_renders lower than its chunk's maximum means the extra
        # variants were rendered but are not wanted in the corpus.
        if k >= int(row.get("n_renders") or 1):
            continue
        # Split directory only - class is NOT a directory. index.csv carries the
        # labels, and it is the only thing a training script should read. A
        # class directory would also have to collapse multi-label records
        # ("AF+LVH") into one string, which loses the multi-label head.
        split = row["split"]
        n_var = _variants_for(split)
        for v in range(n_var):
            # Variant index is global across renders of one record, so a record
            # with 2 renders x 2 variants yields a0..a3 and every filename and
            # RNG seed stays unique.
            a = k * n_var + v
            dst = C.IMAGES / split / f"{record}_a{a}.jpg"
            job = {"src": str(png), "dst": str(dst), "record": record, "k": a}
            job.update({key: row.get(key, "") for key in CARRY})
            jobs.append(job)
    return jobs


def _recover_meta(job: dict) -> dict:
    """Metadata for an image that already exists, without redoing the work.

    Without this an interrupted-and-resumed build writes blank grid_palette /
    rotation_deg / has_annotation into index.csv for every image that survived
    from the first attempt - so the palette-vs-class audit would silently
    describe only the images made after the last restart.
    """
    dst = Path(job["dst"])
    side = dst.with_suffix(".json")
    if side.exists():
        try:
            g = json.loads(side.read_text()).get("augmentation_geometry", {})
            return {"grid_palette": g.get("grid_palette", ""),
                    "rotation_deg": g.get("rotation_deg", ""),
                    "has_annotation": 1}
        except Exception:                              # noqa: BLE001
            pass
    # No sidecar. The palette is the FIRST draw from this image's RNG, which is
    # seeded on (record, variant) alone, so it is recoverable exactly.
    rng = _rng((job["record"], job["k"]))
    names = list(C.GRID_PALETTES)
    w = np.array([C.GRID_PALETTE_WEIGHTS.get(n, 0.0) for n in names], dtype=np.float64)
    w = w / w.sum()
    return {"grid_palette": names[int(rng.choice(len(names), p=w))],
            "rotation_deg": "", "has_annotation": 0}


def _work(job: dict) -> tuple[dict, str | None]:
    dst = Path(job["dst"])
    if dst.exists() and dst.stat().st_size > 0:
        job.update(_recover_meta(job))                 # idempotent re-runs
        return job, None
    try:
        meta = augment(Path(job["src"]), dst, (job["record"], job["k"]))
        job.update(meta)
    except Exception as exc:                          # noqa: BLE001 - report, don't abort
        return job, f"{type(exc).__name__}: {exc}"
    return job, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--workers", type=int, default=C.WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-i", "--input_dir", help="standalone mode: augment a folder")
    ap.add_argument("-o", "--output_dir")
    ap.add_argument("--renders", type=int, default=1)
    ap.add_argument("--palette", default=None, choices=list(C.GRID_PALETTES),
                    help="standalone: force one grid palette instead of drawing "
                         "it at random, so palettes can be compared side by side")
    args = ap.parse_args()

    # ---- standalone mode, used for spot checks -------------------------
    # Useful while tuning the effects above: point it at a handful of PNGs and
    # look at the output, without touching the corpus or its index.
    if args.input_dir:
        if not args.output_dir:
            print("standalone mode needs -o/--output_dir", file=sys.stderr)
            return 1
        src_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
        pngs = sorted(src_dir.rglob("*.png"))[: args.limit or None]
        for png in pngs:
            stem = png.stem.rsplit("-", 1)[0]
            for k in range(args.renders):
                tmp = out_dir / f"{stem}_a{k}.jpg"
                meta = augment(png, tmp, (stem, k), args.palette)
                # name the sample after the palette it drew, so the contact
                # sheet is self-describing when you inspect it. The annotation
                # has to follow the rename or plot_annotations cannot find it.
                final = out_dir / f"{stem}_a{k}_{meta['grid_palette']}.jpg"
                tmp.rename(final)
                if tmp.with_suffix(".json").exists():
                    tmp.with_suffix(".json").rename(final.with_suffix(".json"))
        sizes = [p.stat().st_size for p in out_dir.glob("*.jpg")]
        print(f"wrote {len(sizes)} images -> {out_dir}")
        print(f"mean size {sum(sizes)/max(len(sizes),1)/1024:.0f} KB")
        return 0

    # ---- corpus mode ---------------------------------------------------
    jobs = _plan()
    if not jobs:
        print("no rendered PNGs found - run stage3 first", file=sys.stderr)
        return 1
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(jobs)} images on {args.workers} workers")
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            job, err = fut.result()
            if err:
                failures.append((job["record"], err))
            else:
                row = {"image_path": job["dst"], "record": job["record"],
                       "render_k": job["k"],
                       "grid_palette": job.get("grid_palette", ""),
                       "rotation_deg": job.get("rotation_deg", ""),
                       "has_annotation": job.get("has_annotation", 0)}
                row.update({key: job.get(key, "") for key in CARRY})
                rows.append({k: row.get(k, "") for k in INDEX_FIELDS})
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}")

    rows.sort(key=lambda r: (r["cls"], r["record"], r["render_k"]))
    C.INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with C.INDEX_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)

    sizes = [Path(r["image_path"]).stat().st_size for r in rows]
    print(f"\nwrote {len(rows)} images -> {C.IMAGES}")
    print(f"mean {sum(sizes)/max(len(sizes),1)/1024:.0f} KB, "
          f"total {sum(sizes)/1e9:.1f} GB")
    print(f"index -> {C.INDEX_CSV}")
    if failures:
        print(f"{len(failures)} failures, first few:")
        for rec, err in failures[:10]:
            print(f"    {rec}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
