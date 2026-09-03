"""Stage 4 (photo) - turn a clean kit render into a smartphone photograph.

This replaces stage4_augment.py for the photo-realistic build. The difference
is not "more effects": it is that the effects are applied in the order the
physical chain applies them, at the resolution each one physically acts at.

    clean vector render (kit, 300 dpi, thin ink, grey grid)
      |
      +-- PRESS   ink separation, then ink physics per layer:
      |             density mottle, fibre grain, diffusion, bleed, Beer-Lambert
      |             opacity. The trace is a hot stylus; the grid is offset
      |             print. They are not the same ink and no longer look it.
      |
      +-- SHEET   paper stock: colour, fibre, wrinkle field, fold lines
      |
      +-- CAMERA  pose (roll/yaw/pitch/distance -> true keystone), placement on
                  a surface with a cast shadow, illumination, then DOWNSAMPLE
                  to the frame, then lens blur, vignette, chromatic shift,
                  shot+read noise, white balance, JPEG.

The downsample is the load-bearing step. Area-averaging a 300 dpi sheet into a
2600 px frame is what a lens and sensor physically do, and it is what removes
the hard vector edges. Blurring a 150 dpi render instead just makes a blurry
vector graphic - that was the previous output's main tell.

Every random draw is keyed on the output identity only (record, variant),
never on the class, so no visual property can carry label information.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import config as C
import photo_config as P
from stage4_augment import _rng, _texture_files, INDEX_FIELDS

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------- fields
def _fbm(shape, scale_px: float, rng, octaves: int = 3) -> np.ndarray:
    """Smooth zero-mean noise field with feature size ~scale_px.

    Built by upsampling coarse white noise rather than by FFT: at 8 Mpx the
    FFT costs more than the whole rest of the stage, and the visual difference
    at these amplitudes is nil.
    """
    h, w = shape
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = max(2.0, scale_px / (2 ** o))
        gh, gw = max(2, int(h / s) + 2), max(2, int(w / s) + 2)
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        out += amp * np.asarray(
            Image.fromarray(g).resize((w, h), Image.BICUBIC), np.float32)
        total += amp
        amp *= 0.5
    out /= total
    sd = float(out.std())
    return out / sd if sd > 1e-6 else out


def _blur_mm(arr: np.ndarray, mm: float, ppmm: float) -> np.ndarray:
    """Gaussian blur specified in millimetres on the paper, not pixels.

    Keeps every optical/ink constant physical, so changing RENDER_DPI rescales
    the whole model correctly instead of silently changing how it looks.
    """
    sigma = mm * ppmm
    if sigma < 0.35:
        return arr
    im = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255.0


# ---------------------------------------------------------------- press
def separate_ink(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the kit's greyscale sheet into a trace layer and a grid layer.

    The kit draws the grid at two exact greys (major 102, minor 191), bare
    paper above 250 and the trace at 0, so the levels are separable by
    construction. Returned as ink AMOUNTS, not alphas: amount 1.0 means "one
    unit of ink", and the minor gridline is the unit for the grid layer, so
    the major line arrives at ~2.5 units on its own and needs no fudge factor.
    """
    trace = np.clip((P.INK_TRACE_CUT - grey) / P.INK_TRACE_CUT, 0, 1)

    span = P.INK_PAPER_LEVEL - P.INK_TRACE_CUT
    grid = np.clip((P.INK_PAPER_LEVEL - grey) / span, 0, 1)
    minor_unit = (P.INK_PAPER_LEVEL - C.GRID_MINOR_LEVEL) / span
    grid = grid / minor_unit
    grid *= (1.0 - trace)                    # the trace is printed over the grid
    return trace.astype(np.float32), grid.astype(np.float32)


def press(grey: np.ndarray, rng, ppmm: float, force_palette=None):
    """Lay ink on paper. Returns (sheet RGB float32, palette name)."""
    h, w = grey.shape
    trace, grid = separate_ink(grey)

    # --- trace: hot stylus on thermal paper -----------------------------
    mottle = _fbm((h, w), rng.uniform(*P.TRACE_MOTTLE_SCALE_MM) * ppmm, rng)
    grain = _fbm((h, w), max(2.0, 0.25 * ppmm), rng, octaves=2)
    trace = trace * np.clip(
        1.0 + rng.uniform(*P.TRACE_MOTTLE_AMP) * mottle
            + rng.uniform(*P.TRACE_GRAIN_AMP) * grain, 0.35, 1.7)

    # Diffusion into the fibre, plus a wider low-amplitude bleed halo. Together
    # these are why printed ink has a soft shoulder and a vector stroke does not.
    diff = _blur_mm(trace, rng.uniform(*P.TRACE_DIFFUSE_MM), ppmm)
    halo = _blur_mm(trace, rng.uniform(*P.TRACE_DIFFUSE_MM) * 2.4, ppmm)
    b = rng.uniform(*P.TRACE_BLEED)
    trace = diff * (1.0 - b) + halo * b * 1.25

    # --- grid: offset print, lighter ink, roller pressure varies --------
    gm = _fbm((h, w), rng.uniform(*P.GRID_MOTTLE_SCALE_MM) * ppmm, rng)
    grid = grid * np.clip(1.0 + rng.uniform(*P.GRID_MOTTLE_AMP) * gm, 0.35, 1.6)
    grid = _blur_mm(grid, rng.uniform(*P.GRID_DIFFUSE_MM), ppmm)

    # --- Beer-Lambert opacity -------------------------------------------
    # alpha = 1 - exp(-k * amount), with k set so one unit of ink hits the
    # configured density. Physical, and it means the major gridline's extra
    # ink darkens it without ever clipping to solid.
    kt = -np.log(1.0 - rng.uniform(*P.TRACE_DENSITY))
    kg = -np.log(1.0 - rng.uniform(*P.GRID_DENSITY))
    a_trace = 1.0 - np.exp(-kt * trace)
    a_grid = 1.0 - np.exp(-kg * grid)

    # --- composite paper <- grid <- trace --------------------------------
    names = list(P.GRID_INK_RGB)
    wts = np.array([P.GRID_PALETTE_WEIGHTS.get(n, 0.0) for n in names], float)
    drawn = names[int(rng.choice(len(names), p=wts / wts.sum()))]
    palette = force_palette if force_palette in P.GRID_INK_RGB else drawn

    paper = np.array(P.PAPER_RGB, np.float32) + rng.normal(0, P.PAPER_TINT_JITTER, 3)
    gink = np.array(P.GRID_INK_RGB[palette], np.float32) + rng.normal(0, 5, 3)
    tink = np.array(P.TRACE_INK_RGB, np.float32) + rng.normal(0, P.TRACE_INK_JITTER, 3)

    sheet = np.broadcast_to(paper, (h, w, 3)).astype(np.float32).copy()
    fibre = 1.0 + rng.uniform(*P.PAPER_FIBRE_AMP) * _fbm((h, w), max(2.0, 0.3 * ppmm), rng)
    sheet *= fibre[..., None]

    sheet = sheet * (1 - a_grid[..., None]) + gink * a_grid[..., None]
    sheet = sheet * (1 - a_trace[..., None]) + tink * a_trace[..., None]
    return sheet, palette


# ---------------------------------------------------------------- sheet
def sheet_effects(sheet: np.ndarray, rng, ppmm: float) -> np.ndarray:
    """Wrinkle field and fold lines - lighting on paper, never ink."""
    h, w = sheet.shape[:2]
    files = _texture_files()
    if files and rng.random() < P.WRINKLE_PROB:
        tex = Image.open(files[int(rng.integers(len(files)))]).convert("L")
        tw, th = tex.size
        s = rng.uniform(0.5, 1.0)
        cw, ch = max(16, int(tw * s)), max(16, int(th * s))
        x0 = int(rng.integers(0, max(1, tw - cw + 1)))
        y0 = int(rng.integers(0, max(1, th - ch + 1)))
        tex = tex.crop((x0, y0, x0 + cw, y0 + ch))
        if rng.random() < 0.5:
            tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            tex = tex.transpose(Image.ROTATE_90)
        f = np.asarray(tex.resize((w, h), Image.LANCZOS), np.float32) / 255.0
        # HIGH-PASS: the crease photographs carry large dark blobs that read as
        # stains. Subtracting a blur at WRINKLE_HIGHPASS_MM keeps the fine
        # relief of the paper and discards the blob scale entirely.
        f = f - _blur_mm(f, P.WRINKLE_HIGHPASS_MM, ppmm)
        sd = float(f.std())
        if sd > 1e-4:
            f = 1.0 + (f / sd) * rng.uniform(*P.WRINKLE_AMP)
            sheet = sheet * np.clip(f, *P.WRINKLE_CLIP)[..., None]

    # Fold lines: an odd (derivative-of-Gaussian) profile, bright on one side
    # and dark on the other, which is what a physical crease does to light.
    if rng.random() < P.CREASE_PROB:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        field = np.ones((h, w), np.float32)
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() < 0.5:
                d = yy - (rng.uniform(0.15, 0.85) * h + rng.uniform(-.05, .05) * (xx - w / 2))
            else:
                d = xx - (rng.uniform(0.15, 0.85) * w + rng.uniform(-.05, .05) * (yy - h / 2))
            wd = rng.uniform(0.008, 0.022) * max(h, w)
            g = np.exp(-(d ** 2) / (2 * wd ** 2))
            field *= 1.0 + rng.uniform(*P.CREASE_AMP) * g * (d / wd) / 1.6
        sheet = sheet * np.clip(field, 0.93, 1.07)[..., None]
    return sheet


def illuminate(sheet: np.ndarray, rng) -> np.ndarray:
    """Ambient light across the sheet, then normalise to a measured paper level."""
    h, w = sheet.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    field = 1.0 + rng.uniform(*P.LIGHT_RAMP) * 2.0 * (
        np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
    r = np.sqrt((xx / w - .5) ** 2 + (yy / h - .5) ** 2)
    field *= 1.0 - rng.uniform(*P.LIGHT_FALLOFF) * (r / r.max()) ** 2
    if rng.random() < P.LIGHT_SHADOW_PROB:
        d = {0: xx / w, 1: 1 - xx / w, 2: yy / h, 3: 1 - yy / h}[int(rng.integers(0, 4))]
        field *= 1.0 - rng.uniform(*P.LIGHT_SHADOW_AMP) * np.exp(-d / rng.uniform(.05, .16))
    sheet = sheet * field[..., None]

    lum = sheet @ np.array([.299, .587, .114], np.float32)
    p98 = float(np.percentile(lum, 98))
    if p98 > 1.0:
        sheet *= rng.uniform(*P.PAPER_TARGET_P98) / p98
    return sheet


# ---------------------------------------------------------------- camera
def _homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def _apply_h(H, pts):
    p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ H.T
    return p[:, :2] / p[:, 2:3]


def camera_quad(sw: int, sh: int, fw: int, fh: int, rng):
    """Project the sheet's four corners through a simulated camera pose.

    The sheet is a rectangle in 3-D; it is rotated about its own in-plane axes
    (yaw, pitch) and about the view axis (roll), then projected through a
    pinhole at CAM_DISTANCE_W sheet-widths. That yields TRUE keystone - the
    far edge is genuinely shorter than the near edge - which a random corner
    jitter cannot produce and which the eye reads immediately.
    """
    ry = np.radians(rng.uniform(*P.CAM_YAW_DEG))
    rx = np.radians(rng.uniform(*P.CAM_PITCH_DEG))
    rz = np.radians(rng.uniform(*P.CAM_ROLL_DEG))
    corners = np.array([[-.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0]], float)
    corners[:, 0] *= sw / sh                                  # sheet aspect
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    pts = corners @ (Rz @ Rx @ Ry).T

    d = rng.uniform(*P.CAM_DISTANCE_W)
    proj = pts[:, :2] / (d - pts[:, 2:3]) * d                 # pinhole, unit focal

    fill = rng.uniform(*P.SHEET_FILL)
    span = max(proj[:, 0].ptp() / (fw / fh), proj[:, 1].ptp())
    proj *= fill * fh / span
    proj += np.array([fw / 2, fh / 2]) + np.array(
        [rng.uniform(*P.CAM_OFFSET) * fw, rng.uniform(*P.CAM_OFFSET) * fh])
    return proj                                               # UL, LL, LR, UR


def background(fw: int, fh: int, rng, ppmm: float) -> np.ndarray:
    """The surface the sheet is lying on. Procedural - no assets to ship."""
    base = np.array(P.BG_RGB_CHOICES[int(rng.integers(len(P.BG_RGB_CHOICES)))], np.float32)
    bg = np.broadcast_to(base, (fh, fw, 3)).astype(np.float32).copy()
    bg *= (1.0 + rng.uniform(*P.BG_SHADE_AMP) * _fbm((fh, fw), 30 * ppmm, rng))[..., None]
    bg *= (1.0 + 0.05 * _fbm((fh, fw), 1.2 * ppmm, rng))[..., None]   # surface grain
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    bg *= (1.0 - 0.20 * ((xx / fw - .5) ** 2 + (yy / fh - .5) ** 2))[..., None]
    # The surface is off the focal plane, so it is blurred BEFORE the sensor
    # noise is added - blurring afterwards would smooth the noise too.
    bg = _blur_mm(np.clip(bg, 0, 255) / 255.0, rng.uniform(*P.BG_BLUR_MM), ppmm) * 255.0
    bg += rng.normal(0, rng.uniform(*P.BG_NOISE), bg.shape)
    return bg


def compose(sheet: np.ndarray, rng, ppmm: float):
    """Place the lit sheet in a camera frame. Returns (frame, forward homography)."""
    sh, sw = sheet.shape[:2]
    aspect = rng.uniform(*P.OUT_ASPECT)
    fw = int(round(sw * 1.06))
    fh = int(round(fw / aspect))
    if fh < sh * 1.02:
        fh = int(round(sh * 1.02)); fw = int(round(fh * aspect))

    quad = camera_quad(sw, sh, fw, fh, rng)
    src = np.array([[0, 0], [0, sh], [sw, sh], [sw, 0]], float)
    H = _homography(src, quad)

    frame = background(fw, fh, rng, ppmm)

    # Warp the sheet and its alpha with one inverse map. PIL wants dst->src.
    Hi = np.linalg.inv(H); Hi = Hi / Hi[2, 2]
    coeffs = tuple(Hi.flatten()[:8])
    sheet_u8 = Image.fromarray(np.clip(sheet, 0, 255).astype(np.uint8))
    warped = np.asarray(sheet_u8.transform((fw, fh), Image.PERSPECTIVE, coeffs,
                                           resample=Image.BICUBIC), np.float32)
    mask = Image.new("L", (sw, sh), 255).transform(
        (fw, fh), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
    a = np.asarray(mask, np.float32)[..., None] / 255.0

    # Cast shadow: the sheet's own silhouette, offset, blurred and darkening.
    off = int(rng.uniform(*P.SHADOW_OFFSET_MM) * ppmm)
    sm = np.asarray(mask.filter(ImageFilter.GaussianBlur(
        rng.uniform(*P.SHADOW_BLUR_MM) * ppmm)), np.float32) / 255.0
    sm = np.roll(np.roll(sm, off, 0), int(off * rng.uniform(-1, 1)), 1)
    frame *= (1.0 - rng.uniform(*P.SHADOW_STRENGTH) * np.clip(sm - a[..., 0], 0, 1))[..., None]

    return frame * (1 - a) + warped * a, H


def sensor(frame: np.ndarray, rng, out_long: int):
    """Optical integration, then the camera itself.

    Downsampling here - not blurring earlier - is what makes the print read as
    photographed: every output pixel is an area average of many render pixels,
    exactly as a sensor well integrates the light falling on it.
    """
    fh, fw = frame.shape[:2]
    scale = out_long / max(fw, fh)
    ow, oh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).resize(
        (ow, oh), Image.BOX if scale < 1 else Image.BICUBIC)

    arr = np.asarray(img, np.float32)

    # Lateral chromatic aberration: channels scaled apart about the centre.
    cs = rng.uniform(*P.CHROMA_SHIFT_PX)
    if cs > 0.05:
        out = np.empty_like(arr)
        for ch, k in ((0, 1.0 + cs / max(ow, oh)), (1, 1.0), (2, 1.0 - cs / max(ow, oh))):
            if abs(k - 1.0) < 1e-6:
                out[..., ch] = arr[..., ch]; continue
            im = Image.fromarray(arr[..., ch].astype(np.uint8))
            big = im.resize((max(1, int(ow * k)), max(1, int(oh * k))), Image.BICUBIC)
            l, t = (big.width - ow) // 2, (big.height - oh) // 2
            out[..., ch] = np.asarray(big.crop((l, t, l + ow, t + oh)), np.float32)
        arr = out

    arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(rng.uniform(*P.LENS_BLUR_PX))), np.float32)

    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    r2 = (xx / ow - .5) ** 2 + (yy / oh - .5) ** 2
    arr *= (1.0 - rng.uniform(*P.LENS_VIGNETTE) * r2 / r2.max())[..., None]

    arr *= (1.0 + rng.normal(0, P.WB_JITTER, 3))[None, None, :]

    shot = rng.uniform(*P.SENSOR_SHOT_NOISE) * np.sqrt(np.clip(arr, 0, None) / 255.0)
    arr += rng.standard_normal(arr.shape) * (shot + rng.uniform(*P.SENSOR_READ_NOISE))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), scale


# ---------------------------------------------------------------- driver
def photograph(src: Path, dst: Path, seed_parts: tuple, force_palette=None) -> dict:
    rng = _rng(seed_parts)
    ppmm = P.RENDER_PX_PER_MM
    grey = np.asarray(Image.open(src).convert("L"), np.float32)
    sh, sw = grey.shape

    sheet, palette = press(grey, rng, ppmm, force_palette)
    sheet = sheet_effects(sheet, rng, ppmm)
    sheet = illuminate(sheet, rng)
    frame, H = compose(sheet, rng, ppmm)
    del sheet
    img, scale = sensor(frame, rng, P.OUT_LONG_EDGE)
    del frame

    q = int(rng.integers(P.JPEG_QUALITY[0], P.JPEG_QUALITY[1] + 1))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=q, subsampling=P.JPEG_SUBSAMPLING, optimize=True)

    geom = {"H": H.tolist(), "scale": scale, "src_size": [sw, sh],
            "out_size": list(img.size)}
    wrote = _write_annotation(src, dst, geom, palette)
    return {"grid_palette": palette, "jpeg_quality": q,
            "px_per_mm_out": round(ppmm * scale, 2),
            "out_w": img.size[0], "out_h": img.size[1],
            "has_annotation": int(wrote)}


def _forward(pts_yx, H, scale):
    """Map kit [y, x] coordinates onto the finished photograph."""
    pts = np.asarray(pts_yx, float)
    if pts.size == 0:
        return []
    xy = _apply_h(np.asarray(H), pts[:, ::-1].copy()) * scale
    return [[round(float(b), 2), round(float(a), 2)] for a, b in xy]


def _write_annotation(src: Path, dst: Path, geom: dict, palette: str) -> bool:
    ann_src = src.with_suffix(".json")
    if not ann_src.exists():
        ann_src = src.parent / (src.stem + ".json")
    if not ann_src.exists():
        return False
    try:
        ann = json.load(open(ann_src))
        H, s = np.asarray(geom["H"]), geom["scale"]
        out = dict(ann)
        out["width"], out["height"] = geom["out_size"]
        out["photo_geometry"] = {"homography": geom["H"], "downsample": s,
                                 "grid_palette": palette,
                                 "note": "coords are [y, x] from top-left of THIS image"}
        leads = []
        for e in ann.get("leads", []):
            e = dict(e)
            for key in ("lead_bounding_box", "text_bounding_box"):
                box = e.get(key)
                if isinstance(box, dict) and box:
                    order = sorted(box, key=lambda k: int(k))
                    moved = _forward([box[k] for k in order], H, s)
                    e[key] = dict(zip(order, moved))
            if e.get("plotted_pixels"):
                e["plotted_pixels"] = _forward(e["plotted_pixels"], H, s)
            leads.append(e)
        out["leads"] = leads
        json.dump(out, open(dst.with_suffix(".json"), "w"))
        return True
    except Exception:                       # noqa: BLE001 - never abort a build
        return False


def _job(a):
    row, k, out = a
    src = (C.RENDERED / C.slug(row["cls"]) / f"r{min(k, 0)}" / row["chunk"]
           / f"{row['record']}-0.png")
    if not src.exists():
        cand = list((C.RENDERED / C.slug(row["cls"])).glob(f"r*/{row['chunk']}/{row['record']}-*.png"))
        if not cand:
            return None
        src = cand[0]
    meta = photograph(src, out, (row["record"], k))
    rec = {f: row.get(f, "") for f in INDEX_FIELDS if f in row}
    rec.update({"image_path": str(out), "render_k": k, **meta})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--workers", type=int, default=max(1, (C.WORKERS or 4) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--palette", default=None, choices=list(P.GRID_INK_RGB))
    args = ap.parse_args()

    if not C.STAGED_MAP.exists():
        print("staged_map.csv missing - run stages 1-3 first", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(C.STAGED_MAP.open()))
    if args.limit:
        rows = rows[: args.limit]

    jobs = []
    for r in rows:
        split = r.get("split") or "train"
        n = 1 if split in C.EVAL_SPLITS else max(1, C.VARIANTS_PER_SPLIT.get(split, 1))
        for k in range(n):
            jobs.append((r, k, C.IMAGES / split / f"{r['record']}_a{k}.jpg"))

    print(f"{len(jobs)} photograph(s) on {args.workers} worker(s)")
    out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_job, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                out.append(r)
            print(f"  [{i}/{len(jobs)}] {'ok' if r else 'MISSING RENDER'}")

    if not out:
        print("nothing written", file=sys.stderr)
        return 1
    fields = list(dict.fromkeys(list(INDEX_FIELDS) + list(out[0])))
    C.INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with C.INDEX_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in out)
    sizes = [Path(r["image_path"]).stat().st_size for r in out]
    print(f"\nwrote {len(out)} images -> {C.IMAGES}\n"
          f"mean {np.mean(sizes)/1024:.0f} KB   {out[0]['out_w']}x{out[0]['out_h']}   "
          f"{out[0]['px_per_mm_out']} px/mm on the sheet")
    print(f"index -> {C.INDEX_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
