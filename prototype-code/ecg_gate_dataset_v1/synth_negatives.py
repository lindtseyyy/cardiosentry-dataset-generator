"""Synthetic hard negatives for the ECG / not-ECG gate.

Every image is paper that looks like an ECG sheet in some way but carries no
ECG: ECG-style grid paper with no traces, graph paper with smooth plotted
curves, notebook grids with lines of printed text, and smooth curves on an
ECG-style grid. Nothing here draws spiky QRS-like waveforms, lead labels, a
calibration pulse or a multi-row lead layout: those are the things that make a
sheet an ECG, and the gate must learn them rather than "pink grid = ECG".

Half the images get a synthetic phone-photo treatment (perspective onto a
background, uneven light, noise, blur, JPEG) so the clean-render vs photograph
axis is present on the negative side too. Rendering is deterministic per seed.
"""

from __future__ import annotations

import io
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

KINDS = ("blank_ecg_grid", "graph_paper_curves", "notebook_grid", "ecg_grid_smooth_curves")

ECG_GRID_COLOURS = {
    "red": (238, 120, 120),
    "pink": (245, 170, 180),
    "orange": (240, 160, 100),
    "grey": (170, 170, 170),
    "green": (130, 190, 140),
}
GRAPH_GRID_COLOURS = {
    "blue": (140, 175, 225),
    "green": (140, 200, 150),
    "grey": (185, 185, 185),
    "cyan": (130, 200, 210),
}
PAPER_COLOURS = ((255, 255, 255), (252, 250, 243), (250, 246, 236), (246, 246, 246))


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _random_text(rng: np.random.Generator, n_min: int, n_max: int) -> str:
    alphabet = string.ascii_letters + string.digits + "     :/-.,"
    return "".join(rng.choice(list(alphabet), size=int(rng.integers(n_min, n_max))))


def _canvas(rng: np.random.Generator) -> tuple[Image.Image, float]:
    landscape = rng.random() < 0.75
    w, h = (1600, 1200) if landscape else (1200, 1600)
    paper = PAPER_COLOURS[int(rng.integers(len(PAPER_COLOURS)))]
    px_per_mm = float(rng.uniform(4.5, 7.5))
    return Image.new("RGB", (w, h), paper), px_per_mm


def _draw_grid(draw: ImageDraw.ImageDraw, size: tuple[int, int], step: float, colour, major_every: int,
               minor_width: int = 1, major_width: int = 2, dotted: bool = False) -> None:
    w, h = size
    light = tuple(int(c + (255 - c) * 0.55) for c in colour)
    n_x = int(w / step) + 1
    n_y = int(h / step) + 1
    if dotted:
        for i in range(n_x):
            for j in range(n_y):
                x, y = i * step, j * step
                draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=colour)
        return
    for i in range(n_x):
        x = i * step
        major = major_every and i % major_every == 0
        draw.line([(x, 0), (x, h)], fill=colour if major else light, width=major_width if major else minor_width)
    for j in range(n_y):
        y = j * step
        major = major_every and j % major_every == 0
        draw.line([(0, y), (w, y)], fill=colour if major else light, width=major_width if major else minor_width)


def _header_text(draw: ImageDraw.ImageDraw, rng: np.random.Generator, w: int) -> None:
    font = _font(int(rng.integers(16, 26)))
    x0 = int(rng.integers(30, 90))
    y = int(rng.integers(20, 50))
    for _ in range(int(rng.integers(2, 5))):
        draw.text((x0, y), _random_text(rng, 12, 40), fill=(30, 30, 30), font=font)
        if rng.random() < 0.6:
            draw.text((w // 2 + int(rng.integers(0, 150)), y), _random_text(rng, 8, 28), fill=(30, 30, 30), font=font)
        y += int(font.size * 1.5)


def _smooth_series(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    kind = int(rng.integers(5))
    if kind == 0:
        y = np.sin(2 * np.pi * rng.uniform(0.5, 4) * t + rng.uniform(0, 2 * np.pi))
    elif kind == 1:
        y = np.exp(-rng.uniform(1, 5) * t) * np.cos(2 * np.pi * rng.uniform(2, 6) * t)
    elif kind == 2:
        coeffs = rng.normal(0, 1, size=4)
        y = np.polyval(coeffs, t * 2 - 1)
    elif kind == 3:
        y = 1 / (1 + np.exp(-rng.uniform(6, 15) * (t - rng.uniform(0.3, 0.7))))
    else:
        steps = rng.normal(0, 1, size=n)
        kernel = np.hanning(max(5, n // 12))
        y = np.convolve(np.cumsum(steps), kernel / kernel.sum(), mode="same")
    y = y - y.min()
    return y / (y.max() + 1e-9)


def _plot_curves(draw: ImageDraw.ImageDraw, rng: np.random.Generator, box: tuple[int, int, int, int],
                 n_curves: int, width_range=(2, 5)) -> None:
    x0, y0, x1, y1 = box
    n = max(50, (x1 - x0) // 3)
    xs = np.linspace(x0, x1, n)
    palette = [(20, 20, 20), (30, 60, 170), (190, 30, 30), (20, 120, 50), (120, 40, 140)]
    for _ in range(n_curves):
        ys = _smooth_series(rng, n)
        lo, hi = sorted(rng.uniform(0.1, 0.9, size=2))
        ys = y1 - (lo + ys * max(hi - lo, 0.15)) * (y1 - y0)
        colour = palette[int(rng.integers(len(palette)))]
        draw.line(list(zip(xs.tolist(), ys.tolist())), fill=colour, width=int(rng.integers(*width_range)), joint="curve")


def _axes(draw: ImageDraw.ImageDraw, rng: np.random.Generator, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    font = _font(int(rng.integers(14, 22)))
    draw.line([(x0, y1), (x1, y1)], fill=(20, 20, 20), width=3)
    draw.line([(x0, y0), (x0, y1)], fill=(20, 20, 20), width=3)
    for i in range(int(rng.integers(5, 11)) + 1):
        x = x0 + (x1 - x0) * i / 10
        draw.line([(x, y1), (x, y1 + 10)], fill=(20, 20, 20), width=2)
        draw.text((x - 8, y1 + 14), str(i * int(rng.integers(1, 20))), fill=(20, 20, 20), font=font)
    for j in range(6):
        y = y1 - (y1 - y0) * j / 5
        draw.line([(x0 - 10, y), (x0, y)], fill=(20, 20, 20), width=2)
        draw.text((x0 - 60, y - 8), f"{j * rng.uniform(0.1, 50):.1f}", fill=(20, 20, 20), font=font)
    draw.text(((x0 + x1) // 2 - 80, y0 - 50), _random_text(rng, 10, 30), fill=(20, 20, 20), font=_font(28))


def _blank_ecg_grid(rng: np.random.Generator) -> Image.Image:
    img, ppm = _canvas(rng)
    draw = ImageDraw.Draw(img)
    colour = list(ECG_GRID_COLOURS.values())[int(rng.integers(len(ECG_GRID_COLOURS)))]
    _draw_grid(draw, img.size, ppm, colour, major_every=5)
    if rng.random() < 0.6:
        _header_text(draw, rng, img.size[0])
    return img


def _graph_paper_curves(rng: np.random.Generator) -> Image.Image:
    img, ppm = _canvas(rng)
    draw = ImageDraw.Draw(img)
    colour = list(GRAPH_GRID_COLOURS.values())[int(rng.integers(len(GRAPH_GRID_COLOURS)))]
    _draw_grid(draw, img.size, ppm * rng.uniform(1.0, 2.0), colour, major_every=int(rng.choice([0, 5, 10])))
    w, h = img.size
    box = (int(w * 0.12), int(h * 0.15), int(w * 0.92), int(h * 0.85))
    _axes(draw, rng, box)
    _plot_curves(draw, rng, box, n_curves=int(rng.integers(1, 4)))
    return img


def _notebook_grid(rng: np.random.Generator) -> Image.Image:
    img, ppm = _canvas(rng)
    draw = ImageDraw.Draw(img)
    colour = list(GRAPH_GRID_COLOURS.values())[int(rng.integers(len(GRAPH_GRID_COLOURS)))]
    style = int(rng.integers(3))
    step = ppm * rng.uniform(4.0, 7.0)
    if style == 0:
        _draw_grid(draw, img.size, step, colour, major_every=0)
    elif style == 1:
        _draw_grid(draw, img.size, step, colour, major_every=0, dotted=True)
    else:
        w, h = img.size
        for j in range(int(h / step) + 1):
            draw.line([(0, j * step), (w, j * step)], fill=colour, width=2)
        draw.line([(int(w * 0.1), 0), (int(w * 0.1), h)], fill=(220, 120, 120), width=2)
    w, h = img.size
    ink = (25, 35, 110) if rng.random() < 0.6 else (20, 20, 20)
    font = _font(int(rng.integers(22, 36)))
    y = int(rng.uniform(0.06, 0.12) * h)
    while y < h * 0.9:
        x = int(rng.uniform(0.08, 0.15) * w)
        words = " ".join(_random_text(rng, 2, 9).strip() for _ in range(int(rng.integers(3, 10))))
        draw.text((x, y), words, fill=ink, font=font)
        y += int(max(step, font.size * 1.3) * rng.uniform(1.0, 1.6))
    return img


def _ecg_grid_smooth_curves(rng: np.random.Generator) -> Image.Image:
    img, ppm = _canvas(rng)
    draw = ImageDraw.Draw(img)
    colour = list(ECG_GRID_COLOURS.values())[int(rng.integers(len(ECG_GRID_COLOURS)))]
    _draw_grid(draw, img.size, ppm, colour, major_every=5)
    w, h = img.size
    box = (int(w * 0.08), int(h * 0.2), int(w * 0.94), int(h * 0.85))
    if rng.random() < 0.5:
        _axes(draw, rng, box)
    _plot_curves(draw, rng, box, n_curves=int(rng.integers(1, 3)), width_range=(2, 4))
    return img


RENDERERS = {
    "blank_ecg_grid": _blank_ecg_grid,
    "graph_paper_curves": _graph_paper_curves,
    "notebook_grid": _notebook_grid,
    "ecg_grid_smooth_curves": _ecg_grid_smooth_curves,
}


def photo_treatment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Place the sheet on a background with perspective, light falloff, noise, blur and JPEG."""
    sheet = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = sheet.shape[:2]
    bg_h, bg_w = int(h * rng.uniform(1.1, 1.35)), int(w * rng.uniform(1.1, 1.35))
    base = rng.uniform(40, 200, size=3)
    background = np.clip(base + rng.normal(0, 12, size=(bg_h, bg_w, 3)), 0, 255).astype(np.float32)
    background = cv2.GaussianBlur(background, (0, 0), 3)
    ox, oy = (bg_w - w) / 2, (bg_h - h) / 2
    jitter = lambda s: rng.uniform(-s, s)  # noqa: E731
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [ox + jitter(0.06 * w), oy + jitter(0.06 * h)],
        [ox + w + jitter(0.06 * w), oy + jitter(0.06 * h)],
        [ox + w + jitter(0.06 * w), oy + h + jitter(0.06 * h)],
        [ox + jitter(0.06 * w), oy + h + jitter(0.06 * h)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, matrix, (bg_w, bg_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), matrix, (bg_w, bg_h))[..., None]
    out = warped * mask + background * (1 - mask)
    yy, xx = np.mgrid[0:bg_h, 0:bg_w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * xx / bg_w + np.sin(angle) * yy / bg_h)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-9)
    light = 1.0 - rng.uniform(0.1, 0.4) * ramp
    tint = rng.uniform(0.92, 1.05, size=3)
    out = out * light[..., None] * tint
    out = out + rng.normal(0, rng.uniform(2, 7), size=out.shape)
    sigma = rng.uniform(0.3, 1.4)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), sigma)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=int(rng.integers(60, 92)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render(kind: str, seed: int, photo: bool) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = RENDERERS[kind](rng)
    return photo_treatment(img, rng) if photo else img
