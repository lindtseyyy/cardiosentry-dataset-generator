"""Aspect-preserving resize to the 1024x768 model frame (PLAN 7.3).

LANCZOS fit to height 768 (width follows aspect: 994 for the 2200x1700
master) + symmetric paper-coloured padding to 1024. No stretch, ever:
the aspect ratio is asserted, not assumed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import FINAL_W, FINAL_H, RESIZE_H, PAD_EACH_SIDE
from .util import V1Error
from .variants import paper_colour


def to_final(img: Image.Image) -> tuple[Image.Image, dict]:
    w, h = img.size
    scale = RESIZE_H / h
    new_w = round(w * scale)
    small = img.resize((new_w, RESIZE_H), Image.LANCZOS)
    # aspect check: resized frame must match source aspect to <0.5 px total
    if abs((w / h) - (new_w / RESIZE_H)) > 0.5 / RESIZE_H + 1e-9:
        raise V1Error(f"aspect drift: source {w}x{h} -> {new_w}x{RESIZE_H}")
    pad_total = FINAL_W - new_w
    if pad_total < 0:
        raise V1Error(f"resized width {new_w} exceeds final {FINAL_W}")
    left = pad_total // 2
    right = pad_total - left
    fill = paper_colour(np.asarray(small.convert("RGB"), dtype=np.float32))
    canvas = Image.new("RGB", (FINAL_W, FINAL_H), fill)
    canvas.paste(small.convert("RGB"), (left, 0))
    if canvas.size != (FINAL_W, FINAL_H):
        raise V1Error(f"final size {canvas.size}, need {(FINAL_W, FINAL_H)}")
    return canvas, {"src_size": [w, h], "scale": scale,
                    "resized_size": [new_w, RESIZE_H],
                    "pad_left": left, "pad_right": right,
                    "pad_rgb": list(fill)}
