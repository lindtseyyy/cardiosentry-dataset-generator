#!/usr/bin/env python3
"""Overlay an ECG sheet's annotations on the image, to check they line up.

Works on both the clean render and the augmented JPEG, because stage 4 writes
a transformed annotation next to every image it produces.

    python plot_annotations.py IMAGE [-o OUT] [--what all|boxes|trace|text]
    python plot_annotations.py ../data/.../images/train/ECG000123_a0.jpg
    python plot_annotations.py --lead V2 --zoom sheet.jpg     # crop one lead

The JSON is found automatically: <image>.json, or <stem>.json beside it.

Kit coordinate convention (verified against a real render): every coordinate
is a [y, x] pair, y measured DOWN from the top-left of the image the JSON
describes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Distinct hues so overlapping panels stay readable.
PALETTE = [
    (230, 40, 40), (40, 140, 230), (30, 170, 90), (240, 150, 20),
    (170, 60, 220), (0, 175, 175), (220, 60, 150), (120, 120, 40),
    (60, 90, 230), (200, 90, 40), (80, 180, 40), (200, 40, 90), (90, 90, 90),
]


def find_json(img_path: Path) -> Path | None:
    for cand in (img_path.with_suffix(".json"),
                 img_path.parent / (img_path.stem + ".json"),
                 img_path.parent / (img_path.stem.rsplit("-", 1)[0] + ".json")):
        if cand.exists():
            return cand
    return None


def box_xy(box: dict) -> list:
    """dict of '0'..'3' -> [y, x] into a polygon of (x, y)."""
    return [(v[1], v[0]) for _, v in sorted(box.items(), key=lambda kv: int(kv[0]))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-j", "--json", default=None)
    ap.add_argument("--what", default="all",
                    choices=["all", "boxes", "trace", "text"])
    ap.add_argument("--lead", default=None, help="only this lead (e.g. V2)")
    ap.add_argument("--zoom", action="store_true",
                    help="crop to the selected lead's box")
    ap.add_argument("--stride", type=int, default=3,
                    help="plot every Nth traced pixel (default %(default)s)")
    args = ap.parse_args(argv)

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"no such image: {img_path}", file=sys.stderr)
        return 1
    jpath = Path(args.json) if args.json else find_json(img_path)
    if not jpath:
        print(f"no annotation JSON found beside {img_path.name}.\n"
              "  Clean renders keep it as <record>-0.json; augmented images get\n"
              "  one written by stage 4 only if the render's JSON was present.",
              file=sys.stderr)
        return 1

    ann = json.loads(jpath.read_text())
    img = Image.open(img_path).convert("RGB")

    if (ann.get("width"), ann.get("height")) != img.size:
        print(f"  ! annotation says {ann.get('width')}x{ann.get('height')}, "
              f"image is {img.size[0]}x{img.size[1]} - overlay will not line up",
              file=sys.stderr)

    d = ImageDraw.Draw(img, "RGBA")
    leads = ann.get("leads", [])
    if args.lead:
        leads = [e for e in leads if e.get("lead_name", "").upper() == args.lead.upper()]
        if not leads:
            print(f"lead {args.lead!r} not in {jpath.name}", file=sys.stderr)
            return 1

    crop = None
    for i, e in enumerate(leads):
        col = PALETTE[i % len(PALETTE)]
        name = e.get("lead_name", "?")

        if args.what in ("all", "boxes") and e.get("lead_bounding_box"):
            poly = box_xy(e["lead_bounding_box"])
            d.polygon(poly, outline=col + (255,), width=3)
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            d.text((min(xs) + 4, min(ys) - 16), name, fill=col + (255,))
            if args.zoom and crop is None:
                crop = (min(xs), min(ys), max(xs), max(ys))

        if args.what in ("all", "text") and e.get("text_bounding_box"):
            d.polygon(box_xy(e["text_bounding_box"]),
                      outline=(0, 0, 0, 255), width=2)

        if args.what in ("all", "trace"):
            px = e.get("plotted_pixels") or []
            for y, x in px[::max(1, args.stride)]:
                d.ellipse((x - 1.1, y - 1.1, x + 1.1, y + 1.1),
                          fill=col + (190,))

    if crop:
        pad = 60
        img = img.crop((max(0, crop[0] - pad), max(0, crop[1] - pad),
                        min(img.width, crop[2] + pad), min(img.height, crop[3] + pad)))

    out = Path(args.out) if args.out else img_path.with_name(img_path.stem + "_annotated.jpg")
    img.save(out, "JPEG", quality=92)
    g = ann.get("augmentation_geometry")
    print(f"{out}   {len(leads)} leads"
          + (f"   [rot {g['rotation_deg']}deg, "
             f"quad {'yes' if g['perspective_quad'] else 'no'}, "
             f"margin {g['margin_px']}px]" if g else "   [clean render]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
