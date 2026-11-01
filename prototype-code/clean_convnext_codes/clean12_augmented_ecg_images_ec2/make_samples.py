#!/usr/bin/env python3
"""Preview: the augmented sheets the EC2 notebook writes for a few records.

Uses records from a correct-prediction sample folder (2 per condition) whose
grey kit masters were kept by export_augmented_preview.py (_work/masters,
rendered with the clean12 kit settings and checked pixel-identical to the
clean12 sheets). Plans the WHOLE clean12 manifest with the notebook's seed,
so each record gets exactly the levels the EC2 run assigns it, and augments
the masters with augment_worker.py. Writes, into samples/:

    NN_<CODE>_<site>_<file_id>_clean.png
    NN_<CODE>_<site>_<file_id>_augmented_<grid>_tilt<deg>.png
    index.csv          plan levels + every effect value per sheet
    contact_sheet.png  clean | augmented, one row per record (downscaled)

    python3 make_samples.py [--per-condition 2]
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SAMPLES_SRC = (ROOT / "prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42"
               / "correct_prediction_samples")
CLEAN12_MANIFEST = ROOT / "dataset-files/HEEDB/extracted/ECG_500Hz_clean12/manifest.csv"
CONDITIONS = ["SB", "STACH", "LBBB", "LQT", "AF", "AFL", "3AVB", "2AVB", "1AVB", "STEMI", "STE", "NORM"]
WRINKLES = "ecg_image_v1_code/ecg-image-generator/CreasesWrinkles/wrinkles-dataset/"


def notebook_seed() -> int:
    nb = json.loads((HERE / "Clean12_Augmented_ECG_Images_EC2.ipynb").read_text())
    config = "".join(nb["cells"][1]["source"])
    return int(re.search(r"^AUGMENT_SEED = (\d+)", config, re.M).group(1))


def pick_records(per_condition: int) -> list[tuple[str, dict]]:
    """(condition, index row) pairs: single-label folders, STEMI from STEMI+STE."""
    with (SAMPLES_SRC / "index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    picked = []
    for condition in CONDITIONS:
        folder = "combo_2/STEMI+STE" if condition == "STEMI" else None
        pool = [r for r in rows if (r["folder"] == folder if folder else
                                    r["folder"].startswith(f"single/{condition}_"))]
        picked += [(condition, r) for r in pool[:per_condition]]
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-condition", type=int, default=2)
    parser.add_argument("--out", type=Path, default=HERE / "samples")
    args = parser.parse_args()

    seed = notebook_seed()
    with CLEAN12_MANIFEST.open(newline="") as handle:
        records = [{"record_path": r["record_path"], "site": r["site"], "labels": r["labels"]}
                   for r in csv.DictReader(handle)]
    picked = pick_records(args.per_condition)

    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    rows, thumbs = [], []
    with tempfile.TemporaryDirectory() as tmp:
        # The notebook's two worker modules: clean12 render worker + augment_worker.
        sys.path.insert(0, str(HERE))
        import make_notebook
        (Path(tmp) / "ecg_clean_worker.py").write_text(make_notebook.clean12_worker_source())
        sys.path.insert(0, tmp)
        AW = importlib.import_module("augment_worker")
        plan = AW.balanced_plan(records, seed)
        textures = Path(tmp) / "wrinkles"
        textures.mkdir()
        with tarfile.open(HERE / "ecg_image_v1_code.tar.gz", "r:gz") as tar:
            for info in tar:
                if info.isfile() and info.name.startswith(WRINKLES):
                    (textures / Path(info.name).name).write_bytes(tar.extractfile(info).read())
        AW.set_texture_dir(textures)

        for n, (condition, row) in enumerate(picked, 1):
            rel = row["record_id"]
            stem = f"{n:02d}_{condition}_{row['site']}_{row['file_id']}"
            master_path = SAMPLES_SRC / "_work" / "masters" / f"{row['site']}_{row['file_id']}.png"
            with Image.open(master_path) as master:
                final, meta = AW.augment_master(master, seed, rel, plan[rel])
            AW.check_plan(meta, plan[rel])
            shutil.copyfile(SAMPLES_SRC / row["clean_image"], args.out / f"{stem}_clean.png")
            name = f"{stem}_augmented_{meta['grid_palette']}_tilt{meta['tilt_deg']:+.1f}.png"
            final.save(args.out / name, compress_level=3)
            rows.append({"image": name, "condition": condition, "labels": row["true_labels"],
                         "record_path": rel, **meta, "ink_bin": plan[rel]["ink_bin"],
                         "tilt_bin": plan[rel]["tilt_bin"]})
            effects = ", ".join(k for k in ("perspective", "fold_creases", "edge_shadow", "blur") if meta[k])
            with Image.open(args.out / f"{stem}_clean.png") as clean:
                clean_thumb = clean.convert("RGB").resize((512, 384), Image.LANCZOS)
            thumbs.append((clean_thumb, final.resize((512, 384), Image.LANCZOS),
                           f"{condition} | {meta['grid_palette']}, tilt {meta['tilt_deg']:+.1f}, "
                           f"ink {meta['grid_saturation']:.2f} | {effects or 'no optional effects'}"))
            print(f"{name}: {effects or '-'}")

    fields = ["image", "condition", "labels", "record_path", "ink_bin", "tilt_bin",
              *[f for f in AW.MANIFEST_FIELDS if f not in ("record_path", "image_path", "status", "error",
                                                           "image_bytes", "ink_bin", "tilt_bin")]]
    with (args.out / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)

    sheet = Image.new("RGB", (1024, 24 + 400 * len(thumbs)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "clean (left) | augmented (right): the sheets the EC2 run writes for these records", fill="black")
    for i, (left, right, caption) in enumerate(thumbs):
        y = 24 + 400 * i
        sheet.paste(left, (0, y + 14))
        sheet.paste(right, (512, y + 14))
        draw.text((8, y), caption, fill="black")
    sheet.save(args.out / "contact_sheet.png", optimize=True)
    print(f"Wrote {len(rows)} augmented + {len(rows)} clean sheets to {args.out}")


if __name__ == "__main__":
    main()
