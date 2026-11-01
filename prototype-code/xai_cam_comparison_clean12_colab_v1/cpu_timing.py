"""Explanation Time on THIS machine's CPU (fp32), for the deployment-hardware row.

The Colab notebook times the methods on its GPU and on Colab's CPU; the app runs
on a laptop CPU, so this script times the same code (`xai_cam.py`) there.

    cd prototype-code/xai_cam_comparison_clean12_colab_v1
    ../../.venv/bin/python cpu_timing.py --images <folder of clean12 PNGs> --out timing_laptop.csv

Defaults: the app's served checkpoint, 3 images, the top-scoring label of each.
Grad-CAM and Grad-CAM++ are measured. Score-CAM is extrapolated from
--scorecam-probe masked forwards (activation pass + channels x seconds per
masked forward), because a full CPU Score-CAM map takes hours; --scorecam-full
runs one for real.

Close the browser and other heavy programs first: on an 8-12 GB laptop, memory
pressure alone can double the forward time (36-78 s per forward was measured on
2026-10-05 with ~3 GB free).
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xai_cam as X  # noqa: E402

APP_CHECKPOINT = (HERE.parents[1] / "cardiosentry-application" / "models" / "weights"
                  / "convnext_v1_base_clean12_best.pt")
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def available_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def load_model(path, model_name, num_classes):
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(blob, dict) and "model" in blob:
        source = blob.get("selected_weight_source") or "raw"
        state = blob["model_ema" if source == "ema" else "model"]
    else:
        state = blob
    if state and all(k.startswith("_orig_mod.") for k in state):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    del blob
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_input(path, hw):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (hw[1], hw[0]):
            image = image.resize((hw[1], hw[0]), Image.BICUBIC)
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.0)
    return ((pixels[None] - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", required=True, help="a PNG, or a folder searched for *.png")
    parser.add_argument("--checkpoint", default=str(APP_CHECKPOINT))
    parser.add_argument("--model-name", default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--input-hw", default="768,1024")
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0, help="0 = every core")
    parser.add_argument("--scorecam-probe", type=int, default=16)
    parser.add_argument("--scorecam-batch", type=int, default=4)
    parser.add_argument("--scorecam-full", action="store_true")
    parser.add_argument("--out", default="timing_cpu_local.csv")
    args = parser.parse_args(argv)

    hw = tuple(int(v) for v in args.input_hw.split(","))
    source = Path(args.images)
    paths = [source] if source.is_file() else sorted(source.rglob("*.png"))[:args.n_images]
    if not paths:
        raise SystemExit(f"no PNGs under {source}")
    threads = args.threads or os.cpu_count()
    torch.set_num_threads(threads)
    device = torch.device("cpu")
    note = f"{cpu_name()} ({threads} threads, fp32)"
    print(f"{note}; {available_gib():.1f} GiB available; {len(paths)} images")

    model = load_model(args.checkpoint, args.model_name, args.num_classes)
    split = X.ModelSplit(model)
    inputs = [load_input(p, hw) for p in paths]
    with torch.no_grad():
        # The top-scoring label of each image; this pass also warms the model up.
        labels = [int(model(x)[0].argmax()) for x in inputs]

    rows = []
    for method in X.GRADIENT_METHODS:
        X.explain(split, inputs[0], labels[0], method)                       # warm-up
        for path, x, k in zip(paths, inputs, labels):
            seconds, (_, forwards) = X.timed(lambda: X.explain(split, x, k, method), device)
            rows.append({"device": note, "method": method, "image": path.name, "label_index": k,
                         "seconds": seconds, "forwards": forwards, "measured": "measured"})
            print(f"{method:10s} {path.name}: {seconds:.2f} s")
    for path, x, k in zip(paths[:1] if args.scorecam_full else paths, inputs, labels):
        if args.scorecam_full:
            seconds, (_, forwards) = X.timed(
                lambda: X.explain(split, x, k, "scorecam", None, args.scorecam_batch), device)
            measured = "measured"
        else:
            t_activations, a = X.timed(lambda: X.activations(split, x), device)
            probe = min(args.scorecam_probe, a.shape[1])
            t_probe, (_, probe_forwards) = X.timed(
                lambda: X.scorecam_channel_scores(split, x, a[:, :probe], args.scorecam_batch), device)
            flat = a[0].flatten(1)
            valid = int(((flat.max(dim=1).values - flat.min(dim=1).values) > 0).sum())
            seconds = t_activations + valid * t_probe / max(probe_forwards, 1)
            forwards = 1 + valid
            measured = f"extrapolated from {probe_forwards} masked forwards"
        rows.append({"device": note, "method": "scorecam", "image": path.name, "label_index": k,
                     "seconds": seconds, "forwards": forwards, "measured": measured})
        print(f"scorecam   {path.name}: {seconds:.1f} s ({measured})")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby(["method", "measured"])[["seconds", "forwards"]].median()
    print(summary.to_string())
    print("wrote", args.out)
    return frame


if __name__ == "__main__":
    main()
