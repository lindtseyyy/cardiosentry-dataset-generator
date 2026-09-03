#!/usr/bin/env python3
"""
Check everything that can fail BEFORE committing the GPU hour.

    python preflight.py --data /content/prototype_1152_768

One model, one run folder - but the checks are the same ones that protected
the four-head round: GPU, dataset integrity, the 768px stored-size check, one
real loader batch, and a real forward+backward at the micro-batch the run will
actually use, reporting peak VRAM.
"""
from __future__ import annotations
import argparse, os, time, traceback

import torch

import config as C
import data as D
import manifest
import models


def check_gpu() -> bool:
    print("-" * 78)
    print("GPU")
    print("-" * 78)
    if not torch.cuda.is_available():
        print("  NO GPU. Runtime > Change runtime type > GPU (T4 minimum).")
        return False
    pr = torch.cuda.get_device_properties(0)
    gb = pr.total_memory / 1024**3
    print(f"  {pr.name}   {gb:.1f} GB   torch {torch.__version__}   "
          f"cuda {torch.version.cuda}")
    if gb < 14:
        print(f"  WARNING: {gb:.1f} GB. Micro-batch 8 is tuned for 16 GB; "
              f"lowering it would change BatchNorm statistics and break "
              f"comparability with the reference - prefer a bigger runtime.")
    return True


def check_data(root) -> bool:
    print("-" * 78)
    print("DATASET")
    print("-" * 78)
    try:
        df = manifest.load(root)
        rep = manifest.check_integrity(df, verify_files=False)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False
    print(manifest.describe(df))
    print(f"  {rep['rows']:,} rows | {rep['records']:,} records | "
          f"{rep['patients']:,} patients")

    # The check most likely to bite: pointing --data at the 384px round-1 copy.
    from PIL import Image
    with Image.open(df["abs_path"].iloc[0]) as im:
        w, h = im.size
    print(f"  stored image size: {w}x{h}")
    if min(w, h) < C.LOAD_SIZE:
        print(f"\n  STOP: images are {w}x{h} but the run needs {C.LOAD_SIZE}.")
        print(f"  This looks like round 1's 384px copy. The binary model must see")
        print(f"  the SAME bytes the four-head reference saw - re-extract")
        print(f"  prototype_1152_768.tar (built by the hybrids round's 0_Prepare).")
        return False
    ppm = (150 / 25.4) * (min(w, h) / 1311)
    print(f"  measurement precision: {ppm:.2f} px/mm vertical ({ppm*10:.1f} px per mV)")
    return True


def check_loader(root) -> bool:
    """One real batch off disk. Catches collation and worker issues."""
    print("-" * 78)
    print("DATALOADER  (one real batch)")
    print("-" * 78)
    try:
        views = models.views("resnet50_lvh_binary")
        loaders = D.build_loaders(root, models.micro_batch("resnet50_lvh_binary"),
                                  views, workers=D.resolve_workers())
        t0 = time.time()
        batch_views, y, idx = next(iter(loaders["train"]))
        el = time.time() - t0
        for s, t in sorted(batch_views.items()):
            print(f"  view {s:>4}: {tuple(t.shape)}  "
                  f"mean {t.mean():+.3f}  std {t.std():.3f}")
        print(f"  labels {tuple(y.shape)}   first batch in {el:.1f}s")
        return True
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def check_model(micro_batch: int | None = None, quick: bool = False) -> bool:
    """Build the binary model and run a real forward+backward at the micro-batch
    the run will actually use, reporting peak VRAM."""
    print("-" * 78)
    print("MODEL  (build, one real forward+backward, peak VRAM)")
    print("-" * 78)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vram = (torch.cuda.get_device_properties(0).total_memory / 1024**3
                  if device.type == "cuda" else 0.0)
    try:
        t0 = time.time()
        model, info = models.build("resnet50_lvh_binary", pretrained=not quick)
        model.to(device).train()
        if info.get("channels_last") and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        mb = models.micro_batch("resnet50_lvh_binary", micro_batch)
        views = models.views("resnet50_lvh_binary")
        x = {s: torch.randn(mb, 3, s, s, device=device) for s in views}
        if info.get("channels_last") and device.type == "cuda":
            x = {k: v.contiguous(memory_format=torch.channels_last)
                 for k, v in x.items()}
        y = torch.zeros(mb, C.NUM_CLASSES, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=C.AMP and device.type == "cuda"):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        peak = (torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda" else 0.0)
        print(f"  views {'+'.join(map(str, views))}   params "
              f"{info['params_m_actual']:.1f} M   micro-batch {mb}"
              f"   peak VRAM {peak:.1f}G of {total_vram:.1f}G"
              f"   head width {info['num_classes']}   ({time.time()-t0:.1f}s)")
        if total_vram and total_vram - peak < 2.0:
            print("  under 2 GB spare - consider a bigger runtime. Do NOT lower")
            print("  the micro-batch: it changes BatchNorm statistics and breaks")
            print("  comparability with the reference run (micro 8).")
            return False
        return True
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="test at THIS micro-batch, matching how train.py will "
                         "run. Defaults to $MICRO_BATCH, then the registry (8).")
    ap.add_argument("--quick", action="store_true",
                    help="skip downloading pretrained weights (structure only)")
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print("LVH binary - negative-transfer test - preflight")
    print(f"protocol fingerprint {C.protocol_fingerprint()}")
    print(f"shared   fingerprint {C.shared_fingerprint()}")
    print(f"task     fingerprint {C.task_fingerprint()}")
    print("=" * 78)

    results = {"gpu": check_gpu(), "data": check_data(a.data)}
    if not results["data"]:
        print("\nPREFLIGHT FAILED on the dataset - fix that before anything else.")
        return 1
    results["loader"] = check_loader(a.data)
    if not a.skip_model:
        mb = a.micro_batch or (int(os.environ["MICRO_BATCH"])
                               if os.environ.get("MICRO_BATCH") else None)
        results["model"] = check_model(mb, a.quick)

    print("=" * 78)
    bad = [k for k, v in results.items() if not v]
    for k, v in results.items():
        print(f"  {k:<10}{'OK' if v else 'FAILED'}")
    print("=" * 78)
    if bad:
        print(f"PREFLIGHT FAILED: {', '.join(bad)}")
        return 1
    print("all clear - safe to start the run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
