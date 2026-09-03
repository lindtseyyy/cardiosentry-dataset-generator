#!/usr/bin/env python3
"""
Check everything that can fail BEFORE committing GPU hours.

    python preflight.py --data /content/prototype_1152_768

Turns "OOM at epoch 7 of the 768 dual-stream run" into a message in two minutes.
That matters more this round than last: two of the seven models hold two
backbones and a 768px activation stack on the same 16 GB card.
"""
from __future__ import annotations
import argparse, os, sys, time, traceback

import torch

import config as C
import data as D
import hybrids
import manifest


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
        print(f"  WARNING: {gb:.1f} GB. The 768px models are tuned for 16 GB; "
              f"pass --micro-batch 2 and expect the batch to take longer.")
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

    # The one check that is new this round, and the one most likely to bite:
    # pointing --data at last round's 384px copy.
    from PIL import Image
    with Image.open(df["abs_path"].iloc[0]) as im:
        w, h = im.size
    print(f"  stored image size: {w}x{h}")
    if min(w, h) < C.LOAD_SIZE:
        print(f"\n  STOP: images are {w}x{h} but config.LOAD_SIZE is {C.LOAD_SIZE}.")
        print(f"  This looks like round 1's prototype_1152_384. The 768px models")
        print(f"  would be fed upsampled 384px data and the whole resolution")
        print(f"  comparison would measure nothing.")
        print(f"  Run: python prepare_dataset.py --src /content/prototype_1152")
        return False
    ppm = (150 / 25.4) * (min(w, h) / 1311)
    print(f"  measurement precision: {ppm:.2f} px/mm vertical ({ppm*10:.1f} px per mV)")
    return True


def check_models(root, quick: bool, only: list[str] | None,
                 micro_batch: int | None = None, parallel: int = 1) -> bool:
    """Build every model and run a real forward+backward at the micro-batch the
    BATCH WILL ACTUALLY USE.

    This used to test the per-model registry defaults, which is the wrong number
    whenever run_hybrids.sh is driven with MICRO_BATCH - and MICRO_BATCH is the
    recommended way to run, because a uniform micro-batch is what keeps
    BatchNorm statistics comparable across runs. Testing 4 and then training at
    8 is how an OOM reaches you at notebook B instead of here.
    """
    print("-" * 78)
    print("MODELS  (build, one real forward+backward, peak VRAM)")
    if micro_batch:
        print(f"  micro-batch {micro_batch} for every model "
              f"(uniform - keeps BatchNorm comparable)")
    else:
        print("  per-model registry defaults - NOT uniform. If you intend to run")
        print("  with MICRO_BATCH set, pass --micro-batch here too or this test")
        print("  measures a batch size you will not use.")
    print("-" * 78)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = list(only) if only else hybrids.MODEL_NAMES
    ok = True
    total_vram = (torch.cuda.get_device_properties(0).total_memory / 1024**3
                  if device.type == "cuda" else 0.0)
    print(f"  {'model':<22}{'views':>10}{'params M':>10}{'micro':>7}"
          f"{'peak VRAM':>12}{'headroom':>10}{'fwd+bwd':>10}")
    peaks = {}
    for name in names:
        try:
            t0 = time.time()
            model, info = hybrids.build(name, pretrained=not quick)
            model.to(device).train()
            if info.get("channels_last") and device.type == "cuda":
                model = model.to(memory_format=torch.channels_last)
            mb = hybrids.micro_batch(name, micro_batch)
            views = hybrids.views(name)
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
            peaks[name] = peak
            head = total_vram - peak
            flag = "  <-- TIGHT" if total_vram and head < 2.0 else ""
            print(f"  {name:<22}{'+'.join(map(str,views)):>10}"
                  f"{info['params_m_actual']:10.1f}{mb:7d}"
                  f"{peak:11.1f}G{head:9.1f}G{time.time()-t0:9.1f}s{flag}")
            del model, x, y, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            ok = False
            oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            print(f"  {name:<22}{'OUT OF MEMORY' if oom else 'FAILED'}: "
                  f"{type(exc).__name__}")
            if oom:
                print(f"      at micro-batch {hybrids.micro_batch(name, micro_batch)} "
                      f"on {total_vram:.1f} GB.")
                print(f"      Lower MICRO_BATCH for EVERY run, not just this one -")
                print(f"      a mixed micro-batch means mixed BatchNorm statistics")
                print(f"      and the comparison stops being controlled.")
            else:
                traceback.print_exc()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if peaks and total_vram:
        worst = max(peaks, key=peaks.get)
        print(f"\n  worst case: {worst} at {peaks[worst]:.1f}G of {total_vram:.1f}G")
        if total_vram - peaks[worst] < 2.0:
            print("  under 2 GB spare - fragmentation over 20 epochs may still OOM.")
            print("  Consider one step down in MICRO_BATCH, applied to every run.")

    # PARALLEL=N puts N of these on the card AT ONCE. The number that has to fit
    # is the sum over the heaviest group, not the worst single model - and the
    # right fix for a failure here is to lower PARALLEL, never MICRO_BATCH,
    # because lowering the micro-batch changes BatchNorm statistics and
    # invalidates every run already finished at the current value.
    if parallel > 1 and peaks and total_vram:
        order = sorted(peaks, key=peaks.get, reverse=True)
        group = order[:parallel]
        need = sum(peaks[m] for m in group)
        # CUDA context, cuDNN workspaces and allocator fragmentation are per
        # PROCESS and none of them show up in max_memory_allocated.
        overhead = 0.7 * parallel
        total_need = need + overhead
        print(f"\n  PARALLEL={parallel}: heaviest concurrent group would be")
        for m in group:
            print(f"      {m:<24}{peaks[m]:6.1f}G")
        print(f"      {'+ per-process overhead':<24}{overhead:6.1f}G")
        print(f"      {'= peak together':<24}{total_need:6.1f}G of {total_vram:.1f}G")
        if total_need > total_vram - 2.0:
            print(f"  TOO TIGHT. Use PARALLEL={max(1, parallel - 1)} instead.")
            print("  Do NOT lower MICRO_BATCH to make this fit: that changes BN")
            print("  statistics and breaks comparability with the finished runs.")
            ok = False
        else:
            spare = total_vram - total_need
            print(f"  fits with {spare:.1f}G spare - concurrency is safe here.")
            print("  micro-batch is unchanged, so BN statistics and comparability")
            print("  are unaffected: the processes share SMs, nothing else.")
    return ok


def check_loader(root, only: list[str] | None) -> bool:
    """One real batch off disk. Catches multi-view collation and worker issues.

    Prefers a dual-view model so BOTH views get exercised, but only among the
    models actually being checked.
    """
    print("-" * 78)
    print("DATALOADER  (one real batch, both views)")
    print("-" * 78)
    candidates = list(only) if only else hybrids.MODEL_NAMES
    name = next((n for n in candidates if len(hybrids.views(n)) > 1), candidates[0])
    try:
        views = hybrids.views(name)
        loaders = D.build_loaders(root, hybrids.micro_batch(name), views,
                                  workers=D.resolve_workers())
        t0 = time.time()
        batch_views, y, idx = next(iter(loaders["train"]))
        el = time.time() - t0
        for s, t in sorted(batch_views.items()):
            print(f"  view {s:>4}: {tuple(t.shape)}  "
                  f"mean {t.mean():+.3f}  std {t.std():.3f}")
        print(f"  labels {tuple(y.shape)}   first batch in {el:.1f}s")
        if len(views) > 1:
            lo, hi = min(views), max(views)
            import torch.nn.functional as F
            down = F.interpolate(batch_views[hi], size=lo, mode="bicubic",
                                 align_corners=False, antialias=True)
            mad = (down - batch_views[lo]).abs().mean().item()
            print(f"  view agreement: mean |resize(hi) - lo| = {mad:.4f}")
            print(f"  (both views come from ONE decode of ONE file, so this should")
            print(f"   be small - it is only nonzero because PIL bicubic and torch")
            print(f"   bicubic differ slightly. A large value means the views are")
            print(f"   not the same sheet.)")
            if mad > 0.25:
                print("  WARNING: views disagree more than expected.")
        return True
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", nargs="+", default=None,
                    help="check only these models. run_hybrids.sh passes exactly "
                         "the ones it is about to train, so a model you are not "
                         "training cannot block the batch.")
    ap.add_argument("--quick", action="store_true",
                    help="skip downloading pretrained weights (structure only)")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="test every model at THIS micro-batch, matching how "
                         "run_hybrids.sh will train. Defaults to $MICRO_BATCH.")
    ap.add_argument("--parallel", type=int, default=1,
                    help="how many models run_hybrids.sh will train AT ONCE. "
                         "Checks that the heaviest concurrent group fits in "
                         "VRAM before six hours are committed to finding out.")
    a = ap.parse_args()

    print("=" * 78)
    print("CardioSentry hybrids - preflight")
    print(f"protocol fingerprint {C.protocol_fingerprint()}")
    print(f"views available      base {C.BASE_SIZE}  hi {C.HI_SIZE}  "
          f"stored {C.LOAD_SIZE}")
    print("=" * 78)

    results = {"gpu": check_gpu(), "data": check_data(a.data)}
    if not results["data"]:
        print("\nPREFLIGHT FAILED on the dataset - fix that before anything else.")
        return 1
    results["loader"] = check_loader(a.data, a.model)
    mb = a.micro_batch or (int(os.environ["MICRO_BATCH"])
                           if os.environ.get("MICRO_BATCH") else None)
    if not a.skip_models:
        results["models"] = check_models(a.data, a.quick, a.model, mb, a.parallel)

    print("=" * 78)
    bad = [k for k, v in results.items() if not v]
    for k, v in results.items():
        print(f"  {k:<10}{'OK' if v else 'FAILED'}")
    print("=" * 78)
    if bad:
        print(f"PREFLIGHT FAILED: {', '.join(bad)}")
        return 1
    print("all clear - safe to start the batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
