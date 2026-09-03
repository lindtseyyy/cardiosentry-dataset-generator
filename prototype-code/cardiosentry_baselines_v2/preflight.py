#!/usr/bin/env python3
"""
Check everything BEFORE committing hours of GPU time.

    python preflight.py --data /content/prototype_1152

Verifies, in the order things actually go wrong:
  0. the input geometry is sane and the header crop still fits this corpus
  1. GPU present, VRAM known
  2. dataset extracts, loads, and passes every integrity check
  3. all nine timm tags resolve and download          <- the common failure
  4. each model survives a real forward+backward at the configured micro-batch
     AND THE CONFIGURED (H, W), and reports peak VRAM so you can fix OOM here
     rather than at epoch 7

Nine models at a rectangular input is a bigger surface than v1's five squares:
ViT has to resample its position embeddings and Swin-V2 has to pad, and both of
those fail at construction time rather than at epoch 7 - but only if you look.

    CS_DIM=sq768 python preflight.py --data <DATA>     # check another geometry

Takes a couple of minutes. Worth it every time.
"""
from __future__ import annotations
import argparse, sys, time, traceback

import torch

import config as C
import dimensions as DIM
import models


def hr(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t)
        print("=" * 74)


def check_geometry() -> bool:
    """Everything about the input that can be known without a GPU or the data."""
    hr("0. input geometry")
    g = DIM.geometry(C.DIM_NAME)
    print(f"  CS_DIM={C.DIM_NAME}   {g['H']}x{g['W']}   "
          f"{'header cropped' if g['crop'] else 'full sheet'}")
    print(f"  {DIM.DIMENSIONS[C.DIM_NAME]['note']}")
    print()
    print(f"  aspect            {g['aspect']:.3f}  vs source {g['source_aspect']:.3f}"
          f"   -> distortion {g['aspect_distortion']:.3f}")
    print(f"  physical scale    {g['px_per_mm_x']:.3f} px/mm TIME, "
          f"{g['px_per_mm_y']:.3f} px/mm VOLTAGE")
    print(f"  1 mm ST step      {g['px_per_1mm_st']:.2f} px"
          + ("   <- under 3 px: the STEMI decision is 2-3 pixels tall"
             if g["px_per_1mm_st"] < 3.0 else ""))
    print(f"  100 ms QRS        {g['px_per_qrs']:.1f} px"
          + ("   <- under 8 px: normal vs wide QRS is a marginal call"
             if g["px_per_qrs"] < 8.0 else ""))
    print(f"  time Nyquist      {g['time_nyquist_hz']:.1f} Hz"
          + ("   <- BELOW the 40 Hz signal band: the resize is lossy in time"
             if g["time_nyquist_hz"] < 40.0 else "   (clears the 40 Hz band)"))
    print(f"  pixels            {g['pixels']/1e6:.3f} M  "
          f"({g['px_rel_sq768']:.2f}x the 768x768 v1_768 runs)")

    ok = True
    for h, ax in ((g["H"], "H"), (g["W"], "W")):
        if h % 32:
            print(f"  FAIL: {ax}={h} is not a multiple of 32. Some CNN stage will")
            print(f"        round it down and silently drop a row/column.")
            ok = False
        if h % 16:
            print(f"  FAIL: {ax}={h} is not a multiple of 16 - ViT-B/16 cannot tile it.")
            ok = False
    print()
    if C.REDACT_HEADER:
        print(f"  ID REDACTION ON: a fixed rectangle {DIM.REDACT_BOX} covers the")
        print("  printed ID / Age / Sex block. Without it the record ID separates")
        print("  STEMI EXACTLY in this corpus (<=ECG007796 vs >=ECG007797) and any")
        print("  STEMI score carries an unmeasured shortcut. See dimensions.py.")
    elif g["crop"]:
        print("  header CROPPED: the printed ID is outside the image entirely.")
        print("  NOTE: crop_audit.py measured that a full-width cut also clips")
        print("  waveform on ~7 extreme LVH/STEMI sheets. Redaction on a full-sheet")
        print("  geometry avoids that; this geometry is kept for the ablation.")
    else:
        print("  *** NEITHER CROP NOR REDACTION IS ACTIVE. The printed record ID is")
        print("      in the image and it is a perfect STEMI label. Do not report a")
        print("      STEMI score from this configuration without saying so.")
    if DIM.EXCLUDED_RECORDS:
        print()
        print(f"  excluding {len(DIM.EXCLUDED_RECORDS)} corrupt record(s): "
              f"{', '.join(DIM.EXCLUDED_RECORDS)}")
    return ok


def check_gpu() -> bool:
    hr("1. compute")
    if not torch.cuda.is_available():
        print("  NO GPU. In Colab: Runtime > Change runtime type > T4 GPU (or better).")
        print(f"  Training on CPU is not viable at {C.IMG_H}x{C.IMG_W}.")
        return False
    pr = torch.cuda.get_device_properties(0)
    vram = pr.total_memory / 1024**3
    print(f"  GPU:   {pr.name}")
    print(f"  VRAM:  {vram:.1f} GB")
    print(f"  torch: {torch.__version__}  cuda {torch.version.cuda}")
    if vram < 14:
        print("  WARNING: under 14 GB. Halve --micro-batch if you hit OOM;")
        print("           gradient accumulation keeps the effective batch constant.")
    return True


def check_data(root: str) -> bool:
    hr("2. dataset")
    try:
        import manifest
        df = manifest.load(root)
        rep = manifest.check_integrity(df, verify_files=False)
        print(manifest.describe(df))
        print("\n  integrity checks: PASSED")
        print("    no patient spans splits, no record spans splits,")
        print("    val/test are exactly 1 image per ECG, NORMAL never co-occurs")

        import random
        from PIL import Image
        sample = df.sample(min(20, len(df)), random_state=0)
        t0 = time.time()
        sizes = set()
        for p in sample["abs_path"]:
            with Image.open(p) as im:
                sizes.add(im.size)
        print(f"  read 20 images in {time.time()-t0:.1f}s   sizes: {sizes}")

        # Guard against training at a resolution the files cannot support. With a
        # crop geometry the comparison is against the CROPPED height, not the
        # stored one, or a pre-cropped copy looks like it is too small.
        for (w, h) in sorted(sizes):
            eff_h = h * (DIM.CONTENT_CROP[1] - DIM.CONTENT_CROP[0]) if C.CROP_HEADER else h
            src_px, want_px = w * eff_h, C.IMG_H * C.IMG_W
            if w < C.IMG_W or eff_h < C.IMG_H:
                print(f"  WARNING: stored images are {w}x{h}"
                      f"{f' ({w}x{eff_h:.0f} after the crop)' if C.CROP_HEADER else ''}"
                      f" but this geometry wants {C.IMG_H}x{C.IMG_W}.")
                print(f"           They would be UPSCALED - detail you never had.")
                print(f"           Regenerate with: CS_DIM={C.DIM_NAME} python "
                      f"resize_dataset.py --src <original>")
            elif abs(src_px - want_px) / want_px < 0.02:
                print(f"  images are already at the target geometry - pre-resized, "
                      f"no decode overhead. Good.")
            else:
                print(f"  stored {w}x{h}; every epoch decodes ~{src_px/want_px:.1f}x "
                      f"more pixels than it uses.")
                print(f"           If data-wait is high, run: CS_DIM={C.DIM_NAME} "
                      f"python resize_dataset.py --src <original>")
        if time.time() - t0 > 6:
            print("  WARNING: image reads are slow. Are you reading straight off a")
            print("           mounted Drive? Extract the tar to /content/ first.")
        return True
    except Exception:
        print("  FAILED:")
        traceback.print_exc()
        return False


def check_models(only: str | None, quick: bool) -> bool:
    hr("3. model construction + 4. forward/backward")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [only] if only else models.MODEL_NAMES
    all_ok = True
    suggestions: list[tuple] = []
    vram = (torch.cuda.get_device_properties(0).total_memory / 1024**3
            if torch.cuda.is_available() else 0.0)

    print(f"{'model':18}{'params M':>10}{'micro':>7}{'accum':>7}{'fed HxW':>19}"
          f"{'peak VRAM':>11}{'fwd+bwd':>9}   pretrained tag")
    print("-" * 119)

    for name in names:
        try:
            model, info = models.build(name, pretrained=True)
            mb = models.micro_batch(name)
            accum = models.accumulation_steps(name)
            fed = info["input_geometry"]["fed"]
            over = info["input_geometry"]["pixel_overhead"]
            fed_s = f"{fed[0]}x{fed[1]}" + (f" ({over:.2f}x)" if over > 1.001 else "")
            line = (f"{name:18}{info['params_m_actual']:10.1f}{mb:7d}{accum:7d}"
                    f"{fed_s:>19}")

            if quick or dev.type == "cpu":
                print(line + f"{'-':>11}{'-':>9}   {info['resolved_tag']}")
                del model
                continue

            model.to(dev).train()
            x = torch.randn(mb, 3, C.IMG_H, C.IMG_W, device=dev)
            y = torch.randint(0, 2, (mb, C.NUM_CLASSES), device=dev).float()
            crit = torch.nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler("cuda", enabled=C.AMP)
            # Real optimizer: AdamW allocates two state tensors per parameter on
            # its first step (8 bytes/param). A bare forward+backward never shows
            # that, so measuring without it under-reports peak VRAM and leads to
            # a micro_batch suggestion that OOMs in training.
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

            def _step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=C.AMP):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            _step()                       # warmup: cudnn autotune + optimizer state
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            _step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3

            print(line + f"{peak:10.1f}G{dt:9.2f}s   {info['resolved_tag']}")
            suggestions.append((name, mb, peak, vram, info["params_m_actual"]))
            del model, x, y, opt
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            all_ok = False
            print(f"{name:18}  OOM at micro_batch={models.micro_batch(name)} - "
                  f"lower it in models.REGISTRY (effective batch is preserved)")
            torch.cuda.empty_cache()
        except Exception as exc:
            all_ok = False
            print(f"{name:18}  FAILED: {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()

    if suggestions:
        _suggest_batches(suggestions)
    return all_ok


def _suggest_batches(rows) -> None:
    """Recommend the largest micro-batch that still fits, keeping the EFFECTIVE
    batch fixed.

    Raising micro_batch is a pure speedup: accumulation steps fall to compensate,
    so config.EFFECTIVE_BATCH - and therefore the optimisation trajectory - is
    unchanged, and the comparison stays controlled. Raising EFFECTIVE_BATCH
    itself would NOT be safe; that changes optimisation and must be done for
    every model, with a new fingerprint.
    """
    # micro_batch must divide EFFECTIVE_BATCH exactly, or models would end up
    # with different effective batches.
    divisors = [d for d in range(1, C.EFFECTIVE_BATCH + 1)
                if C.EFFECTIVE_BATCH % d == 0]
    print()
    print("VRAM headroom - suggested micro_batch (effective batch stays "
          f"{C.EFFECTIVE_BATCH})")
    print(f"{'model':18}{'now':>6}{'peak':>9}{'fixed':>8}{'per-sample':>12}"
          f"{'suggest':>9}{'est peak':>10}")
    print("-" * 72)
    changed = []
    for name, mb, peak, vram, params_m in rows:
        # Fixed cost is derived, not guessed: weights(4) + grads(4) + AdamW
        # state(8) = 16 bytes per parameter, independent of batch size.
        fixed = params_m * 1e6 * 16 / 1024**3
        per_sample = max((peak - fixed) / mb, 1e-6)   # activations scale with batch
        budget = vram * 0.80            # headroom for fragmentation + eval pass
        best = mb
        for d in divisors:
            if fixed + per_sample * d <= budget:
                best = max(best, d)
        est = fixed + per_sample * best
        flag = "  <-- raise" if best > mb else ""
        print(f"{name:18}{mb:6d}{peak:8.1f}G{fixed:7.1f}G{per_sample:11.2f}G"
              f"{best:9d}{est:9.1f}G{flag}")
        if best > mb:
            changed.append((name, best))
    if changed:
        print()
        print("  To apply, edit models.REGISTRY:")
        for name, best in changed:
            print(f'      REGISTRY["{name}"]["micro_batch"] = {best}')
        print("  Then re-run preflight to confirm it fits. This is a pure speedup:")
        print("  fewer accumulation steps, identical effective batch, same results.")
    else:
        print("\n  Current micro-batches already use the available VRAM.")


def list_tags(name: str) -> None:
    import timm
    stem = models.REGISTRY[name]["timm"].split(".")[0]
    print(f"\npretrained tags available for '{stem}' in timm {timm.__version__}:\n")
    for t in timm.list_pretrained(f"{stem}*"):
        print("   ", t)
    print("\nset one via models.REGISTRY or --timm-tag. Prefer an in1k tag so the")
    print("pretraining corpus matches the other backbones.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="extracted prototype folder")
    ap.add_argument("--model", choices=models.MODEL_NAMES, help="check just one")
    ap.add_argument("--quick", action="store_true", help="skip forward/backward")
    ap.add_argument("--list-tags", metavar="MODEL", choices=models.MODEL_NAMES)
    a = ap.parse_args()

    if a.list_tags:
        list_tags(a.list_tags)
        return 0

    print("=" * 74)
    print("CardioSentry baselines v2 - preflight")
    print(f"geometry:  {C.geometry_str()}")
    print(f"fingerprint: {C.fingerprint()}   protocol: {C.protocol_fingerprint()}")
    print(f"effective batch: {C.EFFECTIVE_BATCH}   epochs: {C.EPOCHS}   "
          f"models: {len(models.MODEL_NAMES)}")
    print("=" * 74)

    ok = check_geometry()
    ok &= check_gpu()
    if a.data:
        ok &= check_data(a.data)
    else:
        print("\n(no --data given; skipping dataset checks)")
    ok &= check_models(a.model, a.quick)

    hr()
    if ok:
        print("PREFLIGHT PASSED.\n")
        print("Next, prove the training path end to end in ~2 minutes:")
        print(f"  CS_DIM={C.DIM_NAME} python train.py --model resnet50 "
              f"--data {a.data or '<DATA>'} --smoke")
    else:
        print("PREFLIGHT FAILED - fix the above before spending GPU hours.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
