#!/usr/bin/env python3
"""
Assert the things that would otherwise be wrong SILENTLY.

    python selftest.py                 # no GPU, no dataset, no weights needed
    python selftest.py --data <DATA>    # also check against the real corpus

None of these checks is about accuracy. Every one of them is about a bug whose
symptom is a model that trains normally, converges normally, and reports a number
that means something other than what you think it means:

  * (H, W) vs (W, H). torchvision.Resize takes (h, w); PIL takes (w, h). Swap
    them and you train on a portrait ECG. Loss goes down. Nothing warns you.
  * cropping twice. resize_dataset bakes the crop in; if the runtime crop also
    fires, the second cut lands inside the first trace row.
  * the crop window drifting out of the gap between the header and the traces as
    the corpus changes.
  * a geometry whose H or W is not a multiple of 32, so a CNN stage silently
    truncates.
  * micro_batch not dividing EFFECTIVE_BATCH, so two models train at different
    effective batches and the "controlled" comparison is not.
  * the two fingerprints failing to separate protocol from geometry, which is
    what lets compare.py tell an architecture ranking from a resolution study.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import config as C
import dimensions as DIM
import hybrids as H
import models

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}" + (f"  -- {detail}" if detail else ""))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def geometry_checks() -> None:
    print(f"\ngeometry ({C.DIM_NAME})")
    for nm in DIM.DIM_NAMES:
        g = DIM.geometry(nm)
        check(f"{nm}: H,W multiples of 32", g["H"] % 32 == 0 and g["W"] % 32 == 0,
              f"{g['H']}x{g['W']}")
    g = DIM.geometry(C.DIM_NAME)
    check("config.IMG_SIZE is (H, W)", tuple(C.IMG_SIZE) == (g["H"], g["W"]),
          f"{C.IMG_SIZE}")
    # y1 == 1.0 is CORRECT and deliberate: crop_audit.py found the rhythm strip
    # can reach y=1304 of 1311 after rotation, so no bottom cut is safe. The
    # footer caption is 40 px of constant text and costs nothing to keep.
    check("crop window is inside the sheet",
          0.0 < DIM.CONTENT_CROP[0] < DIM.CONTENT_CROP[1] <= 1.0,
          f"{DIM.CONTENT_CROP}")


def transform_checks() -> None:
    """The (H, W) vs (W, H) trap, exercised on a real transform pipeline.

    A non-square PROBE is essential: at 768x768 a transposed resize is invisible,
    which is exactly why v1_768 could not have caught this.
    """
    print("\ntransforms")
    import data as D

    probe = Image.new("RGB", (DIM.SOURCE_W, DIM.SOURCE_H), (255, 255, 255))

    for crop in (False, True):
        for red in (False, True):
            t = D.build_transforms(train=False, apply_crop=crop, apply_redact=red)
            out = t(probe)
            check(f"resize -> (3, {C.IMG_H}, {C.IMG_W})  crop={crop} redact={red}",
                  tuple(out.shape) == (3, C.IMG_H, C.IMG_W), f"got {tuple(out.shape)}")

    # Redaction must never run on top of a crop: the crop already discarded the
    # band the box sits in, so redacting there would paint over waveform.
    names_cr = [type(x).__name__
                for x in D.build_transforms(False, apply_crop=True,
                                            apply_redact=True).transforms]
    check("redaction is skipped when the header is cropped",
          "RedactHeader" not in names_cr, " ".join(names_cr))

    # ...and when it does run, it must run AFTER the resize (cheaper, and it
    # avoids bicubic smearing the rectangle's edges).
    names_r = [type(x).__name__
               for x in D.build_transforms(False, apply_crop=False,
                                           apply_redact=True).transforms]
    check("redaction runs after the resize",
          "RedactHeader" in names_r
          and names_r.index("RedactHeader") > names_r.index("Resize"),
          " ".join(names_r))

    # The box must actually erase the text: no dark ink left inside it.
    box = DIM.redact_box(DIM.SOURCE_W, DIM.SOURCE_H)
    check("redact box is inside the sheet and non-empty",
          0 <= box[0] < box[2] <= DIM.SOURCE_W and 0 <= box[1] < box[3] <= DIM.SOURCE_H,
          str(box))

    # crop must remove height from the TOP and keep full width
    box = DIM.crop_box(DIM.SOURCE_W, DIM.SOURCE_H)
    check("crop keeps full width", box[0] == 0 and box[2] == DIM.SOURCE_W, str(box))
    check("crop removes a band from the top only",
          box[1] > 0 and box[3] == DIM.SOURCE_H, str(box)
          + "  (bottom kept: no bottom cut clears the rhythm strip)")

    # train transform must be photometric-only: same shape, and no geometry op
    names = [type(s).__name__ for s in D.build_transforms(train=True).transforms]
    forbidden = {"RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
                 "RandomRotation", "RandomAffine", "RandomPerspective"}
    check("no forbidden geometric augmentation", not (set(names) & forbidden),
          " ".join(names))


def redaction_effect_check(data: Path) -> None:
    """Prove on a REAL sheet that the box leaves no legible ink behind.

    A ghost that a human can read is a ghost a CNN can read, and the first
    attempt at this (erasing only the ink) failed exactly that way. The test is
    therefore not "did we draw a rectangle" but "is the ink gone".
    """
    print("\nredaction effect")
    import pandas as pd
    from PIL import Image
    import data as D

    df = pd.read_csv(data / "index.csv")
    rng = np.random.default_rng(0)
    sel = df.iloc[rng.choice(len(df), min(25, len(df)), replace=False)]
    worst = 0.0
    for _, r in sel.iterrows():
        with Image.open(data / r["rel_path"]) as im:
            im = im.convert("RGB").resize((C.IMG_W, C.IMG_H), Image.BICUBIC)
        box = DIM.redact_box(im.width, im.height)
        before = np.asarray(im.crop(box).convert("L"), dtype=np.float32)
        after = np.asarray(D.RedactHeader()(im.copy()).crop(box).convert("L"),
                           dtype=np.float32)
        paper = np.percentile(before, 75)
        worst = max(worst, float((after < 0.60 * paper).mean()))
    check("no ink survives inside the redaction box",
          worst < 1e-6, f"worst residual dark fraction {worst:.6f} over 25 sheets")


def batching_checks() -> None:
    print("\nbatching")
    for m in models.MODEL_NAMES:
        mb = models.micro_batch(m)
        ok = C.EFFECTIVE_BATCH % mb == 0
        check(f"{m}: micro {mb} divides effective {C.EFFECTIVE_BATCH}", ok,
              f"accum {C.EFFECTIVE_BATCH // mb}" if ok else "would change the "
              "effective batch")


def fingerprint_checks() -> None:
    """The protocol/geometry split has to actually split.

    Re-imports config under a different CS_DIM in a subprocess, because config
    reads the environment once at import and there is no honest way to fake that
    in-process.
    """
    print("\nfingerprints")
    import json, os, subprocess
    other = "sq768" if C.DIM_NAME != "sq768" else "sq384"
    env = dict(os.environ, CS_DIM=other)
    code = ("import json, config as C; "
            "print(json.dumps([C.fingerprint(), C.protocol_fingerprint()]))")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, cwd=Path(__file__).parent)
    if out.returncode != 0:
        check(f"config imports under CS_DIM={other}", False, out.stderr.strip()[-200:])
        return
    fp2, pf2 = json.loads(out.stdout.strip().splitlines()[-1])
    check("geometry changes the config fingerprint", C.fingerprint() != fp2,
          f"{C.DIM_NAME}={C.fingerprint()}  {other}={fp2}")
    check("geometry does NOT change the protocol fingerprint",
          C.protocol_fingerprint() == pf2, f"{C.protocol_fingerprint()}")


def registry_checks() -> None:
    print("\nregistry")
    expected = len(models.BACKBONES) * len(models.LADDER)
    check(f"{expected} hybrids registered", len(models.MODEL_NAMES) == expected,
          f"{len(models.BACKBONES)} backbones x {len(models.LADDER)} rungs")
    check(f"the default batch is the complete {expected}-cell grid",
          len(models.RUN_ORDER) == expected and not models.RUN_ORDER_OPTIONAL,
          f"{len(models.RUN_ORDER)} in RUN_ORDER, "
          f"{len(models.RUN_ORDER_OPTIONAL)} held back "
          f"({', '.join(models.RUN_ORDER_OPTIONAL) or 'none'})")
    # Every backbone must carry every rung, or a cross-backbone claim about one
    # component is really a claim about whichever backbones happened to have it.
    for bb in models.BACKBONES:
        rungs = [n for n in models.RUN_ORDER
                 if models.REGISTRY[n]["backbone"] == bb]
        check(f"{bb}: all {len(models.LADDER)} rungs are in the default batch",
              len(rungs) == len(models.LADDER),
              f"{len(rungs)} rungs")
    check("every registered model appears in RUN_ORDER or RUN_ORDER_OPTIONAL",
          set(models.RUN_ORDER) | set(models.RUN_ORDER_OPTIONAL)
          == set(models.MODEL_NAMES))
    check("RUN_ORDER groups by backbone (a cut-short night leaves whole families)",
          [models.REGISTRY[n]["backbone"] for n in models.RUN_ORDER]
          == sorted([models.REGISTRY[n]["backbone"] for n in models.RUN_ORDER],
                    key=[models.REGISTRY[n]["backbone"]
                         for n in models.RUN_ORDER].index),
          " -> ".join(dict.fromkeys(models.REGISTRY[n]["backbone"]
                                    for n in models.RUN_ORDER)))
    # A fusion rule needs two streams; a single stream must not claim one. A
    # registry typo here builds a model that trains fine and is not the model
    # the run is named after.
    for n, spec in models.REGISTRY.items():
        streams = int(spec["use_cnn"]) + int(spec["use_lt"])
        ok = (streams == 2) if spec["fusion"] != "none" else (streams == 1)
        check(f"{n}: {streams} stream(s) matches fusion '{spec['fusion']}'", ok)
    for m in models.MODEL_NAMES:
        g = models.input_geometry(m)
        check(f"{m}: fed at the configured size", g["fed"] == tuple(C.IMG_SIZE),
              f"{g['fed'][0]}x{g['fed'][1]}")


def hybrid_checks() -> None:
    """The claims hybrids.py makes about itself, tested rather than asserted.

    All of these are silent-wrongness bugs. An attention module that is not the
    identity at init still trains and still reports a number - a worse one, for a
    reason nothing in the logs would reveal. A pos-embed that does not resample
    still runs and scrambles which token is which. A stride that is not 32 gives
    the transformer a different token grid than its position embedding was built
    for, and it converges anyway.
    """
    import torch
    print("\nhybrid components")
    torch.set_grad_enabled(False)
    torch.manual_seed(C.SEED)

    x = torch.randn(2, 64, 12, 16)
    for kind, cls in H.ATTENTION.items():
        mod = cls(64).eval()
        check(f"{kind}: gamma is initialised to exactly 0",
              float(mod.gamma) == 0.0, f"gamma={float(mod.gamma)}")
        check(f"{kind}: is EXACTLY the identity at step 0",
              torch.equal(mod(x), x),
              "so the pretrained features arrive intact - see hybrids.py")
        with torch.enable_grad():
            mod.train()
            mod(x.requires_grad_(False)).sum().backward()
        check(f"{kind}: gamma receives gradient (it can turn itself on)",
              mod.gamma.grad is not None and float(mod.gamma.grad) != 0.0,
              f"grad={float(mod.gamma.grad) if mod.gamma.grad is not None else None:.4g}")

    check("ECA kernel follows the paper's rule for all three widths",
          (H.eca_kernel_size(512), H.eca_kernel_size(1280),
           H.eca_kernel_size(2048)) == (5, 5, 7),
          f"512->{H.eca_kernel_size(512)}  1280->{H.eca_kernel_size(1280)}  "
          f"2048->{H.eca_kernel_size(2048)}")

    # The transformer's position embedding is 2-D and must survive a grid it was
    # not built for, or this package silently breaks at any other geometry.
    lt = H.LightweightTransformer(64, dim=32, depth=1, heads=4, grid=(6, 8)).eval()
    check("LT pooled output is (B, dim)", tuple(lt(torch.randn(2, 64, 6, 8)).shape)
          == (2, 32))
    try:
        y = lt(torch.randn(2, 64, 9, 11))
        check("LT resamples its position embedding to an unseen grid",
              tuple(y.shape) == (2, 32), "bicubic, 2-D - a flat table would scramble")
    except Exception as exc:
        check("LT resamples its position embedding to an unseen grid", False,
              f"{type(exc).__name__}: {exc}")

    # Per-stream LayerNorm is what makes ablate.py valid: zeroing one stream must
    # not disturb the other stream's contribution to the fused vector.
    from collections import OrderedDict
    dims = OrderedDict([("cnn", 8), ("lt", 6)])
    feats = OrderedDict([("cnn", torch.randn(3, 8)), ("lt", torch.randn(3, 6))])
    cat = H.ConcatFusion(dims).eval()
    full, abl = cat(feats), cat(feats, ablate="lt")
    check("concat fusion: ablating one stream leaves the other untouched",
          torch.equal(full[:, :8], abl[:, :8]) and torch.count_nonzero(abl[:, 8:]) == 0,
          "per-stream LayerNorm, normalised BEFORE the concat")

    gated = H.GatedFusion(dims, gate_dim=8).eval()
    g_full, g_abl = gated(feats), gated(feats, ablate="cnn")
    check("gated fusion: ablation changes the output", not torch.equal(g_full, g_abl))
    check("gated fusion records a gate mean on eval",
          0.0 < float(gated.gate_mean) < 1.0, f"{float(gated.gate_mean):.4f}")
    gated.reset_gate_stats()
    check("gated fusion gate stats can be reset before the test pass",
          float(gated.gate_mean) == 0.0 and float(gated.gate_seen) == 0.0)


def nesting_checks(build: bool) -> None:
    """The claim the whole package rests on: the baseline is NESTED in the hybrid.

    If `HybridModel` with no components is not parameter-for-parameter the model
    baselines_v2 trained, then every delta in compare.py is measuring the harness
    as well as the component, and nobody would be able to tell from the numbers.
    """
    import torch
    print("\nbaseline nesting  (why the baselines_v2 numbers are comparable)")
    if not build:
        print("  (needs to construct backbones; pass --build to run it)")
        return
    import timm
    torch.set_grad_enabled(False)
    for bname, b in models.BACKBONES.items():
        try:
            ref = timm.create_model(b["timm"], pretrained=False,
                                    num_classes=C.NUM_CLASSES)
            n_ref = sum(p.numel() for p in ref.parameters())
            bare = timm.create_model(b["timm"], pretrained=False, num_classes=0)
            hyb = H.HybridModel(bare, feat_ch=bare.num_features,
                                pooled_dim=getattr(bare, "head_hidden_size", None)
                                or bare.num_features,
                                attn=None, use_cnn=True, use_lt=False,
                                fusion="none")
            n_hyb = sum(p.numel() for p in hyb.parameters())
            check(f"{bname}: componentless hybrid == the baselines_v2 model",
                  n_ref == n_hyb, f"{n_ref:,} vs {n_hyb:,} parameters")
            check(f"{bname}: the head is a single Linear, as in baselines_v2",
                  isinstance(hyb.classifier, torch.nn.Linear)
                  and isinstance(ref.get_classifier(), torch.nn.Linear)
                  and hyb.classifier.in_features == ref.get_classifier().in_features,
                  f"Linear({hyb.classifier.in_features}, {C.NUM_CLASSES})")
            # Recorded BN count vs reality. It gates whether micro_batch may be
            # changed, so a stale number here would licence exactly the silent
            # confound the gate exists to prevent.
            n_bn = sum(1 for m in bare.modules()
                       if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
            check(f"{bname}: recorded BatchNorm count is correct",
                  n_bn == b["batchnorm"],
                  f"recorded {b['batchnorm']}, real {n_bn}"
                  + ("   -> micro_batch is FREE to change"
                     if n_bn == 0 else "   -> micro_batch is LOCKED"))

            # The transformer's token grid is derived from a stride of 32. If a
            # backbone ever stops being stride-32 the pos-embed is built for the
            # wrong grid and everything still runs.
            f = bare.forward_features(torch.randn(1, 3, 256, 320))
            check(f"{bname}: forward_features is stride {H.BACKBONE_STRIDE}",
                  f.shape[-2] == 256 // H.BACKBONE_STRIDE
                  and f.shape[-1] == 320 // H.BACKBONE_STRIDE,
                  f"{tuple(f.shape)} from 256x320")
            del ref, bare, hyb
        except Exception as exc:
            check(f"{bname}: nesting", False, f"{type(exc).__name__}: {exc}")


def model_checks(build: bool) -> None:
    print("\nmodels")
    if not build:
        print("  (skipping the real forward pass; pass --build to construct all "
              f"{len(models.MODEL_NAMES)} and run it)")
        return

    import torch
    torch.set_grad_enabled(False)
    for m in models.MODEL_NAMES:
        try:
            net, info = models.build(m, pretrained=False)
            net.eval()
            y = net(torch.randn(1, 3, C.IMG_H, C.IMG_W))
            ok = tuple(y.shape) == (1, C.NUM_CLASSES)
            cp = info["component_params"]
            check(f"{m}: forward -> (1, {C.NUM_CLASSES})", ok,
                  f"{tuple(y.shape)}  +{cp['added_over_backbone']/1e6:.3f}M "
                  f"({cp['added_pct']:+.2f}%)")
            # Every stream must be ablatable, or ablate.py reports nothing.
            for st in net.stream_names:
                net.ablate = st
                net(torch.randn(1, 3, C.IMG_H, C.IMG_W))
            net.ablate = None
            del net
        except Exception as exc:
            check(f"{m}: forward", False, f"{type(exc).__name__}: {exc}")


def corpus_checks(data: Path) -> None:
    """The crop is a measurement, so re-measure it against whatever is on disk."""
    print(f"\ncorpus  {data}")
    import manifest
    import pandas as pd

    try:
        geo = manifest.check_geometry(data)
        check("folder geometry matches CS_DIM", True, geo["note"])
    except RuntimeError as exc:
        check("folder geometry matches CS_DIM", False, str(exc).splitlines()[0])
        return

    df = pd.read_csv(data / "index.csv")
    check("corrupt records are excluded by manifest.load",
          not set(DIM.EXCLUDED_RECORDS) & set(manifest.load(data)["record"]),
          f"{list(DIM.EXCLUDED_RECORDS)} dropped")
    rng = np.random.default_rng(0)
    sel = df.iloc[rng.choice(len(df), min(60, len(df)), replace=False)]

    if geo["apply_crop"]:
        # Checked against crop_audit.py's stored bounds, NOT re-measured here.
        # An earlier version of this check searched for the header only within
        # y < 0.20H and reported the search bound (0.1991) as if it were data,
        # which made a correct crop look like a failure.
        M = DIM.MEASURED
        FH = M["final_height"]
        y0, y1 = DIM.CONTENT_CROP
        check("crop clears the worst-case header",
              y0 * FH > M["header_bottom_max"],
              f"cut {y0*FH:.0f} px > header {M['header_bottom_max']} px "
              f"(margin +{y0*FH - M['header_bottom_max']:.0f})")
        check("crop clears the worst-case waveform",
              y0 * FH < M["trace_top_min"],
              f"cut {y0*FH:.0f} px < waveform {M['trace_top_min']} px "
              f"(margin +{M['trace_top_min'] - y0*FH:.0f}); "
              f"{M['clipped_by_crop']} extreme sheets are clipped regardless")
        check("no bottom cut (the rhythm strip reaches the sheet edge)",
              y1 * FH >= M["rhythm_bottom_max"],
              f"cut {y1*FH:.0f} px >= rhythm {M['rhythm_bottom_max']} px")

    # The leak this whole crop exists for. Assert it is real, so the claim in the
    # writeup is a measurement rather than a memory.
    if "record" in df.columns and "cls" in df.columns:
        num = df["record"].str.extract(r"(\d+)")[0].astype(int)
        stemi = num[df.cls == "STEMI"]
        other = num[df.cls != "STEMI"]
        if len(stemi) and len(other):
            sep = stemi.min() > other.max()
            check("printed record ID separates STEMI (the leak redaction removes)",
                  sep,
                  f"STEMI ids {stemi.min()}..{stemi.max()}, "
                  f"others {other.min()}..{other.max()}"
                  + ("  <- PERFECTLY SEPARABLE" if sep else "  (overlapping - the "
                     "leak does not apply to this corpus)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--build", action="store_true",
                    help="also construct every model and run a forward pass "
                         "(no pretrained download)")
    a = ap.parse_args()

    print("=" * 78)
    print(f"CardioSentry hybrids v2 - selftest")
    print(f"geometry: {C.geometry_str()}")
    print("=" * 78)

    geometry_checks()
    transform_checks()
    batching_checks()
    fingerprint_checks()
    registry_checks()
    hybrid_checks()
    nesting_checks(a.build)
    model_checks(a.build)
    if a.data:
        corpus_checks(a.data)
        if C.REDACT_HEADER:
            redaction_effect_check(a.data)

    print("\n" + "=" * 78)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFAILURES:")
        for f in FAIL:
            print("   ", f)
        print("\nDo not train until these are fixed - each one is a bug whose")
        print("symptom is a plausible-looking number rather than an error.")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
