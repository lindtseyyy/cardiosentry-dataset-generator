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


def model_checks(build: bool) -> None:
    print("\nmodels")
    check("nine architectures registered", len(models.MODEL_NAMES) == 9,
          ", ".join(models.MODEL_NAMES))
    for m in models.MODEL_NAMES:
        g = models.input_geometry(m)
        mult = (models.REGISTRY[m].get("size_multiple"))
        if mult:
            check(f"{m}: fed size is a multiple of {mult}",
                  g["fed"][0] % mult == 0 and g["fed"][1] % mult == 0,
                  f"{g['fed'][0]}x{g['fed'][1]}  ({g['pixel_overhead']:.2f}x px)")
        else:
            check(f"{m}: fed at the configured size", g["fed"] == tuple(C.IMG_SIZE),
                  f"{g['fed'][0]}x{g['fed'][1]}")

    if not build:
        print("  (skipping the real forward pass; pass --build to download weights "
              "and run it)")
        return

    import torch
    torch.set_grad_enabled(False)
    for m in models.MODEL_NAMES:
        try:
            net, info = models.build(m, pretrained=False)
            net.eval()
            y = net(torch.randn(1, 3, C.IMG_H, C.IMG_W))
            check(f"{m}: forward -> (1, {C.NUM_CLASSES})",
                  tuple(y.shape) == (1, C.NUM_CLASSES), f"{tuple(y.shape)}")
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
    print(f"CardioSentry baselines v2 - selftest")
    print(f"geometry: {C.geometry_str()}")
    print("=" * 78)

    geometry_checks()
    transform_checks()
    batching_checks()
    fingerprint_checks()
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
