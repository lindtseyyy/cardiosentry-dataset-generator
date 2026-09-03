#!/usr/bin/env python3
"""
Stream ablation on FINISHED runs. No retraining, no GPU hours.

    python ablate.py --runs runs/ --data /content/prototype_nat768
    python ablate.py --run  runs/resnet50_lt_gated@nat768 --data <DATA>

WHAT IT ANSWERS

A two-stream hybrid that beats its baseline has a follow-up question the headline
cannot answer: did the TRANSFORMER contribute, or did the model get better for
some other reason and carry a passenger? This runs the test set three times on
the trained weights - once whole, once with the CNN stream zeroed, once with the
transformer stream zeroed - and reports what each removal costs.

    d(no transformer) near 0   the transformer is a passenger. The gain, if any,
                               came from the extra head capacity, and a wider
                               head would have been cheaper.
    d(no transformer) large    the transformer stream is load-bearing.
    d(no cnn) near 0           the model is running on the transformer alone,
                               which for a 2-block, 256-wide module on top of a
                               frozen-ish backbone is worth a second look.

WHY THIS IS A VALID THING TO DO, AND WHY IT WOULD NOT BE WITHOUT THE LAYERNORMS

Each stream is LayerNorm'd BEFORE fusion, so zeroing one leaves the other's
statistics exactly as the head saw them in training. Normalising after the concat
instead would mean that dropping half the vector shifts the mean and variance of
the half that remains, and the "ablation" would measure a distribution shift
rather than the stream's contribution. selftest.py asserts the property.

The thresholds are the ones frozen from validation and used to score test in the
original run - read out of result.json, never re-tuned. Re-tuning them per
ablation would let each ablated model pick its own best operating point and the
comparison would flatter the ablations.

An ablated model is OFF-DISTRIBUTION: it was never trained with a stream zeroed.
So these numbers bound what a stream CONTRIBUTES to this trained model; they are
not an estimate of how a single-stream model would have scored if trained alone.
For that, read the `lt`-only rung of the ladder, which IS trained alone.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

import config as C
import data as D
import engine
import metrics
import models


def ablate_run(run_dir: Path, data: str, workers: int = 2,
               loaders: dict | None = None) -> dict | None:
    res_path = run_dir / "result.json"
    ckpt = run_dir / "weights" / "best.pt"
    if not res_path.exists():
        print(f"  {run_dir.name}: no result.json - unfinished run, skipping")
        return None
    if not ckpt.exists():
        print(f"  {run_dir.name}: no weights/best.pt - skipping")
        return None

    res = json.loads(res_path.read_text())
    name = res["model"]
    if name not in models.MODEL_NAMES:
        print(f"  {run_dir.name}: '{name}' is not in this registry "
              f"(a baselines_v2 row?) - skipping")
        return None
    if res.get("fingerprint") != C.fingerprint():
        print(f"  {run_dir.name}: fingerprint {res.get('fingerprint')} != current "
              f"{C.fingerprint()} - the config changed since this run; skipping")
        return None

    streams = (res.get("hybrid") or {}).get("streams", [])
    if len(streams) < 2:
        print(f"  {run_dir.name}: single-stream ({streams}) - nothing to ablate")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = models.build(name, pretrained=False)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    thresholds = res["thresholds"]
    rows = {}
    for abl in [None] + list(streams):
        model.ablate = abl
        p, y, _ = engine.predict(model, loaders["test"], device)
        rows["full" if abl is None else f"no_{abl}"] = metrics.evaluate(y, p, thresholds)
    model.ablate = None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = {"run": run_dir.name, "model": name, "streams": streams,
           "thresholds": thresholds,
           "gate_mean": (res.get("hybrid") or {}).get("gate_mean"),
           "results": rows}
    (run_dir / "ablation.json").write_text(json.dumps(out, indent=2, default=str))
    (run_dir / "ablation.txt").write_text(format_run(out))
    return out


def format_run(out: dict) -> str:
    L = [f"stream ablation - {out['run']}",
         f"streams: {' + '.join(out['streams'])}"
         + (f"   mean gate {out['gate_mean']:.4f}"
            if out.get("gate_mean") is not None else ""),
         "thresholds frozen from validation, never re-tuned", ""]
    base = out["results"]["full"]
    hdr = (f"  {'condition':16}{'mAUROC':>9}{'d':>9}{'mAUPRC':>9}{'d':>9}"
           f"{'mF1':>9}{'d':>9}{'STEMI AUPRC':>13}{'d':>9}")
    L += [hdr, "  " + "-" * (len(hdr) - 2)]
    for cond, r in out["results"].items():
        d = (lambda k: r[k] - base[k])
        ds = (lambda k: "" if cond == "full" else f"{d(k):+9.4f}")
        sa = r["per_label"]["STEMI"]["auprc"]
        sab = base["per_label"]["STEMI"]["auprc"]
        L.append(f"  {cond:16}{r['macro_auroc']:9.4f}{ds('macro_auroc') or '':>9}"
                 f"{r['macro_auprc']:9.4f}{ds('macro_auprc') or '':>9}"
                 f"{r['macro_f1']:9.4f}{ds('macro_f1') or '':>9}"
                 f"{sa:13.4f}"
                 + ("" if cond == "full" else f"{sa - sab:+9.4f}"))
    L += ["",
          "  'no_lt' = the transformer stream zeroed after its LayerNorm.",
          "  A delta near zero means that stream was not load-bearing in this",
          "  trained model. An ablated model is off-distribution - it was never",
          "  trained this way - so read these as the stream's CONTRIBUTION here,",
          "  not as an estimate of a single-stream model trained from scratch."]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path, help="one run directory")
    g.add_argument("--runs", type=Path, help="a folder of run directories")
    ap.add_argument("--data", required=True, help="the extracted corpus")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None,
                    help="combined report (default <runs>/ablation.txt)")
    a = ap.parse_args()

    dirs = [a.run] if a.run else sorted(p for p in a.runs.iterdir() if p.is_dir())
    print(f"loading the test split once for {len(dirs)} run(s) ...")
    # Built ONCE and reused: the loader is identical for every run by
    # construction (same geometry, same manifest), and rebuilding it per run on
    # Colab costs more than the ablation itself.
    loaders = D.build_loaders(a.data, micro_batch=8, workers=a.workers)

    outs = []
    for d in dirs:
        o = ablate_run(d, a.data, a.workers, loaders)
        if o:
            outs.append(o)
            print()
            print(format_run(o))

    if not outs:
        print("\nno two-stream runs to ablate. The lt_concat, lt_gated and "
              "msa_lt_gated rungs are the ones with two streams.")
        return 1

    root = a.runs or a.run.parent
    lines = ["CardioSentry hybrids v2 - stream ablation", ""]
    hdr = (f"{'run':32}{'gate':>8}{'mAUPRC':>9}{'no cnn':>10}{'no lt':>10}"
           f"   what carried it")
    lines += [hdr, "-" * len(hdr)]
    for o in outs:
        r = o["results"]
        full = r["full"]["macro_auprc"]
        dc = r.get("no_cnn", {}).get("macro_auprc")
        dl = r.get("no_lt", {}).get("macro_auprc")
        read = []
        if dl is not None:
            read.append("transformer is a passenger" if abs(full - dl) < 0.01
                        else "transformer is load-bearing")
        if dc is not None and abs(full - dc) < 0.01:
            read.append("CNN stream is a passenger")
        gm = o.get("gate_mean")
        # Built by concatenation rather than by nesting f-strings inside
        # f-strings: same-quote nesting is a 3.12 feature (PEP 701) and this
        # package has to run on whatever Python Colab is serving that week.
        cells = [f"{o['run']:32}",
                 f"{gm:8.3f}" if gm is not None else f"{'-':>8}",
                 f"{full:9.4f}",
                 f"{dc - full:+10.4f}" if dc is not None else f"{'-':>10}",
                 f"{dl - full:+10.4f}" if dl is not None else f"{'-':>10}",
                 "   " + "; ".join(read)]
        lines.append("".join(cells))
    lines += ["", "  'no cnn' / 'no lt' are DELTAS in test macro AUPRC from zeroing",
              "  that stream. Thresholds are the frozen val-tuned ones throughout."]
    txt = "\n".join(lines)
    p = a.out or (root / "ablation.txt")
    p.write_text(txt)
    print("\n" + txt)
    print(f"\ncombined report -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
