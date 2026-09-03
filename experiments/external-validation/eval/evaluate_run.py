#!/usr/bin/env python3
"""
Score a TRAINED RUN from prototype-results on the external STEMI corpus.

    python eval/evaluate_run.py                          # the default run below
    python eval/evaluate_run.py resnet50_cbam@nat768     # any other run
    python eval/evaluate_run.py --list                   # what is scoreable

WHY THIS EXISTS ALONGSIDE evaluate.py
-------------------------------------
`evaluate.py` scores the five cardiosentry_runs_v1 baselines, whose run folders
are named after the model (`resnet50/weights/best.pt`) and whose geometry is
fixed at import time. Everything from baselines_v2 onward is named
`<model>@<dim>` and the geometry is chosen by the CS_DIM environment variable,
which config.py reads ONCE, at import. A run directory therefore has to be
resolved, and CS_DIM set, BEFORE the training code is imported - which is the
one thing an already-imported module cannot do for itself.

So this file is the front end: it resolves the run, reads the geometry the run
recorded in its own result.json, sets CS_DIM to it, and only then imports the
training code. The scoring itself - cohorts, per-head metrics, bootstrap CIs -
is `evaluate.py`'s, imported and reused, so both entry points produce numbers
that mean the same thing and `selftest.py` still covers the path.

THE MODEL IS A COMMAND-LINE ARGUMENT, ON PURPOSE
------------------------------------------------
DEFAULT_RUN below is the one currently under examination. Passing any other run
name (or an absolute path to a run folder) swaps it out; passing several scores
them all into one report so they sit in the same table. Nothing else about the
protocol moves when the model does - see "what is held fixed" below.

WHAT IS HELD FIXED, AND WHY IT MATTERS THAT IT IS
-------------------------------------------------
1. GEOMETRY comes from the run, not from this file. A checkpoint trained at
   nat768 (768x1024, ID redacted) fed 768x768 squares would score badly for a
   reason that has nothing to do with the question being asked. The geometry is
   read from result.json and cross-checked against the config fingerprint the
   checkpoint carries; a mismatch stops the run rather than producing a number.

2. TRANSFORMS are the training code's own `data.build_transforms(train=False)`
   - the same resize, the same header redaction. The redaction is not optional
   here: the printed record ID is a STEMI leak in the training corpus, the model
   was trained never to see it, and un-redacting it at test time would show the
   model a band it has never been given.

3. THRESHOLDS are the ones frozen in best.pt on the ORIGINAL validation split,
   applied unchanged. Re-tuning on this corpus would be tuning on test: it would
   hand a model that ranks external positives barely above chance an operating
   point that manufactures sensitivity. AUROC and AUPRC are reported alongside
   for the same reason - they cannot be rescued by an operating point.

READING THE OUTPUT
------------------
The verdict comes from the two-by-two in evaluate.py's docstring: STEMI is only
interpretable next to AF/LVH/NORMAL, which had multi-source training, carry no
source confound, and ride on the identical sheets. STEMI collapses while they
hold -> the confound is real. Everything collapses together -> domain shift,
inconclusive. STEMI holds -> the confound was in the data and the model did not
use it.

Eight gold LUDB positives. Every interval is wide, and the width is the finding.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../external-validation/eval
EXP = HERE.parent                               # .../external-validation


def _repo_root(start: Path) -> Path:
    """Walk up to the folder holding prototype-code/ and prototype-results/.

    Searched rather than hard-coded because this experiment folder has already
    moved once (repo root -> experiments/), and a relative `parents[n]` silently
    pointed at the wrong directory afterwards.
    """
    for p in (start, *start.parents):
        if (p / "prototype-code").is_dir() and (p / "prototype-results").is_dir():
            return p
    raise SystemExit(
        f"could not find the repo root above {start} - expected a directory "
        f"containing both prototype-code/ and prototype-results/.\n"
        f"  pass --code and --runs-root explicitly.")


REPO = _repo_root(HERE)

# The run under examination. Change this line, or pass a run on the command
# line, to score something else.
DEFAULT_RUN = "efficientnetv2_s_cbam@nat768"

DEFAULT_RUNS_ROOT = REPO / "prototype-results"
DEFAULT_CODE = REPO / "prototype-code" / "cardiosentry_hybrids_v2"
DEFAULT_TESTSET = EXP / "data" / "external_testset"


# ====================================================================== runs
def find_runs(runs_root: Path) -> list[Path]:
    """Every folder under runs_root that holds weights/best.pt, nested included."""
    return sorted({p.parent.parent for p in runs_root.rglob("weights/best.pt")})


def resolve_run(spec: str, runs_root: Path, dim_override: str | None = None) -> dict:
    """A run spec -> everything needed to score it, read off the run itself."""
    p = Path(spec).expanduser()
    if not p.is_dir():
        cand = runs_root / spec
        if not cand.is_dir():
            avail = [str(r.relative_to(runs_root)) for r in find_runs(runs_root)]
            raise SystemExit(
                f"no run '{spec}' - looked at {p} and {cand}.\n"
                f"  runs under {runs_root}:\n    "
                + ("\n    ".join(avail) if avail else "(none found)"))
        p = cand
    p = p.resolve()

    ckpt = p / "weights" / "best.pt"
    if not ckpt.exists():
        alt = p / "best.pt"
        if not alt.exists():
            raise SystemExit(f"{p} has no weights/best.pt (nor best.pt)")
        ckpt = alt

    res: dict = {}
    rj = p / "result.json"
    if rj.exists():
        res = json.loads(rj.read_text())
    else:
        # config_snapshot.json is written at the START of training, so it exists
        # even for a run that died before result.json. It carries the geometry,
        # which is the field that must not be guessed.
        cs = p / "config_snapshot.json"
        if cs.exists():
            res = json.loads(cs.read_text())

    run_name = p.name
    model = res.get("model") or run_name.split("@")[0]
    dim = dim_override or res.get("dim_name") or (
        run_name.split("@")[1] if "@" in run_name else None)
    if dim is None:
        raise SystemExit(
            f"cannot tell what geometry {run_name} was trained at: no result.json "
            f"or config_snapshot.json, and the folder name carries no '@<dim>'.\n"
            f"  pass --dim to state it explicitly.")

    # The run's own internal test scores. Quoted in the report as the figure the
    # external number falls FROM - a STEMI AUROC of 0.99 internally and 0.50
    # here is the whole finding, and it is not legible without both halves.
    internal = {}
    t = res.get("test") or {}
    if t:
        internal = {
            "macro_auroc": t.get("macro_auroc"),
            "stemi_auroc": (t.get("per_label", {}).get("STEMI", {}) or {}).get("auroc"),
            "stemi_auprc": (t.get("per_label", {}).get("STEMI", {}) or {}).get("auprc"),
        }

    return {"dir": p, "run_name": run_name, "model": model, "dim": dim,
            "ckpt": ckpt, "result": res, "internal": internal,
            "fingerprint": res.get("fingerprint")}


# ================================================================= inference
def infer(model, df, device, batch: int, workers: int,
          apply_crop: bool, apply_redact: bool, C, quiet: bool = False):
    """Sigmoid probabilities, [n, 4], in df row order.

    Deliberately mirrors engine.predict: same eval transforms, same autocast,
    same float32 sigmoid. shuffle=False is what keeps the output aligned with
    the index, so nothing downstream has to re-sort.
    """
    import numpy as np
    import torch
    import data as D

    ds = D.ECGSheetDataset(df, train=False, apply_crop=apply_crop,
                           apply_redact=apply_redact)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda")

    amp = device.type == "cuda" and C.AMP
    model.eval()
    out, t0 = [], time.time()
    with torch.no_grad():
        for bi, (images, _, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp):
                logits = model(images)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
            if not quiet and (bi + 1) % 10 == 0:
                done = min((bi + 1) * batch, len(ds))
                rate = done / max(time.time() - t0, 1e-9)
                print(f"      {done}/{len(ds)} sheets  ({rate:.1f}/s)", flush=True)
    return np.concatenate(out, axis=0)


def load_checkpoint(run: dict, device, MO):
    """Model with best.pt loaded, plus the thresholds frozen with it."""
    import torch

    blob = torch.load(run["ckpt"], map_location="cpu", weights_only=False)
    if run["model"] not in MO.MODEL_NAMES:
        raise SystemExit(
            f"'{run['model']}' is not a model in {os.environ['CARDIOSENTRY_CODE']}.\n"
            f"  known there: {', '.join(MO.MODEL_NAMES)}\n"
            f"  if this run came from another package, pass --code.")
    # pretrained=False: the ImageNet weights are overwritten a line later, and
    # downloading them only to discard them costs minutes on a fresh runtime.
    model, info = MO.build(run["model"], pretrained=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"{run['ckpt']} does not fit '{run['model']}' as built here: "
            f"{len(missing)} missing / {len(unexpected)} unexpected tensors.\n"
            f"  e.g. missing {list(missing)[:3]}  unexpected {list(unexpected)[:3]}\n"
            f"  usually means --code points at the wrong training package, or "
            f"timm changed the backbone.")
    model.to(device)

    thr = blob.get("thresholds") if isinstance(blob, dict) else None
    if not thr:
        # Falling back to 0.5 would make this model's sensitivity incomparable
        # with every other model's, which is worse than stopping.
        raise SystemExit(
            f"{run['ckpt']} carries no frozen thresholds. They were tuned on the "
            f"original validation split and must not be re-derived here.")
    thr = [float(t) for t in thr]

    # result.json records the same thresholds. Disagreement means the two files
    # came from different training runs and one of them is stale.
    rt = run["result"].get("thresholds")
    if rt and [round(float(a), 6) for a in rt] != [round(t, 6) for t in thr]:
        print(f"    WARNING: best.pt thresholds {[round(t,3) for t in thr]} differ "
              f"from result.json {[round(float(t),3) for t in rt]}; using best.pt's")
    return model, thr, info, blob


# ======================================================================= main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=None,
                    help=f"run folder name under --runs-root, or a path. "
                         f"default: {DEFAULT_RUN}")
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ap.add_argument("--code", default=str(DEFAULT_CODE),
                    help="training package holding models.py / data.py / config.py")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET))
    ap.add_argument("--out", default=None,
                    help="default: results/<code package name>/")
    ap.add_argument("--dim", default=None,
                    help="override the geometry (normally read from the run)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="default: the model's training micro-batch")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="resamples for the percentile CIs")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N sheets - SMOKE TEST ONLY; the "
                         "numbers are meaningless and are written to a _smoke/ "
                         "subfolder so they cannot be mistaken for results")
    ap.add_argument("--merge", action="store_true",
                    help="keep the runs already scored into --out instead of "
                         "replacing them, so the report compares this run with "
                         "earlier ones. Refused unless the stored geometry, "
                         "fingerprint and corpus all match this invocation.")
    ap.add_argument("--list", action="store_true", help="list scoreable runs and exit")
    ap.add_argument("--no-report", action="store_true",
                    help="write metrics only; skip report.py")
    ap.add_argument("--allow-fingerprint-mismatch", action="store_true",
                    help="score anyway when the run's config fingerprint differs "
                         "from this checkout's - the numbers stop being comparable")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    # Line buffering, so a long CPU run shows progress as it happens and the
    # report.py subprocess's output does not overtake this process's.
    sys.stdout.reconfigure(line_buffering=True)
    runs_root = Path(args.runs_root).expanduser().resolve()

    if args.list:
        found = find_runs(runs_root)
        print(f"scoreable runs under {runs_root}:")
        for r in found:
            rj = r / "result.json"
            note = ""
            if rj.exists():
                d = json.loads(rj.read_text())
                note = f"  [{d.get('model','?')} @ {d.get('dim_name','?')}]"
            print(f"  {r.relative_to(runs_root)}{note}")
        if not found:
            print("  (none)")
        return 0

    specs = args.runs or [DEFAULT_RUN]
    runs = [resolve_run(s, runs_root, args.dim) for s in specs]

    # config.py reads CS_DIM once, at import, so one process scores exactly one
    # geometry. Mixing them would silently feed some checkpoints the wrong input
    # size, so it is refused rather than worked around.
    dims = {args.dim or r["dim"] for r in runs}
    if len(dims) > 1:
        return _die(f"runs span more than one geometry: {sorted(dims)}.\n"
                    f"  config.py fixes the input size at import, so score one "
                    f"geometry per invocation.")
    dim = dims.pop()

    code = Path(args.code).expanduser().resolve()
    if not (code / "models.py").exists():
        return _die(f"no models.py in {code} - pass --code at the training package")

    # ---- everything above this line runs before the training code is imported.
    os.environ["CS_DIM"] = dim
    os.environ["CARDIOSENTRY_CODE"] = str(code)
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(HERE))

    import numpy as np
    import pandas as pd
    import torch

    import config as C
    import manifest as MF
    import models as MO
    import evaluate as EV        # cohorts, per-head stats, bootstrap - reused

    if args.bootstrap != EV.BOOTSTRAP:
        _orig = EV.boot_ci
        EV.boot_ci = (lambda y, p, fn, n=args.bootstrap, seed=C.SEED:
                      _orig(y, p, fn, n, seed))

    out = Path(args.out) if args.out else EXP / "results" / code.name
    if args.limit:
        out = out / "_smoke"
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    testset = Path(args.testset).expanduser().resolve()
    df = EV.load_index(testset)
    EV.check_testset(df)
    if args.limit:
        print(f"!! --limit {args.limit}: SMOKE TEST. These numbers are not results.")
        df = df.head(args.limit).reset_index(drop=True)

    # A pre-resized copy of the corpus has the crop/redaction baked in already;
    # applying either a second time would cut into the first trace row.
    if hasattr(MF, "check_geometry"):
        geo = MF.check_geometry(testset)
    else:
        geo = {"apply_crop": getattr(C, "CROP_HEADER", False),
               "apply_redact": getattr(C, "REDACT_HEADER", False),
               "note": "training package predates geometry markers"}

    masks = EV.cohort_masks(df)
    y_true = df[C.LABELS].to_numpy().astype(int)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    si = C.LABELS.index("STEMI")

    print(f"corpus     {testset}")
    print(f"           {len(df)} sheets   STEMI+ {int(y_true[:, si].sum())} "
          f"(LUDB {int(y_true[masks['LUDB'], si].sum())} gold / "
          f"PTB-XL {int(y_true[masks['PTBXL_MI'], si].sum())} proxy)")
    print(f"geometry   {C.DIM_NAME}  {C.IMG_H}x{C.IMG_W}  "
          f"crop={geo['apply_crop']}  redact={geo['apply_redact']}   ({geo['note']})")
    print(f"code       {code}")
    print(f"device     {device}   fingerprint {C.fingerprint()}")
    print()

    results: dict[str, dict] = {}
    for run in runs:
        name = run["run_name"]
        print(f"--- {name}   ({run['model']})")

        fp = run["fingerprint"]
        if fp and fp != C.fingerprint():
            msg = (f"    run fingerprint {fp} != this checkout's {C.fingerprint()} - "
                   f"the training protocol or geometry has changed since it ran")
            if not args.allow_fingerprint_mismatch:
                return _die(msg + "\n  pass --allow-fingerprint-mismatch to score "
                                  "anyway (the numbers stop being comparable).")
            print("    WARNING:" + msg[4:])

        model, thresholds, info, blob = load_checkpoint(run, device, MO)
        batch = args.batch or MO.micro_batch(run["model"])
        print(f"    epoch {blob.get('epoch', '?')}  "
              f"thresholds {[round(t, 3) for t in thresholds]} (frozen on original val)  "
              f"batch {batch}")

        prob = infer(model, df, device, batch, args.workers,
                     geo["apply_crop"], geo["apply_redact"], C)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        keep = [c for c in ("record", "source", "cohort", "label_confidence",
                            "seen_in_training", "original_id", "printed_id")
                if c in df.columns]
        pred_df = df[keep].copy()
        for k, lab in enumerate(C.LABELS):
            pred_df[f"y_{lab}"] = y_true[:, k]
            pred_df[f"p_{lab}"] = prob[:, k]
            pred_df[f"pred_{lab}"] = (prob[:, k] >= thresholds[k]).astype(int)
        pred_df.to_csv(out / "predictions" / f"{name}.csv", index=False)

        entry = {"model": run["model"], "run_dir": str(run["dir"]),
                 "dim": C.DIM_NAME, "thresholds": thresholds,
                 "epoch": blob.get("epoch"),
                 "params_m": info.get("params_m_actual"),
                 "internal": run["internal"], "cohorts": {}}
        for cname, mask in masks.items():
            if mask.sum() == 0:
                continue
            # CIs only where they are read: 2000 resamples x every cohort x every
            # head is minutes of pure overhead otherwise.
            with_ci = cname in {"ALL", "LUDB", "PTBXL_MI", "PTBXL_ALL"}
            per_head = {lab: EV.head_stats(y_true[mask, k], prob[mask, k],
                                           thresholds[k], with_ci)
                        for k, lab in enumerate(C.LABELS)}
            macro = {}
            for key in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
                vals = [per_head[l][key] for l in C.LABELS
                        if not np.isnan(per_head[l][key])]
                macro[f"macro_{key}"] = float(np.mean(vals)) if vals else float("nan")
            entry["cohorts"][cname] = {"n": int(mask.sum()),
                                       "per_label": per_head, **macro}
        results[name] = entry
        _print_summary(entry, run)

    geometry = {"name": C.DIM_NAME, "H": C.IMG_H, "W": C.IMG_W,
                "crop": geo["apply_crop"], "redact": geo["apply_redact"]}
    if args.merge:
        results = _merge_previous(out, results, C.fingerprint(), geometry,
                                  testset, args.bootstrap)

    payload = {
        "config_fingerprint": C.fingerprint(),
        "protocol_fingerprint": (C.protocol_fingerprint()
                                 if hasattr(C, "protocol_fingerprint") else None),
        "geometry": geometry,
        "labels": C.LABELS,
        "testset": str(testset),
        "code": str(code),
        "weights_root": str(runs_root),
        "n_sheets": int(len(df)),
        "smoke_limit": args.limit,
        "cohort_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "positives": {k: {lab: int(y_true[v, i].sum())
                          for i, lab in enumerate(C.LABELS)}
                      for k, v in masks.items()},
        "bootstrap_resamples": args.bootstrap,
        "models": results,
    }
    (out / "external_metrics.json").write_text(json.dumps(payload, indent=2))

    rows = [{"model": n, "cohort": cn, "label": lab, **m}
            for n, e in results.items()
            for cn, c in e["cohorts"].items()
            for lab, m in c["per_label"].items()]
    pd.DataFrame(rows).to_csv(out / "external_metrics.csv", index=False)

    print(f"wrote {out / 'external_metrics.json'}")
    print(f"      {out / 'external_metrics.csv'}")
    print(f"      {out / 'predictions'}/<run>.csv")

    if not args.no_report:
        rep = out / "EXTERNAL_VALIDATION_REPORT.md"
        subprocess.run([sys.executable, str(HERE / "report.py"),
                        "--metrics", str(out / "external_metrics.json"),
                        "--predictions", str(out / "predictions"),
                        "--index", str(testset / "index.csv"),
                        "-o", str(rep)], check=True)
    return 0


def _merge_previous(out: Path, results: dict, fingerprint: str, geometry: dict,
                    testset: Path, bootstrap: int) -> dict:
    """Carry forward runs already scored into `out`, so the report compares them.

    Every guard below exists because a silently merged row is worse than no row:
    a table that puts a nat768 score beside an sq768 one, or a score from an
    older corpus beside a newer one, reads as a comparison and is not one. When
    anything differs the previous file is left alone and only this invocation's
    runs are written, which is the same behaviour as not passing --merge.
    """
    prev_path = out / "external_metrics.json"
    if not prev_path.exists():
        return results
    prev = json.loads(prev_path.read_text())

    mismatch = []
    if prev.get("config_fingerprint") != fingerprint:
        mismatch.append(f"fingerprint {prev.get('config_fingerprint')} != {fingerprint}")
    if prev.get("geometry") != geometry:
        mismatch.append(f"geometry {prev.get('geometry')} != {geometry}")
    if prev.get("testset") != str(testset):
        mismatch.append(f"corpus {prev.get('testset')} != {testset}")
    if prev.get("smoke_limit"):
        mismatch.append(f"previous file is smoke output (--limit {prev['smoke_limit']})")
    if mismatch:
        print("    NOT merging - the stored results are not comparable with these:")
        for m in mismatch:
            print(f"      {m}")
        return results

    if prev.get("bootstrap_resamples") != bootstrap:
        print(f"    note: stored CIs used {prev.get('bootstrap_resamples')} resamples, "
              f"this run {bootstrap}; point estimates are unaffected")

    carried = {}
    for name, entry in (prev.get("models") or {}).items():
        if name in results:
            continue                      # re-scored now; the new one wins
        # report.py's score-by-finding table reads these; a carried row without
        # its predictions would drop out of that table only, which is confusing.
        if not (out / "predictions" / f"{name}.csv").exists():
            print(f"    note: carrying {name} forward without its predictions CSV")
        carried[name] = entry
    if carried:
        print(f"    merged with {len(carried)} run(s) already scored here: "
              f"{', '.join(carried)}")
    return {**carried, **results}


def _print_summary(entry: dict, run: dict) -> None:
    """The three lines that answer the question, before the report is written."""
    nan = float("nan")
    c = entry["cohorts"]
    gold = c.get("LUDB", {}).get("per_label", {}).get("STEMI", {})
    proxy = c.get("PTBXL_MI", {}).get("per_label", {}).get("STEMI", {})
    intern = (run["internal"] or {}).get("stemi_auroc")
    print(f"    STEMI gold  (LUDB, n+={gold.get('n_pos', 0)})   "
          f"AUROC {gold.get('auroc', nan):.3f}  sens {gold.get('sensitivity', nan):.3f}  "
          f"mean p(pos) {gold.get('mean_prob_pos', nan):.3f}"
          + (f"   [internal test AUROC {intern:.3f}]" if intern else ""))
    print(f"    STEMI proxy (PTB-XL MI, n+={proxy.get('n_pos', 0)})  "
          f"sens {proxy.get('sensitivity', nan):.3f}  "
          f"mean p(pos) {proxy.get('mean_prob_pos', nan):.3f}")
    ludb = c.get("LUDB", {}).get("per_label", {})
    if ludb:
        print("    control heads (LUDB, same sheets): " + "   ".join(
            f"{l} AUROC {ludb[l]['auroc']:.3f}" for l in ("AF", "LVH", "NORMAL")))
    print()


def _die(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
