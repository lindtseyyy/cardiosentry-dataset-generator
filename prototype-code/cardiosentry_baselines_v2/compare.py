#!/usr/bin/env python3
"""
Tabulate finished runs - and first PROVE they are comparable.

    python compare.py --runs runs/                 # rank ARCHITECTURES
    python compare.py --runs runs/ --across-dims   # rank INPUT GEOMETRIES

TWO COMPARISONS, TWO RULES
--------------------------
Ranking nine architectures is only meaningful if every run shared the SAME input
geometry as well as the same protocol: `fingerprint` must match. Ranking seven
geometries is only meaningful if every run shared the same protocol and
DELIBERATELY differed in geometry: `protocol_fingerprint` must match and
`fingerprint` must not.

The default mode enforces the first rule. `--across-dims` enforces the second.
Neither one is `--force`; there is a --force, and it is for when you already know
what you are looking at.

This script refuses to present a ranking when the rule for the mode it is in is
broken, because a table that silently mixes experimental conditions is worse than
no table: it looks authoritative and is not.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import config as C


def load_runs(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
            r.setdefault("run_name", r["model"])
            r.setdefault("dim_name", "?")
            out.append(r)
        except Exception as exc:
            print(f"  skipping unreadable {f}: {exc}")
    return out


def _shared(runs, key, label, notes):
    vals = {r["run_name"]: key(r) for r in runs}
    if len(set(map(str, vals.values()))) > 1:
        notes.append(f"{label} DIFFER:")
        for m, v in vals.items():
            notes.append(f"    {m:26} {v}")
        return False
    return True


def check_comparable(runs: list[dict], across_dims: bool) -> tuple[bool, list[str]]:
    notes, ok = [], True

    if across_dims:
        # the protocol must be identical...
        if not _shared(runs, lambda r: r.get("protocol_fingerprint"),
                       "PROTOCOL FINGERPRINTS", notes):
            ok = False
            notes.append("  Something other than the geometry changed between "
                         "these runs. Re-run the stale ones.")
        # ...and every run must be a DIFFERENT geometry of the SAME architecture,
        # or the table conflates two variables at once.
        arch = {r["model"] for r in runs}
        if len(arch) > 1:
            notes.append(f"  NOTE: {len(arch)} architectures present "
                         f"({', '.join(sorted(arch))}). The table below groups by "
                         f"geometry; read one architecture at a time.")
        dims = [r["dim_name"] for r in runs]
        if len(set(dims)) == 1:
            ok = False
            notes.append(f"ALL RUNS USE THE SAME GEOMETRY ({dims[0]}) - there is "
                         f"nothing to compare across dimensions. Drop "
                         f"--across-dims.")
    else:
        if not _shared(runs, lambda r: r.get("fingerprint"),
                       "CONFIG FINGERPRINTS", notes):
            ok = False
            pf = {r.get("protocol_fingerprint") for r in runs}
            if len(pf) == 1:
                notes.append("  The PROTOCOL fingerprints all match, so the only "
                             "difference is the input geometry.")
                notes.append("  That is the resolution study, not an architecture "
                             "ranking: re-run with --across-dims.")
            else:
                notes.append("  config.py changed between runs. Re-run the stale "
                             "ones.")

    if not _shared(runs, lambda r: r["data"]["root"], "DATASETS", notes):
        ok = False
    if not _shared(runs, lambda r: r["data"].get("rows"), "ROW COUNTS", notes):
        ok = False

    smoke = [r["run_name"] for r in runs if r.get("smoke")]
    if smoke:
        ok = False
        notes.append(f"SMOKE RUNS PRESENT (metrics meaningless): {smoke}")

    lrs = {r["run_name"]: r.get("lr_used") for r in runs}
    if len(set(lrs.values())) > 1 and not all(r.get("sweep") for r in runs):
        notes.append(f"  NOTE: learning rates differ {lrs} and not every run used "
                     f"--sweep. Either use one LR everywhere, or --sweep everywhere.")

    pre = {r["run_name"]: r["model_info"].get("pretrain") for r in runs}
    if len(set(pre.values())) > 1:
        notes.append("  NOTE: pretraining corpora differ - a real asymmetry, not a bug:")
        for m, p in sorted(pre.items()):
            notes.append(f"    {m:26} {p}")

    # Padding overhead: Swin-V2 is fed more pixels than the others at any
    # geometry that is not a multiple of 256. That does not affect its ACCURACY
    # numbers, but it does make its latency and VRAM incomparable.
    padded = {r["run_name"]: r["model_info"].get("input_geometry", {})
              for r in runs}
    over = {m: g.get("pixel_overhead", 1.0) for m, g in padded.items()
            if g.get("pixel_overhead", 1.0) > 1.001}
    if over:
        notes.append("  NOTE: these runs were fed MORE pixels than the geometry "
                     "asks for, to satisfy a")
        notes.append("        window-attention size constraint. Accuracy is "
                     "comparable; cost is not:")
        for m, v in sorted(over.items()):
            notes.append(f"    {m:26} {v:.2f}x pixels "
                         f"({padded[m].get('adaptation','')})")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--across-dims", action="store_true",
                    help="rank input geometries instead of architectures")
    ap.add_argument("--sort", default="macro_auprc",
                    choices=["macro_auprc", "macro_auroc", "macro_f1",
                             "macro_sensitivity", "stemi_auprc", "stemi_sensitivity"])
    ap.add_argument("--csv", type=Path, help="also write the table as CSV")
    ap.add_argument("--out", type=Path, default=None,
                    help="text report path (default <runs>/comparison.txt)")
    ap.add_argument("--force", action="store_true",
                    help="print the table even if the runs are not comparable")
    a = ap.parse_args()

    runs = load_runs(a.runs)
    if not runs:
        print(f"no result.json under {a.runs}/ - train something first")
        return 1

    mode = "input geometry" if a.across_dims else "architecture"
    print("=" * 104)
    print(f"CardioSentry baselines v2 - {mode} comparison   ({len(runs)} runs)")
    print("=" * 104)

    ok, notes = check_comparable(runs, a.across_dims)
    for n in notes:
        print(n)
    if not ok and not a.force:
        print("\nREFUSING to rank non-comparable runs. Fix the above, or pass --force.")
        return 2
    if ok:
        if a.across_dims:
            print(f"comparability: OK - all runs share protocol "
                  f"{runs[0]['protocol_fingerprint']} and differ only in geometry")
        else:
            print(f"comparability: OK - all runs share fingerprint "
                  f"{runs[0]['fingerprint']}")
            g = runs[0].get("geometry", {})
            if g:
                print(f"geometry: {g['name']}  {g['H']}x{g['W']}"
                      f"{'  header cropped' if g['crop'] else '  full sheet'}  |  "
                      f"{g['px_per_mm_x']:.2f} px/mm time, "
                      f"{g['px_per_mm_y']:.2f} px/mm voltage")
        d = runs[0]["data"]
        print(f"dataset: {d['rows']:,} images, {d['records']:,} records, "
              f"{d['patients']:,} patients")

    key = {"macro_auprc": lambda r: r["test"]["macro_auprc"],
           "macro_auroc": lambda r: r["test"]["macro_auroc"],
           "macro_f1": lambda r: r["test"]["macro_f1"],
           "macro_sensitivity": lambda r: r["test"]["macro_sensitivity"],
           "stemi_auprc": lambda r: r["test"]["per_label"]["STEMI"]["auprc"],
           "stemi_sensitivity": lambda r: r["test"]["per_label"]["STEMI"]["sensitivity"],
           }[a.sort]
    runs.sort(key=key, reverse=True)

    # In --across-dims the row label is the geometry; otherwise it is the model.
    label = (lambda r: f"{r['dim_name']} ({r['model']})") if a.across_dims \
        else (lambda r: r["model"])
    W = 26

    out_lines = []

    def emit(line=""):
        print(line)
        out_lines.append(line)

    # ---- geometry legend, only useful in the dimension study --------------
    if a.across_dims:
        import dimensions as DIM
        emit()
        emit("GEOMETRIES IN THIS TABLE")
        emit()
        emit(DIM.table(sorted({r["dim_name"] for r in runs}
                              & set(DIM.DIM_NAMES),
                              key=DIM.DIM_NAMES.index)))

    # ---- headline ---------------------------------------------------------
    emit()
    emit("TEST SET  (thresholds tuned on val, frozen before test)")
    emit()
    hdr = (f"{'run':{W}}{'mAUROC':>9}{'mAUPRC':>9}{'mSens':>8}{'mSpec':>8}"
           f"{'mPrec':>8}{'mF1':>8}{'mFPR':>8}{'ExactAcc':>10}")
    emit(hdr); emit("-" * len(hdr))
    for r in runs:
        t = r["test"]
        emit(f"{label(r):{W}}{t['macro_auroc']:9.4f}{t['macro_auprc']:9.4f}"
             f"{t['macro_sensitivity']:8.4f}{t['macro_specificity']:8.4f}"
             f"{t['macro_precision']:8.4f}{t['macro_f1']:8.4f}"
             f"{t['macro_fpr']:8.4f}{t['exact_match_accuracy']:10.4f}")

    # ---- cost -------------------------------------------------------------
    emit()
    emit("COST  (accuracy is only half the comparison)")
    emit()
    hdr = (f"{'run':{W}}{'params M':>10}{'size MB':>10}{'ms/img b1':>11}"
           f"{'img/s batch':>13}{'train min':>11}{'epochs':>8}{'px fed':>9}")
    emit(hdr); emit("-" * len(hdr))
    for r in runs:
        c = r.get("complexity", {}); b = r.get("inference", {})
        ig = r["model_info"].get("input_geometry", {})
        fed = ig.get("fed", [0, 0])
        emit(f"{label(r):{W}}{c.get('params_total_m', float('nan')):10.1f}"
             f"{c.get('size_mb', float('nan')):10.1f}"
             f"{b.get('latency_ms_b1_mean', float('nan')):11.2f}"
             f"{b.get('throughput_ips_batched', float('nan')):13.1f}"
             f"{r['train_seconds']/60:11.1f}{r.get('epochs_run', 0):8d}"
             f"{fed[0]*fed[1]/1e6:9.3f}M")

    # ---- per label --------------------------------------------------------
    for metric, nice in (("auroc", "AUROC"), ("auprc", "AUPRC"),
                         ("sensitivity", "SENSITIVITY (recall)"),
                         ("specificity", "SPECIFICITY"),
                         ("precision", "PRECISION"), ("f1", "F1"),
                         ("fpr", "FALSE POSITIVE RATE"), ("accuracy", "ACCURACY")):
        emit()
        emit(f"per-label {nice} (test)")
        emit()
        hdr = f"{'run':{W}}" + "".join(f"{l:>10}" for l in C.LABELS) + f"{'MACRO':>10}"
        emit(hdr); emit("-" * len(hdr))
        for r in runs:
            emit(f"{label(r):{W}}"
                 + "".join(f"{r['test']['per_label'][l][metric]:10.4f}" for l in C.LABELS)
                 + f"{r['test']['macro_'+metric]:10.4f}")

    # ---- source confound --------------------------------------------------
    if any(r.get("test_per_source") for r in runs):
        emit()
        emit("per-source macro AUROC (test) - large spread = source confound")
        emit()
        srcs = sorted({s for r in runs for s in r.get("test_per_source", {})})
        hdr = f"{'run':{W}}" + "".join(f"{s:>11}" for s in srcs) + f"{'spread':>10}"
        emit(hdr); emit("-" * len(hdr))
        for r in runs:
            ps = r.get("test_per_source", {})
            vals = [ps[s]["macro_auroc"] for s in srcs if s in ps]
            row = "".join(f"{ps[s]['macro_auroc']:11.4f}" if s in ps else f"{'-':>11}"
                          for s in srcs)
            spread = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
            emit(f"{label(r):{W}}{row}{spread:10.4f}")
        emit()
        emit("  All STEMI comes from one source, and stage 5 measured residual source")
        emit("  signal (0.515 vs 0.372 chance). Report these alongside the headline.")

    # ---- the header-leak read, only in the dimension study ----------------
    if a.across_dims:
        emit()
        emit("HEADER-CROP ABLATION  (STEMI head)")
        emit()
        emit("  The printed record ID separates STEMI EXACTLY in this corpus")
        emit("  (CHAPMAN/PTBXL <= ECG007796 < ECG007797 <= STEMI). It is INSIDE the")
        emit("  image on a full-sheet geometry and OUTSIDE it on a crop* one, so the")
        emit("  gap between the two groups below is an upper bound on how much of the")
        emit("  STEMI score was reading a digit rather than an ST segment.")
        emit()
        hdr = (f"{'run':{W}}{'header':>9}{'STEMI AUROC':>13}{'STEMI AUPRC':>13}"
               f"{'STEMI Sens':>12}{'STEMI FPR':>11}")
        emit(hdr); emit("-" * len(hdr))
        for r in sorted(runs, key=lambda r: (not r.get("geometry", {}).get("crop", False),
                                             r["dim_name"])):
            s = r["test"]["per_label"]["STEMI"]
            vis = "cropped" if r.get("geometry", {}).get("crop") else "VISIBLE"
            emit(f"{label(r):{W}}{vis:>9}{s['auroc']:13.4f}{s['auprc']:13.4f}"
                 f"{s['sensitivity']:12.4f}{s['fpr']:11.4f}")

    txt = a.out or (a.runs / ("comparison_dims.txt" if a.across_dims
                              else "comparison.txt"))
    txt.write_text("\n".join(out_lines))
    print(f"\ntext report -> {txt}")

    if a.csv:
        import csv as _csv
        with a.csv.open("w", newline="") as f:
            w = _csv.writer(f)
            MAC = ["auroc", "auprc", "sensitivity", "specificity", "precision",
                   "f1", "fpr", "accuracy"]
            PL = ["auroc", "auprc", "sensitivity", "specificity", "precision",
                  "f1", "fpr", "accuracy", "threshold", "support"]
            w.writerow(["run_name", "model", "dim", "img_h", "img_w", "crop",
                        "px_per_mm_x", "px_per_mm_y", "aspect_distortion"]
                       + [f"macro_{m}" for m in MAC]
                       + ["exact_match_accuracy", "hamming_accuracy",
                          "micro_f1", "micro_precision", "micro_sensitivity"]
                       + [f"{l}_{m}" for l in C.LABELS for m in PL]
                       + ["params_total", "params_total_m", "size_mb",
                          "latency_ms_b1", "latency_p95_b1", "throughput_ips_batched",
                          "px_fed", "pixel_overhead",
                          "train_min", "epochs", "lr", "pretrain", "timm_tag",
                          "fingerprint", "protocol_fingerprint"])
            for r in runs:
                t = r["test"]; c = r.get("complexity", {}); b = r.get("inference", {})
                g = r.get("geometry", {})
                ig = r["model_info"].get("input_geometry", {})
                fed = ig.get("fed", [0, 0])
                w.writerow([r["run_name"], r["model"], r["dim_name"],
                            g.get("H"), g.get("W"), g.get("crop"),
                            g.get("px_per_mm_x"), g.get("px_per_mm_y"),
                            g.get("aspect_distortion")]
                           + [t[f"macro_{m}"] for m in MAC]
                           + [t["exact_match_accuracy"], t["hamming_accuracy"],
                              t["micro_f1"], t["micro_precision"], t["micro_sensitivity"]]
                           + [t["per_label"][l][m] for l in C.LABELS for m in PL]
                           + [c.get("params_total"), c.get("params_total_m"),
                              c.get("size_mb"), b.get("latency_ms_b1_mean"),
                              b.get("latency_ms_b1_p95"), b.get("throughput_ips_batched"),
                              fed[0] * fed[1], ig.get("pixel_overhead"),
                              round(r["train_seconds"] / 60, 1), r.get("epochs_run"),
                              r.get("lr_used"), r["model_info"].get("pretrain"),
                              r["model_info"].get("resolved_tag"), r.get("fingerprint"),
                              r.get("protocol_fingerprint")])
        print(f"csv -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
