#!/usr/bin/env python3
"""
Tabulate finished runs - and first PROVE they are comparable.

    python compare.py --runs runs/

The comparison is only meaningful if every run shared the controlled conditions.
This script refuses to present a ranking when the config fingerprints disagree,
because a table that silently mixes experimental conditions is worse than no
table: it looks authoritative and is not.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import config as C


def load_runs(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/result.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception as exc:
            print(f"  skipping unreadable {f}: {exc}")
    return out


def check_comparable(runs: list[dict]) -> tuple[bool, list[str]]:
    notes, ok = [], True
    fps = {r["model"]: r.get("fingerprint") for r in runs}
    if len(set(fps.values())) > 1:
        ok = False
        notes.append("CONFIG FINGERPRINTS DIFFER - these runs are NOT comparable:")
        for m, fp in fps.items():
            notes.append(f"    {m:18} {fp}")
        notes.append("  config.py changed between runs. Re-run the stale ones.")

    roots = {r["model"]: r["data"]["root"] for r in runs}
    if len(set(roots.values())) > 1:
        ok = False
        notes.append("DIFFERENT DATASETS used:")
        for m, d in roots.items():
            notes.append(f"    {m:18} {d}")

    rows = {r["model"]: r["data"].get("rows") for r in runs}
    if len(set(rows.values())) > 1:
        ok = False
        notes.append(f"DIFFERENT ROW COUNTS: {rows}")

    smoke = [r["model"] for r in runs if r.get("smoke")]
    if smoke:
        ok = False
        notes.append(f"SMOKE RUNS PRESENT (metrics meaningless): {smoke}")

    lrs = {r["model"]: r.get("lr_used") for r in runs}
    if len(set(lrs.values())) > 1 and not all(r.get("sweep") for r in runs):
        notes.append(f"  NOTE: learning rates differ {lrs} and not every run used "
                     f"--sweep. Either use one LR everywhere, or --sweep everywhere.")

    pre = {r["model"]: r["model_info"].get("pretrain") for r in runs}
    if len(set(pre.values())) > 1:
        notes.append("  NOTE: pretraining corpora differ - a real asymmetry, not a bug:")
        for m, p in pre.items():
            notes.append(f"    {m:18} {p}")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", type=Path)
    ap.add_argument("--sort", default="macro_auprc",
                    choices=["macro_auprc", "macro_auroc", "macro_f1",
                             "macro_sensitivity", "stemi_auprc", "stemi_sensitivity"])
    ap.add_argument("--csv", type=Path, help="also write the table as CSV")
    ap.add_argument("--force", action="store_true",
                    help="print the table even if the runs are not comparable")
    a = ap.parse_args()

    runs = load_runs(a.runs)
    if not runs:
        print(f"no result.json under {a.runs}/ - train something first")
        return 1

    print("=" * 96)
    print(f"CardioSentry architecture comparison   ({len(runs)} runs)")
    print("=" * 96)

    ok, notes = check_comparable(runs)
    for n in notes:
        print(n)
    if not ok and not a.force:
        print("\nREFUSING to rank non-comparable runs. Fix the above, or pass --force.")
        return 2
    if ok:
        print(f"comparability: OK - all runs share fingerprint {runs[0]['fingerprint']}")
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

    # ---- headline ---------------------------------------------------------
    out_lines = []
    def emit(line=""):
        print(line)
        out_lines.append(line)

    emit()
    emit("TEST SET  (thresholds tuned on val, frozen before test)")
    emit()
    hdr = (f"{'model':18}{'mAUROC':>9}{'mAUPRC':>9}{'mSens':>8}{'mSpec':>8}"
           f"{'mPrec':>8}{'mF1':>8}{'mFPR':>8}{'ExactAcc':>10}")
    emit(hdr); emit("-" * len(hdr))
    for r in runs:
        t = r["test"]
        emit(f"{r['model']:18}{t['macro_auroc']:9.4f}{t['macro_auprc']:9.4f}"
             f"{t['macro_sensitivity']:8.4f}{t['macro_specificity']:8.4f}"
             f"{t['macro_precision']:8.4f}{t['macro_f1']:8.4f}"
             f"{t['macro_fpr']:8.4f}{t['exact_match_accuracy']:10.4f}")

    # ---- cost -------------------------------------------------------------
    emit()
    emit("COST  (accuracy is only half the comparison)")
    emit()
    hdr = (f"{'model':18}{'params M':>10}{'size MB':>10}{'ms/img b1':>11}"
           f"{'img/s batch':>13}{'train min':>11}{'epochs':>8}")
    emit(hdr); emit("-" * len(hdr))
    for r in runs:
        c = r.get("complexity", {}); b = r.get("inference", {})
        emit(f"{r['model']:18}{c.get('params_total_m', float('nan')):10.1f}"
             f"{c.get('size_mb', float('nan')):10.1f}"
             f"{b.get('latency_ms_b1_mean', float('nan')):11.2f}"
             f"{b.get('throughput_ips_batched', float('nan')):13.1f}"
             f"{r['train_seconds']/60:11.1f}{r.get('epochs_run', 0):8d}")

    # ---- per label --------------------------------------------------------
    for metric, nice in (("auroc", "AUROC"), ("auprc", "AUPRC"),
                         ("sensitivity", "SENSITIVITY (recall)"),
                         ("specificity", "SPECIFICITY"),
                         ("precision", "PRECISION"), ("f1", "F1"),
                         ("fpr", "FALSE POSITIVE RATE"), ("accuracy", "ACCURACY")):
        emit()
        emit(f"per-label {nice} (test)")
        emit()
        hdr = f"{'model':18}" + "".join(f"{l:>10}" for l in C.LABELS) + f"{'MACRO':>10}"
        emit(hdr); emit("-" * len(hdr))
        for r in runs:
            emit(f"{r['model']:18}"
                 + "".join(f"{r['test']['per_label'][l][metric]:10.4f}" for l in C.LABELS)
                 + f"{r['test']['macro_'+metric]:10.4f}")

    # ---- source confound --------------------------------------------------
    if any(r.get("test_per_source") for r in runs):
        emit()
        emit("per-source macro AUROC (test) - large spread = source confound")
        emit()
        srcs = sorted({s for r in runs for s in r.get("test_per_source", {})})
        hdr = f"{'model':18}" + "".join(f"{s:>11}" for s in srcs) + f"{'spread':>10}"
        emit(hdr); emit("-" * len(hdr))
        for r in runs:
            ps = r.get("test_per_source", {})
            vals = [ps[s]["macro_auroc"] for s in srcs if s in ps]
            row = "".join(f"{ps[s]['macro_auroc']:11.4f}" if s in ps else f"{'-':>11}"
                          for s in srcs)
            spread = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
            emit(f"{r['model']:18}{row}{spread:10.4f}")
        emit()
        emit("  All STEMI comes from one source, and stage 5 measured residual source")
        emit("  signal (0.515 vs 0.372 chance). Report these alongside the headline.")

    txt = a.runs / "comparison.txt"
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
            w.writerow(["model"] + [f"macro_{m}" for m in MAC]
                       + ["exact_match_accuracy", "hamming_accuracy",
                          "micro_f1", "micro_precision", "micro_sensitivity"]
                       + [f"{l}_{m}" for l in C.LABELS for m in PL]
                       + ["params_total", "params_total_m", "size_mb",
                          "latency_ms_b1", "latency_p95_b1", "throughput_ips_batched",
                          "train_min", "epochs", "lr", "pretrain", "timm_tag",
                          "fingerprint"])
            for r in runs:
                t = r["test"]; c = r.get("complexity", {}); b = r.get("inference", {})
                w.writerow([r["model"]] + [t[f"macro_{m}"] for m in MAC]
                           + [t["exact_match_accuracy"], t["hamming_accuracy"],
                              t["micro_f1"], t["micro_precision"], t["micro_sensitivity"]]
                           + [t["per_label"][l][m] for l in C.LABELS for m in PL]
                           + [c.get("params_total"), c.get("params_total_m"),
                              c.get("size_mb"), b.get("latency_ms_b1_mean"),
                              b.get("latency_ms_b1_p95"), b.get("throughput_ips_batched"),
                              round(r["train_seconds"] / 60, 1), r.get("epochs_run"),
                              r.get("lr_used"), r["model_info"].get("pretrain"),
                              r["model_info"].get("resolved_tag"), r.get("fingerprint")])
        print(f"\ncsv -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
