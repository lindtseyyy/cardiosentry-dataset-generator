#!/usr/bin/env python3
"""
Local smoke test for the AF x LVH subgroup round.

Runs analyze.py against the repo's OWN result files (the local copy of the
hybrid_v1 resnet50_hires run folder and the prototype_1152 index), so the
whole pipeline - cross-checks, counts, metrics, bootstrap, verdict - is
exercised before the notebook ships. The model_verify checkpoint load is
exercised separately with --checkpoint-only (no 768px corpus lives in the
repo, so the inference half cannot run here).

    python selftest.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROUND = Path(__file__).resolve().parent

PREDICTIONS = (REPO / "prototype-results/hybrid_v1/resnet50_hires/predictions/"
               "test_predictions.csv")
RESULT_JSON = REPO / "prototype-results/hybrid_v1/resnet50_hires/result.json"
WEIGHTS = REPO / "prototype-results/hybrid_v1/resnet50_hires/weights/best.pt"
INDEX = (REPO / "dataset-combination-code/ecg_dataset_pipeline/data/releases/"
         "v2/prototype_1152/index.csv")
OUT = ROUND / "_selftest_out"

EXPECTED = {
    # the four headline counts on the frozen test split
    "clean_lvh_pos": 227,     # LVH-only ECGs
    "af_lvh_pos": 28,         # AF+LVH ECGs
    "stemi_lvh_pos": 0,       # STEMI+LVH ECGs
    "af_stemi_lvh_pos": 0,    # AF+STEMI+LVH ECGs
    "pooled_auprc": 0.9248,   # frozen pooled LVH AUPRC
    "pooled_tp": 216,
    "clean_auroc": 0.9663,
    "af_auroc": 0.8973,
}


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd))
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit(1)
    return r.stdout


def main():
    for p in (PREDICTIONS, RESULT_JSON, WEIGHTS, INDEX):
        assert Path(p).exists(), f"missing {p}"

    print("== 1/2  analyze.py end to end")
    run([sys.executable, str(ROUND / "analyze.py"),
         "--predictions", str(PREDICTIONS),
         "--result-json", str(RESULT_JSON),
         "--index", str(INDEX),
         "--out", str(OUT)])

    import pandas as pd
    counts = pd.read_csv(OUT / "subgroup_counts_test.csv", index_col=0)
    for ctx, col in (("clean", "lvh_pos"), ("af", "lvh_pos"),
                     ("stemi", "lvh_pos"), ("af_stemi", "lvh_pos")):
        got = int(counts.loc[ctx, col])
        want = EXPECTED[f"{ctx}_lvh_pos"]
        assert got == want, f"test counts: {ctx} LVH+ = {got}, want {want}"
        print(f"  counts test [{ctx}] LVH+ = {got}  OK")

    mets = pd.read_csv(OUT / "subgroup_metrics.csv")
    by_name = dict(zip(mets.stratum, mets.to_dict("records")))
    pool = next(r for r in by_name.values() if r["stratum"] == "POOLED (all test)")
    assert abs(pool["auprc"] - EXPECTED["pooled_auprc"]) < 1e-4, pool
    assert int(pool["tp"]) == EXPECTED["pooled_tp"], pool
    clean = by_name["LVH only (no AF, no STEMI)"]
    af = by_name["AF + LVH (AF present)"]
    assert abs(clean["auroc"] - EXPECTED["clean_auroc"]) < 1e-4, clean
    assert abs(af["auroc"] - EXPECTED["af_auroc"]) < 1e-4, af
    print(f"  pooled AUPRC {pool['auprc']:.4f}  clean AUROC {clean['auroc']:.4f}"
          f"  AF AUROC {af['auroc']:.4f}  OK")

    deltas = pd.read_csv(OUT / "bootstrap_deltas.csv")
    d = dict(zip(deltas.metric, deltas.to_dict("records")))
    for m in ("auprc", "auroc", "sensitivity", "specificity", "fpr",
              "precision", "f1", "sens_at_fpr"):
        assert m in d and d[m]["delta_ci_low"] == d[m]["delta_ci_low"], \
            f"metric {m} missing/NaN in bootstrap output"
    print("  bootstrap rows:", len(d), "metrics OK")

    report = (OUT / "report.txt").read_text()
    assert "VERDICT:" in report and "SUPPORTED" in report, \
        "expected the SUPPORTED verdict in report.txt"
    print("  verdict line found in report.txt")

    print()
    print("== 2/2  model_verify checkpoint-only (weights on this machine)")
    run([sys.executable, str(ROUND / "model_verify.py"),
         "--ckpt", str(WEIGHTS), "--device", "cpu", "--checkpoint-only"])
    print()
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
