#!/usr/bin/env python3
"""
Local smoke test for the AF x LVH joint-head round.

Runs analyze.py against the repo's OWN result files (the local copy of the
hybrid_v1 resnet50_hires run folder), so the whole pipeline - cross-checks,
per-head rows, joint breakdown, exact tests, verdict - is exercised before
the notebook ships. The expected values are transcribed from the same frozen
files, so a mismatch here means the pipeline drifted from the reference.

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
OUT = ROUND / "_selftest_out"

EXPECTED = {
    # frozen pooled rows (cross-check anchors)
    "af_pooled": {"tp": 268, "fp": 15, "fn": 17, "tn": 638, "sens": 0.9404},
    "lvh_pooled": {"tp": 216, "fp": 34, "fn": 39, "tn": 649, "sens": 0.8471},
    # the subgroup rows
    "af_af_lvh": {"tp": 26, "fn": 2, "sens": 0.9286},
    "lvh_af_lvh": {"tp": 18, "fn": 10, "sens": 0.6429},
    # joint breakdown
    "joint": {"af_ok_lvh_ok": 16, "af_ok_lvh_miss": 10,
              "af_miss_lvh_ok": 2, "af_miss_lvh_miss": 0},
    # exact tests
    "mcnemar": 0.0386, "fisher_af": 0.5142, "fisher_lvh": 0.0038,
    "verdict": "SUPPORTED",
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
    for p in (PREDICTIONS, RESULT_JSON):
        assert Path(p).exists(), f"missing {p}"

    run([sys.executable, str(ROUND / "analyze.py"),
         "--predictions", str(PREDICTIONS),
         "--result-json", str(RESULT_JSON),
         "--out", str(OUT)])

    import pandas as pd
    af = pd.read_csv(OUT / "af_head_rows.csv")
    lvh = pd.read_csv(OUT / "lvh_head_rows.csv")
    joint = pd.read_csv(OUT / "joint_breakdown.csv").set_index("category")
    import json
    summary = json.load(open(OUT / "summary.json"))

    for lab, want, rows in (("AF", EXPECTED["af_pooled"], af),
                            ("LVH", EXPECTED["lvh_pooled"], lvh)):
        pool = rows.iloc[-1]
        for k in ("tp", "fp", "fn", "tn"):
            assert int(pool[k]) == want[k], f"{lab} pooled {k}"
        assert abs(pool["sensitivity"] - want["sens"]) < 1e-4, f"{lab} pooled sens"
    a = af.iloc[0]
    l = lvh.iloc[0]
    assert int(a["tp"]) == 26 and int(a["fn"]) == 2, "AF head on AF+LVH"
    assert int(l["tp"]) == 18 and int(l["fn"]) == 10, "LVH head on AF+LVH"
    for k, v in EXPECTED["joint"].items():
        assert int(joint.loc[f"{k}".replace("af_ok_lvh_ok", "AF correct + LVH correct")
                        .replace("af_ok_lvh_miss", "AF correct + LVH missed")
                        .replace("af_miss_lvh_ok", "AF missed  + LVH correct")
                        .replace("af_miss_lvh_miss", "AF missed  + LVH missed"),
                        "count"]) == v, f"joint {k}"
    t = summary["tests"]
    assert abs(t["mcnemar"] - EXPECTED["mcnemar"]) < 1e-3, "mcnemar"
    assert abs(t["fisher_af"] - EXPECTED["fisher_af"]) < 1e-3, "fisher_af"
    assert abs(t["fisher_lvh"] - EXPECTED["fisher_lvh"]) < 1e-3, "fisher_lvh"
    assert summary["verdict"].startswith(EXPECTED["verdict"]), "verdict"
    report = (OUT / "report.txt").read_text()
    assert "VERDICT: SUPPORTED" in report, "verdict line in report.txt"
    print("  pooled AF row  268/15/17/638 OK")
    print("  pooled LVH row 216/34/39/649 OK")
    print("  AF head on AF+LVH: TP 26 FN 2   OK")
    print("  LVH head on AF+LVH: TP 18 FN 10  OK")
    print("  joint breakdown 16/10/2/0        OK")
    print(f"  McNemar p={t['mcnemar']:.4f}  Fisher AF p={t['fisher_af']:.4f}  "
          f"Fisher LVH p={t['fisher_lvh']:.4f}  OK")
    print("  verdict:", summary["verdict"][:60], "...")
    print()
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
