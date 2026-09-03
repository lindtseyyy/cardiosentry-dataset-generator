#!/usr/bin/env python3
"""
Exercise the scoring and reporting path on SIMULATED predictions.

    python selftest.py

Why this exists: the real run happens in Colab against weights that live in
Drive, so the first time `evaluate.py` and `report.py` execute together would
otherwise be on a GPU runtime, minutes into a session, with no easy way to
iterate on a traceback. This runs everything except the forward pass - cohort
masks, per-head metrics, bootstrap intervals, the metrics JSON schema, and the
whole of report.py - against the REAL index.csv, on this machine, in seconds.

It also answers a question worth answering before the experiment is run: what
does each possible outcome actually look like on paper? Three models are
simulated, one per row of the decision table:

    perfect_generalist   ranks STEMI well on every source
                         -> the report should read TRANSFERS

    source_memoriser     ranks the control heads well and STEMI at chance,
                         because it learnt Chongqing paper and none of these
                         sheets are on it
                         -> the report should read CONFOUND CONFIRMED

    domain_shifted       everything at chance
                         -> the report should read INCONCLUSIVE

If the verdict column does not come back with those three answers in that
order, the classifier in report.py is not measuring what it claims to.

THE NUMBERS THIS WRITES ARE FABRICATED. They go to results/_selftest/ and are
never mixed with real output.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate as E          # noqa: E402  (torch is imported lazily inside it)

OUT = HERE.parent / "results" / "_selftest"

# How each simulated model ranks a positive versus a negative. Probabilities are
# drawn from Beta distributions whose separation encodes the behaviour; the exact
# shapes do not matter, only that "good" separates and "chance" does not.
SCENARIOS = {
    "perfect_generalist": {
        "STEMI": {"pos": (9, 2), "neg": (2, 9)},
        "control": {"pos": (9, 2), "neg": (2, 9)},
    },
    "source_memoriser": {
        # Silent on external STEMI positives AND negatives alike: the paper is
        # wrong, so nothing here looks like the thing it memorised, and it
        # cannot rank one against the other. Positives and negatives draw from
        # the SAME low distribution on purpose - that is what "learnt the
        # scanner" looks like off-scanner. It is also why this model scores
        # perfect specificity on every mimic, which is exactly why perfect
        # specificity is not evidence of anything on its own.
        "STEMI": {"pos": (2, 9), "neg": (2, 9)},
        "control": {"pos": (9, 2), "neg": (2, 9)},
    },
    "domain_shifted": {
        "STEMI": {"pos": (4, 5), "neg": (4, 5)},
        "control": {"pos": (4, 5), "neg": (4, 5)},
    },
}

EXPECTED = {
    "perfect_generalist": "TRANSFERS",
    "source_memoriser": "CONFOUND CONFIRMED",
    "domain_shifted": "INCONCLUSIVE",
}


def simulate(y: np.ndarray, spec: dict, rng: np.random.Generator) -> np.ndarray:
    prob = np.empty(len(y))
    pos, neg = y == 1, y == 0
    a, b = spec["pos"]
    prob[pos] = rng.beta(a, b, pos.sum())
    a, b = spec["neg"]
    prob[neg] = rng.beta(a, b, neg.sum())
    return prob


def main() -> int:
    testset = HERE.parent / "data" / "external_testset"
    if not (testset / "index.csv").exists():
        sys.exit(f"missing {testset}/index.csv - run build/03_merge.py first")

    df = E.load_index(testset)
    E.check_testset(df)
    masks = E.cohort_masks(df)
    y_true = df[E.LABELS].to_numpy().astype(int)
    rng = np.random.default_rng(0)

    print(f"index: {len(df)} sheets")
    for name, mask in masks.items():
        n_stemi = int(y_true[mask, E.LABELS.index("STEMI")].sum())
        print(f"  {name:<22} n={int(mask.sum()):>4}  STEMI+={n_stemi}")
    print()

    results = {}
    for name, spec in SCENARIOS.items():
        prob = np.empty_like(y_true, dtype=float)
        for k, lab in enumerate(E.LABELS):
            key = "STEMI" if lab == "STEMI" else "control"
            prob[:, k] = simulate(y_true[:, k], spec[key], rng)
        thresholds = [0.5] * len(E.LABELS)

        entry = {"thresholds": thresholds, "epoch": "SIMULATED",
                 "params_m": None, "cohorts": {}}
        for cname, mask in masks.items():
            if mask.sum() == 0:
                continue
            with_ci = cname in {"ALL", "LUDB", "PTBXL_MI", "PTBXL_ALL"}
            per_head = {lab: E.head_stats(y_true[mask, k], prob[mask, k],
                                          thresholds[k], with_ci)
                        for k, lab in enumerate(E.LABELS)}
            macro = {}
            for key in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
                vals = [per_head[l][key] for l in E.LABELS
                        if not np.isnan(per_head[l][key])]
                macro[f"macro_{key}"] = float(np.mean(vals)) if vals else float("nan")
            entry["cohorts"][cname] = {"n": int(mask.sum()),
                                       "per_label": per_head, **macro}
        results[name] = entry
        print(f"  simulated {name}")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_fingerprint": C_FINGERPRINT,
        "labels": E.LABELS,
        "testset": str(testset),
        "weights_root": "SIMULATED - no weights were loaded",
        "n_sheets": int(len(df)),
        "cohort_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "positives": {k: {lab: int(y_true[v, i].sum())
                          for i, lab in enumerate(E.LABELS)}
                      for k, v in masks.items()},
        "bootstrap_resamples": E.BOOTSTRAP,
        "models": results,
    }
    metrics_path = OUT / "external_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {metrics_path}")

    report_path = OUT / "SELFTEST_REPORT.md"
    proc = subprocess.run(
        [sys.executable, str(HERE / "report.py"),
         "--metrics", str(metrics_path), "-o", str(report_path)],
        capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print("\nFAILED: report.py errored")
        return 1

    text = report_path.read_text()
    print()
    ok = True
    for name, want in EXPECTED.items():
        # The verdict table row for this model, e.g. "| source_memoriser | ... |"
        line = next((l for l in text.splitlines()
                     if l.startswith(f"| {name} |")), "")
        got = "**" + want + "**" in line
        print(f"  {'PASS' if got else 'FAIL'}  {name:<20} expected {want}")
        if not got:
            print(f"        row: {line.strip() or '(row not found)'}")
        ok = False if not got else ok

    print(f"\nreport: {report_path}")
    print("(simulated numbers - never mix these with a real run)")
    return 0 if ok else 1


try:
    sys.path.insert(0, str(E.TRAIN_CODE))
    import config as _cfg
    C_FINGERPRINT = _cfg.fingerprint()
except Exception:                                  # noqa: BLE001
    C_FINGERPRINT = "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
