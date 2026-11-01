"""Refit the clinical_v1 threshold policy for the clean12 + augmented V1 Tiny RFPN run.

The policy (rules per label) is the v1-rfpn study's `clinical_v1`, imported unchanged
from ../v1-rfpn/threshold_research; only the scores differ. Fitting uses the run's
VALIDATION predictions pooled over both views (clean + augmented sheet of every
record), the same rows the run used for its own max-F1 thresholds. Caveat: each
record contributes two correlated rows, so the Wilson lower bounds of the
sensitivity/specificity-constrained rules (NORM, 3AVB, STEMI) see ~2x the
effective n. The test split is read only after freezing, to report the operating
point per view.

Run (needs scikit-learn, e.g. the repo venv):
    .venv/bin/python3 threshold-configurations/v1-rfpn-aug/fit_clinical_v1.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v1-rfpn" / "threshold_research"))

from common import LABELS, predict, sha256_file  # noqa: E402
from fit_thresholds import fit_rule, operational_thresholds  # noqa: E402
from metrics import per_label_table  # noqa: E402
from policies import POLICIES  # noqa: E402
from threshold_methods import max_fbeta  # noqa: E402

RUN_DIR = HERE.parents[1] / "prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_aug_colab_a100_seed42"
MODEL_DIR = RUN_DIR / "convnext_tiny_rfpn"
OUTPUT_DIR = HERE / "outputs"
FROZEN = OUTPUT_DIR / "thresholds_clinical_v1.json"
POLICY = "clinical_v1"


def load(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(MODEL_DIR / f"{split}_predictions.csv", dtype={"record_id": str})
    if set(frame["split"]) != {split}:
        raise ValueError(f"{split}_predictions.csv holds splits {sorted(set(frame['split']))}")
    if frame.groupby("record_id")["variant_id"].apply(lambda v: sorted(v) != ["augmented", "clean"]).any():
        raise ValueError(f"{split}: every record must have exactly one clean and one augmented row")
    config = json.loads((MODEL_DIR / "config_snapshot.json").read_text())
    if tuple(config["identity"]["label_order"]) != LABELS:
        raise ValueError("run label order differs from the clinical_v1 study's")
    y = frame[[f"true_{label}" for label in LABELS]].to_numpy(dtype=np.int8)
    s = frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=np.float32)
    return frame, y, s


def main() -> None:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists: thresholds are frozen. Delete it deliberately to refit.")
    _, y_val, s_val = load("validation")

    # The reimplemented max-F1 must reproduce the run's own validation-tuned thresholds
    # on the same pooled rows, or these scores are not what the run tuned on.
    saved = json.loads((MODEL_DIR / "validation_tuned_thresholds.json").read_text())
    ours = [max_fbeta(y_val[:, i], s_val[:, i], 1.0).threshold for i in range(len(LABELS))]
    mismatched = [label for label, t in zip(LABELS, ours) if np.float32(t) != np.float32(saved[label])]
    if mismatched:
        raise RuntimeError(f"max-F1 reimplementation does not reproduce the run's thresholds: {mismatched}")

    rules = POLICIES[POLICY]
    results = [fit_rule(rules[label], y_val[:, i], s_val[:, i]) for i, label in enumerate(LABELS)]
    thresholds = [float(np.float32(t)) for t in operational_thresholds(results)]
    OUTPUT_DIR.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps({
        "schema_version": 1,
        "policy_version": f"{POLICY}@research_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": RUN_DIR.name,
        "model_checkpoint_sha256": {name: sha256_file(MODEL_DIR / name) for name in ("best.pt", "inference.pt")},
        "validation_predictions_sha256": sha256_file(MODEL_DIR / "validation_predictions.csv"),
        "source_split": "validation (clean + augmented view of every record, pooled)",
        "label_order": list(LABELS),
        "score_definition": "post_sigmoid_float32_probability",
        "comparison_operator": ">=",
        "selected_thresholds": thresholds,
        "selection_methods": [r.method for r in results],
        "selection_params": [r.params for r in results],
        "status_per_label": [r.status if r.status == "ok" else f"{r.status}:fallback_0_5_unvalidated"
                             for r in results],
        "notes_per_label": [r.note for r in results],
        "max_f1_reproduces_run_thresholds": True,
        "caveat": "two correlated rows per record: Wilson bounds of NORM/3AVB/STEMI see ~2x effective n",
        "policy_source": "threshold-configurations/v1-rfpn/threshold_research/policies.py",
        "fit_code_sha256": sha256_file(Path(__file__)),
    }, indent=2))

    # Post-freeze report: test operating point per view, clinical_v1 vs the run's max-F1.
    test, y_test, s_test = load("test")
    tables = []
    for policy, values in ((POLICY, thresholds), ("max_f1", [saved[label] for label in LABELS])):
        for view in ("clean", "augmented"):
            rows = (test["variant_id"] == view).to_numpy()
            table = per_label_table(y_test[rows], s_test[rows], values, policy, f"test_{view}")
            tables.append(table)
    report = pd.concat(tables)
    report.to_csv(OUTPUT_DIR / "test_operating_points.csv", index=False)
    cols = ["split", "policy", "label", "threshold", "sensitivity", "specificity", "ppv", "f1"]
    print(json.dumps(dict(zip(LABELS, thresholds)), indent=1))
    print(report[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    exact = {}
    for view in ("clean", "augmented"):
        rows = (test["variant_id"] == view).to_numpy()
        exact[view] = float((predict(s_test[rows], thresholds) == y_test[rows]).all(axis=1).mean())
    print("exact-match accuracy, clinical_v1:", exact)


if __name__ == "__main__":
    main()
