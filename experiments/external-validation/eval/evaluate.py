#!/usr/bin/env python3
"""
Score the five trained CardioSentry checkpoints on the external test corpus.

    python evaluate.py --weights /content/drive/MyDrive/cardiosentry_runs_v1 \
                       --testset ../data/external_testset \
                       --out ../results

WHAT IS BEING ASKED
-------------------
The training corpus made one question unanswerable. Every STEMI positive came
from the Chongqing ACS set and no other source contributed one, so within that
corpus "detects ST elevation" and "detects the Chongqing scanner" imply exactly
the same predictions. The prototype's own report says so: the per-source probe
returns nan on the STEMI head because that source has no STEMI negatives and
every other source has no STEMI positives.

This corpus separates them. It contains STEMI positives from sources the models
never saw a STEMI label from, so the two hypotheses now predict OPPOSITE things:

    learnt ST elevation   ->  fires on LUDB and PTB-XL STEMI sheets
    learnt the scanner    ->  near-silent on them, because the paper is wrong

THE CONTROL THAT MAKES THE RESULT READABLE
------------------------------------------
A STEMI collapse on its own proves nothing. Any corpus shift - different
cardiograph, different filtering, different population - depresses every head
at once, and that is not a confound, it is domain shift.

AF, LVH and NORMAL are the control. Those three had multi-source training
(PTB-XL and Chapman both contribute all three), so they carry no source
confound, and they ride on the identical sheets through the identical renderer.
That gives a clean two-by-two:

    STEMI falls, AF/LVH/NORMAL hold      -> the confound is real and specific.
                                            The STEMI head learnt the source.
    everything falls together            -> generic domain shift. Says nothing
                                            about the confound either way, and
                                            the experiment is inconclusive.
    STEMI holds                          -> the head generalises. The confound
                                            existed in the data but the model
                                            did not exploit it.

Every number this script prints is in service of reading that table. Nothing
here can be interpreted from the STEMI row alone.

THRESHOLDS ARE NEVER RE-TUNED
-----------------------------
Each checkpoint carries the per-head thresholds frozen from its ORIGINAL
validation split, and those are what get applied. Re-tuning on this corpus
would be tuning on test - it would hand a model that ranks STEMI sheets barely
above chance a threshold that manufactures respectable sensitivity, and it
would destroy exactly the comparison being made. Threshold-free metrics (AUROC,
AUPRC) are reported alongside for the same reason: they cannot be rescued by an
operating point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# The training code is the source of truth for transforms, model construction
# and metric definitions. Reimplementing any of them would mean the external
# numbers are not comparable with the numbers they are being contrasted against.
TRAIN_CODE = Path(os.environ.get("CARDIOSENTRY_CODE",
                                 REPO / "prototype-code" / "cardiosentry_runs_v1"))
if not (TRAIN_CODE / "models.py").exists():
    sys.exit(f"training code not found at {TRAIN_CODE}\n"
             "  set CARDIOSENTRY_CODE to the folder holding models.py / metrics.py")
sys.path.insert(0, str(TRAIN_CODE))

# config.py and metrics.py are pure numpy/stdlib. torch, torchvision, timm and
# the training code's data.py/models.py are imported LAZILY inside the two
# functions that need them, so the cohort logic, the metric plumbing and
# `--help` all work on a machine with no deep-learning stack installed - which
# is how selftest.py exercises this file before it ever reaches Colab.
import config as C          # noqa: E402
import metrics as M         # noqa: E402

LABELS = C.LABELS
BOOTSTRAP = 2000


# ====================================================================== loading
def load_index(testset: Path) -> pd.DataFrame:
    idx = testset / "index.csv"
    if not idx.exists():
        sys.exit(f"missing {idx} - run build/03_merge.py")
    df = pd.read_csv(idx)
    df["abs_path"] = df["rel_path"].map(lambda p: str(testset / p))
    for lab in LABELS:
        df[lab] = df[lab].astype("float32")
    return df


def check_testset(df: pd.DataFrame) -> None:
    """The subset of manifest.check_integrity that still applies here.

    Two of its checks are deliberately dropped rather than satisfied:

      * "every row must carry at least one positive label" - on an evaluation
        corpus an all-negative sheet is not a defect, it is the most useful
        kind of negative there is. 59 of them are here on purpose.
      * "all three splits present" - this corpus is test and nothing else.

    Everything that protects the validity of a score is kept, and one check is
    ADDED that the training version has no need for: no image may be duplicated,
    since a repeated sheet would silently double a patient's weight.
    """
    problems = []

    per_record = df.groupby("record").size()
    if (per_record != 1).any():
        problems.append(f"{int((per_record != 1).sum())} record(s) have more than "
                        f"one sheet; evaluation needs exactly 1 canonical image per ECG")

    per_patient = df.groupby("patient_id").size()
    dupes = per_patient[per_patient > 1]
    if len(dupes):
        # Not fatal - LUDB is one record per subject and PTB-XL selection is
        # near enough - but a patient contributing several sheets is weighted
        # several times, so it has to be visible.
        print(f"  note: {len(dupes)} patient(s) contribute more than one sheet "
              f"(max {int(dupes.max())}); they carry proportionally more weight")

    path_cols = [l for l in LABELS if l != "NORMAL"]
    both = df[(df["NORMAL"] == 1) & (df[path_cols].sum(axis=1) > 0)]
    if len(both):
        problems.append(f"{len(both)} row(s) are NORMAL and also carry a pathology")

    missing = [p for p in df["abs_path"] if not Path(p).exists()]
    if missing:
        problems.append(f"{len(missing)} image(s) missing, e.g. {missing[:3]}")

    # The sentinel from the unlabelled render pass must not have survived the
    # merge. If it did, 59 hard negatives are scored as NORMAL positives and
    # every specificity figure is wrong.
    if "label_confidence" in df.columns:
        sent = df[(df["NORMAL"] == 1) & (df["n_positive_labels"] == 0)]
        if len(sent):
            problems.append(f"{len(sent)} row(s) are NORMAL=1 with zero positives - "
                            f"the render-pass sentinel leaked through 03_merge.py")

    if problems:
        raise RuntimeError("EXTERNAL TEST SET INTEGRITY CHECK FAILED:\n  - "
                           + "\n  - ".join(problems))


# =================================================================== inference
def infer(model, df: pd.DataFrame, device, micro_batch: int) -> np.ndarray:
    """Sigmoid probabilities, [n, 4]. Eval transforms only - see data.build_transforms."""
    import torch
    import data as D                      # training code's loader; needs torchvision

    with torch.no_grad():
        ds = D.ECGSheetDataset(df, train=False)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=micro_batch, shuffle=False, num_workers=2,
            pin_memory=torch.cuda.is_available())
        model.eval()
        out = []
        amp = torch.cuda.is_available() and C.AMP
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                logits = model(images)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_checkpoint(path: Path, name: str, device):
    import torch
    import models as MO

    blob = torch.load(path, map_location="cpu", weights_only=False)
    # pretrained=False: the ImageNet weights are about to be overwritten, and
    # downloading them on a Colab runtime just to discard them wastes minutes.
    model, info = MO.build(name, pretrained=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.to(device)

    thr = blob.get("thresholds") if isinstance(blob, dict) else None
    if not thr:
        # Silently defaulting to 0.5 would make this model's sensitivity
        # incomparable with the others', which is worse than stopping.
        raise ValueError(
            f"{path} carries no frozen thresholds. They were tuned on the "
            f"original validation split and must not be re-derived here; "
            f"without them this model cannot be compared with the rest.")
    return model, list(thr), info, blob


# ===================================================================== cohorts
def cohort_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The slices the report is built from.

    Named rather than derived on the fly, because several of them encode a
    clinical argument and that argument should be readable in one place.
    """
    src = df["source"].to_numpy()
    labs = {l: df[l].to_numpy() for l in LABELS}
    text = df["original_labels"].fillna("").str.lower().to_numpy()
    n_pos = df["n_positive_labels"].to_numpy()

    masks = {
        "ALL": np.ones(len(df), dtype=bool),
        # The gold cohort: a source seen in no class, with STEMI asserted in
        # words by the reporting cardiologist.
        "LUDB": src == "LUDB",
        # Proxy: a source seen extensively, but never once labelled STEMI.
        "PTBXL_MI": src == "PTBXL_MI",
        "PTBXL_NORM": src == "PTBXL_NORM",
        "PTBXL_ALL": np.isin(src, ["PTBXL_MI", "PTBXL_NORM"]),
    }

    # STEMI's differential, and the sharpest test in the corpus. Early
    # repolarisation is BENIGN ST elevation - a sheet that genuinely shows the
    # elevation a STEMI shows, on a patient who is not infarcting. A model
    # reading ST morphology may reasonably be fooled. A model reading paper
    # cannot be fooled, because the paper is not Chongqing paper. High
    # specificity here is therefore NOT automatically good news, and the
    # report has to say so.
    def contains(*needles: str) -> np.ndarray:
        return np.array([any(n in t for n in needles) for t in text], dtype=bool)

    masks["MIMIC_early_repol"] = contains("early repolarization")
    masks["MIMIC_repol_abnorm"] = contains("non-specific repolarization")
    masks["MIMIC_ischemia_scar"] = contains("ischemia", "scar", "nstemi")
    # All negatives that are nonetheless abnormal: the honest denominator for
    # "does it fire on anything that looks unwell?"
    masks["HARD_NEG"] = (labs["STEMI"] == 0) & (n_pos == 0)
    masks["STEMI_NEG_ALL"] = labs["STEMI"] == 0
    return masks


# =================================================================== bootstrap
def boot_ci(y: np.ndarray, p: np.ndarray, fn, n: int = BOOTSTRAP,
            seed: int = C.SEED) -> tuple[float, float]:
    """Percentile CI, resampling sheets with replacement.

    With eight gold positives the point estimate is nearly uninformative on its
    own, so nothing from the LUDB STEMI head should ever be quoted without the
    interval that comes back from here - it will be very wide, and that width
    is the honest finding.
    """
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(float(fn(y[idx], p[idx])))
        except ValueError:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def head_stats(y: np.ndarray, prob: np.ndarray, thr: float, with_ci: bool) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = dict(M.binary_metrics(y, prob, thr))
    out["n"] = int(len(y))
    out["n_pos"] = int(y.sum())
    out["n_neg"] = int((y == 0).sum())
    out["threshold"] = float(thr)
    out["mean_prob_pos"] = float(prob[y == 1].mean()) if y.sum() else float("nan")
    out["mean_prob_neg"] = float(prob[y == 0].mean()) if (y == 0).sum() else float("nan")
    if with_ci:
        lo, hi = boot_ci(y, prob, roc_auc_score)
        out["auroc_lo"], out["auroc_hi"] = lo, hi
        lo, hi = boot_ci(y, prob, average_precision_score)
        out["auprc_lo"], out["auprc_hi"] = lo, hi
    return out


# ======================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="run root holding <model>/weights/best.pt "
                         "(e.g. /content/drive/MyDrive/cardiosentry_runs_v1)")
    ap.add_argument("--testset", default=str(HERE.parent / "data" / "external_testset"))
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names (default: everything found)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Deferred to here rather than module scope so the cohort and metric code
    # above stays importable without a deep-learning stack; see the note beside
    # the config/metrics imports.
    import torch
    import models as MO

    testset, out = Path(args.testset), Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    df = load_index(testset)
    check_testset(df)
    masks = cohort_masks(df)
    y_true = df[LABELS].to_numpy().astype(int)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}   sheets: {len(df)}   "
          f"STEMI+: {int(y_true[:, LABELS.index('STEMI')].sum())} "
          f"(LUDB {int(y_true[masks['LUDB'], LABELS.index('STEMI')].sum())} gold / "
          f"PTB-XL {int(y_true[masks['PTBXL_MI'], LABELS.index('STEMI')].sum())} proxy)")
    print(f"config fingerprint: {C.fingerprint()}")

    root = Path(args.weights)
    names = args.models or [m for m in MO.MODEL_NAMES
                            if (root / m / "weights" / "best.pt").exists()]
    if not names:
        sys.exit(f"no <model>/weights/best.pt under {root}\n"
                 f"  looked for: {', '.join(MO.MODEL_NAMES)}")
    print(f"models: {', '.join(names)}\n")

    results: dict[str, dict] = {}
    for name in names:
        ckpt = root / name / "weights" / "best.pt"
        print(f"--- {name}")
        model, thresholds, info, blob = load_checkpoint(ckpt, name, device)
        print(f"    checkpoint epoch {blob.get('epoch', '?')}  "
              f"thresholds {[round(t, 3) for t in thresholds]}  (frozen on original val)")

        prob = infer(model, df, device, MO.micro_batch(name))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pred_df = df[["record", "source", "cohort", "label_confidence",
                      "seen_in_training", "original_id", "printed_id"]].copy()
        for k, lab in enumerate(LABELS):
            pred_df[f"y_{lab}"] = y_true[:, k]
            pred_df[f"p_{lab}"] = prob[:, k]
            pred_df[f"pred_{lab}"] = (prob[:, k] >= thresholds[k]).astype(int)
        pred_df.to_csv(out / "predictions" / f"{name}.csv", index=False)

        entry = {"thresholds": thresholds, "epoch": blob.get("epoch"),
                 "params_m": info.get("params_m_actual"), "cohorts": {}}
        for cname, mask in masks.items():
            if mask.sum() == 0:
                continue
            # CIs only where they will be read; 2000 resamples x every cohort
            # x every head is minutes of pure overhead otherwise.
            with_ci = cname in {"ALL", "LUDB", "PTBXL_MI", "PTBXL_ALL"}
            per_head = {}
            for k, lab in enumerate(LABELS):
                per_head[lab] = head_stats(y_true[mask, k], prob[mask, k],
                                           thresholds[k], with_ci)
            macro = {}
            for key in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
                vals = [per_head[l][key] for l in LABELS
                        if not np.isnan(per_head[l][key])]
                macro[f"macro_{key}"] = float(np.mean(vals)) if vals else float("nan")
            entry["cohorts"][cname] = {"n": int(mask.sum()), "per_label": per_head,
                                       **macro}
        results[name] = entry

        s = entry["cohorts"]
        gold = s.get("LUDB", {}).get("per_label", {}).get("STEMI", {})
        proxy = s.get("PTBXL_MI", {}).get("per_label", {}).get("STEMI", {})
        print(f"    STEMI  gold(LUDB)  sens {gold.get('sensitivity', float('nan')):.3f}  "
              f"AUROC {gold.get('auroc', float('nan')):.3f}  "
              f"mean p(pos) {gold.get('mean_prob_pos', float('nan')):.3f}")
        print(f"    STEMI  proxy(PTBXL) sens {proxy.get('sensitivity', float('nan')):.3f}  "
              f"mean p(pos) {proxy.get('mean_prob_pos', float('nan')):.3f}")
        ctl = s.get("ALL", {})
        print(f"    control heads (ALL): " + "  ".join(
            f"{l} AUROC {ctl['per_label'][l]['auroc']:.3f}"
            for l in ("AF", "LVH", "NORMAL")))
        print()

    payload = {
        "config_fingerprint": C.fingerprint(),
        "labels": LABELS,
        "testset": str(testset),
        "weights_root": str(root),
        "n_sheets": int(len(df)),
        "cohort_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "positives": {k: {lab: int(y_true[v, i].sum()) for i, lab in enumerate(LABELS)}
                      for k, v in masks.items()},
        "bootstrap_resamples": BOOTSTRAP,
        "models": results,
    }
    (out / "external_metrics.json").write_text(json.dumps(payload, indent=2))

    rows = []
    for name, entry in results.items():
        for cname, c in entry["cohorts"].items():
            for lab, m in c["per_label"].items():
                rows.append({"model": name, "cohort": cname, "label": lab, **m})
    pd.DataFrame(rows).to_csv(out / "external_metrics.csv", index=False)

    print(f"wrote {out / 'external_metrics.json'}")
    print(f"      {out / 'external_metrics.csv'}")
    print(f"      {out / 'predictions'}/<model>.csv")
    print("\nnow run:  python report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
