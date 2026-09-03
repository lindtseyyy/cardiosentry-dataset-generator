#!/usr/bin/env python3
"""
Compare the LVH-only binary run against the shared four-head reference -
and first PROVE the comparison is valid.

    python compare.py --runs runs/ --reference ../cardiosentry_hybrids_v1/runs

THE QUESTION THIS ROUND ASKS

Did removing STEMI, AF and NORMAL from the shared training objective improve
LVH? The reference is round 2's `resnet50_hires`: ResNet-50@768 with four
sigmoid heads, trained on the same frozen splits, same protocol, same bytes.
The only intended difference is the task (1 head vs 4).

THREE CHECKS BEFORE ANY NUMBER IS PRINTED AS A COMPARISON

  1. SHARED PROTOCOL - the binary run's protocol settings, with the task
     fields removed, must equal the reference's recorded settings field by
     field. Any change to seed, augmentation, optimiser, schedule, effective
     batch, epochs, monitor or threshold rule refuses the comparison.
  2. FROZEN REFERENCE - when a live reference result.json is provided, its
     LVH numbers must equal the frozen transcription in baseline.py. A moving
     reference cannot be trusted to anchor a delta.
  3. DATASET IDENTITY - same row counts, same data root, same micro-batch,
     same views, same input fingerprint. BatchNorm normalises over the
     micro-batch, so a micro-batch mismatch is a real confound, not a VRAM
     detail.

PRIMARY READ

PTB-XL LVH AUPRC. The STEMI database contributes 146 images and ZERO LVH
positives, so a model that merely learns to recognise that database cleans up
its pooled false positives without reading a single QRS complex. PTB-XL
carries 234 of the 255 LVH positives and is internally single-source - the
one subgroup the shortcut cannot reach.

Paired bootstrap CIs (1,200 resamples, same indices for both models) are
computed on the saved per-image predictions for AUPRC, AUROC, sensitivity,
specificity, FPR, precision and F1 - threshold metrics at each model's OWN
val-tuned frozen threshold, exactly as reported.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

import baseline
import bootstrap
import config as C
import subgroup

TASK_KEYS = ("LABELS", "NUM_CLASSES")

METRIC_NAMES = {"auprc": "AUPRC", "auroc": "AUROC", "sensitivity": "SENS",
                "specificity": "SPEC", "precision": "PREC", "f1": "F1",
                "fpr": "FPR"}


# ---------------------------------------------------------------- loading
def load_runs(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
            r["_dir"] = f.parent
            out.append(r)
        except Exception as exc:
            print(f"  skipping unreadable {f}: {exc}")
    return out


def find_reference(path: Path | None) -> Path | None:
    """Locate the four-head reference result.json: a file, a run dir, or the
    hybrid_v1 round folder."""
    if path is None:
        return None
    path = Path(path)
    if path.is_file():
        return path
    for cand in (path / f"{baseline.REFERENCE_MODEL}/result.json",
                 path / "result.json"):
        if cand.exists():
            return cand
    return None


def load_live_reference(path: Path | None) -> dict | None:
    """Return the live reference result.json contents, or None to fall back to
    the frozen baseline.py transcription."""
    p = find_reference(path)
    if p is None:
        if path is not None:
            print(f"  NOTE: no reference result.json found under {path} - "
                  f"using the frozen transcription in baseline.py.")
        return None
    return json.loads(p.read_text())


def load_predictions(run_dir: Path) -> dict | None:
    """{record: (true_lvh, prob_lvh, pred_lvh, source)} from the saved CSV."""
    import pandas as pd
    p = Path(run_dir) / "predictions" / "test_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    need = {"record", "source", "true_LVH", "prob_LVH", "pred_LVH"}
    if not need <= set(df.columns):
        return None
    df = df.drop_duplicates(subset="record", keep="first")
    return df.set_index("record").to_dict("index")


# ---------------------------------------------------------------- protocol checks
def shared_settings_of(run: dict) -> dict:
    s = dict(run.get("protocol_settings") or {})
    for k in TASK_KEYS:
        s.pop(k, None)
    return s


def _rehash(settings: dict) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _normalize(v):
    """JSON round-trip so tuples from live config objects and lists from
    result.json compare equal - the hashes already treat them identically."""
    return json.loads(json.dumps(v, default=str))


def check_comparable(binary_runs: list[dict], live_ref: dict | None) -> tuple[bool, list[str]]:
    notes, ok = [], True

    fps = {r["model"]: r.get("protocol_fingerprint") for r in binary_runs}
    if len(set(fps.values())) > 1:
        ok = False
        notes.append("BINARY RUNS DISAGREE ON PROTOCOL - refusing:")
        for m, fp in fps.items():
            notes.append(f"    {m:24} {fp}")

    # 1. shared protocol vs the reference (live settings, else frozen)
    ref_settings = (live_ref.get("protocol_settings")
                    if live_ref else baseline.REFERENCE_PROTOCOL_SETTINGS)
    ref_shared = _normalize({k: v for k, v in dict(ref_settings).items()
                             if k not in TASK_KEYS})
    ref_hash = _rehash(ref_shared)
    for r in binary_runs:
        mine = _normalize(shared_settings_of(r))
        my_hash = _rehash(mine)
        if my_hash != ref_hash:
            ok = False
            notes.append(f"{r['model']}: SHARED PROTOCOL MISMATCH vs the reference")
            notes.append(f"    binary   {my_hash}")
            notes.append(f"    reference {ref_hash}")
            for k in sorted(set(ref_shared) | set(mine)):
                if ref_shared.get(k) != mine.get(k):
                    notes.append(f"    {k}: binary {mine.get(k)!r} != "
                                 f"reference {ref_shared.get(k)!r}")
        if my_hash != baseline.REFERENCE_SHARED_FINGERPRINT:
            ok = False
            notes.append(f"{r['model']}: shared fingerprint {my_hash} != frozen "
                         f"{baseline.REFERENCE_SHARED_FINGERPRINT} in baseline.py")

    # 2. frozen cross-check, only when a live reference exists
    if live_ref:
        lt = live_ref["test"]["per_label"]["LVH"]
        lp = live_ref["test_per_source"]["PTBXL"]["per_label"]["LVH"]
        for key, (frozen, live) in {
            "pooled AUPRC": (baseline.REFERENCE_LVH["pooled"]["auprc"], lt["auprc"]),
            "pooled AUROC": (baseline.REFERENCE_LVH["pooled"]["auroc"], lt["auroc"]),
            "PTBXL AUPRC": (baseline.REFERENCE_LVH["PTBXL"]["auprc"], lp["auprc"]),
            "PTBXL AUROC": (baseline.REFERENCE_LVH["PTBXL"]["auroc"], lp["auroc"]),
        }.items():
            if abs(frozen - live) > 1e-6:
                ok = False
                notes.append(f"LIVE REFERENCE DISAGREES WITH FROZEN on {key}: "
                             f"live {live:.6f} vs frozen {frozen:.6f}")
        if live_ref.get("micro_batch") != baseline.REFERENCE_META["micro_batch"]:
            ok = False
            notes.append(f"live reference micro_batch {live_ref.get('micro_batch')} "
                         f"!= frozen {baseline.REFERENCE_META['micro_batch']}")

    # 3. dataset identity
    for r in binary_runs:
        d = r.get("data", {})
        if live_ref:
            rd = live_ref.get("data", {})
            for field in ("rows", "records", "patients"):
                if d.get(field) != rd.get(field):
                    ok = False
                    notes.append(f"{r['model']}: {field} {d.get(field)} != reference "
                                 f"{rd.get(field)} - different dataset")
            if Path(d.get("root", "")).name != Path(rd.get("root", "")).name:
                ok = False
                notes.append(f"{r['model']}: data root {d.get('root')} != reference "
                             f"{rd.get('root')}")
        else:
            if d.get("rows") not in (None, 6670):
                ok = False
                notes.append(f"{r['model']}: {d.get('rows')} rows - expected 6670")
        if r.get("micro_batch") != baseline.REFERENCE_META["micro_batch"]:
            ok = False
            notes.append(f"{r['model']}: micro_batch {r.get('micro_batch')} != "
                         f"reference {baseline.REFERENCE_META['micro_batch']} - "
                         f"BatchNorm statistics differ; the comparison would be confounded")
        views = "+".join(map(str, r.get("model_info", {}).get("views", [])))
        if views != "768":
            ok = False
            notes.append(f"{r['model']}: views {views} != reference 768")
        if r.get("input_fingerprint") != baseline.REFERENCE_META["input_fingerprint"]:
            ok = False
            notes.append(f"{r['model']}: input fingerprint "
                         f"{r.get('input_fingerprint')} != reference "
                         f"{baseline.REFERENCE_META['input_fingerprint']}")

    # declared differences, stated so nobody reads them as faults
    notes.append("  DECLARED DIFFERENCE (the experiment):")
    for r in binary_runs:
        ts = r.get("task_settings") or {}
        notes.append(f"    {r['model']:24} labels {ts.get('LABELS')}   |   "
                     f"reference labels {baseline.REFERENCE_PROTOCOL_SETTINGS['LABELS']}")
    return ok, notes


# ---------------------------------------------------------------- metrics access
def lvh_metrics(run: dict, subgroup_name: str | None = None) -> dict:
    """LVH metrics for one run; subgroup None = pooled test."""
    if subgroup_name is None:
        return run["test"]["per_label"]["LVH"]
    ps = run.get("test_per_source", {})
    if subgroup_name not in ps:
        return {}
    return ps[subgroup_name]["per_label"]["LVH"]


def reference_metrics(subgroup_name: str | None = None) -> dict:
    if subgroup_name is None:
        return baseline.REFERENCE_LVH["pooled"]
    if subgroup_name == "PTBXL":
        return baseline.REFERENCE_LVH["PTBXL"]
    return baseline.REFERENCE_LVH.get(subgroup_name, {})


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", type=Path,
                    help="folder containing the binary run(s)")
    ap.add_argument("--reference", default=None, type=Path,
                    help="the four-head round folder, run dir or result.json "
                         "(default: frozen transcription in baseline.py)")
    ap.add_argument("--csv", type=Path, help="also write the headline table as CSV")
    ap.add_argument("--n-resamples", type=int, default=bootstrap.N_RESAMPLES,
                    help=f"bootstrap resamples (default {bootstrap.N_RESAMPLES})")
    ap.add_argument("--force", action="store_true",
                    help="print the table even if the checks fail")
    a = ap.parse_args()

    binary_runs = load_runs(a.runs)
    if not binary_runs:
        print(f"no result.json under {a.runs}/ - train the binary model first")
        return 1
    live_ref = load_live_reference(a.reference)

    out_lines = []
    def emit(line=""):
        print(line)
        out_lines.append(line)

    emit("=" * 108)
    emit("LVH BINARY vs FOUR-HEAD SHARED - is the multi-label objective hurting LVH?")
    emit("=" * 108)
    emit(f"  reference: {baseline.REFERENCE_MODEL} (round 2, {baseline.REFERENCE_TASK})")
    emit(f"            ResNet-50@768, micro-batch 8, "
         f"{baseline.REFERENCE_META['train_min']:.1f} min, "
         f"{baseline.REFERENCE_META['epochs_run']} epochs")
    if live_ref:
        emit(f"  live reference loaded from {find_reference(a.reference)}")
    else:
        emit("  reference numbers are the FROZEN transcription in baseline.py")

    ok, notes = check_comparable(binary_runs, live_ref)
    for n in notes:
        emit(n)
    if not ok and not a.force:
        emit()
        emit("REFUSING to compare - fix the above, or pass --force.")
        return 2
    if ok:
        emit("comparability: OK - shared protocol matches the four-head reference;")
        emit("              the ONLY difference is the task (1 head vs 4).")
    emit()

    # reference test predictions, for the paired bootstrap and matched-FPR reads
    ref_preds = None
    if live_ref and find_reference(a.reference):
        ref_preds = load_predictions(find_reference(a.reference).parent)

    # ---------------------------------------------------------------- headline
    emit("=" * 108)
    emit("POOLED TEST LVH  (val-tuned threshold, frozen - 938 images, 255 positives)")
    emit("=" * 108)
    emit("  shown for context. THE PRIMARY READ IS THE PTB-XL TABLE BELOW: the")
    emit("  STEMI database contributes 146 LVH-negative images, so pooled deltas")
    emit("  can be database recognition, not ECG reading.")
    emit()
    hdr = (f"{'model':24}{'task':24}{'AUPRC':>9}{'d':>8}{'AUROC':>9}{'d':>8}"
           f"{'Sens':>8}{'Spec':>8}{'Prec':>8}{'F1':>8}{'FPR':>8}"
           f"{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}")
    emit(hdr); emit("-" * len(hdr))
    rp = reference_metrics()
    emit(f"{baseline.REFERENCE_MODEL:24}{baseline.REFERENCE_TASK:24}"
         f"{rp['auprc']:9.4f}{'':>8}{rp['auroc']:9.4f}{'':>8}"
         f"{rp['sensitivity']:8.4f}{rp['specificity']:8.4f}{rp['precision']:8.4f}"
         f"{rp['f1']:8.4f}{rp['fpr']:8.4f}"
         f"{rp['tp']:5d}{rp['fp']:5d}{rp['fn']:5d}{rp['tn']:5d}")
    csv_rows = []
    for r in binary_runs:
        m = lvh_metrics(r)
        emit(f"{r['model']:24}{'LVH-only binary':24}"
             f"{m['auprc']:9.4f}{m['auprc']-rp['auprc']:+8.4f}"
             f"{m['auroc']:9.4f}{m['auroc']-rp['auroc']:+8.4f}"
             f"{m['sensitivity']:8.4f}{m['specificity']:8.4f}{m['precision']:8.4f}"
             f"{m['f1']:8.4f}{m['fpr']:8.4f}"
             f"{m['tp']:5d}{m['fp']:5d}{m['fn']:5d}{m['tn']:5d}")
        csv_rows.append({"model": r["model"], "subset": "pooled",
                         **{k: m[k] for k in METRIC_NAMES}})
    emit()
    emit("  Read the FPR delta before the AUPRC delta. A model that gains")
    emit("  sensitivity and gains FPR has moved its operating point, not improved.")
    emit("  Round 1's LVH failure was FALSE-POSITIVE dominated (333 FP vs 229 FN")
    emit("  over five backbones).")

    # ---------------------------------------------------------------- PTB-XL
    emit()
    emit("=" * 108)
    emit("PTB-XL LVH  -  THE PRIMARY READ  (619 images, 234 positives)")
    emit("=" * 108)
    emit("  PTB-XL carries 234 of the 255 LVH positives and is internally")
    emit("  single-source, so this is the subgroup the database shortcut cannot")
    emit("  reach. Rank on AUPRC here, not on the pooled column above.")
    emit()
    hdr = (f"{'model':24}{'AUPRC':>9}{'d':>8}{'AUROC':>9}{'d':>8}{'Sens':>8}"
           f"{'Spec':>8}{'Prec':>8}{'F1':>8}{'FPR':>8}{'sens@.08':>10}")
    emit(hdr); emit("-" * len(hdr))
    rq = reference_metrics("PTBXL")
    emit(f"{baseline.REFERENCE_MODEL:24}{rq['auprc']:9.4f}{'':>8}"
         f"{rq['auroc']:9.4f}{'':>8}{rq['sensitivity']:8.4f}"
         f"{rq['specificity']:8.4f}{rq['precision']:8.4f}{rq['f1']:8.4f}"
         f"{rq['fpr']:8.4f}{rq.get('sens_at_matched_fpr', float('nan')):10.4f}")
    for r in binary_runs:
        m = lvh_metrics(r, "PTBXL")
        csv_rows.append({"model": r["model"], "subset": "PTBXL",
                         **{k: m.get(k, float("nan")) for k in METRIC_NAMES}})
        # sens@.08 needs per-image predictions; the reference's is frozen.
        s08 = float("nan")
        bin_preds = load_predictions(r["_dir"])
        if bin_preds is not None:
            rows = [v for v in bin_preds.values() if v["source"] == "PTBXL"]
            s08 = subgroup.sens_at_fpr(
                np.array([v["true_LVH"] for v in rows], dtype=float),
                np.array([v["prob_LVH"] for v in rows], dtype=float),
                subgroup.MATCHED_FPR)
        emit(f"{r['model']:24}{m['auprc']:9.4f}{m['auprc']-rq['auprc']:+8.4f}"
             f"{m['auroc']:9.4f}{m['auroc']-rq['auroc']:+8.4f}"
             f"{m['sensitivity']:8.4f}{m['specificity']:8.4f}"
             f"{m['precision']:8.4f}{m['f1']:8.4f}{m['fpr']:8.4f}{s08:10.4f}")
    emit()
    emit(subgroup.MATCHED_FPR_NOTE)

    # ---------------------------------------------------------------- bootstrap
    emit()
    emit("=" * 108)
    emit(f"PAIRED BOOTSTRAP  ({a.n_resamples:,} resamples, same indices for both models)")
    emit("=" * 108)
    emit("  deltas are binary - four-head, so positive = the LVH-only model wins.")
    emit("  threshold metrics use each model's OWN val-tuned frozen threshold.")

    for subset_label, key in (("POOLED", None), ("PTB-XL", "PTBXL")):
        ref_key = reference_metrics(key)
        for r in binary_runs:
            mine = lvh_metrics(r, key)
            # per-source blocks carry no threshold: the operating point is the
            # model's OWN pooled val-tuned frozen threshold, same for reference
            thr_b = (r["test"]["per_label"]["LVH"].get("threshold")
                     if key is not None else mine.get("threshold", 0.5))
            thr_s = (baseline.REFERENCE_LVH["pooled"]["threshold"]
                     if key is not None
                     else ref_key.get("threshold", baseline.REFERENCE_LVH["pooled"]["threshold"]))
            # need per-image predictions from BOTH models for the paired draw
            bin_preds = load_predictions(r["_dir"])
            if bin_preds is None or ref_preds is None:
                emit(f"\n  {subset_label}, {r['model']}: per-image predictions missing "
                     f"(binary {bin_preds is not None}, reference {ref_preds is not None}) "
                     f"- bootstrap skipped.")
                continue
            records = list(bin_preds.keys())
            if not set(records) == set(ref_preds.keys()):
                emit(f"\n  {subset_label}, {r['model']}: prediction rows do not align "
                     f"between runs - bootstrap skipped.")
                continue
            y = np.array([bin_preds[rc]["true_LVH"] for rc in records], dtype=float)
            pb = np.array([bin_preds[rc]["prob_LVH"] for rc in records], dtype=float)
            ps = np.array([ref_preds[rc]["prob_LVH"] for rc in records], dtype=float)
            if key == "PTBXL":
                mask = np.array([bin_preds[rc]["source"] == "PTBXL"
                                 for rc in records])
                y, pb, ps = y[mask], pb[mask], ps[mask]
            emit(f"\n  {subset_label}  ({len(y)} images, {int(y.sum())} positives)")
            emit(f"  {'metric':14}{'four-head':>11}{'binary':>11}{'delta':>9}"
                 f"{'95% CI':>22}{'P(bin>shared)':>15}")
            for row in bootstrap.bootstrap_table(
                    y, pb, ps, thr_b, thr_s,
                    metrics=("auprc", "auroc", "sensitivity", "specificity",
                             "fpr", "precision", "f1"),
                    n_resamples=a.n_resamples):
                nice = METRIC_NAMES[row["metric"]]
                emit(f"  {nice:14}{row['shared']:11.4f}{row['binary']:11.4f}"
                     f"{row['delta']:+9.4f}"
                     f"{'['+f'{row['ci_low']:.4f}'+', '+f'{row['ci_high']:.4f}'+']':>22}"
                     f"{row['p_binary_better']:15.3f}")

    # ---------------------------------------------------------------- verdict
    emit()
    emit("=" * 108)
    emit("VERDICT - does this support NEGATIVE TRANSFER from the multi-label setup?")
    emit("=" * 108)
    emit("  Rule, stated before the numbers so the call is not made post hoc:")
    emit("    * PTB-XL LVH AUPRC delta CI entirely above +0.005  -> SUPPORTS")
    emit("    * point estimate positive but CI includes 0         -> NOT SEPARATED")
    emit("    * point estimate at or below 0                      -> NO EVIDENCE")
    for r in binary_runs:
        m = lvh_metrics(r, "PTBXL")
        dq = m["auprc"] - rq["auprc"]
        dp = lvh_metrics(r)["auprc"] - rp["auprc"]
        ci = None
        bin_preds = load_predictions(r["_dir"])
        if bin_preds is not None and ref_preds is not None and \
                set(bin_preds) == set(ref_preds):
            records = list(bin_preds.keys())
            y = np.array([bin_preds[rc]["true_LVH"] for rc in records], dtype=float)
            pb = np.array([bin_preds[rc]["prob_LVH"] for rc in records], dtype=float)
            ps = np.array([ref_preds[rc]["prob_LVH"] for rc in records], dtype=float)
            mask = np.array([bin_preds[rc]["source"] == "PTBXL" for rc in records])
            ci = bootstrap.paired_delta(
                y[mask], pb[mask], ps[mask],
                r["test"]["per_label"]["LVH"].get("threshold", 0.5),
                baseline.REFERENCE_LVH["pooled"]["threshold"],
                "auprc", n_resamples=a.n_resamples)
        emit(f"\n  {r['model']}:")
        emit(f"    PTB-XL LVH AUPRC   four-head {rq['auprc']:.4f}  ->  binary "
             f"{m['auprc']:.4f}   delta {dq:+.4f}"
             + (f"   95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
                f"P(bin>shared)={ci['p_binary_better']:.3f}" if ci else ""))
        emit(f"    pooled   LVH AUPRC four-head {rp['auprc']:.4f}  ->  binary "
             f"{lvh_metrics(r)['auprc']:.4f}   delta {dp:+.4f}")
        share = subgroup.shortcut_share(dp, dq)
        if share == share:
            emit(f"    shortcut share {share*100:.0f}% - the fraction of the pooled "
                 f"gain that does NOT reproduce inside PTB-XL")
        if ci and ci["ci_low"] > 0.005:
            verdict = ("SUPPORTS negative transfer: the LVH-only objective "
                       "improves PTB-XL LVH AUPRC beyond the noise floor")
        elif dq > 0:
            verdict = ("NOT SEPARATED: positive point estimate, CI includes 0 - "
                       "weak, directional evidence only")
        else:
            verdict = ("NO EVIDENCE of negative transfer: removing STEMI/AF/NORMAL "
                       "did not help LVH inside PTB-XL")
        emit(f"    -> {verdict}")
        # operating-point check
        ds = m["sensitivity"] - rq["sensitivity"]
        dfpr = m["fpr"] - rq["fpr"]
        if ds > 0.005 and dfpr > 0.005:
            emit(f"    NOTE: binary gains sensitivity {ds:+.4f} AND FPR {dfpr:+.4f} "
                 f"inside PTB-XL - part of the change is an operating-point move,")
            emit(f"          not a better detector. The AUPRC delta and sens@.08 "
                 f"read answer that; F1 at frozen thresholds does not.")
    emit()
    emit("  CONTEXT ROWS, same frozen test split:")
    emit(f"    round 1 resnet50@384 (4 heads): pooled LVH AUPRC "
         f"{baseline.ROUND1_RESNET50_LVH['auprc']:.4f}")
    emit(f"    round 2 resnet50_base@384 (4 heads): pooled LVH AUPRC "
         f"{baseline.ROUND2_BASE_384_LVH['auprc']:.4f}, PTB-XL "
         f"{baseline.ROUND2_BASE_384_LVH['PTBXL_auprc']:.4f}")
    emit(f"    round 2 resnet50_hires@768 (4 heads): pooled LVH AUPRC "
         f"{rp['auprc']:.4f}, PTB-XL {rq['auprc']:.4f}   <- THE REFERENCE")

    # ---------------------------------------------------------------- per source
    emit()
    emit("=" * 108)
    emit("LVH BY SOURCE - pooled gain is not automatically an LVH gain")
    emit("=" * 108)
    emit(f"  {'model':24}{'pooled AUPRC':>14}{'PTBXL':>9}{'CHAPMAN':>9}"
         f"{'STEMI-db FP':>13}")
    emit("  " + "-" * 60)
    emit(f"  {baseline.REFERENCE_MODEL:24}{rp['auprc']:14.4f}"
         f"{rq['auprc']:9.4f}{reference_metrics('CHAPMAN').get('auprc', float('nan')):9.4f}"
         f"{baseline.REFERENCE_LVH['STEMI']['fp']:13d}")
    for r in binary_runs:
        m = lvh_metrics(r)
        ps_ = r.get("test_per_source", {})
        ptb = ps_.get("PTBXL", {}).get("per_label", {}).get("LVH", {}).get("auprc",
                                                                          float("nan"))
        ch = ps_.get("CHAPMAN", {}).get("per_label", {}).get("LVH", {}).get("auprc",
                                                                           float("nan"))
        stemi_fp = ""
        if "STEMI" in ps_:
            d = ps_["STEMI"]["per_label"]["LVH"]
            stemi_fp = f"{d['fpr']*ps_['STEMI']['n']:.0f}/{ps_['STEMI']['n']}"
        emit(f"  {r['model']:24}{m['auprc']:14.4f}{ptb:9.4f}{ch:9.4f}{stemi_fp:>13}")

    # ---------------------------------------------------------------- cost
    emit()
    emit("=" * 108)
    emit("COST - the binary model should cost the same, since only the head shrank")
    emit("=" * 108)
    emit(f"  {'model':24}{'params M':>10}{'size MB':>10}{'ms/img b1':>11}"
         f"{'train min':>11}{'epochs':>8}")
    emit("  " + "-" * 60)
    emit(f"  {baseline.REFERENCE_MODEL:24}"
         f"{baseline.REFERENCE_META['params_total_m']:10.1f}"
         f"{baseline.REFERENCE_META['size_mb']:10.1f}"
         f"{baseline.REFERENCE_META['latency_ms_b1']:11.2f}"
         f"{baseline.REFERENCE_META['train_min']:11.1f}"
         f"{baseline.REFERENCE_META['epochs_run']:8d}")
    for r in binary_runs:
        c = r.get("complexity", {})
        b = r.get("inference", {})
        emit(f"  {r['model']:24}{c.get('params_total_m', float('nan')):10.1f}"
             f"{c.get('size_mb', float('nan')):10.1f}"
             f"{b.get('latency_ms_b1_mean', float('nan')):11.2f}"
             f"{r['train_seconds']/60:11.1f}{r.get('epochs_run', 0):8d}")

    txt = a.runs / "comparison.txt"
    txt.write_text("\n".join(out_lines))
    emit(f"\ntext report -> {txt}")

    if a.csv:
        import csv as _csv
        with a.csv.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["model", "subset"] + list(METRIC_NAMES))
            w.writeheader()
            w.writerow({"model": baseline.REFERENCE_MODEL, "subset": "pooled",
                        **{k: reference_metrics()[k] for k in METRIC_NAMES}})
            w.writerow({"model": baseline.REFERENCE_MODEL, "subset": "PTBXL",
                        **{k: reference_metrics("PTBXL")[k] for k in METRIC_NAMES}})
            w.writerows(csv_rows)
        print(f"csv -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
