#!/usr/bin/env python3
"""
Turn results/external_metrics.json into a report a human can act on.

    python report.py                       # -> results/EXTERNAL_VALIDATION_REPORT.md

The report is organised around ONE decision table, and every section exists to
support or qualify it:

    STEMI collapses while AF/LVH/NORMAL hold   -> confound confirmed, and it is
                                                  specific to the STEMI head
    everything collapses together              -> domain shift; inconclusive
    STEMI holds                                -> the head generalises

The verdict is computed, not written by hand, so it cannot drift away from the
numbers underneath it. `classify_outcome` below is where the thresholds for
"collapses" and "holds" are declared; they are deliberately in one place and
deliberately conservative.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
CONTROL = ["AF", "LVH", "NORMAL"]

# The prototype's own held-out test scores, read from
# prototype-results/cardiosentry_runs_v1/<model>/metrics.csv (split=test).
# Quoted so the external numbers have something to fall FROM - a STEMI AUROC of
# 0.98 internally and 0.5 externally is the whole finding, and it is not legible
# without both halves side by side.
#
# The STEMI columns here are the CONFOUNDED ones. They are the measurement this
# corpus exists to re-take, not a target to be matched.
INTERNAL = {
    "resnet50":        {"macro_auroc": 0.9770, "stemi_auroc": 0.9861, "stemi_auprc": 0.9560},
    "convnext_base":   {"macro_auroc": 0.9694, "stemi_auroc": 0.9815, "stemi_auprc": 0.9463},
    "vgg16":           {"macro_auroc": 0.9662, "stemi_auroc": 0.9592, "stemi_auprc": 0.9175},
    "vit_b_384":       {"macro_auroc": 0.9632, "stemi_auroc": 0.9558, "stemi_auprc": 0.8659},
    "efficientnet_b3": {"macro_auroc": 0.9376, "stemi_auroc": 0.9375, "stemi_auprc": 0.8129},
}

# Where the verdict's language comes from. Stated as constants because a reader
# who disagrees with the wording should be able to see exactly what produced it.
COLLAPSE_AUROC = 0.65     # at or below this a head is not usefully ranking
HOLD_AUROC = 0.85         # at or above this a head has clearly transferred


def f(x, nd=4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def ci(m, key) -> str:
    lo, hi = m.get(f"{key}_lo"), m.get(f"{key}_hi")
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)):
        return f(m.get(key))
    return f"{f(m.get(key))} [{f(lo, 3)}, {f(hi, 3)}]"


def classify_outcome(stemi_auroc: float, control_auroc: float,
                     worst_control: float = float("nan")) -> tuple[str, str]:
    """The two-by-two from evaluate.py's docstring, applied to one model.

    `worst_control` is the weakest single control head, and it is passed
    separately because the MEAN hides exactly the case that matters. In this
    corpus AF transfers near-perfectly while LVH collapses; averaging the three
    reports a healthy ~0.80 control baseline that no individual head actually
    has. The mean still decides the verdict - one collapsed control head does
    not mean the corpus shifted under every head - but a collapsed head is
    surfaced in the reason rather than silently averaged away.
    """
    if math.isnan(stemi_auroc):
        return "UNDETERMINED", "the STEMI head could not be scored on this cohort"
    if control_auroc <= COLLAPSE_AUROC:
        return "INCONCLUSIVE", (
            "the control heads fell too, so the corpus shifted under every head "
            "at once; this says nothing about the STEMI confound either way")
    caveat = ""
    if not math.isnan(worst_control) and worst_control <= COLLAPSE_AUROC:
        caveat = (f" (note: one control head is itself at {worst_control:.2f}, so "
                  f"the control baseline is not uniformly healthy)")
    if stemi_auroc <= COLLAPSE_AUROC:
        return "CONFOUND CONFIRMED", (
            "STEMI collapsed while the multi-source heads held — the failure is "
            "specific to the one head whose positives came from a single source"
            + caveat)
    if stemi_auroc >= HOLD_AUROC:
        return "TRANSFERS", (
            "STEMI ranks external positives above negatives at least as well as "
            "the control heads do; the confound existed in the data but this "
            "model did not rely on it" + caveat)
    return "PARTIAL", (
        "STEMI degraded well beyond the control heads without collapsing — "
        "consistent with a model using both ST morphology and source cues"
        + caveat)


def score_gradient(pred_dir, index_path, names) -> dict:
    """Mean STEMI probability per finding-group, on LUDB sheets only.

    Returns {} when the prediction CSVs are not to hand - the rest of the report
    is built from the metrics JSON alone and must still render without them.
    """
    if not pred_dir:
        return {}
    pred_dir, index_path = Path(pred_dir), Path(index_path)
    if not pred_dir.is_dir() or not index_path.exists():
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}

    idx = pd.read_csv(index_path)
    text = idx.set_index("record")["original_labels"].fillna("").str.lower()
    out: dict = {}
    for name in names:
        f = pred_dir / f"{name}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["t"] = d["record"].map(text).fillna("")
        d = d[d["source"] == "LUDB"]
        pos = d[d["y_STEMI"] == 1]
        neg = d[d["y_STEMI"] == 0]
        early = neg[neg["t"].str.contains("early repolarization")]
        isch = neg[neg["t"].str.contains("ischemia|scar|nstemi", regex=True)]
        repol = neg[neg["t"].str.contains("non-specific repolarization")]
        rest = neg.drop(index=early.index.union(isch.index).union(repol.index))
        out[name] = {"pos": pos["p_STEMI"].mean(), "early": early["p_STEMI"].mean(),
                     "isch": isch["p_STEMI"].mean(), "repol": repol["p_STEMI"].mean(),
                     "other": rest["p_STEMI"].mean()}
        out["_n"] = {"pos": len(pos), "early": len(early), "isch": len(isch),
                     "repol": len(repol), "other": len(rest)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(HERE.parent / "results" / "external_metrics.json"))
    ap.add_argument("-o", "--out", default=str(HERE.parent / "results"
                                               / "EXTERNAL_VALIDATION_REPORT.md"))
    ap.add_argument("--predictions", default=None,
                    help="folder of <model>.csv prediction files; enables the "
                         "score-by-finding section, which is the single most "
                         "diagnostic table in the report")
    ap.add_argument("--index", default=str(HERE.parent / "data" / "external_testset"
                                           / "index.csv"))
    args = ap.parse_args()

    p = Path(args.metrics)
    if not p.exists():
        raise SystemExit(f"missing {p} — run evaluate.py first")
    data = json.loads(p.read_text())
    models = data["models"]
    sizes, pos = data["cohort_sizes"], data["positives"]

    def cell(name, cohort, label, key):
        return (models[name]["cohorts"].get(cohort, {})
                .get("per_label", {}).get(label, {}).get(key, float("nan")))

    def macro_control(name, cohort):
        vals = [cell(name, cohort, l, "auroc") for l in CONTROL]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def worst_control(name, cohort):
        vals = [cell(name, cohort, l, "auroc") for l in CONTROL]
        vals = [v for v in vals if not math.isnan(v)]
        return min(vals) if vals else float("nan")

    L: list[str] = []
    A = L.append

    A("# CardioSentry — External STEMI Validation")
    A("")
    A("**Does the STEMI head detect ST elevation, or does it detect the Chongqing "
      "scanner?** The training corpus could not tell the difference. This one can.")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| **Sheets** | {data['n_sheets']} · one canonical image per ECG |")
    A(f"| **Gold STEMI** | {pos['LUDB']['STEMI']} (LUDB — asserted, unseen source) |")
    A(f"| **Proxy STEMI** | {pos['PTBXL_MI']['STEMI']} (PTB-XL acute MI — seen source, "
      f"never labelled STEMI) |")
    A(f"| **Hard negatives** | {sizes.get('HARD_NEG', 0)} abnormal sheets carrying no "
      f"target class |")
    A(f"| **Config fingerprint** | `{data['config_fingerprint']}` |")
    A(f"| **Bootstrap** | {data['bootstrap_resamples']} resamples, percentile CIs |")
    A("")
    A("---")
    A("")

    # ---------------------------------------------------------------- verdict
    A("## Verdict")
    A("")
    A("Read this table row by row. The STEMI column alone means nothing — it is "
      "only interpretable against the control column beside it, because a corpus "
      "shift depresses every head at once and that is not a confound.")
    A("")
    A("| Model | STEMI AUROC — internal (confounded) | STEMI AUROC — external gold | Drop | Control AUROC (AF/LVH/NORMAL, same sheets) | Reading |")
    A("|---|---:|---:|---:|---:|---|")
    outcomes = {}
    for name in models:
        s = cell(name, "LUDB", "STEMI", "auroc")
        # Control measured on LUDB too, NOT on the whole corpus. The comparison
        # only means anything if both halves come off the same sheets: same
        # cardiograph, same population, same render. Scoring STEMI on LUDB and
        # the controls on everything would let a difference in cohort masquerade
        # as a difference between heads.
        c = macro_control(name, "LUDB")
        verdict, _ = classify_outcome(s, c, worst_control(name, "LUDB"))
        outcomes[name] = verdict
        # A v2 run records its own internal test scores in the metrics JSON
        # (evaluate_run.py reads them off result.json); the table above is
        # the fallback for the v1 baselines, which predate that field.
        internal = (models[name].get("internal") or {}).get("stemi_auroc")
        if internal is None:
            internal = INTERNAL.get(name, {}).get("stemi_auroc")
        drop = (internal - s) if (internal is not None and not math.isnan(s)) else None
        A(f"| {name} | {f(internal)} | {f(s)} | {f(drop)} | {f(c)} | **{verdict}** |")
    A("")
    A("The *internal* column is the number the prototype reported on its own "
      "held-out test set — where every STEMI positive was a Chongqing sheet and "
      "every Chongqing sheet was a STEMI positive. It is the figure this corpus "
      "was built to re-take, not a target to be matched.")
    A("")
    for name in models:
        _, why = classify_outcome(cell(name, "LUDB", "STEMI", "auroc"),
                                  macro_control(name, "LUDB"),
                                  worst_control(name, "LUDB"))
        A(f"- **{name}** — {why}.")
    A("")

    tally = {}
    for v in outcomes.values():
        tally[v] = tally.get(v, 0) + 1
    A(f"**Across all {len(models)} models:** " + ", ".join(
        f"{n} {v.lower()}" for v, n in sorted(tally.items(), key=lambda kv: -kv[1])) + ".")
    A("")
    A("---")
    A("")

    # ------------------------------------------------------- the STEMI head
    A("## 1. The STEMI head, cohort by cohort")
    A("")
    A("Three cohorts, three different questions.")
    A("")
    A(f"- **LUDB** ({pos['LUDB']['STEMI']} positives / {sizes['LUDB']} sheets) — a "
      "source seen in no class, with STEMI asserted in words by the reporting "
      "cardiologist. The gold standard, and far too small to carry a precise "
      "estimate. Always read the interval, never the point.")
    A(f"- **PTB-XL acute MI** ({pos['PTBXL_MI']['STEMI']} positives) — a source seen "
      "extensively in training but **never once labelled STEMI**. A model keying on "
      "paper cannot fire here; a model reading ST morphology can. The label is a "
      "proxy (`mi` + Stadium I, i.e. acute), not an asserted ST-elevation label.")
    A(f"- **PTB-XL normal** ({sizes['PTBXL_NORM']} sheets) — negative controls from "
      "the same familiar source, which is what keeps source from being a clean "
      "proxy for the label inside the PTB-XL arm.")
    A("")
    A("| Model | Cohort | n | pos | AUROC (95% CI) | AUPRC (95% CI) | Sens @ frozen thr | Spec | mean p(pos) | mean p(neg) |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in models:
        for cohort in ("LUDB", "PTBXL_MI", "PTBXL_ALL", "ALL"):
            m = (models[name]["cohorts"].get(cohort, {})
                 .get("per_label", {}).get("STEMI"))
            if not m:
                continue
            A(f"| {name} | {cohort} | {m['n']} | {m['n_pos']} | {ci(m, 'auroc')} | "
              f"{ci(m, 'auprc')} | {f(m['sensitivity'], 3)} | {f(m['specificity'], 3)} | "
              f"{f(m['mean_prob_pos'], 3)} | {f(m['mean_prob_neg'], 3)} |")
    A("")
    A("> **`PTBXL_MI` shows `n/a` for AUROC and AUPRC, and that is correct rather "
      "than broken.** Every sheet in that cohort is a STEMI positive, so there is "
      "nothing to rank it against and both metrics are mathematically undefined — "
      "the same degeneracy that made the prototype's internal per-source probe "
      "return `nan`. Read **sensitivity** and **mean p(pos)** on that row, and use "
      "**`PTBXL_ALL`** (the MI arm plus its 155 same-source normal controls) when "
      "you want a ranking metric from PTB-XL.")
    A("")
    A("> **`mean p(pos)` is the column to read when sensitivity is zero.** "
      "Sensitivity depends on a threshold frozen from the original validation "
      "split; a model can rank external positives correctly and still fire on "
      "none of them because its threshold sits above every probability it "
      "assigned. Mean probability separates *cannot see it* from *sees it but "
      "will not commit*. Those need different fixes — retraining versus "
      "recalibration.")
    A("")

    # ------------------------------------------------------------- control
    A("## 2. The control heads")
    A("")
    A("AF, LVH and NORMAL had multi-source training — PTB-XL and Chapman both "
      "contribute all three — so they carry no source confound. They ride on the "
      "same sheets through the same renderer. Whatever they lose here is the price "
      "of the corpus, not of a confound, and it is the baseline the STEMI drop has "
      "to be measured against.")
    A("")
    A(f"Scored **on LUDB only** ({sizes['LUDB']} sheets: {pos['LUDB']['AF']} AF, "
      f"{pos['LUDB']['LVH']} LVH, {pos['LUDB']['NORMAL']} NORMAL), so the "
      "control heads and the gold "
      "STEMI head are measured on the identical sheets — same cardiograph, same "
      "population, same render. Anything else would let a difference in cohort "
      "masquerade as a difference between heads.")
    A("")
    A("| Model | AF AUROC | LVH AUROC | NORMAL AUROC | Control mean | STEMI (gold) | Gap |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name in models:
        vals = [cell(name, "LUDB", l, "auroc") for l in CONTROL]
        cm = macro_control(name, "LUDB")
        s = cell(name, "LUDB", "STEMI", "auroc")
        gap = cm - s if not (math.isnan(cm) or math.isnan(s)) else float("nan")
        A(f"| {name} | " + " | ".join(f(v) for v in vals) +
          f" | {f(cm)} | {f(s)} | {f(gap)} |")
    A("")
    A("For reference, the same heads across the whole corpus (LUDB + PTB-XL):")
    A("")
    A("| Model | AF AUROC | LVH AUROC | NORMAL AUROC | Control mean |")
    A("|---|---:|---:|---:|---:|")
    for name in models:
        vals = [cell(name, "ALL", l, "auroc") for l in CONTROL]
        A(f"| {name} | " + " | ".join(f(v) for v in vals) +
          f" | {f(macro_control(name, 'ALL'))} |")
    A("")
    A("A large positive gap is the signature the experiment was built to detect: "
      "the heads that were trained across sources transferred, and the one head "
      "whose positives came from a single source did not.")
    A("")

    # -------------------------------------------------------------- mimics
    A("## 3. Specificity on STEMI mimics")
    A("")
    A("These are the sheets that separate *reads ST morphology* from *reads paper*, "
      "and they need reading carefully in the right direction.")
    A("")
    for key, blurb in (
        ("MIMIC_early_repol",
         "**Early repolarisation syndrome** — benign ST elevation. Genuinely "
         "elevated ST segments on a patient who is not infarcting."),
        ("MIMIC_repol_abnorm",
         "**Non-specific repolarisation abnormality** — ST-T change without "
         "infarction."),
        ("MIMIC_ischemia_scar",
         "**Ischemia / scar / suspected NSTEMI** — coronary disease that is "
         "specifically *not* ST-elevation MI."),
    ):
        n = sizes.get(key, 0)
        if not n:
            continue
        A(f"- {blurb} ({n} sheets)")
    A("")
    A("| Model | " + " | ".join(k.replace("MIMIC_", "") + " FP rate"
                                for k in ("MIMIC_early_repol", "MIMIC_repol_abnorm",
                                          "MIMIC_ischemia_scar")) + " | hard-neg FP rate |")
    A("|---|---:|---:|---:|---:|")
    for name in models:
        cells = []
        for key in ("MIMIC_early_repol", "MIMIC_repol_abnorm",
                    "MIMIC_ischemia_scar", "HARD_NEG"):
            m = (models[name]["cohorts"].get(key, {})
                 .get("per_label", {}).get("STEMI"))
            if not m or m["n_neg"] == 0:
                cells.append("—")
            else:
                spec = m["specificity"]
                cells.append("n/a" if math.isnan(spec) else f"{1 - spec:.3f}")
        A(f"| {name} | " + " | ".join(cells) + " |")
    A("")
    A("> **A perfect score here is not automatically good news.** A model that has "
      "learnt \"Chongqing paper ⇒ STEMI\" scores 0.000 on every column of this "
      "table, because none of these sheets are on Chongqing paper. Zero false "
      "positives is only meaningful from a model that also *fired on the true "
      "positives* in section 1. Read the two tables together or not at all.")
    A("")

    # ------------------------------------------------- what the head fires on
    grad = score_gradient(args.predictions, args.index, list(models))
    if grad:
        A("## 3b. What the STEMI head actually fires on")
        A("")
        A("Mean predicted STEMI probability on LUDB sheets, grouped by what the "
          "cardiologist's report says. This is the most diagnostic table in the "
          "report, because the competing explanations predict *different "
          "orderings* rather than merely different magnitudes.")
        A("")
        A("- A model reading **the Chongqing scanner**, or the record ID printed "
          "on the sheet, scores every LUDB sheet alike — true positives included "
          "— because none of them carry the cue. Flat row.")
        A("- A model reading **raw ST elevation** also fires on early "
          "repolarisation: that finding *is* ST elevation, in a patient who is "
          "not infarcting.")
        A("- A model reading **infarct morphology** separates the two: high on "
          "STEMI, low on early repolarisation, intermediate on ischemia and "
          "scar, which share ST-T changes without the injury pattern.")
        A("")
        A("| Model | STEMI+ | ischemia / scar / NSTEMI | non-specific repol. "
          "| other negatives | **early repol (benign STE)** |")
        A("|---|---:|---:|---:|---:|---:|")
        for name in models:
            g = grad.get(name)
            if not g:
                continue
            A(f"| {name} | **{g['pos']:.3f}** | {g['isch']:.3f} | "
              f"{g['repol']:.3f} | {g['other']:.3f} | {g['early']:.3f} |")
        A("")
        n = grad["_n"]
        A(f"Group sizes: {n['pos']} STEMI positives, {n['isch']} ischemia/scar, "
          f"{n['repol']} non-specific repolarisation, {n['other']} other "
          f"negatives, {n['early']} early repolarisation.")
        A("")

    # ---------------------------------------------------------- limitations
    A("## 4. What this corpus cannot tell you")
    A("")
    A(f"1. **Eight gold positives.** LUDB contributes {pos['LUDB']['STEMI']} asserted "
      "STEMI records and that is the entire supply in the database. Every LUDB "
      "STEMI interval in this report is correspondingly wide. It is enough to "
      "detect a collapse — a head scoring near 0.5 with 8 positives is still "
      "clearly not working — and nowhere near enough to certify that a head works.")
    A("2. **The PTB-XL arm is a proxy, and it is not a new source.** "
      "`dgedon/nstemi_prediction` publishes a curated selection *of PTB-XL*, which "
      "is already a training source. Its value is one-directional: PTB-XL "
      "contributed zero STEMI labels, so firing here cannot be explained by paper. "
      "But the label is `mi` + Stadium I (acute), and acute MI still admits NSTEMI. "
      "Do not merge it with the LUDB figure into a single sensitivity.")
    A("3. **Rendering is synthetic on both sides.** These sheets and the training "
      "sheets came from the same `ecg-image-kit` at the same DPI with the same "
      "augmentation. That is deliberate — it isolates *source of signal* from "
      "*style of paper* — but it means this measures generalisation across ECG "
      "databases, not across real cardiograph printouts or phone photographs.")
    A("4. **Prevalence is not clinical.** LVH sits near 54% in LUDB against a far "
      "lower rate in the training corpus. AUROC is prevalence-independent; AUPRC, "
      "F1 and precision are not, and should not be compared across cohorts.")
    A("5. **NORMAL in LUDB is an assertion by absence.** LUDB publishes no "
      "\"normal ECG\" statement, so a record is called NORMAL when its full "
      "cardiologist report lists a sinus rhythm, an axis, and nothing else. That "
      "is weaker than PTB-XL's explicit `NORM` and yields 25 records.")
    A("")

    A("## 5. Provenance")
    A("")
    A(f"- Test corpus: `{data['testset']}`")
    A(f"- Weights: `{data['weights_root']}`")
    A(f"- Controlled-config fingerprint: `{data['config_fingerprint']}` — matches the "
      "training runs, so transforms, normalisation and image size are identical.")
    A("- Thresholds: frozen per model on the **original validation split** and applied "
      "unchanged. Never re-tuned here; doing so would be tuning on test.")
    A("- Per-model checkpoint epochs: " + ", ".join(
        f"{n} = {models[n].get('epoch', '?')}" for n in models))
    A("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out}  ({len(L)} lines)")
    for name, v in outcomes.items():
        print(f"  {name:<18} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
