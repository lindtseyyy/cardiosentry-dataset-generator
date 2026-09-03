#!/usr/bin/env python3
"""
Read-only analysis: AF+LVH test cases in prototype_1152 vs frozen
ResNet-50 @768 predictions, traced back to original dataset annotations.

Inputs (never modified):
  - prototype-results/hybrid_v1/resnet50_hires/predictions/test_predictions.csv
  - dataset-combination-code/ecg_dataset_pipeline/data/releases/v2/prototype_1152/index.csv
  - dataset-combination-code/ecg_dataset_pipeline/data/harmonized/harmonized_ecg_metadata.csv
  - dataset-files/ptb-xl-dataset/ptbxl_database.csv (raw cross-check)
  - dataset-files/chapman-dataset/ConditionNames_SNOMED-CT.csv (SNOMED decode)

Outputs (written to af_lvh_error_analysis/):
  - test_af_lvh_case_table.csv
  - train_af_lvh_extra_condition_audit.csv
  - summary.txt
"""
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/lindtseyxvii/Codes/cardiosentry")
PRED = ROOT / "prototype-results/hybrid_v1/resnet50_hires/predictions/test_predictions.csv"
PROTO = ROOT / "dataset-combination-code/ecg_dataset_pipeline/data/releases/v2/prototype_1152/index.csv"
HARM = ROOT / "dataset-combination-code/ecg_dataset_pipeline/data/harmonized/harmonized_ecg_metadata.csv"
PTBXL_DB = ROOT / "dataset-files/ptb-xl-dataset/ptbxl_database.csv"
CHAP_NAMES = ROOT / "dataset-files/chapman-dataset/ConditionNames_SNOMED-CT.csv"
OUTDIR = ROOT / "af_lvh_error_analysis"

def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))

preds = load_csv(PRED)
proto = load_csv(PROTO)
harm = load_csv(HARM)
harm_by_id = {r["global_record_id"]: r for r in harm}
proto_by_rec = {r["record"]: r for r in proto}

chapman_names = {}
with open(CHAP_NAMES, newline="", encoding="utf-8-sig") as fh:
    for row in csv.DictReader(fh):
        code = (row.get("Snomed_CT") or "").strip()
        if code:
            chapman_names[code] = (row.get("Acronym Name", "").strip(),
                                   row.get("Full Name", "").strip())

# Extra SNOMED names not in the shipped CSV (public SNOMED-CT definitions)
extra_snomed = {
    "164934002": ("TINV", "T wave inversion"),
    "17338001": ("VPB", "Ventricular premature beats"),
    "111975006": ("AVB", "Atrioventricular block"),
    "59118001": ("RBBB", "Right bundle branch block"),
    "698252002": ("LBBB", "Left bundle branch block"),
    "429622005": ("STDD", "ST drop down"),
    "428750005": ("STTC", "ST-T change"),
}

# PTB-XL SCP categories (beyond AFIB/LVH)
CONDUCTION = {"ILBBB", "IRBBB", "CLBBB", "CRBBB", "IVCD", "ABQRS", "LAFB", "LPFB",
              "LAO/LAE", "RAO/RAE", "1AVB", "2AVB", "3AVB", "AVB", "IVB", "WPW"}
MI_ISCHEMIA = {"IMI", "ASMI", "AMI", "ALMI", "ILMI", "LMI", "PMI", "IPMI", "IPLMI",
               "ISC_", "ISCAL", "ISCAS", "ISCIL", "ISCIN", "ISCLA", "ISCAN",
               "INJAS", "INJAL", "INJIN", "INJIL", "INJLA", "ANEUR"}
REPOL = {"STD_", "INVT", "LOWT", "NST_", "NT_", "QWAVE", "STE_", "VCLVH",
         "STTC", "STDD", "TINV"}
RHYTHM_OTHER = {"PVC", "BIGU", "SVTAC", "PSVT", "AFLT", "SVARR", "ABERR", "PACE",
                "VPB", "SBRAD", "STACH", "SARRH"}
DRUG = {"DIG", "NDT"}
HYP_OTHER = {"RVH", "SEHYP", "RAH", "LVQRSAL"}
MI_CODES = {"IMI", "ASMI", "AMI", "ALMI", "ILMI", "LMI", "PMI", "IPMI", "IPLMI",
            "MISW", "OMI"}

def categorize(extra_labels, source):
    out = {"conduction": [], "mi": [], "ischemia": [], "repol": [],
           "rhythm_other": [], "drug": [], "other": []}
    for lab in extra_labels:
        if lab in CONDUCTION:
            out["conduction"].append(lab)
        elif lab in MI_CODES:
            out["mi"].append(lab)
        elif lab in MI_ISCHEMIA:
            out["ischemia"].append(lab)
        elif lab in REPOL:
            out["repol"].append(lab)
        elif lab in RHYTHM_OTHER:
            out["rhythm_other"].append(lab)
        elif lab in DRUG:
            out["drug"].append(lab)
        elif lab in HYP_OTHER:
            out["other"].append(lab)
        else:
            out["other"].append(lab)
    return out

def readable_extra(lab, source):
    if source == "CHAPMAN":
        if lab in chapman_names:
            a, n = chapman_names[lab]
            return f"{lab}({n})"
        if lab in extra_snomed:
            a, n = extra_snomed[lab]
            return f"{lab}({n})"
        return f"{lab}(UNDEFINED)"
    return lab

# ---------------------------------------------------------------- test cases
test_cases = []
for r in preds:
    if r["true_AF"] == "1" and r["true_LVH"] == "1":
        test_cases.append(r)
test_cases.sort(key=lambda r: -float(r["prob_LVH"]))

rows = []
print(f"{'record':10s} {'src':7s} {'orig_id':8s} {'LVH prob':>8s} {'pred':>4s} {'corr':>4s}  original_labels")
for r in test_cases:
    rec = r["record"]
    p = proto_by_rec[rec]
    h = harm_by_id[p["record_id"]]
    src = h["source_dataset"]
    labs = h["original_labels"].split("|")
    is_corr = r["pred_LVH"] == "1"
    rows.append({
        "record": rec, "record_id": p["record_id"], "source": src,
        "original_record_id": h["original_record_id"], "patient_id": h["patient_id"],
        "split_group_id": h["split_group_id"],
        "prototype_cls": p["cls"], "prototype_AF": p["AF"], "prototype_LVH": p["LVH"],
        "prototype_STEMI": p["STEMI"], "prototype_NORMAL": p["NORMAL"],
        "age": p["age"], "sex": p["sex"],
        "original_labels": h["original_labels"],
        "original_labels_readable": h["original_labels_readable"],
        "accepted_label_mapping": h["accepted_label_mapping"],
        "LVH_prob": float(r["prob_LVH"]), "LVH_pred": int(r["pred_LVH"]),
        "AF_prob": float(r["prob_AF"]), "AF_pred": int(r["pred_AF"]),
        "STEMI_prob": float(r["prob_STEMI"]), "STEMI_pred": int(r["pred_STEMI"]),
        "NORMAL_prob": float(r["prob_NORMAL"]), "NORMAL_pred": int(r["pred_NORMAL"]),
        "LVH_correct": "correct" if is_corr else "INCORRECT",
    })
    extra = [l for l in labs if l not in ("AFIB", "LVH", "164889003", "164873001")]
    cat = categorize(extra, src)
    print(f"{rec:10s} {src:7s} {h['original_record_id']:8s} {float(r['prob_LVH']):8.4f} "
          f"{r['pred_LVH']:>4s} {('OK ' if is_corr else 'MISS'):>5s}  {h['original_labels']}")

# ---------------------------------------------------------------- raw cross-check
raw_db = load_csv(PTBXL_DB)
raw_by_id = {row["ecg_id"]: row for row in raw_db}
cross_check_ok = True
for r in rows:
    if r["source"] != "PTBXL":
        continue
    raw = raw_by_id[r["original_record_id"]]
    raw_codes = set(ast.literal_eval(raw["scp_codes"]).keys())
    harm_codes = set(r["original_labels"].split("|"))
    if raw_codes != harm_codes:
        cross_check_ok = False
        print(f"  CROSS-CHECK MISMATCH {r['record_id']}: raw {sorted(raw_codes - harm_codes)} extra / {sorted(harm_codes - raw_codes)} missing")
print(f"raw ptbxl_database.csv scp_codes == harmonized original_labels for all test rows: {cross_check_ok}")

# ---------------------------------------------------------------- stats
def cond_flags(cat):
    return {
        "any_extra": any(cat[k] for k in cat),
        "n_extra": sum(len(cat[k]) for k in cat),
        "conduction": bool(cat["conduction"]),
        "mi": bool(cat["mi"]),
        "ischemia": bool(cat["ischemia"]),
        "repol": bool(cat["repol"]),
        "rhythm_other": bool(cat["rhythm_other"]),
        "drug": bool(cat["drug"]),
    }

from scipy.stats import fisher_exact

print("\n=== 28 test AF+LVH cases: extra-condition profile ===")
corr_rows = [r for r in rows if r["LVH_correct"] == "correct"]
miss_rows = [r for r in rows if r["LVH_correct"] == "INCORRECT"]
def profile(group):
    out = {}
    for r in group:
        labs = r["original_labels"].split("|")
        extra = [l for l in labs if l not in ("AFIB", "LVH", "164889003", "164873001")]
        cat = categorize(extra, r["source"])
        f = cond_flags(cat)
        for k, v in f.items():
            if k == "n_extra":
                out.setdefault(k, []).append(v)
            else:
                out.setdefault(k, []).append(1 if v else 0)
    return out
cp, mp = profile(corr_rows), profile(miss_rows)
for k in cp:
    if k == "n_extra":
        print(f"  {k:14s} correct={sum(cp[k])/len(cp[k]):.2f}  miss={sum(mp[k])/len(mp[k]):.2f}")
    else:
        a = sum(cp[k]); b = sum(mp[k])
        n1, n2 = len(cp[k]), len(mp[k])
        odds, pval = fisher_exact([[a, n1 - a], [b, n2 - b]])
        print(f"  {k:14s} correct={a}/{n1}  miss={b}/{n2}  OR={odds:.2f}  Fisher p={pval:.4f}")

# ------------------------------------------------ write test table
out_csv = OUTDIR / "test_af_lvh_case_table.csv"
fields = ["record", "record_id", "source", "original_record_id", "patient_id",
          "prototype_cls", "age", "sex",
          "original_labels_readable", "accepted_label_mapping",
          "LVH_prob", "LVH_pred", "LVH_correct", "AF_prob", "AF_pred",
          "STEMI_prob", "STEMI_pred", "NORMAL_prob", "NORMAL_pred"]
with open(out_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda x: (x["LVH_correct"] != "INCORRECT", -x["LVH_prob"])):
        w.writerow({**r, "LVH_prob": f"{r['LVH_prob']:.6f}",
                    "AF_prob": f"{r['AF_prob']:.6f}", "STEMI_prob": f"{r['STEMI_prob']:.6f}",
                    "NORMAL_prob": f"{r['NORMAL_prob']:.6f}"})
print(f"\nwrote {out_csv}")

# ------------------------------------------------ train audit (202 AF+LVH)
train_rows = [p for p in proto if p["split"] == "train" and p["AF"] == "1" and p["LVH"] == "1"]
print(f"\n=== train AF+LVH audit: {len(train_rows)} records ===")
train_out = []
for p in train_rows:
    h = harm_by_id[p["record_id"]]
    labs = h["original_labels"].split("|")
    extra = [l for l in labs if l not in ("AFIB", "LVH", "164889003", "164873001")]
    cat = categorize(extra, h["source_dataset"])
    train_out.append({
        "record": p["record"], "record_id": p["record_id"],
        "source": h["source_dataset"], "original_record_id": h["original_record_id"],
        "original_labels": h["original_labels"],
        "original_labels_readable": h["original_labels_readable"],
        "n_original_labels": len(labs), "n_extra_labels": len(extra),
        "extra_labels": "|".join(extra),
        "conduction": "|".join(cat["conduction"]),
        "mi": "|".join(cat["mi"]), "ischemia": "|".join(cat["ischemia"]),
        "repol": "|".join(cat["repol"]), "rhythm_other": "|".join(cat["rhythm_other"]),
        "drug": "|".join(cat["drug"]), "other": "|".join(cat["other"]),
    })
flags = Counter()
for t in train_out:
    flags["n_records"] += 1
    if t["n_extra_labels"] == 0:
        flags["clean_AF+LVH_only"] += 1
    if t["conduction"]:
        flags["has_conduction"] += 1
    if t["mi"]:
        flags["has_MI_code"] += 1
    if t["ischemia"]:
        flags["has_ischemia"] += 1
    if t["repol"]:
        flags["has_repol"] += 1
    if t["rhythm_other"]:
        flags["has_other_arrhythmia"] += 1
    if t["drug"]:
        flags["has_drug"] += 1
print("  train AF+LVH profile:")
for k in ("n_records", "clean_AF+LVH_only", "has_conduction", "has_MI_code",
          "has_ischemia", "has_repol", "has_other_arrhythmia", "has_drug"):
    print(f"    {k:24s} {flags[k]}")
n_extra_dist = Counter(t["n_extra_labels"] for t in train_out)
print("  n_extra_labels distribution:", dict(sorted(n_extra_dist.items())))

src_dist = Counter(t["source"] for t in train_out)
print("  source:", dict(src_dist))

out_csv2 = OUTDIR / "train_af_lvh_extra_condition_audit.csv"
with open(out_csv2, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(train_out[0].keys()))
    w.writeheader()
    w.writerows(train_out)
print(f"wrote {out_csv2}")

# Chapman-specific: decode every SNOMED code seen on the 10 Chapman train rows
print("\n=== CHAPMAN train AF+LVH rows: decoded SNOMED ===")
chap_rows = [t for t in train_out if t["source"] == "CHAPMAN"]
for t in chap_rows:
    codes = t["original_labels"].split("|")
    decoded = []
    for c in codes:
        if c in chapman_names:
            decoded.append(f"{c}={chapman_names[c][0]}({chapman_names[c][1]})")
        elif c in extra_snomed:
            decoded.append(f"{c}={extra_snomed[c][0]}({extra_snomed[c][1]})*")
        else:
            decoded.append(f"{c}=UNDEFINED")
    print(f"  {t['record_id']:15s} {' | '.join(decoded)}")
