#!/usr/bin/env python3
"""
STEPS 2-3 (+ STEP 11) -- Apply the label mapping and extract candidate records.

Reads config/label_mapping.yaml and applies it to every record in every
source. No label decision is taken here that is not declared in the YAML.

Outputs
  reports/label_mapping_audit.csv      every source label x how it was treated
  reports/extracted_candidates.csv     one row per ACCEPTED candidate record
  reports/excluded_records.csv         one row per EXCLUDED record + reason
  reports/unmapped_labels_report.csv   labels the config refuses to guess at
  reports/ambiguity_report.csv         contradictory / ambiguous label sets
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (TARGET_CLASSES, LabelResolver, assumption, banner, info,
                    load_mapping, load_paths, reports_dir, warn, write_csv)
from sources import build_sources

CANDIDATE_FIELDS = [
    "global_record_id", "source_dataset", "original_record_id", "patient_id",
    "patient_id_is_synthetic", "original_labels", "original_labels_readable",
    "cardiosentry_label", "STEMI", "AF", "LVH", "NORMAL", "n_positive_labels",
    "accepted_label_mapping", "age", "sex", "signal_path", "header_path",
    "data_path", "ambiguity_flag", "ambiguity_note", "source_extra_json",
]

EXCLUDED_FIELDS = [
    "source_dataset", "original_record_id", "patient_id", "original_labels",
    "original_labels_readable", "exclusion_reason", "exclusion_detail",
]


def gid(prefix: str, original_record_id) -> str:
    """Stable global id derived from the SOURCE record id, not from a running
    counter. This matters for auditability: changing a label decision in the
    YAML must not silently renumber every record downstream. PTB-XL ecg_id 123
    is always PTBXL_000123, Chapman JS01234 is always CHAPMAN_001234."""
    digits = re.sub(r"\D", "", str(original_record_id))
    return f"{prefix}_{int(digits):06d}" if digits else f"{prefix}_{original_record_id}"


def _readable(source_key: str, resolver: LabelResolver, raw_labels) -> str:
    out = []
    for lab in raw_labels:
        d = resolver.describe(lab)
        nm = d["source_label_name"] or d["acronym"]
        out.append(f"{lab}" + (f" ({nm})" if nm and nm != str(lab) else ""))
    return "; ".join(out) if out else "(no labels)"


def extract_source(source_key, prefix, records, resolver, mapping,
                   candidates, excluded, ambiguities, unmapped_counter,
                   stats, likelihood_filter=None):
    counter = 0
    for r in records:
        raw = list(r.raw_labels)

        # ---- optional PTB-XL diagnostic-likelihood filter ---------------
        dropped_by_likelihood = []
        if likelihood_filter is not None:
            raw, dropped_by_likelihood = likelihood_filter(r, raw, resolver)

        d = resolver.resolve(raw)
        binary, flag, note = resolver.finalize(d)

        for u in d.unknown:
            unmapped_counter[(source_key, u)] += 1

        readable = _readable(source_key, resolver, r.raw_labels)
        n_pos = sum(binary.values())

        if flag and flag != "RECORD_EXCLUDED_BY_LABEL":
            ambiguities.append({
                "source_dataset": r.source,
                "original_record_id": r.original_record_id,
                "patient_id": r.patient_id or "",
                "original_labels": "|".join(str(x) for x in r.raw_labels),
                "original_labels_readable": readable,
                "ambiguity_flag": flag,
                "ambiguity_note": note,
                "resolved_STEMI": binary["STEMI"], "resolved_AF": binary["AF"],
                "resolved_LVH": binary["LVH"], "resolved_NORMAL": binary["NORMAL"],
            })

        if n_pos == 0:
            reasons = []
            if d.excluded:
                sts = sorted({s for _, s, _ in d.excluded})
                reasons.append("+".join(sts))
            if d.unknown:
                reasons.append("EXCLUDE_UNKNOWN_CODE")
            if not reasons:
                reasons.append("NO_TARGET_LABEL")
            detail = []
            if d.record_excluders:
                reasons = ["EXCLUDE_RECORD"]
                detail.append("record voided by unresolved label(s): "
                              + ", ".join(sorted(set(d.record_excluders))))
            if d.excluded:
                detail.append("excluded labels: " + ", ".join(
                    f"{l}[{s}]" for l, s, _ in d.excluded))
            if d.unknown:
                detail.append("undefined labels: " + ", ".join(sorted(set(d.unknown))))
            if d.normal_asserted and not binary["NORMAL"]:
                detail.append("normal assertion present but disqualified: " + note)
            if dropped_by_likelihood:
                detail.append("below likelihood threshold: " + ", ".join(dropped_by_likelihood))
            excluded.append({
                "source_dataset": r.source,
                "original_record_id": r.original_record_id,
                "patient_id": r.patient_id or "",
                "original_labels": "|".join(str(x) for x in r.raw_labels),
                "original_labels_readable": readable,
                "exclusion_reason": "+".join(sorted(set(reasons))),
                "exclusion_detail": "; ".join(detail) or "no label maps to a CardioSentry class",
            })
            stats[(r.source, "excluded")] += 1
            continue

        counter += 1
        pos_classes = [c for c in TARGET_CLASSES if binary[c]]
        acc = "; ".join(f"{'|'.join(v)}->{k}" for k, v in sorted(d.positives.items()))
        pid = r.patient_id
        synth = 0
        if not pid:
            # Chapman ships no patient identifier. Rather than silently
            # treating all its records as one patient (which would break
            # grouped splitting) or as independent (which would hide any
            # real repeats), assign an explicit synthetic per-record id and
            # flag it so downstream splitting knows it is an assumption.
            pid = f"{prefix}_SYNTH_{r.original_record_id}"
            synth = 1

        candidates.append({
            "global_record_id": gid(prefix, r.original_record_id),
            "source_dataset": r.source,
            "original_record_id": r.original_record_id,
            "patient_id": pid,
            "patient_id_is_synthetic": synth,
            "original_labels": "|".join(str(x) for x in r.raw_labels),
            "original_labels_readable": readable,
            "cardiosentry_label": "+".join(pos_classes),
            "STEMI": binary["STEMI"], "AF": binary["AF"],
            "LVH": binary["LVH"], "NORMAL": binary["NORMAL"],
            "n_positive_labels": n_pos,
            "accepted_label_mapping": acc,
            "age": r.age, "sex": r.sex,
            "signal_path": r.signal_path,
            "header_path": r.header_path,
            "data_path": r.data_path,
            "ambiguity_flag": flag,
            "ambiguity_note": note,
            "source_extra_json": json.dumps(
                {k: v for k, v in r.extra.items() if not k.startswith("_")},
                default=str, ensure_ascii=False),
        })
        for c in pos_classes:
            stats[(r.source, c)] += 1
        stats[(r.source, "valid_candidates")] += 1


def make_ptbxl_likelihood_filter(mapping, scp):
    rules = mapping.get("rules", {}).get("ptbxl_likelihood", {})
    thr = float(rules.get("min_diagnostic_likelihood", 0) or 0)
    apply_rhythm = bool(rules.get("apply_to_rhythm_statements", False))
    if thr <= 0:
        return None

    def _f(rec, raw, resolver):
        kept, dropped = [], []
        lk = rec.extra.get("scp_likelihoods", {})
        for code in raw:
            st = scp.get(code, {})
            is_rhythm = str(st.get("rhythm", "")) not in ("", "nan", "None")
            if is_rhythm and not apply_rhythm:
                kept.append(code)
                continue
            v = lk.get(code, 0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 0.0
            (kept if v >= thr else dropped).append(code if v >= thr else f"{code}@{v}")
        return kept, dropped
    return _f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--mapping", default=None)
    args = ap.parse_args(argv)

    paths = load_paths(args.paths)
    mapping = load_mapping(args.mapping)
    srcs = build_sources(paths)
    rdir = reports_dir(paths)

    candidates, excluded, ambiguities = [], [], []
    unmapped_counter = collections.Counter()
    stats = collections.Counter()

    # ---------------- label-mapping audit (every declared label) --------
    audit = []
    for skey, sconf in mapping["sources"].items():
        res = LabelResolver(mapping, skey)
        for lab in (sconf.get("labels") or {}):
            d = res.describe(lab)
            audit.append({
                "source_dataset": sconf.get("name", skey),
                "source_key": skey,
                "source_label": d["source_label"],
                "acronym": d["acronym"],
                "source_label_name": d["source_label_name"],
                "cardiosentry_label": d["maps_to"] or "(none)",
                "mapping_status": d["status"],
                "disqualifies_normal": d["disqualifies_normal"],
                "reason": d["reason"],
            })

    # ---------------- PTB-XL --------------------------------------------
    if "ptbxl" in srcs and srcs["ptbxl"].available():
        info("extracting PTB-XL ...")
        s = srcs["ptbxl"]
        scp = s.scp_statements()
        res = LabelResolver(mapping, "ptbxl")
        lf = make_ptbxl_likelihood_filter(mapping, scp)
        if lf:
            thr = mapping["rules"]["ptbxl_likelihood"]["min_diagnostic_likelihood"]
            assumption(f"PTB-XL diagnostic statements below likelihood {thr} are being dropped "
                       f"(rules.ptbxl_likelihood.min_diagnostic_likelihood).")
        extract_source("ptbxl", "PTBXL", s.read(), res, mapping,
                       candidates, excluded, ambiguities, unmapped_counter, stats, lf)

    # ---------------- Chapman -------------------------------------------
    if "chapman" in srcs and srcs["chapman"].available():
        info("extracting Chapman/Ningbo ...")
        s = srcs["chapman"]
        res = LabelResolver(mapping, "chapman")
        assumption("Chapman/Ningbo ships no patient identifier. Each record is assigned a "
                   "synthetic per-record patient id (CHAPMAN_SYNTH_<rec>) and flagged with "
                   "patient_id_is_synthetic=1. Repeat ECGs from the same real patient in this "
                   "source cannot be detected and may leak across a train/test split.")
        extract_source("chapman", "CHAPMAN", s.read(), res, mapping,
                       candidates, excluded, ambiguities, unmapped_counter, stats)

    # ---------------- STEMI ---------------------------------------------
    if "stemi" in srcs and srcs["stemi"].available():
        info("extracting 2026 ACS/STEMI ...")
        s = srcs["stemi"]
        res = LabelResolver(mapping, "stemi")
        n_test = s.unlabeled_test_count()
        if n_test:
            assumption(f"2026 ACS/STEMI test.csv holds {n_test} rows with no diagnostic columns; "
                       f"they are unlabelable and excluded wholesale.")
            for _ in range(0):
                pass
        extract_source("stemi", "STEMI", s.read(mapping["sources"]["stemi"]), res, mapping,
                       candidates, excluded, ambiguities, unmapped_counter, stats)
        # record the unlabeled test rows in the exclusion ledger, as a block
        if n_test:
            excluded.append({
                "source_dataset": "STEMI",
                "original_record_id": f"(test.csv: {n_test} rows)",
                "patient_id": "", "original_labels": "", "original_labels_readable": "",
                "exclusion_reason": "NO_LABELS_AVAILABLE",
                "exclusion_detail": f"CSV/test.csv contains {n_test} records but carries no "
                                    f"diagnostic columns at all, so no record in it can be labelled.",
            })

    # ---------------- unmapped labels report ----------------------------
    unmapped_rows = []
    for (skey, lab), n in unmapped_counter.most_common():
        res = LabelResolver(mapping, skey)
        d = res.describe(lab)
        unmapped_rows.append({
            "source_dataset": mapping["sources"][skey].get("name", skey),
            "source_key": skey, "source_label": lab,
            "occurrences": n, "status": d["status"],
            "disqualifies_normal": d["disqualifies_normal"],
            "reason": d["reason"],
            "action_required": ("Define this label in config/label_mapping.yaml if it is "
                                "clinically relevant to STEMI/AF/LVH/NORMAL; otherwise it stays "
                                "excluded and blocks NORMAL."),
        })
    # also surface EXCLUDE_UNCERTAIN entries that ARE declared -- they are
    # deliberate refusals to guess and belong in the same report
    for skey, sconf in mapping["sources"].items():
        res = LabelResolver(mapping, skey)
        for lab in (sconf.get("labels") or {}):
            d = res.describe(lab)
            if d["status"] in ("EXCLUDE_UNCERTAIN", "EXCLUDE_RECORD"):
                unmapped_rows.append({
                    "source_dataset": sconf.get("name", skey), "source_key": skey,
                    "source_label": f"{lab} ({d['acronym'] or d['source_label_name']})",
                    "occurrences": "(declared)", "status": d["status"],
                    "disqualifies_normal": d["disqualifies_normal"], "reason": d["reason"],
                    "action_required": "Deliberate refusal to guess. Review and decide explicitly.",
                })

    write_csv(rdir / "label_mapping_audit.csv", audit,
              ["source_dataset", "source_key", "source_label", "acronym", "source_label_name",
               "cardiosentry_label", "mapping_status", "disqualifies_normal", "reason"])
    write_csv(rdir / "extracted_candidates.csv", candidates, CANDIDATE_FIELDS)
    write_csv(rdir / "excluded_records.csv", excluded, EXCLUDED_FIELDS)
    write_csv(rdir / "unmapped_labels_report.csv", unmapped_rows,
              ["source_dataset", "source_key", "source_label", "occurrences", "status",
               "disqualifies_normal", "reason", "action_required"])
    write_csv(rdir / "ambiguity_report.csv", ambiguities,
              ["source_dataset", "original_record_id", "patient_id", "original_labels",
               "original_labels_readable", "ambiguity_flag", "ambiguity_note",
               "resolved_STEMI", "resolved_AF", "resolved_LVH", "resolved_NORMAL"])

    print(banner("STEP 2-3 -- LABEL MAPPING AND EXTRACTION"))
    print(f"  candidate records extracted : {len(candidates):,}")
    print(f"  records excluded            : {len(excluded):,}")
    print(f"  ambiguous / contradictory   : {len(ambiguities):,}")
    print(f"  undefined labels encountered: {len(unmapped_counter):,}")
    print("\n  candidates per class per source (multi-label: rows may double-count):")
    srcs_seen = sorted({k[0] for k in stats})
    print(f"    {'source':10s} {'STEMI':>7s} {'AF':>7s} {'LVH':>7s} {'NORMAL':>7s} {'records':>9s}")
    for s in srcs_seen:
        print(f"    {s:10s} {stats[(s,'STEMI')]:7d} {stats[(s,'AF')]:7d} "
              f"{stats[(s,'LVH')]:7d} {stats[(s,'NORMAL')]:7d} {stats[(s,'valid_candidates')]:9d}")
    tot = {c: sum(stats[(s, c)] for s in srcs_seen) for c in TARGET_CLASSES}
    print(f"    {'TOTAL':10s} {tot['STEMI']:7d} {tot['AF']:7d} {tot['LVH']:7d} {tot['NORMAL']:7d} "
          f"{sum(stats[(s,'valid_candidates')] for s in srcs_seen):9d}")
    for f in ("label_mapping_audit.csv", "extracted_candidates.csv", "excluded_records.csv",
              "unmapped_labels_report.csv", "ambiguity_report.csv"):
        info(f"wrote {rdir/f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
