"""Union-find split groups + duplicate/conflict audits (PLAN 0.3, 4.1, 4.5).

Atomic groups = connected components over patient edges + waveform-hash
edges. Conflict audit includes an executable mapping check: each member's
target set is recomputed from its own dx codes via diagnosis_mapping.csv.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .config import TARGET_LABELS
from .util import V1Error, log


class UnionFind:
    def __init__(self, items):
        self.par = {x: x for x in items}

    def find(self, a):
        par = self.par
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.par[rb] = ra


def build_groups(records: list[dict], pinfo: dict[str, dict]) -> dict:
    """Return group_id -> list[record]. Group ids G000000.. by min record_id."""
    uf = UnionFind([r["record_id"] for r in records])
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_hash[r["v1_waveform_sha256"]].append(r["record_id"])
    for v in by_hash.values():
        for rid in v[1:]:
            uf.union(v[0], rid)
    by_pid: dict[str, list[str]] = defaultdict(list)
    for r in records:
        pid = pinfo[r["record_id"]]["patient_id"]
        if pid:
            by_pid[pid].append(r["record_id"])
    for v in by_pid.values():
        for rid in v[1:]:
            uf.union(v[0], rid)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[uf.find(r["record_id"])].append(r)
    groups = {}
    for i, key in enumerate(sorted(buckets, key=lambda k: min(
            r["record_id"] for r in buckets[k]))):
        groups[f"G{i:06d}"] = sorted(buckets[key],
                                     key=lambda r: r["record_id"])
    n_multi = sum(1 for v in groups.values() if len(v) > 1)
    log(f"union groups: {len(groups)} "
        f"({n_multi} multi-record, "
        f"{sum(len(v) for v in groups.values() if len(v) > 1)} records)")
    return groups


def group_edge_types(members: list[dict], pinfo: dict[str, dict]) -> str:
    """patient / hash / both / single for one group."""
    if len(members) < 2:
        return "single"
    hashes = {m["v1_waveform_sha256"] for m in members}
    pids = {pinfo[m["record_id"]]["patient_id"] for m in members} - {""}
    has_h = len(hashes) < len(members)
    has_p = len(pids) == 1 and bool(pids)
    if has_h and has_p:
        return "both"
    if has_h:
        return "hash"
    if has_p:
        return "patient"
    # multi-record only via TRANSITIVE union (a-b patient, b-c hash, …)
    return "transitive"


def load_code_to_targets(mapping_csv: str | Path) -> dict[str, set[str]]:
    """SNOMED code -> target labels from MAP_*_TARGET rows only."""
    out: dict[str, set[str]] = defaultdict(set)
    with open(mapping_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["normalization_decision"] in ("MAP_EXACT_TARGET",
                                                 "MAP_CHALLENGE_SCORING_EQUIVALENT"):
                if row["normalized_target_label"] in TARGET_LABELS:
                    out[row["original_diagnosis_code"]].add(
                        row["normalized_target_label"])
    return out


def expected_targets(dx_codes: str, code_map: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for code in (dx_codes or "").split("|"):
        code = code.strip()
        if code:
            out |= code_map.get(code, set())
    return out


def conflict_taxonomy(members: list[dict]) -> str:
    sets = [set(m["target_labels"].split("|")) for m in members]
    inter = set.intersection(*sets)
    union = set.union(*sets)
    if not inter:
        return "DISJOINT"
    if any(s == union for s in sets) and any(s == inter for s in sets) \
            and inter != union:
        return "SUBSET_SUPERSET"
    return "PARTIAL"


def find_conflicting_groups(groups: dict, code_map: dict) -> list[dict]:
    """Identical-waveform subgroups whose members disagree on target labels.

    Crucially scoped to shared-`v1_waveform_sha256` subgroups: same-patient /
    different-ECG members of a union group legitimately carry different
    labels and are NOT conflicts. Includes a per-member mapping recompute.
    """
    out = []
    for gid, members in groups.items():
        by_hash: dict[str, list[dict]] = defaultdict(list)
        for m in members:
            by_hash[m["v1_waveform_sha256"]].append(m)
        for v in by_hash.values():
            if len(v) < 2 or len({m["target_labels"] for m in v}) < 2:
                continue
            per_member = []
            mapping_clean = True
            for m in v:
                exp = expected_targets(m.get("v1_original_dx_codes", ""),
                                       code_map)
                got = set(m["target_labels"].split("|"))
                ok = (exp == got)
                mapping_clean &= ok
                per_member.append({
                    "record_id": m["record_id"], "source": m["source_dataset"],
                    "targets": m["target_labels"],
                    "orig_labels": m.get("v1_all_original_labels", ""),
                    "dx": m.get("v1_original_dx_codes", ""),
                    "mapping_recomputes": ok})
            out.append({
                "dup_group_id": gid,
                "n_members": len(v),
                "taxonomy": conflict_taxonomy(v),
                "members": per_member,
                "mapping_clean": mapping_clean,
                "verdict": ("annotation_disagreement_unresolved"
                            if mapping_clean
                            else "MAPPING_MISMATCH_INVESTIGATE"),
            })
    return out


def write_duplicate_audit(groups: dict, pinfo: dict, path: str | Path) -> int:
    multi = {g: v for g, v in groups.items() if len(v) > 1}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dup_group_id", "edge_types", "n_members", "record_ids",
                    "sources", "patient_ids", "target_label_sets"])
        for gid in sorted(multi):
            v = multi[gid]
            w.writerow([
                gid, group_edge_types(v, pinfo), len(v),
                ";".join(m["record_id"] for m in v),
                ";".join(sorted({m["source_dataset"] for m in v})),
                ";".join(sorted({pinfo[m["record_id"]]["patient_id"]
                                 for m in v} - {""}) or [""]),
                ";".join(sorted({m["target_labels"] for m in v}))])
    log(f"duplicate audit: {len(multi)} groups -> {path}")
    return len(multi)


def write_conflict_audit(conflicts: list[dict], path: str | Path) -> int:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dup_group_id", "taxonomy", "verdict", "mapping_clean",
                    "metric_excluded", "record_id", "source", "targets",
                    "orig_labels", "dx_codes", "mapping_recomputes"])
        for c in conflicts:
            for m in c["members"]:
                w.writerow([c["dup_group_id"], c["taxonomy"], c["verdict"],
                            c["mapping_clean"], True, m["record_id"],
                            m["source"], m["targets"], m["orig_labels"],
                            m["dx"], m["mapping_recomputes"]])
    bad = [c for c in conflicts if not c["mapping_clean"]]
    if bad:
        raise V1Error(f"{len(bad)} conflicting groups FAIL the mapping "
                      f"recompute: {[c['dup_group_id'] for c in bad][:5]} — "
                      "investigate before proceeding")
    log(f"conflict audit: {len(conflicts)} groups "
        f"({sum(c['n_members'] for c in conflicts)} records), "
        "all mapping-clean, all verdict=annotation_disagreement_unresolved")
    return len(conflicts)
