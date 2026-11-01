"""Grouped iterative multi-label stratification (PLAN 4.2-4.4).

Pure greedy pass (Sechidis-style) on union groups; NO blanket rare-combo
forcing (v1.1 amendment 3). Forced placement exists only as a logged minimal
repair if the feasibility gates fail.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .config import (SEED, SPLIT_RATIO, TARGET_LABELS, LABEL_MIN_ABSOLUTE,
                     LABEL_MIN_FRACTION, LABEL_RATIO_TOL, SOURCE_PP_TOL,
                     SOURCE_MIN_RECORDS_GATE)
from .util import V1Error, log, rng_from_parts, atomic_write_text

SPLITS = ("train", "val", "test")


def _group_label_vector(members: list[dict]) -> Counter:
    """OR over members (stratification arithmetic only; labels never merged)."""
    out: Counter = Counter()
    for m in members:
        for lab in m["labels_tuple"]:
            out[lab] += 1
    for lab in out:
        out[lab] = 1
    return out


def iterative_split(groups: dict[str, list[dict]], seed: int = SEED) -> dict[str, str]:
    """Greedy rarest-label-first placement. Deterministic for fixed input."""
    gids = sorted(groups)
    rng = rng_from_parts(seed, "split-order")
    # desired RECORD counts per label per split (groups carry OR-vectors, but
    # floors are counted in records, so desire is record-based)
    label_total = Counter()
    for gid in gids:
        for lab in _group_label_vector(groups[gid]):
            label_total[lab] += len(groups[gid])
    desired = {lab: {s: label_total[lab] * SPLIT_RATIO[s] for s in SPLITS}
               for lab in TARGET_LABELS}
    have = {lab: Counter() for lab in TARGET_LABELS}
    # process groups holding the rarest labels first (stable, seeded ties)
    rarity = {lab: label_total[lab] for lab in TARGET_LABELS}
    order = sorted(gids, key=lambda g: (
        min(rarity[lab] for lab in _group_label_vector(groups[g])),
        rng.random()))
    assign: dict[str, str] = {}
    for gid in order:
        vec = _group_label_vector(groups[gid])
        n = len(groups[gid])
        # Split with the largest total RELATIVE deficit over this group's
        # labels. Relative (not absolute) deficits let the 10% splits
        # compete fairly with train; absolute deficits systematically starve
        # val/test for mid-size labels (observed: LAE 89 vs want 125).
        def deficit(s):
            tot = 0.0
            for lab in vec:
                want = desired[lab][s]
                if want > 0:
                    tot += max(0.0, (want - have[lab][s]) / want)
            return tot
        scored = [(deficit(s), rng.random(), s) for s in SPLITS]
        best = max(scored)[2]
        assign[gid] = best
        for lab in vec:
            have[lab][best] += n
    return assign


def evaluate(assign: dict[str, str], groups: dict) -> dict:
    """Per-label and per-source split counts (records)."""
    label = {lab: Counter() for lab in TARGET_LABELS}
    source = defaultdict(Counter)
    nsplit = Counter()
    for gid, s in assign.items():
        for m in groups[gid]:
            nsplit[s] += 1
            source[m["source_dataset"]][s] += 1
            for lab in m["labels_tuple"]:
                label[lab][s] += 1
    return {"label": label, "source": source, "nsplit": nsplit}


def check_gates(ev: dict, n_records: int) -> list[str]:
    """Return a list of gate violations (empty = pass)."""
    bad = []
    for lab in TARGET_LABELS:
        tot = sum(ev["label"][lab].values())
        floor = max(LABEL_MIN_ABSOLUTE, LABEL_MIN_FRACTION * tot)
        for s in SPLITS:
            got = ev["label"][lab][s]
            if got < floor:
                bad.append(f"label {lab}/{s}: {got} < floor {floor:.0f}")
            want = tot * SPLIT_RATIO[s]
            if want > 0 and abs(got - want) / want > LABEL_RATIO_TOL:
                bad.append(f"label {lab}/{s}: {got} vs want {want:.0f} "
                           f"(>{LABEL_RATIO_TOL:.0%} off)")
    for src, counts in ev["source"].items():
        tot = sum(counts.values())
        if tot < SOURCE_MIN_RECORDS_GATE:
            continue  # documented exception (ptb: 15), reported exactly
        for s in SPLITS:
            pp = 100 * counts[s] / tot
            want_pp = 100 * SPLIT_RATIO[s]
            if abs(pp - want_pp) > SOURCE_PP_TOL:
                bad.append(f"source {src}/{s}: {pp:.1f}% vs {want_pp:.1f}%")
    if sum(ev["nsplit"].values()) != n_records:
        bad.append("split counts do not sum to record count")
    return bad


def repair(assign: dict[str, str], groups: dict, seed: int = SEED,
           max_moves: int = 500) -> tuple[dict[str, str], list[str]]:
    """Minimal logged repair: move smallest feasible groups into deficient
    splits until gates pass. Only called when the greedy pass is infeasible.
    """
    assign = dict(assign)
    moves: list[str] = []
    rng = rng_from_parts(seed, "split-repair")
    n_records = sum(len(v) for v in groups.values())
    for _ in range(max_moves):
        viol = check_gates(evaluate(assign, groups), n_records)
        if not viol:
            return assign, moves
        # parse first violation of form "label L/S: ..." or "source S/S: ..."
        first = viol[0]
        if first.startswith("label"):
            lab, rest = first.split(" ")[1].split("/")
            need_s = rest.rstrip(":")
            cands = [g for g, s in assign.items() if s != need_s
                     and lab in _group_label_vector(groups[g])]
        else:
            src, rest = first.split(" ")[1].split("/")
            need_s = rest.rstrip(":")
            cands = [g for g, s in assign.items()
                     if s != need_s and groups[g][0]["source_dataset"] == src
                     and all(m["source_dataset"] == src for m in groups[g])]
        if not cands:
            raise V1Error(f"repair infeasible: {first} (no movable group)")
        cands.sort(key=lambda g: (len(groups[g]), rng.random()))
        g = cands[0]
        moves.append(f"{g} ({len(groups[g])} rec) {assign[g]}->{need_s}: {first}")
        assign[g] = need_s
    raise V1Error(f"repair exceeded {max_moves} moves; last: "
                  f"{check_gates(evaluate(assign, groups), n_records)[:3]}")


SPLIT_FIELDS = ["record_id", "dup_group_id", "patient_id",
                "patient_recovery_method", "source_dataset",
                "v1_original_record_id", "target_labels",
                *TARGET_LABELS,
                "subset_output_path", "v1_waveform_sha256",
                "upstream_filename_hr", "ecg_id", "strat_fold"]


def save_splits(splits_dir: str | Path, assign: dict[str, str],
                groups: dict, pinfo: dict, moves: list[str]) -> dict:
    """Write train/val/test_records.csv + split_report.txt. Refuses overwrite."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        if (splits_dir / f"{s}_records.csv").exists():
            raise V1Error(f"{s}_records.csv already exists — refusing to "
                          "re-split (resume reuses existing manifests)")
    ev = evaluate(assign, groups)
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for gid, s in assign.items():
        for m in groups[gid]:
            pi = pinfo[m["record_id"]]
            row = {"record_id": m["record_id"], "dup_group_id": gid,
                   "patient_id": pi["patient_id"],
                   "patient_recovery_method": pi["method"],
                   "source_dataset": m["source_dataset"],
                   "v1_original_record_id": m["v1_original_record_id"],
                   "target_labels": m["target_labels"],
                   "subset_output_path": m["subset_output_path"],
                   "v1_waveform_sha256": m["v1_waveform_sha256"],
                   "upstream_filename_hr": pi["upstream_filename_hr"],
                   "ecg_id": pi["ecg_id"], "strat_fold": pi["strat_fold"]}
            for lab in TARGET_LABELS:
                row[lab] = m["multihot"][lab]
            by_split[s].append(row)
    for s in SPLITS:
        by_split[s].sort(key=lambda r: r["record_id"])
        with open(splits_dir / f"{s}_records.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SPLIT_FIELDS)
            w.writeheader()
            w.writerows(by_split[s])
    # combination coverage note (best-effort reporting, not a gate)
    combos = defaultdict(Counter)
    for gid, s in assign.items():
        for m in groups[gid]:
            combos[m["target_labels"]][s] += 1
    uncovered = sum(1 for c, d in combos.items() if len(d) < 3)
    lines = [f"split seed: {SEED}", f"ratio: {SPLIT_RATIO}",
             f"groups: {len(groups)}",
             "records: " + ", ".join(
                 f"{s}={ev['nsplit'][s]}" for s in SPLITS),
             "", "per-label counts (train/val/test):"]
    for lab in TARGET_LABELS:
        c = ev["label"][lab]
        lines.append(f"  {lab:5s} {c['train']:6d} {c['val']:6d} {c['test']:6d}")
    lines += ["", "per-source counts (train/val/test):"]
    for src in sorted(ev["source"]):
        c = ev["source"][src]
        lines.append(f"  {src:28s} {c['train']:6d} {c['val']:6d} {c['test']:6d}")
    lines += ["",
              f"distinct combinations: {len(combos)}; "
              f"in all 3 splits: {len(combos) - uncovered}; "
              f"partial: {uncovered} (best-effort, expected for rare combos)",
              "",
              f"forced placements (repair): {len(moves)}"]
    lines += [f"  {m}" for m in moves]
    atomic_write_text(splits_dir / "split_report.txt", "\n".join(lines) + "\n")
    log(f"splits saved: " + ", ".join(
        f"{s}={ev['nsplit'][s]}" for s in SPLITS) + f" (repair moves: {len(moves)})")
    return {"nsplit": dict(ev["nsplit"]), "moves": moves}
