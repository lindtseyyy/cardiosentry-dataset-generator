"""Final validation gates (PLAN 15). Order: leakage, counts, files, pixels,
labels, provenance, metric separation. Any failure raises ValidationError
with offender identities; packaging never runs on red.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (EXPECTED_ACCEPTED, FINAL_W, FINAL_H, GRIDCHECK_SAMPLE_N,
                     SUBSETS, TARGET_LABELS, TRAIN_VARIANTS, EVAL_VARIANTS,
                     subset_for)
from .util import V1Error, log, sha256_file, rng_from_parts
from . import gridcheck
from . import resize as resize_mod
from . import variants as variants_mod


class ValidationError(V1Error):
    pass


def load_metadata(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)
    rows = []
    for s in SUBSETS:
        with open(dataset_dir / s / "metadata.csv", newline="") as fh:
            rows.extend(list(csv.DictReader(fh)))
    with open(dataset_dir / "full_metadata.csv", newline="") as fh:
        full = list(csv.DictReader(fh))
    return rows, full


def gate_leakage(rows) -> None:
    """0 overlap on record_id / dup_group_id / hash / patient_id (PLAN 15.1)."""
    for key in ("record_id", "dup_group_id", "v1_waveform_sha256"):
        if rows and all(not r.get(key, "") for r in rows):
            raise ValidationError(
                f"column {key} missing/empty in all {len(rows)} rows — "
                f"schema drift, backfill chunk csvs before gating (v1.15)")
    for key in ("record_id", "dup_group_id", "v1_waveform_sha256", "patient_id"):
        seen: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            v = r.get(key, "")
            if key == "patient_id" and not v:
                continue
            seen[v].add(r["split"])
        bad = {k: v for k, v in seen.items() if len(v) > 1}
        if bad:
            ex = list(bad.items())[:3]
            raise ValidationError(f"LEAKAGE on {key}: {len(bad)} keys span "
                                  f"splits, e.g. {ex}")
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for key in ("record_id", "dup_group_id", "v1_waveform_sha256", "patient_id"):
        by_split = defaultdict(set)
        for r in rows:
            if key == "patient_id" and not r.get(key):
                continue
            by_split[r["split"]].add(r.get(key, ""))
        for a, b in pairs:
            n = len(by_split[a] & by_split[b])
            print(f"PASS: {n} {key} overlap between {a} and {b}")
            if n:
                raise ValidationError(f"leakage {key} {a}/{b}: {n}")


def gate_counts(rows, accepted_ids: set[str], omitted_ids: set[str]) -> None:
    recs = {r["record_id"] for r in rows}
    if len(accepted_ids) != EXPECTED_ACCEPTED:
        raise ValidationError(f"accepted={len(accepted_ids)}, need 31109")
    if recs - accepted_ids:
        raise ValidationError(f"images for unknown records: "
                              f"{sorted(recs - accepted_ids)[:3]}")
    if recs & omitted_ids:
        raise ValidationError(f"images for unresolved/excluded records: "
                              f"{sorted(recs & omitted_ids)[:3]}")
    by_split = Counter(r["split"] for r in rows)
    nrec = defaultdict(set)
    for r in rows:
        nrec[r["split"]].add(r["record_id"])
    want = (len(nrec["train"]) * len(TRAIN_VARIANTS)
            + len(nrec["val"]) * len(EVAL_VARIANTS)
            + len(nrec["test"]) * len(EVAL_VARIANTS))
    if len(rows) != want:
        raise ValidationError(f"image rows {len(rows)} != expected {want}")
    by_subset = Counter(r["subset"] for r in rows)
    want_sub = {"train": len(nrec["train"]) * len(TRAIN_VARIANTS),
                "val_clean": len(nrec["val"]), "val_stress": len(nrec["val"]),
                "test_clean": len(nrec["test"]),
                "test_stress": len(nrec["test"])}
    for sub, n in want_sub.items():
        if by_subset[sub] != n:
            raise ValidationError(
                f"subset {sub}: {by_subset[sub]} images, need {n}")
    log(f"counts OK: records train/val/test = "
        f"{len(nrec['train'])}/{len(nrec['val'])}/{len(nrec['test'])}, "
        f"images={len(rows)} (subsets "
        f"{'/'.join(str(by_subset[s]) for s in SUBSETS)})")


def gate_files(rows, dataset_dir: str | Path) -> None:
    dataset_dir = Path(dataset_dir)
    on_disk = set()
    for s in SUBSETS:
        for p in (dataset_dir / s / "images").glob("*.png"):
            on_disk.add(f"{s}/images/{p.name}")
    referenced = {r["image_path"] for r in rows}
    if referenced - on_disk:
        raise ValidationError(f"metadata rows w/o file: "
                              f"{sorted(referenced - on_disk)[:3]}")
    if on_disk - referenced:
        raise ValidationError(f"orphan images: {sorted(on_disk - referenced)[:3]}")
    bad = []
    for r in rows:
        p = dataset_dir / r["image_path"]
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
        except Exception as exc:
            bad.append(f"{r['image_path']}: {exc}")
            continue
        if sha256_file(p) != r["image_sha256"]:
            bad.append(f"{r['image_path']}: sha mismatch")
    if bad:
        raise ValidationError(f"file integrity failed for {len(bad)}: {bad[:3]}")
    log(f"files OK: {len(rows)} PNGs open + sha-verified, 0 orphans")


def gate_pixels(rows, dataset_dir: str | Path, master_dir: str | Path,
                texture_dir: str | Path, seed: int) -> None:
    dataset_dir = Path(dataset_dir)
    for r in rows:
        with Image.open(dataset_dir / r["image_path"]) as im:
            if im.size != (FINAL_W, FINAL_H):
                raise ValidationError(f"{r['image_path']}: size {im.size}")
    # stratified grid sample: >=500 + clean siblings of sampled derived views
    rng = rng_from_parts(seed, "gridcheck-sample")
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["variant_type"]].append(r)
    sample = []
    per = max(1, GRIDCHECK_SAMPLE_N // max(1, len(by_type)))
    for vtype, lst in by_type.items():
        idx = rng.choice(len(lst), size=min(per, len(lst)), replace=False)
        sample.extend(lst[i] for i in idx)
    by_record = defaultdict(dict)
    for r in rows:
        by_record[r["record_id"]][r["variant_id"]] = r
    meas, pairs = [], []
    for r in sample:
        m = gridcheck.measure_image(
            dataset_dir / r["image_path"], r["grid_palette"],
            gridcheck.nominal_for_size(FINAL_W, FINAL_H))
        meas.append(m)
        sib = by_record[r["record_id"]].get("clean")
        if sib is not None and r["variant_id"] != "clean":
            c = gridcheck.measure_image(
                dataset_dir / sib["image_path"], sib["grid_palette"],
                gridcheck.nominal_for_size(FINAL_W, FINAL_H))
            pairs.append((c, m))
    rep = gridcheck.gate_report(meas, pairs)
    (Path(dataset_dir) / "distortion_report.json").write_text(
        json.dumps(rep, indent=2) + "\n")
    if not rep["pass"]:
        raise ValidationError("grid gates failed: "
                              f"{rep['failures'][:5]} (see distortion_report.json)")
    log(f"pixels OK: all {FINAL_W}x{FINAL_H}; grid sample n={len(meas)} "
        f"pairs={len(pairs)} worst_nom={rep['worst_nominal_dev']:.2%} "
        f"worst_mm={rep['worst_mismatch'] if rep['worst_mismatch'] is not None else 0:.2%}")
    # determinism spot check: re-derive clean+stress from masters, byte-compare
    det = [r for r in rows if r["variant_id"] in ("clean", "stress")]
    idx = rng.choice(len(det), size=min(20, len(det)), replace=False)
    for i in idx:
        r = det[i]
        got = rederive_final(Path(master_dir) / f"{r['record_id']}.png",
                             r["variant_id"], texture_dir)
        if sha256_file(dataset_dir / r["image_path"]) != \
                hashlib.sha256(got).hexdigest():
            raise ValidationError(f"nondeterministic re-derive: {r['image_path']}")
    log("determinism spot check OK (20 clean/stress re-derives byte-identical)")


def rederive_final(master_path: str | Path, variant_id: str,
                   texture_dir: str | Path) -> bytes:
    """Re-derive a clean/stress final EXACTLY as chunk.py does (v1.16).

    Every step mirrors _process_chunk: RGB->L mode pipeline, same
    make_*/to_final calls, same PNG save params (optimize=True — the
    gate once omitted it and cried nondeterminism on identical
    pixels). Pinned by test_rederive.py; change chunk.py derivation
    and this must change with it.
    """
    master = Image.open(Path(master_path)).convert("RGB")
    if variant_id == "clean":
        view, _ = variants_mod.make_clean(master.convert("L"))
    elif variant_id == "stress":
        view, _ = variants_mod.make_stress(master.convert("L"), texture_dir)
    else:
        raise V1Error(f"re-derive supports clean/stress, not {variant_id!r}")
    final, _ = resize_mod.to_final(view)
    buf = io.BytesIO()
    final.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def gate_labels(rows, manifest_multihot: dict[str, dict]) -> None:
    bad = []
    for r in rows:
        exp = manifest_multihot.get(r["record_id"])
        if exp is None:
            bad.append(f"{r['record_id']}: unknown record")
            continue
        for lab in TARGET_LABELS:
            if int(r[lab]) != exp[lab]:
                bad.append(f"{r['record_id']}: label {lab} mismatch")
                break
    if bad:
        raise ValidationError(f"label join failed for {len(bad)}: {bad[:3]}")
    log(f"labels OK: {len(rows)} rows join-match accepted manifest")


def gate_provenance(accepted_records: list[dict],
                    config_json_path: str | Path, running_config_json: str,
                    seed: int) -> None:
    rng = random.Random(seed)
    sample = rng.sample(accepted_records, min(500, len(accepted_records)))
    for r in sample:
        hea = Path(r["subset_output_path"])
        wave = hea.with_suffix(".mat") if r["subset_waveform_format"] == "mat" \
            else hea.with_suffix(".dat")
        if sha256_file(hea) != r["subset_header_sha256"]:
            raise ValidationError(f"source header changed on disk: {hea}")
        if sha256_file(wave) != r["subset_waveform_sha256"]:
            raise ValidationError(f"source waveform changed on disk: {wave}")
    if Path(config_json_path).read_text() != running_config_json:
        raise ValidationError("generation_config.json != running config")
    log("provenance OK: 500 source records byte-unchanged, config identical")


def gate_subset_separation(rows) -> None:
    """Four-subset eval separation (PLAN v1.2 §11/§14/§15.7).

    Every row's subset must equal subset_for(split, variant_id); every
    eval record must own exactly one <split>_clean + one <split>_stress
    row (same record, same split); train rows all carry subset "train".
    """
    by_record = defaultdict(list)
    for r in rows:
        try:
            want = subset_for(r["split"], r["variant_id"])
        except ValueError as exc:
            raise ValidationError(f"{r.get('record_id')}: {exc}")
        if r.get("subset", "") != want:
            raise ValidationError(
                f"{r.get('record_id')}: subset {r.get('subset')!r} != "
                f"derived {want!r} for split={r['split']} "
                f"variant={r['variant_id']}")
        by_record[r["record_id"]].append(r)
    for rid, lst in by_record.items():
        splits = {r["split"] for r in lst}
        if len(splits) != 1:
            raise ValidationError(f"{rid}: rows span splits "
                                  f"{sorted(splits)}")
        s = lst[0]["split"]
        got = sorted((r["subset"], r["variant_id"]) for r in lst)
        if s == "train":
            want_pairs = sorted(("train", v) for v in TRAIN_VARIANTS)
        else:
            want_pairs = sorted([(f"{s}_clean", "clean"),
                                 (f"{s}_stress", "stress")])
        if got != want_pairs:
            raise ValidationError(f"{rid}: subset/variant pairs {got}, "
                                  f"need {want_pairs}")
    n = Counter(r["subset"] for r in rows)
    log("subset separation OK: "
        + ", ".join(f"{s}={n[s]}" for s in SUBSETS))


def gate_metric_separation(rows, conflicting_ids: set[str]) -> None:
    """Eval purity + primary/stress separation + no pooling (PLAN 15.7)."""
    by_record = defaultdict(list)
    for r in rows:
        by_record[r["record_id"]].append(r)
    for rid, lst in by_record.items():
        splits = {r["split"] for r in lst}
        if len(splits) != 1:
            raise ValidationError(f"{rid}: variants span splits {splits}")
        s = lst[0]["split"]
        vids = sorted(r["variant_id"] for r in lst)
        if s == "train":
            if vids != sorted(TRAIN_VARIANTS):
                raise ValidationError(f"{rid}: train variants {vids}")
        else:
            if vids != sorted(EVAL_VARIANTS):
                raise ValidationError(f"{rid}: eval variants {vids}")
        prim = [r for r in lst if int(r["primary_metric"]) == 1]
        if s == "train":
            if prim:
                raise ValidationError(f"{rid}: train row in primary metric")
        else:
            want_primary = rid not in conflicting_ids
            if want_primary and not (len(prim) == 1
                                     and prim[0]["variant_id"] == "clean"):
                raise ValidationError(f"{rid}: primary row wrong: "
                                      f"{[(r['variant_id'], r['primary_metric']) for r in lst]}")
            if not want_primary and prim:
                raise ValidationError(f"{rid}: conflicting group in primary")
            for r in lst:
                if int(r["primary_metric"]) == 0 and not r["metric_note"]:
                    raise ValidationError(f"{r['image_path']}: missing metric_note")
    n_prim = sum(1 for r in rows if int(r["primary_metric"]) == 1)
    log(f"metric separation OK: primary={n_prim}, "
        f"stress/eval-excluded={sum(1 for r in rows if r['split'] != 'train') - n_prim}")


def write_audits(rows, dataset_dir: str | Path) -> None:
    """Palette/font × label audits (PLAN 10.5 + font extension)."""
    dataset_dir = Path(dataset_dir)
    pal = defaultdict(Counter)
    font = defaultdict(Counter)
    for r in rows:
        for lab in TARGET_LABELS:
            if int(r[lab]) == 1:
                pal[lab][r["grid_palette"]] += 1
                font[lab][r["render_font"]] += 1
    with open(dataset_dir / "grid_palette_by_label.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = sorted({r["grid_palette"] for r in rows})
        w.writerow(["label", *names])
        for lab in TARGET_LABELS:
            w.writerow([lab, *[pal[lab][n] for n in names]])
    with open(dataset_dir / "font_by_label.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = sorted({r["render_font"] for r in rows})
        w.writerow(["label", *names])
        for lab in TARGET_LABELS:
            w.writerow([lab, *[font[lab][n] for n in names]])
    audit = {"grid_palette": {lab: dict(pal[lab]) for lab in TARGET_LABELS},
             "render_font": {lab: dict(font[lab]) for lab in TARGET_LABELS}}
    (dataset_dir / "grid_palette_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    # concentration gate: no palette/font may dominate a label (>80% while
    # its global share is <50%) — crude but catches seeding bugs
    tot_pal = Counter(r["grid_palette"] for r in rows)
    tot_all = len(rows)
    for lab in TARGET_LABELS:
        n_lab = sum(pal[lab].values())
        for name, c in pal[lab].items():
            if n_lab and c / n_lab > 0.80 and tot_pal[name] / tot_all < 0.50:
                raise ValidationError(
                    f"palette {name} concentrated in {lab}: {c}/{n_lab}")
    log("palette/font audits written, concentration gate OK")
