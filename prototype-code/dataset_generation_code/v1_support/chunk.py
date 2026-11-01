"""Per-chunk build worker (PLAN 13.13). Top-level + picklable so the
ProcessPoolExecutor can run it. One chunk = 200 records, all stages:

  stage Arial -> render Arial -> save masters -> train font pass ->
  variants -> resize -> save finals -> atomic index_parts/chunk csv.

A chunk either completes fully or raises after appending loud rows to
failures.csv. Resume = validate-and-skip finished chunks.
"""
from __future__ import annotations

import csv
import shutil
import time as _time
from pathlib import Path

from PIL import Image

from .config import (FIXED_FONT, MASTER_W, MASTER_H, SEED, TARGET_LABELS,
                     TRAIN_VARIANTS, EVAL_VARIANTS, subset_for)
from .util import V1Error, ensure_dir, log, sha256_file
from . import render as render_mod
from . import resize as resize_mod
from . import stage as stage_mod
from . import variants as variants_mod

CHUNK_FIELDS = ["record_id", "source_record_id", "patient_id",
                "patient_recovery_method", "upstream_filename_hr", "ecg_id",
                "strat_fold", "dup_group_id", "v1_waveform_sha256",
                "variant_id", "variant_type",
                "primary_metric", "metric_note", "split", "subset",
                "image_path",
                *TARGET_LABELS,
                "source_dataset", "grid_palette", "rotation_deg",
                "render_font", "image_sha256", "image_bytes", "master_sha256"]


def chunk_csv_path(index_parts_dir: str | Path, chunk_id: int) -> Path:
    return Path(index_parts_dir) / f"chunk_{chunk_id:05d}.csv"


def chunk_complete(index_parts_dir: str | Path, chunk_id: int,
                   dataset_dir: str | Path) -> bool:
    """A chunk is done iff its csv exists and every listed image exists."""
    csv_path = chunk_csv_path(index_parts_dir, chunk_id)
    if not csv_path.exists():
        return False
    try:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return False
    if not rows:
        return False
    dataset_dir = Path(dataset_dir)
    return all((dataset_dir / r["image_path"]).exists() for r in rows)


def rotate_failures_log(path: str | Path) -> str | None:
    """Resume hygiene for failures.csv (PLAN v1.14).

    Chunk failures append rows and re-raise, killing the §13 cell — but a
    resumed run that OVERCOMES the failure would still trip the end-of-§13
    "failures.csv empty" assert on the stale rows. Call once at §13 start:
    a log with data rows is moved aside (timestamped, audit preserved) so
    the fresh run starts clean; header-only/missing logs are untouched.
    Returns the rotated filename, or None if nothing was rotated.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    stamp = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
    rotated = path.with_name(f"failures_{stamp}.csv")
    path.rename(rotated)
    log(f"rotated stale failures.csv ({len(rows)} rows) -> {rotated.name}")
    return rotated.name


def backfill_hash_column(index_parts_dir: str | Path,
                         hash_by_record_id: dict[str, str]) -> dict:
    """Add v1_waveform_sha256 to chunk csvs that lack it (PLAN v1.15).

    Chunks rendered before v1.15 wrote no hash column, which made
    gate_leakage read '' for every row and cry LEAKAGE. Backfills from
    the accepted manifest (no re-render: images + masters untouched).
    Atomic per file (tmp + replace); idempotent (second run verifies);
    never overwrites a present value — conflicts raise. Every row must
    resolve to a non-empty manifest hash or the backfill refuses.
    """
    files = sorted(Path(index_parts_dir).glob("chunk_*.csv"))
    if not files:
        raise V1Error(f"no chunk_*.csv in {index_parts_dir}")
    added = verified = 0
    for f in files:
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            old_fields = list(reader.fieldnames or [])
            rows = list(reader)
        if "v1_waveform_sha256" in old_fields:
            for r in rows:
                want = hash_by_record_id.get(r["record_id"])
                if not want:
                    raise V1Error(f"{f.name}: no manifest hash for "
                                  f"{r['record_id']}")
                if r["v1_waveform_sha256"] != want:
                    raise V1Error(f"{f.name}: hash conflict for "
                                  f"{r['record_id']} (csv != manifest)")
            verified += len(rows)
            continue
        for r in rows:
            want = hash_by_record_id.get(r["record_id"])
            if not want:
                raise V1Error(f"{f.name}: no manifest hash for "
                              f"{r['record_id']}")
            r["v1_waveform_sha256"] = want
        tmp = f.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=old_fields + ["v1_waveform_sha256"])
            w.writeheader()
            w.writerows(rows)
        tmp.replace(f)
        added += len(rows)
    log(f"hash backfill: {len(files)} files, {added} rows backfilled, "
        f"{verified} rows verified")
    return {"files": len(files), "added": added, "verified": verified}


def _append_failures(path: str | Path, chunk_id: int, where: str,
                     exc: BaseException) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["chunk_id", "stage", "error"])
        w.writerow([chunk_id, where, f"{type(exc).__name__}: {exc}"])


def _save_master(png: Path, master_dir: Path, record_id: str) -> tuple[str, str]:
    with Image.open(png) as im:
        if im.size != (MASTER_W, MASTER_H):
            raise V1Error(f"{record_id}: master size {im.size}, "
                          f"need {(MASTER_W, MASTER_H)}")
    dst = master_dir / f"{record_id}.png"
    shutil.copy2(png, dst)
    return str(dst), sha256_file(dst)


def process_chunk(job: dict) -> dict:
    """Run one chunk. job keys: chunk_id, records, paths, seed, fonts_url.

    records: list of dicts with record_id, split, subset_output_path,
      labels (12-tuple), multihot, dup_group_id, v1_waveform_sha256,
      pinfo, conflicting, source_dataset, target_labels, v1 ids.
    paths: scratch_dir, render_dir, master_dir, dataset_dir, index_parts_dir,
      failures_csv, kit_dir, kit_python, texture_dir.
    """
    cid = job["chunk_id"]
    recs = job["records"]
    P = job["paths"]
    seed = job.get("seed", SEED)
    failures_csv = P["failures_csv"]
    try:
        return _process_chunk(cid, recs, P, seed)
    except Exception as exc:
        _append_failures(failures_csv, cid, "chunk", exc)
        raise


def _process_chunk(cid: int, recs: list[dict], P: dict, seed: int) -> dict:
    scratch = ensure_dir(Path(P["scratch_dir"]) / f"chunk_{cid:05d}")
    render_out = ensure_dir(Path(P["render_dir"]) / f"chunk_{cid:05d}")
    master_dir = ensure_dir(P["master_dir"])
    dataset_dir = ensure_dir(P["dataset_dir"])
    kit_dir, kit_python = P["kit_dir"], P["kit_python"]
    texture_dir = P["texture_dir"]
    fonts = render_mod.kit_fonts(kit_dir)
    T: dict[str, float] = {}

    # --- Arial pass (all records): stage -> render -> masters -------------
    _t0 = _time.perf_counter()
    stage_arial = scratch / "arial"
    stage_mod.stage_chunk(
        [{"record_id": r["record_id"],
          "subset_output_path": r["subset_output_path"]} for r in recs],
        stage_arial)
    render_mod.run_batch(kit_dir, kit_python, stage_arial,
                         render_out / "arial", FIXED_FONT, seed)
    got = render_mod.collect_pngs(render_out / "arial",
                                  {r["record_id"] for r in recs})
    render_mod.clean_render_dir(render_out / "arial")
    master_of, master_sha = {}, {}
    for rid, png in got.items():
        path, sha = _save_master(png, master_dir, rid)
        master_of[rid] = path
        master_sha[rid] = sha
    T["arial"] = _time.perf_counter() - _t0

    # --- train font pass (seeded per-record fonts, transient masters) -----
    _t0 = _time.perf_counter()
    font_master_of, font_master_sha, font_of = {}, {}, {}
    train_recs = [r for r in recs if r["split"] == "train"]
    if train_recs:
        for font, group in render_mod.plan_font_jobs(
                train_recs, fonts, seed).items():
            sub = scratch / ("font_" + "".join(
                c if c.isalnum() else "_" for c in font))
            stage_mod.stage_chunk(
                [{"record_id": r["record_id"],
                  "subset_output_path": r["subset_output_path"]}
                 for r in group], sub)
            fout = render_out / ("font_" + "".join(
                c if c.isalnum() else "_" for c in font))
            render_mod.run_batch(kit_dir, kit_python, sub, fout, font, seed)
            gotf = render_mod.collect_pngs(
                fout, {r["record_id"] for r in group})
            render_mod.clean_render_dir(fout)
            for rid, png in gotf.items():
                with Image.open(png) as im:
                    if im.size != (MASTER_W, MASTER_H):
                        raise V1Error(f"{rid}: font-master size {im.size}")
                font_master_of[rid] = str(png)
                font_master_sha[rid] = sha256_file(png)
                font_of[rid] = font
    T["font"] = _time.perf_counter() - _t0

    # --- variants ----------------------------------------------------------
    T["variants"] = T["resize"] = T["save_png"] = 0.0
    rows = []
    for r in recs:
        rid, split = r["record_id"], r["split"]
        arial = Image.open(master_of[rid]).convert("RGB")
        variants = TRAIN_VARIANTS if split == "train" else EVAL_VARIANTS
        for vid in variants:
            _t1 = _time.perf_counter()
            if vid == "clean":
                view, meta = variants_mod.make_clean(arial.convert("L"))
                font = FIXED_FONT
                msha = master_sha[rid]
            elif vid == "stress":
                view, meta = variants_mod.make_stress(arial.convert("L"),
                                                      texture_dir)
                font = FIXED_FONT
                msha = master_sha[rid]
            else:
                fmaster = Image.open(font_master_of[rid]).convert("L")
                try:
                    view, meta = variants_mod.make_augmented(
                        fmaster, rid, vid, texture_dir, seed)
                finally:
                    fmaster.close()
                font = font_of[rid]
                msha = font_master_sha[rid]
            T["variants"] += _time.perf_counter() - _t1
            _t1 = _time.perf_counter()
            final, _rs = resize_mod.to_final(view)
            T["resize"] += _time.perf_counter() - _t1
            subset = subset_for(split, vid)
            rel = f"{subset}/images/{rid}__{vid}.png"
            dst = dataset_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            _t1 = _time.perf_counter()
            final.save(dst, "PNG", optimize=True)
            T["save_png"] += _time.perf_counter() - _t1
            if r["split"] in ("val", "test") and vid == "clean" \
                    and not r["conflicting"]:
                primary, note = 1, ""
            else:
                primary = 0
                if r["split"] == "train":
                    note = "train_split"
                elif vid == "stress":
                    note = ("conflicting_duplicate_labels:"
                            "annotation_disagreement_unresolved"
                            if r["conflicting"] else "stress_view")
                else:
                    note = ("conflicting_duplicate_labels:"
                            "annotation_disagreement_unresolved")
            row = {"record_id": rid, "source_record_id": rid,
                   "patient_id": r["pinfo"]["patient_id"],
                   "patient_recovery_method": r["pinfo"]["method"],
                   "upstream_filename_hr": r["pinfo"]["upstream_filename_hr"],
                   "ecg_id": r["pinfo"]["ecg_id"],
                   "strat_fold": r["pinfo"]["strat_fold"],
                   "dup_group_id": r["dup_group_id"],
            "v1_waveform_sha256": r["v1_waveform_sha256"],
            "variant_id": vid,
                   "variant_type": meta["variant_type"],
                   "primary_metric": primary, "metric_note": note,
                   "split": split, "subset": subset, "image_path": rel,
                   "source_dataset": r["source_dataset"],
                   "grid_palette": meta["grid_palette"],
                   "rotation_deg": meta["rotation_deg"], "render_font": font,
                   "image_sha256": sha256_file(dst),
                   "image_bytes": dst.stat().st_size, "master_sha256": msha}
            for lab in TARGET_LABELS:
                row[lab] = r["multihot"][lab]
            rows.append(row)
        arial.close()

    # --- atomic chunk csv ---------------------------------------------------
    _t0 = _time.perf_counter()
    csv_path = chunk_csv_path(P["index_parts_dir"], cid)
    tmp = csv_path.with_suffix(".csv.tmp")
    ensure_dir(csv_path.parent)
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CHUNK_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)

    # --- cleanup transient scratch (masters + finals + csv persist) ---------
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.rmtree(render_out, ignore_errors=True)
    T["csv"] = _time.perf_counter() - _t0
    return {"chunk_id": cid, "n_records": len(recs), "n_images": len(rows),
            "bytes": sum(r["image_bytes"] for r in rows),
            "timings": {k: round(v, 2) for k, v in T.items()}}
