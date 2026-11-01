#!/usr/bin/env python3
"""Export test ECGs a clean12 run gets exactly right, clean AND augmented, per label combination.

Reads ``<run>/<architecture>/test_predictions.csv`` (``record_id`` is the HEEDB
record path, e.g. ``I0006/WFDB/2012/MUSE_..._30000.hea``; a clean+augmented run
has one row per view, ``variant_id`` clean / augmented). All predictions use
the operating thresholds in ``--thresholds`` (default: the ``clinical_v1``
file under ``threshold-configurations/`` fitted for this run's
``inference.pt``), which must equal what the app serves. Candidates are the
test ECGs whose stored probabilities, thresholded that way, give exactly the
true label set on all 12 labels in every view the run evaluated, and whose
true label set equals the clean12 manifest's labels for that record.

``--reuse <previous sample set>`` first re-scores that set's own clean and
augmented PNGs with this model; the ECGs still exactly right on both (and
stored-exact for this run) are kept, the others are dropped, and only the
remaining places are screened anew.

Per label combination (single conditions and co-occurring ones) candidates are
tried in a seeded order, distinct patients first (``--seed``; the order of
``export_cardiologist_samples.select_records``). Each tried record is rendered
twice from its WFDB signal (local clean12 extract, symlinked; HEEDB download
only if missing; every signal checked against the manifest SHA-256):

* clean: the clean12 dataset renderer (the image type the model was tested on);
* augmented: the EC2 augmented-dataset pipeline
  (``clean_convnext_codes/clean12_augmented_ecg_images_ec2/augment_worker.py``,
  ``Clean12_Augmented_ECG_Images_EC2.ipynb``) with the record's own levels from
  that notebook's whole-manifest balanced plan and its ``AUGMENT_SEED``, i.e.
  the sheet the EC2 run writes for this record.

Both image files are then scored by the model exactly as the CardioSentry app
serves it (``cardiosentry-application`` loader, the run's ``inference.pt``;
CPU fp32), thresholded with the operating thresholds. A record is exported only if
the predictions for BOTH images equal the ground truth exactly; otherwise the
next candidate replaces it, until ``--per-combination`` pass or the candidates
run out. ``screening.csv`` lists every tried record and why it was kept or not.
After export every image is re-scored from disk (``verification.csv``), and
every level of every balanced augmentation factor must appear among the
exported augmented images. ``augmentation_breakdown.py`` then shows every
augmentation step on its own on one of these ECGs.

Output layout::

    <out>/single/<CODE>_<name>/NN_<site>_<file_id>_clean.png
    <out>/single/<CODE>_<name>/NN_<site>_<file_id>_augmented_<grid>_tilt<deg>.png
    <out>/combo_<k>/<A+B+...>/...       k = 2..5 (NO_EXACT_MATCH.txt if none qualified)
    <out>/augmentation_effects/         one augmentation step per image
    <out>/index.csv                     one row per ECG: record ID, labels, plan, predictions
    <out>/screening.csv                 every tried record: kept, or which image failed
    <out>/verification.csv              every check, per ECG
    <out>/README.txt
    <out>/_work/                        WFDB records (symlinks), kit masters, kit

Usage::

    .venv/bin/python export_correct_combinations.py \\
        --run ../../prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_aug_colab_a100_seed42 \\
        --reuse ../../prototype-results/new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42/correct_prediction_samples

Run it with the repository ``.venv`` (wfdb; matplotlib 3.11.1 reproduces the
clean12 sheets pixel-exactly, older versions differ in the lead labels). The
model is scored on ``--scorer-python`` (default ``/usr/bin/python3``, the
interpreter the app runs on, with its torch/timm).

AWS credentials (``AWS_PROFILE=lindtsey-cardiosentry``) are needed only if a
record is missing from ``--local-root``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

import augmentation_breakdown
import export_cardiologist_samples as base
from export_correct_predictions import LABELS

LOCAL_ROOT = base.HERE.parent.parent / "dataset-files" / "HEEDB" / "extracted" / base.ARCHIVE_ROOT
AUGMENT_DIR = base.HERE.parent / "clean_convnext_codes" / "clean12_augmented_ecg_images_ec2"
AUGMENT_NOTEBOOK = "Clean12_Augmented_ECG_Images_EC2.ipynb"
APP_ROOT = base.HERE.parent.parent / "cardiosentry-application"
THRESHOLD_ROOT = base.HERE.parent.parent / "threshold-configurations"
ORDER = list(base.LABEL_NAMES)  # canonical clean12 order, as in the manifest's `labels`
PLAN_KEYS = ("grid_palette", "ink_bin", "tilt_bin", "perspective", "fold_creases", "edge_shadow", "blur")


def canonical(labels) -> str:
    return "+".join(c for c in ORDER if c in labels)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_thresholds(inference_pt: Path) -> Path:
    """The clinical_v1 file under threshold-configurations/ fitted for this inference.pt."""
    sha = sha256_file(inference_pt)
    hits = [p for p in sorted(THRESHOLD_ROOT.rglob("thresholds_clinical_v1.json"))
            if json.loads(p.read_text())["model_checkpoint_sha256"]["inference.pt"] == sha]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} thresholds_clinical_v1.json files under {THRESHOLD_ROOT} are fitted "
                         f"for {inference_pt} (sha256 {sha[:12]}…); pass --thresholds")
    return hits[0]


def load_thresholds(path: Path, inference_pt: Path) -> tuple[dict[str, float], str]:
    """(label -> threshold, policy name) from a frozen threshold-study file; it must
    have been fitted for this run's inference.pt and use `>=`."""
    frozen = json.loads(path.read_text())
    if frozen["model_checkpoint_sha256"]["inference.pt"] != sha256_file(inference_pt):
        raise SystemExit(f"{path} was fitted for another inference.pt, not {inference_pt}")
    if frozen["comparison_operator"] != ">=":
        raise SystemExit(f"{path}: comparison {frozen['comparison_operator']!r}, expected '>='")
    if sorted(frozen["label_order"]) != sorted(LABELS):
        raise SystemExit(f"{path}: labels {frozen['label_order']} != {LABELS}")
    return dict(zip(frozen["label_order"], frozen["selected_thresholds"])), frozen["policy_version"]


def load_test_predictions(run: Path, architecture: str) -> tuple[list[dict], tuple[str, ...]]:
    """(one row per test ECG with truth and stored probabilities per view, the views).
    Views are `variant_id` (clean; clean + augmented for a clean+augmented run);
    a file without that column holds the clean view only."""
    with (run / "record_split.csv").open(newline="", encoding="utf-8") as handle:
        patients = {r["record_path"]: r["patient_key"] for r in csv.DictReader(handle)}
    rows: dict[str, dict] = {}
    with (run / architecture / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rec, view = row["record_id"], row.get("variant_id") or "clean"
            truth = frozenset(c for c in LABELS if row[f"true_{c}"] == "1")
            entry = rows.setdefault(rec, {"record_path": rec, "site": row["source"],
                                          "file_id": PurePosixPath(rec).stem, "patient_key": patients[rec],
                                          "true": truth, "views": {}})
            if entry["true"] != truth or view in entry["views"] or row["split"] != "test":
                raise SystemExit(f"{rec} ({view}): non-test row, duplicate view or inconsistent truth")
            entry["views"][view] = {c: float(row[f"prob_{c}"]) for c in LABELS}
    views = {tuple(sorted(r["views"], key=lambda v: v != "clean")) for r in rows.values()}
    if len(views) != 1 or next(iter(views))[0] != "clean":
        raise SystemExit(f"test_predictions.csv: views differ between records or lack 'clean': {views}")
    return list(rows.values()), next(iter(views))


def exact_matches(rows: list[dict], manifest: dict[str, dict],
                  thresholds: dict[str, float]) -> list[dict]:
    """Manifest rows of the test ECGs whose stored probabilities, at the operating
    thresholds, give exactly the true label set in every view."""
    keep = []
    for row in rows:
        entry = manifest.get(row["record_path"])
        if entry is None:
            raise SystemExit(f"{row['record_path']}: not in the clean12 manifest")
        truth = frozenset(entry["labels"].split(";"))
        if truth != row["true"]:
            raise SystemExit(f"{row['record_path']}: test_predictions ground truth "
                             f"{canonical(row['true'])} != manifest {entry['labels']}")
        if all(frozenset(c for c in LABELS if probs[c] >= thresholds[c]) == truth
               for probs in row["views"].values()):
            keep.append({**entry, "pred": truth, "prob": row["views"]["clean"], "stored": row["views"]})
    return keep


def stage_signals(selected: list[dict], local_root: Path, signals_dir: Path,
                  access_point: str) -> list[str]:
    """Symlink local WFDB files into signals_dir, download the rest from HEEDB.
    Sets row['record_source']; returns the records that could not be retrieved."""
    missing = []
    for row in selected:
        rel = PurePosixPath(row["record_path"])
        hea = local_root.joinpath(*rel.parts)
        signal = base.header_signal_file(hea)[0] if hea.is_file() else None
        if signal is None or not (hea.parent / signal).is_file():
            missing.append(row)
            continue
        dest = signals_dir.joinpath(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        for link, target in ((dest, hea), (dest.parent / signal, hea.parent / signal)):
            link.unlink(missing_ok=True)
            link.symlink_to(target.resolve())
        row["record_source"] = "local"
    failed = []
    for row in missing:
        try:
            base.download_records([row], signals_dir, access_point)
            row["record_source"] = "HEEDB download"
        except Exception as exc:  # report; the next candidate takes its place
            failed.append(f"{row['record_path']}: {type(exc).__name__}: {exc}")
    for row in selected:
        if "record_source" in row:
            base.verify_signal(signals_dir / row["record_path"], row["sha256"])
    return failed


# --------------------------------------------------------------------------
# EC2 augmentation (pool children)
# --------------------------------------------------------------------------
def notebook_seed(augment_dir: Path) -> int:
    """AUGMENT_SEED of the EC2 augmented-dataset notebook."""
    for source in base.notebook_code_cells(augment_dir / AUGMENT_NOTEBOOK):
        match = re.search(r"^AUGMENT_SEED = (\d+)", source, re.M)
        if match:
            return int(match.group(1))
    raise SystemExit(f"AUGMENT_SEED not found in {augment_dir / AUGMENT_NOTEBOOK}")


def load_augmenter(augment_dir: Path, kit_dir: Path):
    """augment_worker (needs ecg_clean_worker importable: base.prepare_render)."""
    if str(augment_dir) not in sys.path:
        sys.path.insert(0, str(augment_dir))
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(augment_dir),
                                                                 os.environ.get("PYTHONPATH")]))
    aw = importlib.import_module("augment_worker")
    aw.set_texture_dir(kit_dir / "CreasesWrinkles" / "wrinkles-dataset")
    return aw


def _init_augment(augment_dir: str, kit_dir: str) -> None:
    load_augmenter(Path(augment_dir), Path(kit_dir))


def _augment_one(task: tuple) -> dict:
    """Grey kit master -> the EC2 run's augmented sheet for this record."""
    master_png, out_png, seed, rel, plan, level = task
    import augment_worker as aw
    with Image.open(master_png) as master:
        final, meta = aw.augment_master(master, seed, rel, plan)
    aw.check_plan(meta, plan)
    final.save(out_png, compress_level=level)
    return meta


# --------------------------------------------------------------------------
# Model scoring: a child process on the app's interpreter (system python3:
# the app's torch/timm), fed JSON lines. The parent never imports torch.
# --------------------------------------------------------------------------
def serve_scorer(app_root: str, model_id: str) -> None:
    """Child: load the app model, answer one JSON line (list of image paths) per request."""
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # anything the app prints goes to stderr, not into the protocol
    sys.path.insert(0, app_root)
    import numpy as np
    from backend.services import runner
    from backend.services.registry import discover
    ld = discover()[model_id]
    lm = runner.load_model(ld)
    labels = list(ld.descriptor.labels)
    norm = ld.descriptor.preprocess.normalize
    proto.write(json.dumps({"labels": labels, "thresholds": list(lm.thresholds),
                            "checkpoint_sha256": ld.checkpoint_sha256, "weights": str(ld.weights_path),
                            "python": sys.executable}) + "\n")
    proto.flush()
    for line in sys.stdin:
        out = []
        for path in json.loads(line):
            views, _ = runner.derive_views(lm, Path(path), norm.mean, norm.std)
            logits = runner._forward_views(lm, views).float().numpy()[0]
            probs = 1.0 / (1.0 + np.exp(-logits))
            out.append({label: float(p) for label, p in zip(labels, probs)})
        proto.write(json.dumps(out) + "\n")
        proto.flush()


class Scorer:
    def __init__(self, python: str, app_root: Path, model_id: str, thresholds: dict[str, float],
                 inference_pt: Path):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        self.proc = subprocess.Popen([python, str(Path(__file__).resolve()), "--serve-scorer",
                                      str(app_root), model_id],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        self.info = json.loads(self._read())
        run_sha = hashlib.sha256(inference_pt.read_bytes()).hexdigest()
        if self.info["checkpoint_sha256"] != run_sha:
            raise SystemExit(f"app model {model_id} weights {self.info['checkpoint_sha256']} "
                             f"are not the run's {inference_pt} ({run_sha})")
        for label, served in zip(self.info["labels"], self.info["thresholds"]):
            if abs(served - thresholds[label]) > 1e-6:
                raise SystemExit(f"{label}: app serves threshold {served}, operating threshold is "
                                 f"{thresholds[label]}; point the app descriptor at the same policy")
        self.thresholds = thresholds
        print(f"Scoring with app model {model_id} on {self.info['python']} "
              f"({Path(self.info['weights']).name}, sha256 {run_sha[:12]}… = run inference.pt)", flush=True)

    def _read(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"model scorer exited (code {self.proc.wait()})")
        return line

    def score(self, paths: list[Path]) -> list[dict[str, float]]:
        self.proc.stdin.write(json.dumps([str(p) for p in paths]) + "\n")
        self.proc.stdin.flush()
        return json.loads(self._read())

    def predicted(self, probs: dict[str, float]) -> frozenset:
        return frozenset(c for c in LABELS if probs[c] >= self.thresholds[c])

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


# --------------------------------------------------------------------------
# Screening: render, augment and score candidates until each combination is full
# --------------------------------------------------------------------------
def candidate_order(exact: list[dict], seed: int) -> dict[str, list[dict]]:
    """Per label combination: every exactly-correct record, in draw order."""
    groups: dict[str, list[dict]] = {}
    for row in base.select_records(exact, len(exact), seed):
        groups.setdefault(row["labels"], []).append(row)
    return groups


def assess(batch: list[dict], scorer: Scorer) -> list[str]:
    """Score each row's clean and augmented PNG; one status per row."""
    probs = scorer.score([p for r in batch for p in (r["clean_path"], r["augmented_path"])])
    statuses = []
    for i, row in enumerate(batch):
        truth = frozenset(row["labels"].split(";"))
        row["clean_prob"], row["augmented_prob"] = probs[2 * i], probs[2 * i + 1]
        row["clean_pred"] = scorer.predicted(row["clean_prob"])
        row["augmented_pred"] = scorer.predicted(row["augmented_prob"])
        clean_ok, aug_ok = row["clean_pred"] == truth, row["augmented_pred"] == truth
        statuses.append("exported" if clean_ok and aug_ok else "clean_wrong" if aug_ok
                        else "augmented_wrong" if clean_ok else "both_wrong")
    return statuses


def reuse_previous(previous: Path, groups: dict[str, list[dict]], manifest: dict[str, dict],
                   plan: dict[str, dict], scorer: Scorer, work: Path, signals_dir: Path,
                   local_root: Path, access_point: str,
                   per_combination: int) -> tuple[dict[str, list[dict]], list[dict]]:
    """Re-score a previous sample set's own PNGs with this model.
    (rows still exactly right on both images, per combination; one screening row
    per previous ECG, status prefixed `previous_`)."""
    with (previous / "index.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    candidates = {r["record_path"]: r for members in groups.values() for r in members}
    screening, batch = [], []
    for old in index:
        rec = old["record_id"]
        row = candidates.get(rec)
        if row is None:
            screening.append({"record": dict(manifest[rec]), "status": "previous_stored_not_exact"})
            continue
        if any(str(plan[rec][k]) != old[k] for k in PLAN_KEYS):
            raise SystemExit(f"{previous}: {rec} has plan levels other than the EC2 plan's")
        folder = work / "previous" / old["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        row["clean_path"] = Path(shutil.copy2(previous / old["clean_image"], folder))
        row["augmented_path"] = Path(shutil.copy2(previous / old["augmented_image"], folder))
        row["plan"] = plan[rec]
        row["augment_meta"] = {"grid_palette": old["grid_palette"], "grid_saturation": old["grid_saturation"],
                               "tilt_deg": float(old["tilt_deg"])}
        row["previous_master"] = previous / "_work" / "masters" / f"{row['site']}_{row['file_id']}.png"
        batch.append(row)
    print(f"Previous set {previous}: {len(index)} ECGs, {len(batch)} stored-exact for this run; "
          f"re-scoring {2 * len(batch)} images ...", flush=True)
    accepted: dict[str, list[dict]] = {}
    for row, status in zip(batch, assess(batch, scorer)):
        if status == "exported" and len(accepted.get(row["labels"], [])) < per_combination:
            accepted.setdefault(row["labels"], []).append(row)
        else:
            screening.append({"record": row, "status": f"previous_{status}"})
    kept = [r for rows in accepted.values() for r in rows]
    unretrieved = stage_signals(kept, local_root, signals_dir, access_point)
    if unretrieved:
        raise SystemExit("previous ECGs whose signal could not be staged:\n  " + "\n  ".join(unretrieved))
    for row in kept:
        master = augmentation_breakdown.master_path(work, row["site"], row["file_id"])
        if row["previous_master"].is_file():  # grey kit master, for augmentation_effects/
            master.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["previous_master"], master)
        screening.append({"record": row, "status": "previous_kept"})
    print(f"  kept {len(kept)} of {len(index)} previous ECGs", flush=True)
    return accepted, screening


def screen(groups: dict[str, list[dict]], accepted: dict[str, list[dict]], per_combination: int, args,
           work: Path, signals_dir: Path, local_root: Path, aug_seed: int, plan: dict[str, dict],
           scorer: Scorer, level: int) -> tuple[list[dict], list[dict]]:
    """(accepted rows with images and scores, one screening row per tried record).
    `accepted` (per combination) is filled up to `per_combination` from `groups`."""
    candidates_dir = work / "candidates"
    tried = {key: 0 for key in groups}
    accepted = {key: list(accepted.get(key, [])) for key in groups}
    screening: list[dict] = []
    round_no = 0
    while True:
        batch = []
        for key, members in groups.items():
            need = per_combination - len(accepted[key])
            take = members[tried[key]:tried[key] + max(need, 0)]
            tried[key] += len(take)
            batch += take
        if not batch:
            break
        round_no += 1
        print(f"Round {round_no}: {len(batch)} candidates", flush=True)
        unretrieved = stage_signals(batch, local_root, signals_dir, args.heedb_access_point)
        for problem in unretrieved:
            print(f"NOT RETRIEVED (replaced by the next candidate): {problem}")
        for row in batch:
            if "record_source" not in row:
                screening.append({"record": row, "status": "not_retrieved"})
        batch = [r for r in batch if "record_source" in r]
        jobs = [{"rel": r["record_path"], "folder": base.label_folder(r["labels"]).as_posix(),
                 "dir": str(candidates_dir / base.label_folder(r["labels"])),
                 "stem": f"{r['site']}_{r['file_id']}", "row": r,
                 "master_copy": str(augmentation_breakdown.master_path(work, r["site"], r["file_id"]))}
                for r in batch]
        results = base.render(jobs, signals_dir, work, args.notebook.resolve(), args.kit_bundle.resolve(),
                              base.AUGMENT_CODE_DIR, args.workers, args.seed, 0)
        failures = [f"{rel}: {r['error']}" for rel, r in results.items() if "error" in r]
        if failures:
            raise SystemExit(f"{len(failures)} records failed to render:\n  " + "\n  ".join(failures))

        tasks = []
        for job in jobs:
            row = job["row"]
            row["clean_path"] = Path(job["dir"]) / results[job["rel"]]["clean"]
            row["augmented_path"] = Path(job["dir"]) / f"{job['stem']}_augmented.png"
            row["plan"] = plan[row["record_path"]]
            tasks.append((job["master_copy"], str(row["augmented_path"]), aug_seed,
                          row["record_path"], row["plan"], level))
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_augment,
                                 initargs=(str(args.augment_dir.resolve()), str(work / "kit" /
                                           "ecg_image_v1_code" / "ecg-image-generator"))) as pool:
            for row, meta in zip(batch, pool.map(_augment_one, tasks)):
                row["augment_meta"] = meta
        print(f"  augmented {len(batch)}; scoring {2 * len(batch)} images ...", flush=True)

        for row, status in zip(batch, assess(batch, scorer)):
            screening.append({"record": row, "status": status})
            if status == "exported":
                accepted[row["labels"]].append(row)
        print(f"  kept {sum(len(v) for v in accepted.values())} so far", flush=True)
    return [r for rows in accepted.values() for r in rows], screening


def screening_rows(screening: list[dict]) -> list[dict]:
    out = []
    for item in screening:
        row = item["record"]
        out.append({
            "combination": canonical(row["labels"].split(";")), "record_id": row["record_path"],
            "patient_key": row["patient_key"], "status": item["status"],
            "clean_pred_labels": canonical(row["clean_pred"]) if "clean_pred" in row else "",
            "augmented_pred_labels": canonical(row["augmented_pred"]) if "augmented_pred" in row else "",
            **{k: row["plan"][k] if "plan" in row else "" for k in PLAN_KEYS},
        })
    return out


# --------------------------------------------------------------------------
# Verification from the files on disk
# --------------------------------------------------------------------------
def verify_output(out: Path, run: Path, architecture: str,
                  thresholds: dict[str, float], manifest: dict[str, dict],
                  signals_dir: Path, per_combination: int, scorer: Scorer,
                  plan: dict[str, dict], plan_factors: dict[str, tuple]) -> list[dict]:
    """Re-check every exported ECG from the files on disk; raises on any failure."""
    with (out / "index.csv").open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    predictions = {r["record_path"]: r for r in load_test_predictions(run, architecture)[0]}  # test rows only
    with (run / "record_split.csv").open(newline="", encoding="utf-8") as handle:
        split = {r["record_path"]: r["split"] for r in csv.DictReader(handle)}
    print(f"Verifying: re-scoring {2 * len(index)} exported images from disk ...", flush=True)
    rescored = scorer.score([out / r[k] for r in index for k in ("clean_image", "augmented_image")])
    checks = []
    for i, row in enumerate(index):
        rec = row["record_id"]
        pred_row, entry = predictions[rec], manifest[rec]
        truth = frozenset(entry["labels"].split(";"))
        stored = {view: frozenset(c for c in LABELS if probs[c] >= thresholds[c])
                  for view, probs in pred_row["views"].items()}
        clean_pred = scorer.predicted(rescored[2 * i])
        augmented_pred = scorer.predicted(rescored[2 * i + 1])
        try:
            base.verify_signal(signals_dir / rec, entry["sha256"])
            signal_ok = True
        except RuntimeError:
            signal_ok = False
        stem = f"_{entry['site']}_{entry['file_id']}_"
        clean, augmented = out / row["clean_image"], out / row["augmented_image"]
        prefix = clean.name.removesuffix("clean.png")  # NN_<site>_<file_id>_
        images_ok = (clean.name.endswith("_clean.png") and stem in prefix
                     and augmented.name.startswith(prefix + "augmented_") and clean.parent == augmented.parent)
        for path in (clean, augmented):
            images_ok &= path.is_file()
            if path.is_file():
                with Image.open(path) as image:
                    images_ok &= image.size == (1024, 768)
        checks.append({
            "folder": row["folder"], "record_id": rec,
            "test_split": split.get(rec) == "test",
            "truth_matches_manifest": pred_row["true"] == truth and row["true_labels"] == canonical(truth),
            "stored_probabilities_exact": (all(s == truth for s in stored.values())
                                           and row["pred_labels"] == canonical(truth)),
            "model_on_clean_image_exact": clean_pred == truth and row["clean_pred_labels"] == canonical(truth),
            "model_on_augmented_image_exact": (augmented_pred == truth
                                               and row["augmented_pred_labels"] == canonical(truth)),
            "augmentation_plan_matches_ec2": all(str(plan[rec][k]) == row[k] for k in PLAN_KEYS),
            "folder_matches_labels": row["folder"] == base.label_folder(entry["labels"]).as_posix()
                                     and int(row["n_labels"]) == len(truth),
            "signal_sha256_matches_manifest": signal_ok and row["signal_sha256"] == entry["sha256"],
            "clean_and_augmented_present": images_ok,
        })
    per_folder = Counter(row["folder"] for row in index)
    failed = [c for c in checks if not all(v for k, v in c.items() if k not in ("folder", "record_id"))]
    problems = [f"{c['record_id']}: {[k for k, v in c.items() if v is False]}" for c in failed]
    problems += [f"{folder}: {n} ECGs > {per_combination}" for folder, n in per_folder.items()
                 if n > per_combination]
    if len({row["record_id"] for row in index}) != len(index):
        problems.append("a record is exported twice")
    for factor, levels in plan_factors.items():
        absent = sorted(set(map(str, levels)) - {row[factor] for row in index})
        if absent:
            problems.append(f"augmentation factor {factor}: levels {absent} not among the exported images")
    with (out / "verification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    if problems:
        raise SystemExit("Verification failed:\n  " + "\n  ".join(problems))
    print(f"Verified {len(checks)} ECGs x {len(checks[0]) - 2} checks, "
          f"{len(per_folder)} folders, max {max(per_folder.values())} per folder")
    return checks


def coverage_lines(index_rows: list[dict], plan_factors: dict[str, tuple], aw) -> list[str]:
    names = {"ink_bin": lambda i: f"{aw.INK_BINS[i][0]:.2f}-{aw.INK_BINS[i][1]:.2f}",
             "tilt_bin": lambda i: f"{aw.TILT_BINS[i][0]:+.0f} to {aw.TILT_BINS[i][1]:+.0f} deg"}
    lines = []
    for factor, levels in plan_factors.items():
        counts = Counter(str(row[factor]) for row in index_rows)
        label = names.get(factor, lambda level: {0: "off", 1: "on"}.get(level, level))
        lines.append(f"  {factor:<13} " + ", ".join(f"{label(level)}: {counts[str(level)]}" for level in levels))
    return lines


README = """Correctly classified clean12 test ECGs, clean and augmented, per label combination
==================================================================================

Run: {run}
Model: {architecture}; operating thresholds: {policy} (below; from {threshold_file}).
Test set: {n_test:,} ECGs (patient-level split; no test patient was trained on).

Every ECG here is predicted exactly right on BOTH of its images: the model
flags every true condition and nothing else, across all 12 labels, on the
clean sheet and on the augmented (photographed-paper) sheet. Checks per ECG:
  1. the run's stored test probabilities ({views} view), at the operating
     thresholds, give exactly the true label set;
  2. the model, loaded exactly as the CardioSentry app serves it ({app_model};
     weights = the run's inference.pt, CPU fp32; the app serves these same
     thresholds), scores the clean PNG in this folder and the augmented PNG in
     this folder, and both thresholded predictions equal the ground truth;
  3. the ground truth equals the clean12 manifest; the signal matches its
     manifest SHA-256.
{reused}Candidates that failed check 2 were replaced by the next candidate of the same
combination (screening.csv lists every tried ECG and which image failed).
Partially correct predictions are never included. Candidate order is a seeded
draw (seed {seed}), distinct patients first, not the most confident cases; at
most {per_combination} per combination.

Folders
  single/<CODE>_<name>/   ECGs with exactly one condition
  combo_<k>/<A+B+...>/    ECGs with exactly these k conditions together
  augmentation_effects/   each augmentation step on its own (see its README)

Files (per ECG, same number prefix)
  NN_<site>_<id>_clean.png
      The ECG sheet as rendered for the clean12 dataset (the image type the
      model was trained and tested on): 1024x768, 3x4 leads + Lead II rhythm
      strip, 25 mm/s, 10 mm/mV, red grid.
  NN_<site>_<id>_augmented_<grid colour>_tilt<degrees>.png
      The same record as the EC2 augmented-dataset notebook
      (Clean12_Augmented_ECG_Images_EC2.ipynb, augment_worker.py, seed
      {aug_seed}) writes it: paper/photo augmentation with stronger wrinkles
      and creases than the old stage 4, and this record's own balanced-plan
      levels (grid colour, ink strength, tilt, perspective, fold creases, edge
      shadow, blur). Then JPEG q85 and the clean view's 1024x768 fit.
<site>_<id> is the HEEDB record; index.csv has the full HEEDB record ID
(record path), patient key, true labels, the run's stored test prediction and
probabilities (prob_* = clean view; stored_augmented_prob_* = augmented view,
if the run evaluated it), the model's prediction and probabilities on each
image (clean_*, augmented_*), and the augmentation plan levels. verification.csv
lists every check, per ECG.

Records: {sources}.{unretrieved}

Augmentation coverage among the exported augmented images (every level of
every balanced factor appears; checked):
{coverage}

Combinations in the test set:
  ECGs -> stored probabilities exactly correct -> tried here -> exported
  (tried includes previous-set ECGs re-scored here; tried - exported = failed
  on the clean or augmented image, see screening.csv)
{counts}

Operating thresholds ({policy}; prediction = probability >= threshold)
{thresholds}

Label codes
{labels}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run group folder (holds record_split.csv)")
    parser.add_argument("--architecture", default="convnext_tiny_rfpn")
    parser.add_argument("--thresholds", type=Path,
                        help="frozen threshold-study file (label_order + selected_thresholds); default: the "
                             "thresholds_clinical_v1.json under threshold-configurations/ fitted for the run's "
                             "inference.pt. The app model must serve the same values")
    parser.add_argument("--out", type=Path, help="default: <run>/correct_prediction_samples")
    parser.add_argument("--reuse", type=Path,
                        help="previous sample set: keep its ECGs this model still gets exactly right")
    parser.add_argument("--per-combination", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="candidate order")
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT,
                        help="local clean12 extract (holds manifest.csv and the WFDB records)")
    parser.add_argument("--manifest", help="clean12 manifest.csv or archive; default: <local-root>/manifest.csv")
    parser.add_argument("--heedb-access-point", default=base.HEEDB_ACCESS_POINT)
    parser.add_argument("--effects-record", help="<folder>/<NN> for augmentation_effects/; default: chosen")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--notebook", type=Path,
                        default=base.RENDER_CODE_DIR / "Clean12_ECG_Signal_Images_EC2.ipynb")
    parser.add_argument("--kit-bundle", type=Path, default=base.RENDER_CODE_DIR / "ecg_image_v1_code.tar.gz")
    parser.add_argument("--augment-dir", type=Path, default=AUGMENT_DIR,
                        help="EC2 augmented-dataset folder (augment_worker.py + its notebook)")
    parser.add_argument("--app", type=Path, default=APP_ROOT, help="cardiosentry-application root")
    parser.add_argument("--app-model", help="app model id whose weights are the run's inference.pt; "
                                            "default: the app's models/active.yaml")
    parser.add_argument("--scorer-python", default="/usr/bin/python3",
                        help="interpreter the app runs on (scores the images)")
    args = parser.parse_args()
    if args.per_combination < 1 or args.workers < 1:
        parser.error("--per-combination and --workers must be >= 1")
    app_root = args.app.expanduser().resolve()
    app_model = args.app_model or re.search(r"^id:\s*(\S+)", (app_root / "models" / "active.yaml").read_text(),
                                            re.M).group(1)

    run = args.run.expanduser().resolve()
    out = (args.out or run / "correct_prediction_samples").expanduser().resolve()
    previous = args.reuse.expanduser().resolve() if args.reuse else None
    if previous == out:
        parser.error("--reuse must not be the output folder")
    work = out / "_work"
    signals_dir = work / "signals"
    local_root = args.local_root.expanduser().resolve()
    augment_dir = args.augment_dir.expanduser().resolve()
    inference_pt = run / args.architecture / "inference.pt"
    threshold_file = (args.thresholds.expanduser().resolve() if args.thresholds
                      else find_thresholds(inference_pt))
    thresholds, policy = load_thresholds(threshold_file, inference_pt)

    manifest_rows = base.read_manifest(args.manifest or str(local_root / "manifest.csv"))
    manifest = {r["record_path"]: r for r in manifest_rows}
    rows, views = load_test_predictions(run, args.architecture)
    exact = exact_matches(rows, manifest, thresholds)
    groups = candidate_order(exact, args.seed)
    tested = Counter(canonical(r["true"]) for r in rows)
    correct = Counter(canonical(r["labels"].split(";")) for r in exact)
    print(f"Operating thresholds {policy}: {len(rows):,} test ECGs, {len(exact):,} exactly correct "
          f"({' + '.join(views)} view), {len(correct)} of {len(tested)} combinations", flush=True)

    scorer = Scorer(args.scorer_python, app_root, app_model, thresholds, inference_pt)
    try:
        work.mkdir(parents=True, exist_ok=True)
        for stale in [signals_dir, work / "masters", work / "candidates", work / "previous"]:
            shutil.rmtree(stale, ignore_errors=True)
        config = base.prepare_render(work, args.notebook.resolve(), args.kit_bundle.resolve(),
                                     base.AUGMENT_CODE_DIR)
        aw = load_augmenter(augment_dir, Path(config["kit_dir"]))
        aug_seed = notebook_seed(augment_dir)
        plan = aw.balanced_plan([{"record_path": r["record_path"], "site": r["site"], "labels": r["labels"]}
                                 for r in manifest_rows], aug_seed)
        level = config["png_compress_level"]

        accepted, screening = {}, []
        if previous:
            accepted, screening = reuse_previous(previous, groups, manifest, plan, scorer, work, signals_dir,
                                                 local_root, args.heedb_access_point, args.per_combination)
            done = {item["record"]["record_path"] for item in screening}
            groups = {key: [r for r in members if r["record_path"] not in done]
                      for key, members in groups.items()}
        selected, screened_new = screen(groups, accepted, args.per_combination, args, work, signals_dir,
                                        local_root, aug_seed, plan, scorer, level)
        screening += screened_new

        for stale in [out / "single", *out.glob("combo_*"), out / "augmentation_effects"]:
            shutil.rmtree(stale, ignore_errors=True)
        jobs = base.plan_jobs(out, selected)
        index_rows = []
        for job in jobs:
            row = job["row"]
            meta = row["augment_meta"]
            folder = Path(job["dir"])
            folder.mkdir(parents=True, exist_ok=True)
            clean = f"{job['stem']}_clean.png"
            augmented = f"{job['stem']}_augmented_{meta['grid_palette']}_tilt{meta['tilt_deg']:+.1f}.png"
            shutil.copyfile(row["clean_path"], folder / clean)
            shutil.copyfile(row["augmented_path"], folder / augmented)
            index_rows.append({
                "folder": job["folder"], "clean_image": f"{job['folder']}/{clean}",
                "augmented_image": f"{job['folder']}/{augmented}",
                "record_id": row["record_path"], "site": row["site"], "file_id": row["file_id"],
                "patient_key": row["patient_key"], "n_labels": row["n_labels"],
                "true_labels": canonical(row["labels"].split(";")), "pred_labels": canonical(row["pred"]),
                "clean_pred_labels": canonical(row["clean_pred"]),
                "augmented_pred_labels": canonical(row["augmented_pred"]),
                "label_names": "; ".join(base.LABEL_NAMES[c] for c in row["labels"].split(";")),
                "record_source": row["record_source"], "signal_sha256": row["sha256"],
                **{k: row["plan"][k] for k in PLAN_KEYS},
                "grid_saturation": meta["grid_saturation"], "tilt_deg": meta["tilt_deg"],
                **{f"prob_{c}": f"{row['prob'][c]:.4f}" for c in ORDER},
                **{f"clean_prob_{c}": f"{row['clean_prob'][c]:.4f}" for c in ORDER},
                **{f"augmented_prob_{c}": f"{row['augmented_prob'][c]:.4f}" for c in ORDER},
                **{f"stored_{view}_prob_{c}": f"{row['stored'][view][c]:.4f}"
                   for view in views[1:] for c in ORDER},
            })
        with (out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)
        screened = screening_rows(screening)
        with (out / "screening.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(screened[0]))
            writer.writeheader()
            writer.writerows(screened)

        exported = Counter(canonical(r["labels"].split(";")) for r in selected)
        tried = Counter(r["combination"] for r in screened if r["status"] != "previous_stored_not_exact")
        max_k = max(len(r["true"]) for r in rows)
        for k in range(2, max_k + 1):
            if not exported or all(key.count("+") + 1 != k for key in exported):
                folder = out / f"combo_{k}"
                folder.mkdir(parents=True, exist_ok=True)
                combos = [key for key in tested if key.count("+") + 1 == k]
                (folder / "NO_EXACT_MATCH.txt").write_text(
                    f"No {k}-condition test ECG was predicted exactly right on both its clean and its\n"
                    f"augmented image, so this folder is empty.\n"
                    f"{k}-condition combinations in the test set (ECGs, stored probabilities exact, "
                    f"tried, exported):\n"
                    + "".join(f"  {key}: {tested[key]}, {correct[key]}, {tried[key]}, 0\n" for key in combos))
        counts = []
        for k in range(1, max_k + 1):
            keys = sorted((key for key in tested if key.count("+") + 1 == k), key=lambda key: -tested[key])
            title = "Single" if k == 1 else f"Combo {k}"
            counts.append(f"  {title}: {len(keys)} combinations, {sum(key in exported for key in keys)} exported, "
                          f"{sum(exported[key] for key in keys)} ECGs")
            counts += [f"    {key:<26} {tested[key]:>5} -> {correct[key]:>5} -> {tried[key]:>3} -> {exported[key]}"
                       for key in keys]
        sources = Counter(r["record_source"] for r in selected)
        not_retrieved = [r["record_id"] for r in screened if r["status"] == "not_retrieved"]
        previous_status = Counter(r["status"] for r in screened if r["status"].startswith("previous_"))
        reused = (f"{previous_status['previous_kept']} of {sum(previous_status.values())} ECGs were carried "
                  f"over from the previous sample set\n({previous}): its own PNGs, re-scored with this model "
                  f"and still exactly right\non both; the others were dropped "
                  f"({', '.join(f'{n} {s}' for s, n in previous_status.items() if s != 'previous_kept')}).\n"
                  if previous else "")
        (out / "README.txt").write_text(README.format(
            run=run.name, architecture=args.architecture, policy=policy, n_test=len(rows),
            threshold_file=os.path.relpath(threshold_file, base.HERE.parent.parent),
            views=" and ".join(views), reused=reused,
            seed=args.seed, per_combination=args.per_combination,
            app_model=app_model, aug_seed=aug_seed,
            sources=", ".join(f"{n} {source}" for source, n in sources.items()),
            unretrieved="".join(f"\nNOT RETRIEVED (replaced by the next candidate): {p}" for p in not_retrieved),
            coverage="\n".join(coverage_lines(index_rows, aw.PLAN_FACTORS, aw)),
            counts="\n".join(counts),
            thresholds="\n".join(f"  {c:<6} {thresholds[c]:.4f}" for c in ORDER),
            labels="\n".join(f"  {code:<6} {name}" for code, name in base.LABEL_NAMES.items())))

        verify_output(out, run, args.architecture, thresholds, manifest, signals_dir,
                      args.per_combination, scorer, plan, aw.PLAN_FACTORS)
    finally:
        scorer.close()
    shutil.rmtree(work / "candidates", ignore_errors=True)
    shutil.rmtree(work / "previous", ignore_errors=True)
    augmentation_breakdown.export(out, aug_seed, args.effects_record, args.notebook.resolve(),
                                  args.kit_bundle.resolve(), augment_dir, args.workers)
    status = Counter(r["status"] for r in screened)
    print(f"Done: {len(index_rows)} ECGs x (clean + augmented) in "
          f"{len({r['folder'] for r in index_rows})} label folders under {out}; "
          f"screened {len(screened)}: {dict(status)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--serve-scorer"]:
        serve_scorer(*sys.argv[2:4])
    else:
        main()
