#!/usr/bin/env python3
"""Export a small, label-organized HEEDB clean12 review set for a cardiologist.

For every label combination in the clean12 manifest (single conditions and
co-occurring conditions alike, 118 combinations), this script:

1. reads the clean12 record manifest (labels, patient key, signal SHA-256)
   from the head of the clean12 archive in S3 (~10 MB read, then the stream is
   closed), picks at most ``--per-combination`` records per combination
   (distinct patients first, deterministic for a given ``--seed``), and
   downloads only those records' ``.hea`` + signal files from the HEEDB
   (BDSP) S3 access point the clean12 selection was built from, checking every
   signal against the manifest SHA-256 (~1,000 files, ~60 MB at the default);
2. renders each record with the exact clean12 rendering code: the worker and
   the kit settings are loaded from
   ``../clean_convnext_codes/clean12_ecg_signal_images_ec2/Clean12_ECG_Signal_Images_EC2.ipynb``
   and the SHA-256-pinned ``ecg_image_v1_code.tar.gz`` kit next to it, so the
   images match the cloud dataset (1024x768, 3x4 + Lead II rhythm strip,
   25 mm/s, 10 mm/mV);
3. writes an augmented copy of each sheet with the image pipeline's paper/photo
   augmentation, ``stage4_augment.augment`` from
   ``dataset-combination-code/ecg-dataset-pipeline/image_pipeline/`` (the code
   ``prototype-code/cardiosentry_generate_images.ipynb`` runs as stage 4):
   random grid colour (red/orange/pink/grey) and ink fading, tilt of up to
   3 degrees, keystone perspective, wrinkled-paper texture, fold creases,
   uneven lighting and edge shadow, white-balance drift, exposure and contrast
   jitter, sensor noise, mild defocus, and JPEG compression. It is applied to
   the same grey kit master the clean view is recoloured from, then fitted to
   the same 1024x768 frame;
4. organizes both images per label folder with an ``index.csv``.

Output layout::

    <out>/README.txt
    <out>/index.csv
    <out>/single/AF_Atrial_fibrillation/01_<site>_<file_id>_clean.png
    <out>/single/AF_Atrial_fibrillation/01_<site>_<file_id>_augmented_orange_tilt-1.4.png
    <out>/combo_2/LQT+AF/...
    <out>/combo_3/STACH+STEMI+STE/...
    <out>/_work/signals/...   downloaded WFDB records (kept, so a rerun skips the download)

Requirements (Python 3.10+)::

    pip install boto3 numpy scipy Pillow matplotlib pandas qrcode pyyaml "wfdb>=4.3"

AWS credentials need ``s3:GetObject`` on the clean12 archive and BDSP
credentialed access to HEEDB; ``AWS_PROFILE=lindtsey-cardiosentry`` has both.
``--manifest`` also accepts a local ``ECG_500Hz_clean12.tar.gz`` or
``manifest.csv``.

Usage::

    AWS_PROFILE=lindtsey-cardiosentry python export_cardiologist_samples.py --out ~/cardiologist_samples_clean12
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from PIL import Image

HERE = Path(__file__).resolve().parent
RENDER_CODE_DIR = HERE.parent / "clean_convnext_codes" / "clean12_ecg_signal_images_ec2"
AUGMENT_CODE_DIR = HERE.parent.parent / "dataset-combination-code" / "ecg-dataset-pipeline" / "image_pipeline"
DEFAULT_MANIFEST = ("s3://cardiosentry-fcmae-758821198776-us-east-1-an/"
                    "CardioSentry/dataset/clean12/ECG_500Hz_clean12.tar.gz")
HEEDB_ACCESS_POINT = "arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-credentialed-access-point-1"
ARCHIVE_ROOT = "ECG_500Hz_clean12"

# Canonical clean12 label order (the manifest's `labels` column uses it).
LABEL_NAMES = {
    "SB": "Sinus bradycardia",
    "STACH": "Sinus tachycardia",
    "LBBB": "Left bundle branch block",
    "LQT": "QT/QTc prolongation",
    "AF": "Atrial fibrillation",
    "AFL": "Atrial flutter",
    "3AVB": "Third-degree AV block",
    "2AVB": "Second-degree AV block",
    "1AVB": "First-degree AV block",
    "STEMI": "STEMI",
    "STE": "ST elevation",
    "NORM": "Normal ECG",
}


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select_records(rows: list[dict], per_combination: int, seed: int) -> list[dict]:
    """At most `per_combination` rows per `labels` value, distinct patients first."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["labels"], []).append(row)
    picked = []
    for members in groups.values():
        members.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['record_path']}".encode()).hexdigest())
        seen, first, repeat = set(), [], []
        for row in members:
            (repeat if row["patient_key"] in seen else first).append(row)
            seen.add(row["patient_key"])
        picked.extend((first + repeat)[:per_combination])
    return picked


# --------------------------------------------------------------------------
# Manifest and record download
# --------------------------------------------------------------------------
def read_manifest(source: str) -> list[dict]:
    """clean12 manifest rows from a manifest.csv, or from the head of the archive."""
    if source.endswith(".csv"):
        with open(Path(source).expanduser(), newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if source.startswith("s3://"):
        import boto3
        bucket, _, key = source[len("s3://"):].partition("/")
        raw = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
    else:
        raw = Path(source).expanduser().open("rb")
    try:
        # manifest.csv is one of the first members: read it, then close the stream.
        with tarfile.open(fileobj=raw, mode="r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name == f"{ARCHIVE_ROOT}/manifest.csv":
                    text = archive.extractfile(member).read().decode("utf-8")
                    return list(csv.DictReader(io.StringIO(text, newline="")))
                if PurePosixPath(member.name).suffix in {".hea", ".mat", ".dat"}:
                    break
    finally:
        raw.close()
    raise RuntimeError(f"{ARCHIVE_ROOT}/manifest.csv not found at the start of {source}")


def header_signal_file(hea_path: Path) -> tuple[str, str]:
    """(signal file name, WFDB format) from the first signal line."""
    lines = [l for l in hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
             if l.strip() and not l.startswith("#")]
    fields = lines[1].split()
    return fields[0], fields[1]


def verify_signal(hea_path: Path, expected_sha256: str) -> None:
    """Sample bytes must match the manifest sha256 (same hash as download_clean12.py)."""
    name, fmt = header_signal_file(hea_path)
    offset = int(fmt.split("+")[1]) if "+" in fmt else 0
    digest = hashlib.sha256((hea_path.parent / name).read_bytes()[offset:]).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"{hea_path}: signal sha256 {digest} != manifest {expected_sha256}")


def download_records(selected: list[dict], signals_dir: Path, access_point: str,
                     workers: int = 16) -> None:
    """Fetch each record's .hea + signal file from HEEDB (key ECG/<record_path>)."""
    import boto3
    client = boto3.client("s3", region_name="us-east-1")

    def fetch(key: str, dest: Path) -> int:
        if dest.is_file():
            return 0
        body = client.get_object(Bucket=access_point, Key=key)["Body"].read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(body)
        part.replace(dest)
        return len(body)

    def one(row: dict) -> int:
        rel = PurePosixPath(row["record_path"])
        hea = signals_dir.joinpath(*rel.parts)
        nbytes = fetch(f"ECG/{rel}", hea)
        signal = header_signal_file(hea)[0]
        nbytes += fetch(f"ECG/{rel.parent / signal}", hea.parent / signal)
        if not row.get("sha256"):  # no manifest hash for this record (e.g. picked from predictions)
            return nbytes
        try:
            verify_signal(hea, row["sha256"])
        except RuntimeError:
            hea.unlink()
            (hea.parent / signal).unlink()
            raise
        return nbytes

    print(f"Downloading {len(selected)} records from HEEDB ...", flush=True)
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, nbytes in enumerate(pool.map(one, selected), 1):
            total += nbytes
            if done % 100 == 0 or done == len(selected):
                print(f"  {done}/{len(selected)} records ({total / 2**20:.1f} MiB downloaded)", flush=True)
    print(f"Have {len(selected)} records "
          f"({sum(bool(r.get('sha256')) for r in selected)} verified against the manifest sha256)")



# --------------------------------------------------------------------------
# Rendering with the clean12 notebook's own worker and kit settings
# --------------------------------------------------------------------------
def notebook_code_cells(notebook: Path) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in cells if c["cell_type"] == "code"]


def load_render_settings(notebook: Path) -> tuple[dict, str]:
    """(kit settings from §1, worker module source from §6)."""
    cells = notebook_code_cells(notebook)
    try:
        config_cell = next(c for c in cells if c.lstrip().startswith("# 1. Configuration"))
        worker_cell = next(c for c in cells if c.lstrip().startswith("WORKER_SOURCE = r'''"))
    except StopIteration:
        raise RuntimeError(f"{notebook}: configuration or worker cell not found") from None
    namespace: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(config_cell, f"{notebook}:config", "exec"), namespace)
    worker_source = next(
        ast.literal_eval(node.value) for node in ast.parse(worker_cell).body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "WORKER_SOURCE" for t in node.targets))
    settings = {key: namespace[name] for key, name in (
        ("kit_seed", "KIT_SEED"), ("kit_fixed_font", "KIT_FIXED_FONT"),
        ("kit_render_flags", "KIT_RENDER_FLAGS"), ("kit_env", "KIT_ENV"),
        ("max_duration_s", "MAX_DURATION_S"), ("png_compress_level", "PNG_COMPRESS_LEVEL"))}
    return settings, worker_source


def prepare_kit(bundle: Path, kit_root: Path) -> Path:
    want = Path(str(bundle) + ".sha256").read_text().split()[0].lower()
    have = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if have != want:
        raise RuntimeError(f"{bundle}: SHA-256 {have} does not match its sidecar {want}")
    marker = kit_root / ".bundle_sha256"
    if not (marker.is_file() and marker.read_text().strip() == want):
        shutil.rmtree(kit_root, ignore_errors=True)
        with tarfile.open(bundle, "r:gz") as archive:
            try:
                archive.extractall(kit_root, filter="data")
            except TypeError:  # Python without PEP 706 filters; bundle is SHA-256 verified
                archive.extractall(kit_root)
        marker.write_text(want + "\n")
    kit_dir = kit_root / "ecg_image_v1_code" / "ecg-image-generator"
    if not (kit_dir / "gen_ecg_images_from_data_batch.py").is_file():
        raise RuntimeError(f"Kit batch driver missing under {kit_dir}")
    return kit_dir


# --------------------------------------------------------------------------
# Rendering (pool children): clean12 worker functions + stage4_augment
# --------------------------------------------------------------------------
def _init_render(config: dict, signals_dir: str) -> None:
    import ecg_clean_worker
    import stage4_augment
    ecg_clean_worker.init_worker(config, signals_dir, signals_dir)
    # Wrinkled-paper photographs ship with the same kit bundle.
    stage4_augment.C.KIT = Path(config["kit_dir"])


def _write_views(master_png: Path, job: dict, seed: int, variants: int) -> dict:
    import ecg_clean_worker as worker
    import stage4_augment
    folder = Path(job["dir"])
    folder.mkdir(parents=True, exist_ok=True)
    level = worker.CFG["png_compress_level"]
    with Image.open(master_png) as master:
        if master.size != (worker.MASTER_W, worker.MASTER_H):
            raise ValueError(f"master size {master.size}, need {(worker.MASTER_W, worker.MASTER_H)}")
        clean = f"{job['stem']}_clean.png"
        worker.to_final(worker.recolour_clean(master)).save(folder / clean, compress_level=level)
    if job.get("master_copy"):  # grey kit master, for augmenting this sheet later
        Path(job["master_copy"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(master_png, job["master_copy"])
    augmented = []
    for variant in range(variants):
        photo = master_png.with_name(f"{master_png.stem}_a{variant}.jpg")
        meta = stage4_augment.augment(master_png, photo, (seed, job["rel"], variant))
        tilt = float(meta["rotation_deg"])
        number = "" if variants == 1 else str(variant + 1)
        name = f"{job['stem']}_augmented{number}_{meta['grid_palette']}_tilt{tilt:+.1f}.png"
        with Image.open(photo) as image:
            worker.to_final(image).save(folder / name, compress_level=level)
        augmented.append({"image": name, "grid_palette": meta["grid_palette"], "tilt_deg": tilt})
    return {"clean": clean, "augmented": augmented}


def render_batch(jobs: list[dict], seed: int, variants: int) -> dict[str, dict]:
    """Stage, render and write one kit batch; returns rel -> result or error."""
    import ecg_clean_worker as worker
    scratch = Path(worker.CFG["kit_scratch_dir"]) / f"w{os.getpid()}-{time.time_ns()}"
    staged, rendered = scratch / "staged", scratch / "render"
    results: dict[str, dict] = {}
    staged_jobs: dict[str, dict] = {}
    try:
        for job in jobs:
            try:
                header = worker.INPUT / job["rel"]
                rows, names, fs, nsamp, _, comments = worker.parse_header(header)
                record_id = worker.record_id_for(job["rel"])
                worker.stage_record(record_id, worker.load_mv(header, rows, names, nsamp),
                                    fs, comments, staged)
                staged_jobs[record_id] = job
            except Exception as exc:
                results[job["rel"]] = {"error": f"{type(exc).__name__}: {exc}"}
        masters: dict[str, Path] = {}
        if staged_jobs:
            try:
                worker.run_kit(staged, rendered)
                masters = worker.collect_pngs(rendered, set(staged_jobs))
            except Exception as exc:
                for job in staged_jobs.values():
                    results[job["rel"]] = {"error": f"{type(exc).__name__}: {exc}"}
        for record_id, master in masters.items():
            job = staged_jobs[record_id]
            try:
                results[job["rel"]] = _write_views(master, job, seed, variants)
            except Exception as exc:
                results[job["rel"]] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return results


def prepare_render(work: Path, notebook: Path, kit_bundle: Path, augment_code: Path) -> dict:
    """Unpack the kit, write the clean12 worker module and make both importable;
    returns the worker config (`_init_render` / `ecg_clean_worker.init_worker`)."""
    settings, worker_source = load_render_settings(notebook)
    kit_dir = prepare_kit(kit_bundle, work / "kit")
    if not (augment_code / "stage4_augment.py").is_file():
        raise FileNotFoundError(f"stage4_augment.py not found in {augment_code}")
    (work / "ecg_clean_worker.py").write_text(worker_source)
    # Pool children import both modules by name (spawn/forkserver need PYTHONPATH).
    paths = [str(work), str(augment_code)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [*paths, os.environ.get("PYTHONPATH")]))
    return {**settings, "kit_dir": str(kit_dir), "kit_python": sys.executable,
            "kit_scratch_dir": str(work / "kit_scratch")}


def render(jobs: list[dict], signals_dir: Path, work: Path, notebook: Path, kit_bundle: Path,
           augment_code: Path, workers: int, seed: int, variants: int) -> dict[str, dict]:
    config = prepare_render(work, notebook, kit_bundle, augment_code)
    size = max(1, min(48, -(-len(jobs) // workers)))
    batches = [jobs[i:i + size] for i in range(0, len(jobs), size)]
    print(f"Rendering {len(jobs)} records (clean + {variants} augmented each; "
          f"{len(batches)} batches, {workers} workers) ...", flush=True)
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_render,
                             initargs=(config, str(signals_dir))) as pool:
        futures = [pool.submit(render_batch, batch, seed, variants) for batch in batches]
        for future in as_completed(futures):
            results.update(future.result())
            print(f"  rendered {len(results)}/{len(jobs)}", flush=True)
    return results


# --------------------------------------------------------------------------
# Organizing
# --------------------------------------------------------------------------
def label_folder(labels: str) -> Path:
    codes = labels.split(";")
    if len(codes) == 1:
        name = LABEL_NAMES[codes[0]].replace("/", "-").replace(" ", "_")
        return Path("single") / f"{codes[0]}_{name}"
    return Path(f"combo_{len(codes)}") / "+".join(codes)


def plan_jobs(out: Path, selected: list[dict]) -> list[dict]:
    """Label folder and numbered file stem for every selected record."""
    order = {code: i for i, code in enumerate(LABEL_NAMES)}
    selected = sorted(selected, key=lambda r: (int(r["n_labels"]),
                                               [order[c] for c in r["labels"].split(";")]))
    jobs, counters = [], {}
    for row in selected:
        folder = label_folder(row["labels"])
        counters[folder] = counters.get(folder, 0) + 1
        jobs.append({"rel": row["record_path"], "folder": folder.as_posix(),
                     "dir": str(out / folder),
                     "stem": f"{counters[folder]:02d}_{row['site']}_{row['file_id']}", "row": row})
    return jobs


README = """CardioSentry clean12 sample set for clinical review
===================================================

Source: HEEDB clean12 selection (113,358 physician-verified 12-lead ECGs,
native 500 Hz, 10 s). At most {per} ECGs per label combination, different
patients where possible ({n} ECGs, {combos} combinations).

Folders
  single/<CODE>_<name>/   ECGs carrying exactly one label
  combo_<k>/<A+B+...>/    ECGs carrying exactly these k labels together

Files (per ECG, same number prefix)
  NN_<site>_<id>_clean.png
      The synthetic ECG sheet as rendered for the dataset: 1024x768, 3x4 leads
      + Lead II rhythm strip, 25 mm/s, 10 mm/mV, red grid.
  NN_<site>_<id>_augmented_<grid colour>_tilt<degrees>.png
      The same ECG printed and photographed synthetically: grid colour
      (red/orange/pink/grey) with faded or fresh ink, tilt up to 3 degrees,
      slight perspective, wrinkled-paper texture, fold creases, uneven
      lighting and edge shadow, white-balance drift, exposure and contrast
      changes, sensor noise, mild blur, JPEG compression. The waveform
      itself is never changed.

Label codes
{labels}

index.csv lists every ECG with its labels, patient key, acquisition time,
other (unlabelled) findings from the source report, and all image paths.
"""


def write_index(out: Path, jobs: list[dict], results: dict[str, dict],
                per_combination: int) -> list[dict]:
    index_rows, failures = [], []
    for job in jobs:
        result, row = results[job["rel"]], job["row"]
        if "error" in result:
            failures.append(f"{job['rel']}: {result['error']}")
            continue
        folder = job["folder"]
        index_rows.append({
            "folder": folder, "clean_image": f"{folder}/{result['clean']}",
            "augmented_images": ";".join(f"{folder}/{a['image']}" for a in result["augmented"]),
            "grid_palettes": ";".join(a["grid_palette"] for a in result["augmented"]),
            "tilt_deg": ";".join(f"{a['tilt_deg']:.2f}" for a in result["augmented"]),
            "labels": row["labels"].replace(";", "+"),
            "label_names": "; ".join(LABEL_NAMES[c] for c in row["labels"].split(";")),
            "n_labels": row["n_labels"], "site": row["site"], "file_id": row["file_id"],
            "patient_key": row["patient_key"], "acq_time": row["acq_time"],
            "other_findings": row["other_findings"], "avb2_subtype": row["avb2_subtype"],
            "ste_propagated_from_stemi": row["ste_propagated_from_stemi"],
            "record_path": row["record_path"], "signal_sha256": row["sha256"],
        })
    if not index_rows:
        raise RuntimeError("No record rendered:\n  " + "\n  ".join(failures[:10]))
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    labels = "\n".join(f"  {code:<6} {name}" for code, name in LABEL_NAMES.items())
    (out / "README.txt").write_text(README.format(
        per=per_combination, n=len(index_rows), combos=len({r["labels"] for r in index_rows}),
        labels=labels))
    if failures:
        print(f"WARNING: {len(failures)} records failed to render:\n  " + "\n  ".join(failures))
    return index_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="clean12 archive (s3:// or local .tar.gz) or manifest.csv to select from")
    parser.add_argument("--heedb-access-point", default=HEEDB_ACCESS_POINT,
                        help="HEEDB (BDSP) S3 access point ARN the records are downloaded from")
    parser.add_argument("--out", type=Path, default=Path("cardiologist_samples_clean12"))
    parser.add_argument("--per-combination", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="record selection and augmentation draws")
    parser.add_argument("--augmented-variants", type=int, default=1,
                        help="augmented images per ECG (the training corpus used 4)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--notebook", type=Path,
                        default=RENDER_CODE_DIR / "Clean12_ECG_Signal_Images_EC2.ipynb")
    parser.add_argument("--kit-bundle", type=Path,
                        default=RENDER_CODE_DIR / "ecg_image_v1_code.tar.gz")
    parser.add_argument("--augment-code", type=Path, default=AUGMENT_CODE_DIR,
                        help="image_pipeline folder holding stage4_augment.py and config.py")
    args = parser.parse_args()
    if min(args.per_combination, args.workers, args.augmented_variants) < 1:
        parser.error("--per-combination, --workers and --augmented-variants must be >= 1")

    out = args.out.expanduser().resolve()
    work = out / "_work"
    signals_dir = work / "signals"
    selection_file = work / "selection.json"
    key = {"per_combination": args.per_combination, "seed": args.seed}

    selected = None
    if selection_file.is_file():
        saved = json.loads(selection_file.read_text())
        if saved["key"] == key:
            selected = saved["records"]
            print(f"Reusing the saved selection of {len(selected)} records")
    if selected is None:
        print(f"Reading the clean12 manifest from {args.manifest}", flush=True)
        rows = read_manifest(args.manifest)
        selected = select_records(rows, args.per_combination, args.seed)
        print(f"Manifest: {len(rows):,} records, {len({r['labels'] for r in rows})} label "
              f"combinations -> {len(selected)} selected", flush=True)
        shutil.rmtree(signals_dir, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        selection_file.write_text(json.dumps({"key": key, "records": selected}, indent=1))
    download_records(selected, signals_dir, args.heedb_access_point)

    for stale in [out / "single", *out.glob("combo_*")]:
        shutil.rmtree(stale, ignore_errors=True)
    jobs = plan_jobs(out, selected)
    results = render(jobs, signals_dir, work, args.notebook.resolve(), args.kit_bundle.resolve(),
                     args.augment_code.resolve(), args.workers, args.seed, args.augmented_variants)
    index_rows = write_index(out, jobs, results, args.per_combination)
    shutil.rmtree(work / "kit_scratch", ignore_errors=True)
    print(f"Done: {len(index_rows)} ECGs x (clean + {args.augmented_variants} augmented) in "
          f"{len({r['folder'] for r in index_rows})} label folders under {out}")


if __name__ == "__main__":
    main()
