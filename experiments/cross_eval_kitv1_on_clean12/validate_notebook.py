"""CPU checks for CrossEval_KitV1_on_Clean12_Colab.ipynb (no AWS, no GPU).

Run:  ../../.venv/bin/python validate_notebook.py

1. The notebook is regenerated, and every code cell compiles.
2. The library cells are byte-identical to tar_ranges.py and cross_eval_lib.py.
3. The image pipeline cell is verbatim clean12 training code (checked against
   the clean12 training notebook).
4. tar_ranges indexes Python- and GNU-tar archives (pax, gnu, ustar) exactly
   like tarfile, for many segment splits, with fake "ustar" bytes planted in
   member data.
5. S3RangeReader (taken from the notebook) pins the ETag, retries transient
   errors, and stops on a changed object.
6. cross_eval_lib on real local files: the frozen split hash is 4eba80c7…,
   and the exclusion count against kit-v1's record_manifest.csv is computed
   (skipped if the local data is absent).
7. End to end: every notebook cell after the install runs against a fake S3
   that serves synthetic clean12 / kit-v1 TARs, configs, tiny ConvNeXt
   checkpoints and saved predictions. Checks: the exclusion, the EMA/raw
   choice, the AP table, the reproduction guard (it must fire on wrong saved
   predictions), the Drive write, and the refusal to overwrite.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NOTEBOOK = HERE / "CrossEval_KitV1_on_Clean12_Colab.ipynb"
CLEAN12_TRAINING_NOTEBOOK = (REPO / "prototype-code/clean_convnext_codes/convnext_v1_base_clean12_v1/"
                             "CardioSentry_ConvNeXtV1_Base_Clean12_EC2.ipynb")
LOCAL_SPLIT = REPO / "prototype-code/clean_convnext_codes/convnext_v1_base_clean12_v1/splits"
LOCAL_CLEAN12_MANIFEST = REPO / "dataset-files/HEEDB/extracted/ECG_500Hz_clean12/manifest.csv"
LOCAL_KITV1_RECORD_MANIFEST = REPO / "dataset-files/HEEDB/extracted/ECG_500Hz_patient_splits/record_manifest.csv"
sys.path.insert(0, str(HERE))

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def wrap(function):
        try:
            detail = function() or ""
            RESULTS.append((name, True, detail))
            print(f"[PASS] {name} {detail}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, repr(exc)))
            print(f"[FAIL] {name}: {exc!r}")
            import traceback
            traceback.print_exc()
        return function
    return wrap


def code_cells(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def strip_magics(source):
    return "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())


subprocess.run([sys.executable, str(HERE / "make_notebook.py")], check=True)
NB = json.loads(NOTEBOOK.read_text())
CELLS = code_cells(NB)


@check("1 notebook regenerates and every code cell compiles")
def _():
    for index, source in enumerate(CELLS):
        compile(strip_magics(source), f"<cell {index}>", "exec")
    return f"({len(NB['cells'])} cells, {len(CELLS)} code)"


@check("2 library cells are verbatim copies of tar_ranges.py and cross_eval_lib.py")
def _():
    for filename in ("tar_ranges.py", "cross_eval_lib.py"):
        source = (HERE / filename).read_text()
        matches = [cell for cell in CELLS if cell.split("\n", 1)[1:] == [source]]
        assert len(matches) == 1, f"{filename}: {len(matches)} verbatim cells"


@check("3 image pipeline cell is verbatim clean12 training code")
def _():
    training = "\n".join(code_cells(json.loads(CLEAN12_TRAINING_NOTEBOOK.read_text())))
    import make_notebook as maker
    importlib.reload(maker)
    first, rest = maker.TRAINING_PREPROCESSING_SOURCE.split("\n", 1)
    assert first in training, "INPUT_SIZE line not in the training notebook"
    assert rest.strip("\n") in training, "transform/normalization block differs from training"
    assert maker.TRAINING_PREPARE_BATCH_SOURCE.strip("\n") in training, "prepare_image_batch differs from training"
    pipeline_cell = next(cell for cell in CELLS if cell.startswith("# 8."))
    assert maker.TRAINING_PREPROCESSING_SOURCE in pipeline_cell and maker.TRAINING_PREPARE_BATCH_SOURCE in pipeline_cell


class MemoryReader:
    def __init__(self, blob):
        self.blob, self.size = blob, len(blob)

    def read(self, offset, length):
        return self.blob[offset:offset + length]


def make_tree(root: Path, count: int, seed: int) -> Path:
    rng = random.Random(seed)
    top = root / "clean_ecg_signal_images_clean12"
    for index in range(count):
        folder = top / "images" / ("I0001" if index % 2 else "I0006") / "WFDB" / f"S000{index % 4}" / "2006" / "11"
        folder.mkdir(parents=True, exist_ok=True)
        data = bytes(rng.getrandbits(8) for _ in range(rng.choice([0, 1, 511, 512, 513, 3000, 9000])))
        if len(data) > 600:      # a fake header magic inside member data
            data = data[:257] + b"ustar\x0000" + data[265:]
        (folder / f"de_{111246768 + index}_20110103092228_20110103093259_long_tail.png").write_bytes(data)
    for name in ("manifest.csv", "record_manifest.csv", "README.md"):
        (top / name).write_text("x" * rng.randint(1, 2000))
    os.utime(top / "README.md", (1.5e9 + 0.25, 1.5e9 + 0.25))   # float mtime -> pax record
    return top


@check("4 tar_ranges indexes pax/gnu/ustar archives exactly like tarfile")
def _():
    import tar_ranges
    importlib.reload(tar_ranges)
    root = Path(tempfile.mkdtemp())
    top = make_tree(root, 60, 1)
    blobs = {}
    for fmt, name in ((tarfile.PAX_FORMAT, "python-pax"), (tarfile.GNU_FORMAT, "python-gnu"),
                      (tarfile.USTAR_FORMAT, "python-ustar")):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=fmt) as archive:
            archive.add(top, arcname=top.name)
        blobs[name] = buffer.getvalue()
    for fmt in ("gnu", "posix", "ustar"):
        out = root / f"{fmt}.tar"
        done = subprocess.run(["tar", "--format", fmt, "-cf", str(out), "-C", str(root), top.name],
                              capture_output=True)
        if done.returncode == 0:
            blobs[f"gnutar-{fmt}"] = out.read_bytes()
    for name, blob in blobs.items():
        reference = {m.name: (m.offset_data, m.size) for m in tarfile.open(fileobj=io.BytesIO(blob)).getmembers()}
        for segments in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89):
            index = tar_ranges.build_tar_index(MemoryReader(blob), segments=segments, workers=8, log=None,
                                               min_segment_bytes=512, sync_window_bytes=2048)
            got = {key: (value["data_offset"], value["size"]) for key, value in index.items()}
            assert got == reference, (name, segments, sorted(set(reference.items()) ^ set(got.items()))[:3])
        member = next(key for key in reference if key.endswith(".png") and reference[key][1] > 600)
        index = tar_ranges.build_tar_index(MemoryReader(blob), segments=7, workers=4, log=None,
                                           min_segment_bytes=512, sync_window_bytes=1024)
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            assert tar_ranges.read_member(MemoryReader(blob), index, member) == archive.extractfile(member).read()
    # A corrupted header in the chain must be detected, never silently skipped.
    blob = bytearray(blobs["python-pax"])
    victim = sorted(tarfile.open(fileobj=io.BytesIO(bytes(blob))).getmembers(), key=lambda m: m.offset)[30]
    blob[victim.offset + 148] ^= 0x01
    try:
        tar_ranges.build_tar_index(MemoryReader(bytes(blob)), segments=4, workers=4, log=None,
                                   min_segment_bytes=512, sync_window_bytes=2048)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a corrupted header was not detected")
    return f"({len(blobs)} archive formats x 10 segment splits)"


def notebook_definition(cell_prefix: str, name: str) -> str:
    cell = next(cell for cell in CELLS if cell.startswith(cell_prefix))
    tree = ast.parse(strip_magics(cell))
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    return ast.get_source_segment(strip_magics(cell), node)


class FakeS3Error(Exception):
    pass


def client_error(code, operation="GetObject"):
    from botocore.exceptions import ClientError
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakeS3:
    """Just enough of the boto3 S3 client for the notebook."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.fail_next_reads = 0
        self.puts = []

    def add(self, key, data: bytes):
        self.objects[key] = data
        self.etags[key] = '"' + hashlib.md5(data).hexdigest() + '"'

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise client_error("404", "HeadObject")
        return {"ContentLength": len(self.objects[Key]), "ETag": self.etags[Key]}

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        if Key not in self.objects:
            raise client_error("NoSuchKey")
        if IfMatch is not None and IfMatch != self.etags[Key]:
            raise client_error("PreconditionFailed")
        if self.fail_next_reads:
            self.fail_next_reads -= 1
            raise client_error("SlowDown")
        data = self.objects[Key]
        if Range:
            start, end = (int(part) for part in Range.removeprefix("bytes=").split("-"))
            data = data[start:end + 1]
        return {"Body": io.BytesIO(data)}

    def download_file(self, Bucket, Key, Filename, Config=None):
        Path(Filename).write_bytes(self.objects[Key])

    def put_object(self, **kwargs):
        self.puts.append(kwargs["Key"])
        raise AssertionError("the notebook must never write to S3")


@check("5 S3RangeReader pins the ETag, retries, and stops on a changed object")
def _():
    import threading
    import time
    from botocore.exceptions import BotoCoreError, ClientError
    namespace = {"threading": threading, "time": time, "ClientError": ClientError, "BotoCoreError": BotoCoreError}
    exec(notebook_definition("# 2.", "S3RangeReader"), namespace)
    s3 = FakeS3()
    s3.add("a.tar", bytes(range(256)) * 40)
    reader = namespace["S3RangeReader"](s3, "bucket", "a.tar")
    namespace["time"] = type("T", (), {"sleep": staticmethod(lambda s: None)})
    assert reader.read(10, 5) == bytes(range(10, 15))
    s3.fail_next_reads = 2
    assert reader.read(0, 3) == b"\x00\x01\x02"
    assert reader.read(reader.size - 2, 10) == bytes([254, 255])
    s3.add("a.tar", b"changed")
    try:
        reader.read(0, 3)
    except RuntimeError as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("an object change was not detected")


@check("6 real local data: frozen split hash and kit-v1 TRAIN exclusion count")
def _():
    import cross_eval_lib as lib
    importlib.reload(lib)
    split_files = sorted(LOCAL_SPLIT.glob("*.csv")) if LOCAL_SPLIT.is_dir() else []
    split_file = next((f for f in split_files if f.name.endswith("split_v1.csv")), None)
    if split_file is None or not LOCAL_CLEAN12_MANIFEST.is_file() or not LOCAL_KITV1_RECORD_MANIFEST.is_file():
        return "(skipped: local data not present)"
    split = pd.read_csv(split_file, dtype=str, keep_default_na=False).set_index("patient_key")["split"]
    assert lib.split_sha256(split) == "4eba80c71bba27e8bde0ed1a510de9d4819ede5139946b48526461a7bc4841cd"
    manifest = pd.read_csv(LOCAL_CLEAN12_MANIFEST, dtype=str, keep_default_na=False)
    validation = lib.assign_split(manifest, split)
    validation = validation[validation["split"].eq("validation")]
    kit = pd.read_csv(LOCAL_KITV1_RECORD_MANIFEST, dtype=str, keep_default_na=False)
    kit_sha = hashlib.sha256(LOCAL_KITV1_RECORD_MANIFEST.read_bytes()).hexdigest()
    assert kit_sha == "b3da84a99d74b8476b14ba3245a01163da20a91206cd225319fc7484f3913964", kit_sha
    kept, info = lib.exclude_kitv1_patients(validation.reset_index(drop=True), kit)
    assert info["excluded_records"] == info["excluded_records_site_qualified"]
    assert info["kept_records"] + info["excluded_records"] == len(validation) == 11301
    for label in lib.CLEAN12_LABELS:
        positives = int(pd.to_numeric(kept[label]).sum())
        assert 0 < positives < len(kept), (label, positives)
    return (f"(validation 11,301; excluded {info['excluded_records']:,} records / "
            f"{info['excluded_patients']:,} patients; kept {info['kept_records']:,})")


@check("6b cross_eval_lib unit behaviour")
def _():
    import cross_eval_lib as lib
    from sklearn.metrics import average_precision_score
    kit_order = ["Normal ECG", "Atrial Fibrillation", "Atrial Flutter", "Strict STEMI", "ST elevation",
                 "ST depression", "T-wave inversion", "Prior infarct pattern", "LVH", "RVH", "LBBB", "RBBB",
                 "First-degree AV block", "2AVB", "Third-degree / complete AV block", "QT/QTc prolongation",
                 "Sinus bradycardia", "Sinus tachycardia", "LAFB", "LAD", "RAD"]
    columns = lib.kitv1_columns_for_clean12(kit_order)
    assert [kit_order[c] for c in columns] == [lib.CLEAN12_TO_KITV1[label] for label in lib.CLEAN12_LABELS]
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (200, 12))
    a, b = rng.random((200, 12)), rng.random((200, 12))
    table = lib.ap_summary(y, a, b)
    macro = table.set_index("label").loc[lib.MACRO_ROW]
    expected = np.mean([average_precision_score(y[:, i], a[:, i]) for i in range(1, 12)])
    assert abs(macro["kitv1_ap"] - expected) < 1e-12
    y[:, 3] = 0
    try:
        lib.ap_summary(y, a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("AP with no positives must be refused")
    raw = {"w": 1}
    ema = {"_orig_mod.w": 2}
    state, info = lib.select_checkpoint_weights({"model": raw, "model_ema": ema, "selected_weight_source": "ema"})
    assert state == {"w": 2} and info["selected_weight_source"] == "ema"
    state, _ = lib.select_checkpoint_weights({"model": raw, "model_ema": ema, "selected_weight_source": "raw"})
    assert state == raw
    target = Path(tempfile.mkdtemp()) / "out" / "summary.csv"
    lib.save_summary_file(b"a,b\n", target, overwrite=False)
    assert target.read_bytes() == b"a,b\n" and not list(target.parent.glob("*.partial"))
    try:
        lib.save_summary_file(b"new", target, overwrite=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing summary was overwritten")
    lib.save_summary_file(b"new", target, overwrite=True)
    assert target.read_bytes() == b"new"


def png_bytes(seed):
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    image = Image.new("RGB", (1024, 768), (255, 240, 240))
    draw = ImageDraw.Draw(image)
    for _ in range(40):
        draw.line([(rng.randrange(1024), rng.randrange(768)) for _ in range(2)], fill=(0, 0, 0), width=2)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def tar_of(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 1.7e9 + 0.5
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def checkpoint_bytes(model_name, classes, source, fingerprint, label_order):
    import timm
    import torch
    raw = timm.create_model(model_name, pretrained=False, num_classes=classes)
    ema = timm.create_model(model_name, pretrained=False, num_classes=classes)
    buffer = io.BytesIO()
    torch.save({"model": raw.state_dict(), "model_ema": ema.state_dict(), "selected_weight_source": source,
                "epoch": 7, "validation_pathology_only_macro_average_precision": 0.5,
                "config_fingerprint": fingerprint, "numpy_random_state": np.random.get_state(),
                "config": {"identity": {"model": model_name, "label_order": label_order}}}, buffer)
    selected = ema if source == "ema" else raw
    return buffer.getvalue(), selected


@check("7 end to end on CPU against a fake S3 (tiny ConvNeXt, synthetic archives)")
def _():
    import cross_eval_lib as lib
    import torch
    labels = lib.CLEAN12_LABELS
    kit_order = ["Normal ECG", "Atrial Fibrillation", "Atrial Flutter", "Strict STEMI", "ST elevation",
                 "ST depression", "T-wave inversion", "Prior infarct pattern", "LVH", "RVH", "LBBB", "RBBB",
                 "First-degree AV block", "2AVB", "Third-degree / complete AV block", "QT/QTc prolongation",
                 "Sinus bradycardia", "Sinus tachycardia", "LAFB", "LAD", "RAD"]
    top = "clean_ecg_signal_images_clean12"
    records, render, split_rows, kit_rows, images = [], [], [], [], {}
    plan = [("validation", i) for i in range(36)] + [("train", i) for i in range(36, 42)] + [("test", i) for i in range(42, 48)]
    for split, i in plan:
        site = "I0001" if i % 2 else "I0006"
        record_path = f"{site}/WFDB/2015/rec_{i:03d}.hea"
        image_path = f"images/{site}/WFDB/2015/rec_{i:03d}.png"
        label = labels[i % 12]
        row = {"record_path": record_path, "site": site, "patient_key": f"{170000000 + i}", "labels": label,
               "n_labels": 1, **{name: int(name == label) for name in labels}}
        records.append(row)
        data = png_bytes(i)
        images[f"{top}/{image_path}"] = data
        render.append({"source_header": record_path, "image_path": image_path, "status": "success",
                       "image_bytes": len(data)})
        split_rows.append({"patient_key": row["patient_key"], "split": split, "site": site, "records": 1})
        if split == "validation" and i < 12:
            kit_rows.append({"patient_key": row["patient_key"], "split": "train", "site": site})
        elif split == "validation" and i < 18:
            kit_rows.append({"patient_key": row["patient_key"], "split": "validation", "site": site})
    kit_rows.append({"patient_key": "999999999", "split": "train", "site": "I0001"})
    manifest_bytes = pd.DataFrame(render).to_csv(index=False).encode()
    record_bytes = pd.DataFrame(records).to_csv(index=False).encode()
    kit_bytes = pd.DataFrame(kit_rows).to_csv(index=False).encode()
    split_series = pd.DataFrame(split_rows).set_index("patient_key")["split"]
    clean_tar = tar_of({f"{top}/README.md": b"readme", **images,
                        f"{top}/manifest.csv": manifest_bytes, f"{top}/record_manifest.csv": record_bytes})
    kit_tar = tar_of({"clean_ecg_signal_images_kit_v1/images/train/x.png": png_bytes(99),
                      "clean_ecg_signal_images_kit_v1/record_manifest.csv": kit_bytes})
    sha = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    pipeline = {"model": "convnext_atto", "input_size_hw": [768, 1024], "input_color_mode": "RGB",
                "resize_interpolation": "bicubic", "resize_antialias": True, "resize_preserves_aspect_ratio": False,
                "tensor_conversion": "PILToTensor uint8 on CPU -> float32 / 255 on GPU",
                "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]}
    clean_config = {"identity": {**pipeline, "label_order": labels, "dataset_archive_sha256": sha(clean_tar),
                                 "dataset_metadata_sha256": sha(record_bytes), "dataset_manifest_sha256": sha(manifest_bytes),
                                 "split": {"sha256": lib.split_sha256(split_series)}},
                    "runtime": {"split_record_counts": {"train": 6, "validation": 36, "test": 6},
                                "split_image_counts": {"train": 6, "validation": 36, "test": 6}},
                    "fingerprint": "clean12fp"}
    kit_config = {"identity": {**pipeline, "label_order": kit_order, "dataset_metadata_sha256": sha(kit_bytes)},
                  "fingerprint": "kitv1fp"}
    clean_ckpt, clean_selected = checkpoint_bytes("convnext_atto", 12, "ema", "clean12fp", labels)
    kit_ckpt, kit_selected = checkpoint_bytes("convnext_atto", 21, "raw", "kitv1fp", kit_order)

    clean_prefix = "runs/convnext_v1_base_clean12_g6e2xlarge_seed42"
    kit_prefix = "runs/convnext_v1_base_kitv1/convnext_v1_base_kitv1_clean_g6e2xlarge_seed42"
    s3 = FakeS3()
    s3.add("dataset/clean_ecg_signal_images_clean12.tar", clean_tar)
    s3.add("dataset/clean_ecg_signal_images_clean12.tar.sha256", f"{sha(clean_tar)}  x.tar\n".encode())
    s3.add("dataset/clean_ecg_signal_images_kit_v1.tar", kit_tar)
    s3.add(f"{clean_prefix}/config_snapshot.json", json.dumps(clean_config).encode())
    s3.add(f"{kit_prefix}/config_snapshot.json", json.dumps(kit_config).encode())
    s3.add(f"{clean_prefix}/best.pt", clean_ckpt)
    s3.add(f"{kit_prefix}/best.pt", kit_ckpt)
    s3.add(f"{clean_prefix}/patient_split.csv", pd.DataFrame(split_rows).to_csv(index=False).encode())
    wrong = pd.DataFrame({"record_id": [r["record_path"] for r in records], "primary_metric": 1,
                          **{f"true_{name}": [r[name] for r in records] for name in labels},
                          **{f"prob_{name}": np.linspace(0, 1, len(records)) for name in labels}})
    s3.add(f"{clean_prefix}/validation_predictions.csv", wrong.to_csv(index=False).encode())

    scratch = Path(tempfile.mkdtemp())
    drive_dir = scratch / "MyDrive" / "CardioSentry" / "new_runs" / "cross_eval_kitv1_on_clean12"
    drive_dir.parent.mkdir(parents=True)
    os.environ.update({"CS_DRIVE_OUTPUT_DIR": str(drive_dir), "AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "CS_ALLOW_CPU": "1",
                       "CS_SCRATCH_ROOT": str(scratch), "CS_NUM_WORKERS": "0", "CS_EVAL_BATCH_SIZE": "4",
                       "CS_INDEX_WORKERS": "8", "CS_FETCH_WORKERS": "4"})
    namespace: dict = {"__name__": "__notebook__"}
    code = [strip_magics(cell) for cell in CELLS]
    config_index = next(i for i, cell in enumerate(CELLS) if cell.startswith("# 2."))
    exec(compile(code[config_index], "<cell 2>", "exec"), namespace)
    namespace["S3"] = s3
    later = code[config_index + 1:]
    by_number = {cell.split(".", 1)[0]: (i, cell) for i, cell in enumerate(later)}
    for number in ("# 3a", "# 3b", "# 4", "# 5", "# 6", "# 7", "# 8", "# 9", "# 10"):
        index, cell = by_number[number]
        exec(compile(cell, f"<cell {number}>", "exec"), namespace)
    exclusion = namespace["EXCLUSION"]
    assert exclusion["validation_records"] == 36 and exclusion["excluded_records"] == 12, exclusion
    assert exclusion["records_with_patient_in_kitv1_validation"] == 6
    assert len(namespace["eval_df"]) == 24
    assert namespace["KITV1_CHECKPOINT"]["selected_weight_source"] == "raw"
    assert namespace["CLEAN12_CHECKPOINT"]["selected_weight_source"] == "ema"
    for model_key, expected in (("kitv1_model", kit_selected), ("clean12_model", clean_selected)):
        loaded = namespace[model_key].state_dict()
        for key, value in expected.state_dict().items():
            assert torch.equal(loaded[key].cpu(), value), (model_key, key)

    # Wrong saved predictions: the reproduction guard must stop before any write.
    _, guard_cell = by_number["# 11"]
    try:
        exec(compile(guard_cell, "<cell 11>", "exec"), namespace)
    except RuntimeError as exc:
        assert "does not reproduce" in str(exc)
    else:
        raise AssertionError("the reproduction guard did not fire on wrong saved predictions")
    assert not s3.puts

    # Correct saved predictions (the clean12 model's own scores): cells 11-12 complete.
    right = wrong.set_index("record_id").copy()
    for column, name in enumerate(labels):
        right.loc[namespace["eval_df"]["record_path"], f"prob_{name}"] = namespace["clean12_probability"][:, column]
    s3.add(f"{clean_prefix}/validation_predictions.csv", right.reset_index().to_csv(index=False).encode())
    for number in ("# 11", "# 12"):
        exec(compile(by_number[number][1], f"<cell {number}>", "exec"), namespace)
    csv_path = drive_dir / "cross_eval_kitv1_on_clean12.csv"
    json_path = drive_dir / "cross_eval_kitv1_on_clean12.json"
    table = pd.read_csv(csv_path)
    assert list(table["label"]) == labels + [lib.MACRO_ROW]
    assert list(table.columns) == ["label", "kitv1_output", "records", "positives", "prevalence",
                                   "kitv1_ap", "clean12_ap", "ap_difference"]
    assert (table["records"] == 24).all()
    macro = table.set_index("label").loc[lib.MACRO_ROW]
    assert abs(macro["kitv1_ap"] - table.iloc[1:12]["kitv1_ap"].mean()) < 1e-5
    assert sorted(path.name for path in drive_dir.iterdir()) == [
        "cross_eval_kitv1_on_clean12.csv", "cross_eval_kitv1_on_clean12.json"]
    assert not s3.puts and not any(key.startswith(clean_prefix + "/cross_eval") for key in s3.objects)
    provenance = json.loads(json_path.read_text())
    assert provenance["exclusion"]["excluded_records"] == 12
    assert "record_path" not in csv_path.read_text()

    # A second run must refuse to overwrite: at the early check (cell 4) and at the write (cell 12).
    try:
        exec(compile(by_number["# 4"][1], "<cell 4 rerun>", "exec"), namespace)
    except FileExistsError:
        pass
    else:
        raise AssertionError("cell 4 did not stop on an existing result")
    try:
        exec(compile(by_number["# 12"][1], "<cell 12 rerun>", "exec"), namespace)
    except FileExistsError:
        pass
    else:
        raise AssertionError("cell 12 overwrote an existing result")
    return "(24 of 36 validation records kept; guard fired; CSV + JSON on Drive, nothing in S3)"


failed = [name for name, ok, _ in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
sys.exit(1 if failed else 0)
