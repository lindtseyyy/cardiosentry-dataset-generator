"""CPU checks of the generated ConvNeXt-V1 Tiny + Residual FPN clean12 + augmented Colab notebook;
never mounts Drive or contacts Google.

Run: ../../../.venv/bin/python validate_notebook.py

The notebook's own definitions are executed. Sections 3 (two archives, Drive ->
tar streaming against a fake Drive API and a fake mount, PNG or lossless-WebP
storage), 4a/4c (augmented view audit and two-view selection), 6
(architectures), 7 (worker tuning) and 11 (comparison) run as whole cells, and
run_architecture() runs end to end for the real timm V1 Tiny + RFPN and the
optional plain V1 Tiny baseline (random weights, no download) on tiny
synthetic sheets: clean views as PNG, augmented views as lossless WebP. "Drive"
is a temporary folder. CUDA-only calls are stubbed, so this checks the
pipeline logic, not GPU speed or memory.

Checked: the fusion cell is fusion_necks.py verbatim; the recipe (the plain V1
Tiny one) and its overrides; RFPN = plain V1 Tiny at step 0 with stock timm
keys under backbone. and the plain model's LR profile parameter by parameter;
the lossless WebP worker reproduces the PNG pixels of the real clean and
augmented sample sheets; both archives staged, skipped, re-staged as WebP, and
refused on a checksum mismatch; the augmented manifest audit; both views of a
record in one split with clean-only pos_weight; an RFPN run interrupted in
epoch 2 and resumed from Drive; per-view test metrics and bootstraps; skipping
a finished model after the local copy is wiped; continuing past a failing
model; a recipe change being refused; the view-aware paired comparison
including clean-only references re-scored from their test_predictions.csv;
benchmark-only mode; libpng decoding == PIL + resize on the real clean12
sample sheets; one shared raw+EMA validation pass == two separate passes.
CS_CHECK_NATIVE=1 (default) also builds both models at 768x1024 for MACs.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import ast
import contextlib
import functools
import json
import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "CardioSentry_ConvNeXtV1_Tiny_RFPN_Clean12_Aug_Colab.ipynb"
AUG_SAMPLES = HERE.parent / "clean12_augmented_ecg_images_ec2" / "samples"
VIEWS = ("clean", "augmented")
FUSION_FILE = HERE / "fusion_necks.py"
RUN_CELL_PREFIXES = ("# 10a.", "# 10b.", "# 12.")
RFPN, PLAIN = "convnext_tiny_rfpn", "convnext_tiny"
V1_TINY_RECIPE = {"base_lr": 2e-4, "weight_decay": 0.02, "warmup_epochs": 2, "layer_decay": 0.86,
                  "drop_path_rate": 0.1, "head_init_scale": 0.001, "rare_sampling": False}
FULL_CELL_MARKERS = {
    "dataset": "def stream_extract_archive(",
    "augmented_audit": "aug_manifest_df = pd.read_csv(",
    "selection": "def stored_image_path(",
    "architectures": "ARCHITECTURE_INFO = {}",
    "workers": "def tune_decoder_workers(",
    "comparison": "comparison_rows, per_label_rows, histories, draws = [], [], {}, {}",
    "benchmark_table": "# 10c. Benchmark-only cost table",
}
SMALL_SIZE = (32, 48)
CONSTANTS = {
    "LABELS", "NORMAL_LABEL", "PATHOLOGY_LABELS", "LABEL_NAMES", "SUBSETS", "SITES", "TRAIN_VARIANTS",
    "VIEWS", "EVAL_VARIANTS", "WEBP_TRANSCODER_SOURCE", "DATASET_ARCHIVES",
    "FUSION_CHANNELS", "_V1_TINY_MODEL", "_V1_TINY_PRETRAINING",
    "ARCHITECTURE_SPECS", "ARCHITECTURES", "CONTINUE_ON_ERROR", "INPUT_SIZE", "MAX_EPOCHS",
    "EARLY_STOPPING_PATIENCE", "EARLY_STOPPING_MIN_EPOCHS", "MIN_LR_RATIO",
    "DROP_RATE", "EFFECTIVE_BATCH_SIZE", "EMA_DECAY",
    "COLLAPSE_GUARD_ENABLED", "COLLAPSE_LOSS_RATIO", "COLLAPSE_CHECK_EVERY_N_BATCHES",
    "COLLAPSE_PATIENCE_WINDOWS", "COLLAPSE_MIN_VAL_MACRO_AUROC", "TRAIN_BRIGHTNESS_JITTER",
    "TRAIN_CONTRAST_JITTER", "RARE_SAMPLING_THRESHOLD", "BOOTSTRAP_ROUNDS",
    "BOOTSTRAP_CONFIDENCE", "CONTRADICTION_LAMBDA", "GRAD_CLIP_NORM", "SEED", "AUTO_BATCH_CANDIDATES",
    "EVAL_BATCH_CANDIDATES", "WORKER_TUNING_BATCH", "BATCH_PROBE_TIMED_UPDATES", "EVAL_PROBE_TIMED_BATCHES",
    "VRAM_SAFETY_FRACTION", "USE_AMP", "DETERMINISTIC", "USE_FUSED_ADAMW", "TORCH_COMPILE_ENABLED",
    "TORCH_COMPILE_MODE", "BENCHMARK_TRAIN_BATCHES", "BENCHMARK_VAL_BATCHES", "BENCHMARK_WARMUP_BATCHES",
    "WORKER_BENCHMARK_BATCHES", "COMBINATION_LOW_SUPPORT_THRESHOLD", "PREFETCH_FACTOR",
    "DATALOADER_START_METHOD", "PROGRESS_UPDATE_EVERY_N_BATCHES", "INTRA_EPOCH_SYNC_MINUTES",
    "SPLIT_VERSION", "SPLIT_SEED", "SPLIT_RATIOS", "DRIVE_SYNC_EVERY_N_EPOCHS", "HARDWARE_PROFILE",
    "IMAGENET_MEAN", "IMAGENET_STD", "IMAGE_COLOR_MODE", "RESIZE_INTERPOLATION", "RESIZE_ANTIALIAS",
    "EVAL_METADATA_KEYS", "GPU_LUMINANCE", "ARCH_RESULTS", "ARCH_FAILURES", "LOSS_COMPONENT_KEYS",
    "PREDICT_METADATA_KEYS", "SMALL_RESULT_FILES", "CHECKPOINT_FILES",
    "BENCHMARK_ONLY", "COMPUTE_UNITS_PER_HOUR", "BENCHMARK_RESULTS",
    "FUSION_CHOICES", "FUSION_HEAD_CHOICES", "FUSION_STAGE_INDICES",
    "_sync_threads", "_sync_threads_lock", "_copy_key_threads", "_copy_key_lock",
}


def load_notebook():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    namespace = {"__name__": "cardiosentry_notebook", "DEVICE": torch.device("cpu"),
                 "display": lambda *args, **kwargs: None}
    full_cells = {}
    run_cells = []
    fusion_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        source = "".join(cell["source"])
        if "%pip" in source:
            assert '"timm==1.0.28"' in source
            continue
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
        fusion_cells += source == FUSION_FILE.read_text()
        if source.startswith(RUN_CELL_PREFIXES):
            run_cells.append(source)
            continue
        for name, marker in FULL_CELL_MARKERS.items():
            if marker in source:
                full_cells[name] = source
        tree = ast.parse(source)
        for node in tree.body:
            keep = isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) or (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets)
            ) or (  # try: from torchvision.io import decode_webp / except ImportError
                isinstance(node, ast.Try) and all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in node.body)
            )
            if keep:
                exec(compile(ast.Module(body=[node], type_ignores=[]), NOTEBOOK.name, "exec"), namespace)
    assert fusion_cells == 1, "the notebook's fusion cell differs from fusion_necks.py; rerun make_notebook.py"
    assert set(full_cells) == set(FULL_CELL_MARKERS), sorted(full_cells)
    assert [cell.splitlines()[1] for cell in run_cells[:2]] == [
        f'run_architecture("{RFPN}")', f'run_architecture("{PLAIN}")']
    # Default: only the RFPN model trains; the plain V1 Tiny is opt-in.
    assert namespace["ARCHITECTURES"] == [RFPN]
    assert namespace["TORCH_COMPILE_MODE"] == "default" and namespace["BENCHMARK_ONLY"] is False
    specs = namespace["ARCHITECTURE_SPECS"]
    assert set(specs) == {RFPN, PLAIN}
    for arch, fusion in ((RFPN, "rfpn"), (PLAIN, "none")):
        assert specs[arch]["model_name"] == "convnext_tiny.fb_in22k_ft_in1k_384"
        assert specs[arch]["fusion"] == fusion
        # The plain V1 Tiny clean12 recipe, unchanged: the runs differ only in the neck.
        assert specs[arch]["recipe"] == V1_TINY_RECIPE, (arch, specs[arch]["recipe"])
    assert namespace["FUSION_CHANNELS"] == 256
    assert namespace["resolve_fusion"]("rfpn") == ("fpn", "residual")
    with patch.dict(os.environ, {"CS_BASE_LR_CONVNEXT_TINY_RFPN": "3e-4", "CS_RARE_SAMPLING_CONVNEXT_TINY_RFPN": "1"}):
        override = namespace["architecture_recipe"](RFPN, base_lr=2e-4, weight_decay=0.02, warmup_epochs=2,
                                                    rare_sampling=False)
    assert override["base_lr"] == 3e-4 and override["rare_sampling"] is True
    assert specs[RFPN]["model_name"] in timm.list_pretrained("convnext_tiny*")
    namespace["tqdm"] = functools.partial(tqdm, disable=True)
    print("PASS: notebook structure; every code cell compiles; fusion cell == fusion_necks.py; "
          "plain V1 Tiny recipe for both models, overrides; pretrained tag exists in timm", timm.__version__)
    return namespace, full_cells


def synthetic_targets(labels, index):
    targets = np.zeros(len(labels), dtype=np.int8)
    if index % 3 == 0:
        targets[0] = 1  # NORM, exclusive
    else:
        targets[1 + (index % (len(labels) - 1))] = 1
        targets[1 + ((index * 5) % (len(labels) - 1))] = 1
    return targets


def write_synthetic_dataset(ns, folder):
    """Both views of every record: the clean sheet as PNG, the augmented one as lossless WebP."""
    labels = ns["LABELS"]
    rng = np.random.default_rng(0)
    rows = []
    sizes = {"train": 16, "validation": 12, "test": 12}
    for split, count in sizes.items():
        for index in range(count):
            targets = synthetic_targets(labels, index)
            for view, suffix in (("clean", ".png"), ("augmented", ".webp")):
                path = folder / view / "images" / f"{split}_{index}{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = rng.integers(0, 256, (SMALL_SIZE[0], SMALL_SIZE[1], 3), dtype=np.uint8)
                if suffix == ".webp":
                    Image.fromarray(pixels).save(path, "WEBP", lossless=True, exact=True)
                else:
                    Image.fromarray(pixels).save(path)
                rows.append({"split": split, "subset": split, "record_id": f"{split}/{index}",
                             "patient_id": f"p-{split}-{index // 2}", "source_dataset": ("I0001", "I0006")[index % 2],
                             "variant_id": view, "primary_metric": 1, "local_path": str(path),
                             **dict(zip(labels, targets.tolist()))})
    frame = pd.DataFrame(rows)
    frames = {split: frame[frame["split"].eq(split)].reset_index(drop=True) for split in sizes}
    for split in ("validation", "test"):
        missing = [label for label in labels if frames[split][label].nunique() < 2]
        assert not missing, f"synthetic {split} split lacks support for {missing}"
    return frames


def configure_namespace(ns, folder):
    frames = write_synthetic_dataset(ns, folder / "data")
    runs_root = folder / "drive_runs"
    run_dir, drive_dir = folder / "local_run", runs_root / "tiny_rfpn_check"
    run_dir.mkdir()
    drive_dir.mkdir(parents=True)
    labels = ns["LABELS"]
    ns.update(
        DEVICE=torch.device("cpu"), USE_AMP=False, AMP_DTYPE=torch.bfloat16, USE_FUSED_ADAMW=False,
        TORCH_COMPILE_ENABLED=False, INPUT_SIZE=SMALL_SIZE, MAX_EPOCHS=2, EARLY_STOPPING_MIN_EPOCHS=2,
        EARLY_STOPPING_PATIENCE=5, EFFECTIVE_BATCH_SIZE=4, AUTO_BATCH_CANDIDATES=[8, 2, 1],
        EVAL_BATCH_CANDIDATES=[4, 2], BENCHMARK_TRAIN_BATCHES=2, BENCHMARK_VAL_BATCHES=1,
        BENCHMARK_WARMUP_BATCHES=1, WORKER_BENCHMARK_BATCHES=2, WORKER_TUNING_BATCH=2,
        WORKER_CANDIDATES=[0], AUTO_TUNE_WORKERS=True, NUM_WORKERS=0, BOOTSTRAP_ROUNDS=25,
        COLLAPSE_MIN_VAL_MACRO_AUROC=0.0, CONTINUE_ON_ERROR=True,
        # The optional in-notebook baseline is switched on, so both build paths run.
        ARCHITECTURES=[RFPN, PLAIN],
        RUN_NAME="tiny_rfpn_check", RUN_DIR=run_dir, DRIVE_RUN_DIR=drive_dir, DRIVE_RUNS_ROOT=runs_root,
        RUN_LOG_PATH=run_dir / "run.log", DRIVE_SYNC_ENABLED=True, HF_HOME=folder / "hf",
        gpu=SimpleNamespace(name="CPU check", total_memory=80 * 2**30),
        PACKAGE_VERSIONS={"timm": timm.__version__, "torch": torch.__version__, "torchvision": "check"},
        METADATA_SHA256="m" * 64, MANIFEST_SHA256="n" * 64, AUG_MANIFEST_SHA256="a" * 64,
        DATASET_ARCHIVES={view: SimpleNamespace(view=view, tar_name=f"{view}.tar", expected_sha256=view[0] * 64,
                                                drive_path=folder / f"{view}.tar") for view in VIEWS},
        IMAGE_STORAGE_SELECTED="webp", DATASET_MODE="clean+augmented", SPLIT_SHA256="s" * 64,
        POS_WEIGHT_MODE="sqrt",
        train_df=frames["train"], val_df=frames["validation"], test_df=frames["test"],
        selected_frames=frames,
        image_transform=transforms.Compose([
            transforms.Resize(SMALL_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.PILToTensor(),
        ]),
    )
    ns["GPU_MEAN"] = torch.tensor(ns["IMAGENET_MEAN"]).view(1, 3, 1, 1)
    ns["GPU_STD"] = torch.tensor(ns["IMAGENET_STD"]).view(1, 3, 1, 1)
    ns["train_dataset"] = ns["ECGImageDataset"](frames["train"], return_metadata=False, training=True)
    ns["val_dataset"] = ns["ECGImageDataset"](frames["validation"], return_metadata=True)
    ns["test_dataset"] = ns["ECGImageDataset"](frames["test"], return_metadata=True)
    # The augmented rows decode from lossless WebP to exactly the pixels that were written.
    row = int(np.flatnonzero(frames["train"]["variant_id"].eq("augmented"))[0])
    with Image.open(frames["train"].local_path[row]) as image:
        assert torch.equal(ns["train_dataset"][row][0], torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1))
    # Section 4c's sampling plan and section 5's sampler, from the synthetic training rows.
    train_positive = frames["train"][labels].sum(axis=0).astype(np.int64)
    ns.update(train_positive=train_positive, train_label_frequency=train_positive / len(frames["train"]),
              _train_targets=frames["train"][labels].to_numpy(dtype=np.float64))
    ns["SAMPLING_PLANS"] = {False: ns["build_sampling_plan"](False)}
    ns["TRAIN_SAMPLERS"] = {False: ns["RepeatFactorSampler"](ns["SAMPLING_PLANS"][False].repeat_factors, ns["SEED"])}
    assert len(ns["TRAIN_SAMPLERS"][False]) == len(frames["train"]) == 32
    assert ns["SAMPLING_PLANS"][False].constant_predictor_loss > 0


def check_view_cells(ns, full_cells, folder):
    """Sections 4a + 4c: the augmented manifest audit and the two-view selection."""
    labels = ns["LABELS"]
    clean_root, aug_root = folder / "views" / "clean", folder / "views" / "augmented"
    rows = []
    for split, count in (("train", 16), ("validation", 12), ("test", 12)):
        for index in range(count):
            rows.append({"record_path": f"I000{1 + index % 2}/WFDB/{split}_{index}.hea",
                         "site": ("I0001", "I0006")[index % 2], "patient_key": f"p-{split}-{index // 2}",
                         "split": split, **dict(zip(labels, synthetic_targets(labels, index).tolist()))})
    record_df = pd.DataFrame(rows)
    for label in labels:
        record_df[label] = record_df[label].astype(np.int8)
    clean_view = record_df.drop(columns=["split"]).assign(
        image_path=lambda f: "images/" + f["record_path"].str.replace(".hea", ".png", regex=False),
        record_id=record_df["record_path"], dup_group_id=record_df["record_path"], patient_id=record_df["patient_key"],
        source_dataset=record_df["site"], variant_id="clean", variant_type="clean", primary_metric=1)
    split_by_patient = dict(zip(record_df["patient_key"], record_df["split"]))
    for root in (clean_root, aug_root):
        root.mkdir(parents=True)
        record_df.drop(columns=["split"]).to_csv(root / "record_manifest.csv", index=False)
    full_manifest = clean_view[["record_path", "image_path"]].assign(status="success")
    run_dir = folder / "view_run"
    run_dir.mkdir()
    ns.update(record_df=record_df, clean_view_df=clean_view, RUN_DIR=run_dir, KIT_ROOT=clean_root, AUG_ROOT=aug_root,
              RECORD_MANIFEST_CSV=clean_root / "record_manifest.csv",
              AUG_RECORD_MANIFEST_CSV=aug_root / "record_manifest.csv", AUG_MANIFEST_CSV=aug_root / "manifest.csv")
    ns["METADATA_SHA256"] = ns["sha256_file"](ns["RECORD_MANIFEST_CSV"])

    def audit(manifest, record_manifest=None):
        manifest.to_csv(ns["AUG_MANIFEST_CSV"], index=False)
        (record_manifest if record_manifest is not None else record_df.drop(columns=["split"])).to_csv(
            ns["AUG_RECORD_MANIFEST_CSV"], index=False)
        (run_dir / "records_without_augmented_image.csv").unlink(missing_ok=True)
        exec(full_cells["augmented_audit"], ns)
        return ns["metadata_df"]

    # 1. A failed and a missing augmented sheet: those records keep their clean sheet only.
    partial = full_manifest.copy()
    partial.loc[0, "status"] = "failed"
    metadata = audit(partial.drop(index=1))
    assert metadata["variant_id"].value_counts().to_dict() == {"clean": 40, "augmented": 38}
    without = pd.read_csv(run_dir / "records_without_augmented_image.csv")["record_path"].tolist()
    assert without == sorted(record_df.record_path[:2])
    augmented = metadata[metadata["variant_id"].eq("augmented")].set_index("record_id")
    clean = metadata[metadata["variant_id"].eq("clean")].set_index("record_id").loc[augmented.index]
    assert (augmented[labels + ["patient_id", "source_dataset"]] == clean[labels + ["patient_id", "source_dataset"]]).all().all()
    # 2. Same content in other bytes (reordered rows) is accepted; a flipped label, an
    # unknown record and a duplicate are refused.
    audit(full_manifest, record_df.drop(columns=["split"]).iloc[::-1])
    flipped = record_df.drop(columns=["split"]).copy()
    flipped.loc[3, "AF"] = 1 - flipped.loc[3, "AF"]
    for manifest, record_manifest, error_type, needle in (
        (full_manifest, flipped, ValueError, "disagrees"),
        (pd.concat([full_manifest, pd.DataFrame([{"record_path": "x.hea", "image_path": "images/x.png",
                                                   "status": "success"}])]), None, RuntimeError, "absent"),
        (pd.concat([full_manifest, full_manifest.iloc[:1]]), None, ValueError, "more than once"),
    ):
        try:
            audit(manifest, record_manifest)
            raise AssertionError(f"the augmented audit accepted a bad manifest ({needle})")
        except error_type as error:
            assert needle in str(error), error

    # 3. Section 4c on complete views stored as WebP: both sheets of a record in its
    # split, each mapped to its own archive's .webp file; pos_weight equals the
    # clean-only one, since both views carry the same labels.
    metadata = audit(full_manifest).copy()
    metadata["split"] = metadata["patient_id"].map(split_by_patient)
    metadata["subset"] = metadata["split"]
    for view, rel in zip(metadata["variant_id"], metadata["image_path"]):
        path = ns["DATASET_ROOTS"][view] / Path(rel).with_suffix(".webp")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"webp")
    ns.update(metadata_df=metadata, STORED_IMAGE_SUFFIX=".webp", MANIFEST_SHA256="n" * 64,
              AUG_MANIFEST_SHA256="a" * 64, SPLIT_SHA256="s" * 64,
              DATASET_ARCHIVES={view: SimpleNamespace(expected_sha256=view[0] * 64) for view in VIEWS})
    exec(full_cells["selection"], ns)
    train = ns["train_df"]
    assert len(train) == 32 and all(sorted(views) == sorted(VIEWS)
                                    for _, views in train.groupby("record_id")["variant_id"])
    assert all(path.endswith(".webp") and f"/views/{view}/" in path for view, path in zip(train.variant_id, train.local_path))
    clean_only = record_df[record_df["split"].eq("train")][labels].to_numpy(dtype=np.float64)
    negatives = len(clean_only) - clean_only.sum(axis=0)
    expected = np.sqrt(negatives / clean_only.sum(axis=0))
    assert np.allclose(ns["SAMPLING_PLANS"][False].pos_weight.numpy(), expected)
    print("PASS: sections 4a/4c (augmented sheets take their record's audited labels/patient/site; failed or "
          "missing sheets listed and the record kept clean-only; reordered-but-equal record manifest accepted; "
          "a disagreeing label, an unknown record and a duplicate refused; both views of a record in its split, "
          "each mapped to its archive's stored .webp; pos_weight == clean-only)")


@contextlib.contextmanager
def cpu_stubs():
    real_create = timm.create_model

    def create_without_download(name, pretrained=False, cache_dir=None, **kwargs):
        return real_create(name, pretrained=False, **kwargs)

    with contextlib.ExitStack() as stack:
        for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
            stack.enter_context(patch(f"torch.cuda.{name}", lambda *args, **kwargs: None))
        stack.enter_context(patch("torch.cuda.max_memory_allocated", lambda *args: 2**30))
        stack.enter_context(patch("torch.cuda.get_device_properties",
                                  lambda *args: SimpleNamespace(total_memory=80 * 2**30, name="CPU check")))
        stack.enter_context(patch("torch.cuda.get_rng_state_all", lambda: []))
        stack.enter_context(patch("torch.cuda.set_rng_state_all", lambda states: None))
        stack.enter_context(patch("timm.create_model", create_without_download))
        yield


def named_learning_rates(model, optimizer, prefix=""):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {names[id(parameter)][len(prefix):]: (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups for parameter in group["params"]
            if names[id(parameter)].startswith(prefix)}


def check_decoding(ns, folder):
    """Native sheets: libpng decode == PIL RGB + bicubic resize. Other sizes: the PIL path."""
    from torchvision.transforms import functional as TF
    samples = sorted((HERE.parent / "clean12_ecg_signal_images_ec2" / "samples").glob("[0-1][0-9]_*.png"))
    samples = [path for path in samples if Image.open(path).size == (1024, 768)]
    full_transform = transforms.Compose([
        transforms.Resize((768, 1024), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.PILToTensor(),
    ])
    labels = ns["LABELS"]
    frame = pd.DataFrame([{"local_path": str(path), **{label: 0 for label in labels}} for path in samples])
    with patch.dict(ns, {"INPUT_SIZE": (768, 1024), "image_transform": full_transform}):
        dataset = ns["ECGImageDataset"](frame, return_metadata=False)
        for index, path in enumerate(samples):
            with Image.open(path) as image:
                expected = full_transform(image.convert("RGB"))
            assert torch.equal(dataset[index][0], expected), path.name
    # An off-size sheet (and a palette PNG) falls back to PIL + resize.
    odd = folder / "odd.png"
    Image.fromarray(np.random.default_rng(3).integers(0, 256, (40, 60, 3), dtype=np.uint8)).convert("P").save(odd)
    dataset = ns["ECGImageDataset"](pd.DataFrame([{"local_path": str(odd), **{label: 0 for label in labels}}]),
                                    return_metadata=False)
    with Image.open(odd) as image:
        expected = ns["image_transform"](image.convert("RGB"))
    assert tuple(dataset[0][0].shape) == (3, *SMALL_SIZE) and torch.equal(dataset[0][0], expected)
    assert TF.get_image_size(dataset[0][0]) == [SMALL_SIZE[1], SMALL_SIZE[0]]
    print(f"PASS: libpng decoding is pixel-identical to PIL RGB + bicubic resize on {len(samples)} real clean12 "
          "sample sheets; an off-size palette PNG takes the PIL resize path")


def check_webp_transcoder(ns, folder):
    """Section 3's WebP worker on the real clean and augmented sample sheets, with each archive's
    settings: the stored .webp decodes (through the training dataset) to the PNG's exact pixels."""
    import importlib
    import sys
    module_dir = folder / "transcoder"
    module_dir.mkdir()
    (module_dir / "cs_webp_transcode_check.py").write_text(ns["WEBP_TRANSCODER_SOURCE"])
    sys.path.insert(0, str(module_dir))
    try:
        transcoder = importlib.import_module("cs_webp_transcode_check")
    finally:
        sys.path.remove(str(module_dir))
    # The notebook's own per-archive settings (DATASET_ARCHIVES from the config cell); run
    # before configure_namespace replaces DATASET_ARCHIVES.
    settings = {view: (archive.webp_method, archive.webp_quality) for view, archive in ns["DATASET_ARCHIVES"].items()}
    assert set(settings) == set(VIEWS)
    labels = ns["LABELS"]
    full_transform = transforms.Compose([transforms.PILToTensor()])
    png_total = webp_total = checked = 0
    for view, pattern in (("clean", "*_clean.png"), ("augmented", "*_augmented_*.png")):
        for path in sorted(AUG_SAMPLES.glob(pattern))[:4]:
            destination = folder / f"{view}_{path.stem}.webp"
            png_bytes, webp_bytes = transcoder.transcode(path.read_bytes(), str(destination), *settings[view])
            png_total, webp_total = png_total + png_bytes, webp_total + webp_bytes
            frame = pd.DataFrame([{"local_path": str(destination), **{label: 0 for label in labels}}])
            with patch.dict(ns, {"INPUT_SIZE": (768, 1024), "image_transform": full_transform}):
                decoded = ns["ECGImageDataset"](frame, return_metadata=False)[0][0]
            with Image.open(path) as image:
                assert torch.equal(decoded, full_transform(image.convert("RGB"))), path.name
            assert not destination.with_name(destination.name + ".part").exists()
            checked += 1
    assert checked == 8, checked
    print(f"PASS: lossless WebP worker on {checked} real clean/augmented sample sheets: training decodes the "
          f"stored .webp to the PNG's exact pixels; size {webp_total / png_total:.0%} of the PNGs")


def check_shared_cells(ns, full_cells):
    exec(full_cells["architectures"], ns)
    info = ns["ARCHITECTURE_INFO"]
    assert list(info) == ns["ARCHITECTURES"]
    for arch, row in info.items():
        assert row["normalization_mean"] == list(ns["IMAGENET_MEAN"]), arch
        # 20 per-block levels; 0.86**19 puts the stem where Base's 0.925**37 put it (~0.056x).
        assert row["layer_decay_levels"] == 20 and abs(row["layer_lr_scale_min"] - 0.86 ** 19) < 1e-6, row
        assert row["layer_lr_scale_max"] == 1.0, row
        assert row["gmacs_per_sheet"] > 0, row
    assert 25 < info[PLAIN]["parameters_m"] < 31 and info[PLAIN]["fusion_parameters_m"] == 0
    assert 1.9 < info[RFPN]["fusion_parameters_m"] < 2.4, info[RFPN]
    assert math.isclose(info[RFPN]["parameters_m"], info[PLAIN]["parameters_m"] + info[RFPN]["fusion_parameters_m"])
    assert info[RFPN]["fusion_spec"]["neck"] == "fpn" and info[RFPN]["fusion_spec"]["head"] == "residual"
    assert info[RFPN]["fusion_spec"]["channels"] == 256
    assert (ns["RUN_DIR"] / "architectures.json").is_file()
    assert (ns["DRIVE_RUN_DIR"] / "architectures.json").is_file()
    specs = ns["ARCHITECTURE_SPECS"]
    recipe = specs[RFPN]["recipe"]
    torch.manual_seed(0)
    plain = ns["build_model"](specs[PLAIN], pretrained=False).eval()
    rfpn = ns["build_model"](specs[RFPN], pretrained=False).eval()
    assert isinstance(rfpn, ns["ConvNeXtMultiScaleFusion"]) and rfpn.head_mode == "residual"
    assert rfpn.neck_name == "fpn" and [len(layer.weight) for layer in rfpn.neck.lateral] == [256] * 3
    assert [layer.in_channels for layer in rfpn.neck.lateral] == [192, 384, 768]
    # Stock timm keys under backbone., so the pretrained weights load unchanged.
    backbone_keys = [key[len("backbone."):] for key in rfpn.state_dict() if key.startswith("backbone.")]
    assert backbone_keys == list(plain.state_dict())
    assert {key.split(".")[0] for key in rfpn.state_dict()} == {"backbone", "neck", "fuse_norm", "fuse_fc"}
    assert rfpn.get_classifier().weight.abs().max() < 0.01, "head_init_scale 0.001 was not applied"
    assert rfpn.fuse_fc.weight.count_nonzero() == 0 and rfpn.fuse_fc.bias.count_nonzero() == 0
    # Step 0: with the same backbone weights the RFPN model IS the plain V1 Tiny.
    rfpn.backbone.load_state_dict(plain.state_dict())
    sample = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        assert torch.equal(rfpn(sample), plain(sample)), "the zero-init residual branch changed the step-0 output"
    # Optimizer: the backbone LR/WD profile equals the plain model's parameter by
    # parameter; the neck + fusion classifier train at the peak LR.
    plain_optimizer = ns["build_optimizer"](plain, recipe)
    rfpn_optimizer = ns["build_optimizer"](rfpn, recipe)
    plain_profile = named_learning_rates(plain, plain_optimizer)
    backbone_profile = named_learning_rates(rfpn, rfpn_optimizer, prefix="backbone.")
    assert set(plain_profile) == set(backbone_profile)
    for name, (lr, weight_decay) in plain_profile.items():
        assert math.isclose(backbone_profile[name][0], lr, rel_tol=1e-12) and backbone_profile[name][1] == weight_decay, name
    rates = [group["lr"] for group in rfpn_optimizer.param_groups]
    assert np.isclose(max(rates), recipe["base_lr"]) and np.isclose(min(rates), recipe["base_lr"] * 0.86 ** 19)
    fusion_groups = [group for group in rfpn_optimizer.param_groups if group.get("fusion")]
    assert len(fusion_groups) == 2 and all(np.isclose(group["lr"], recipe["base_lr"]) for group in fusion_groups)
    for optimizer, model in ((plain_optimizer, plain), (rfpn_optimizer, rfpn)):
        for group in optimizer.param_groups:
            dims = {parameter.ndim for parameter in group["params"]}
            assert (group["weight_decay"] == 0.0) == (dims == {1}), (group["weight_decay"], dims)
            assert group["weight_decay"] in (0.0, recipe["weight_decay"])
        assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))
    # Backward reaches every parameter; one update moves the fusion classifier off zero.
    rfpn.train()
    rfpn(sample).square().mean().backward()
    assert all(parameter.grad is not None for parameter in rfpn.parameters())
    rfpn_optimizer.step()
    assert rfpn.fuse_fc.weight.count_nonzero() > 0
    # Raw and EMA share one validation pass: identical to scoring each model alone.
    loader = torch.utils.data.DataLoader(ns["val_dataset"], batch_size=5, shuffle=False)
    rfpn.eval()
    shared = ns["predict_loader_models"]({"raw": rfpn, "ema": plain}, loader, None)
    for name, model in (("raw", rfpn), ("ema", plain)):
        alone = ns["predict_loader"](model, loader, None)
        assert np.array_equal(shared[name]["y_true"], alone["y_true"])
        assert np.allclose(shared[name]["y_probability"], alone["y_probability"], atol=1e-6)
        assert shared[name]["metadata"] == alone["metadata"]
    assert not np.allclose(shared["raw"]["y_probability"], shared["ema"]["y_probability"])

    exec(full_cells["workers"], ns)
    assert ns["NUM_WORKERS"] == 0 and (ns["DRIVE_RUN_DIR"] / "worker_benchmark.json").is_file()
    print(f"PASS: section 6 ({info[RFPN]['fusion_parameters_m']:.2f} M RFPN parameters on 192/384/768 -> 256; "
          "stock timm keys under backbone.; RFPN output == plain V1 Tiny at step 0; backbone LR profile "
          "identical to plain (20 levels, stem 0.057x); fusion groups at the peak LR; backward reaches every "
          "parameter; one shared raw+EMA validation pass == two separate passes) and section 7 (worker tuning) cells")


def check_native_size(ns):
    if os.environ.get("CS_CHECK_NATIVE", "1") == "0":
        print("SKIP: native 768x1024 MAC count (CS_CHECK_NATIVE=0)")
        return
    rows = {}
    with patch.dict(ns, {"INPUT_SIZE": (768, 1024)}):
        for arch in ns["ARCHITECTURES"]:
            model = ns["build_model"](ns["ARCHITECTURE_SPECS"][arch], pretrained=False)
            rows[arch] = round(ns["count_forward_gmacs"](model), 1)
            del model
    print("PASS: native 768x1024 forward per model; GMACs per sheet:", rows)


def steps_per_epoch(ns):
    batches = len(ns["TRAIN_SAMPLERS"][False]) // 2  # micro-batch 2 (the probe picks it below)
    return (batches - batches % 2) // 2               # accumulation 2 -> effective batch 4


def load_exported(ns, path):
    exported = torch.load(path, map_location="cpu", weights_only=True)
    model = timm.create_model(exported["model_name"], pretrained=False, num_classes=len(exported["label_order"]))
    if exported["fusion_neck"] != "none":
        model = ns["ConvNeXtMultiScaleFusion"](model, exported["fusion_neck"], exported["fusion_head"],
                                               len(exported["label_order"]), channels=exported["fusion_channels"])
    model.load_state_dict(exported["model"], strict=True)
    return exported, model


def check_full_runs(ns, full_cells):
    run_dir, drive_dir = ns["RUN_DIR"], ns["DRIVE_RUN_DIR"]
    steps = steps_per_epoch(ns)

    # 0. A model left out of CS_ARCHITECTURES (cell 10b by default) is skipped untouched.
    with patch.dict(ns, {"ARCHITECTURES": [RFPN]}):
        assert ns["run_architecture"](PLAIN) is None
    assert not (run_dir / PLAIN).exists()

    # 1. V1 Tiny + RFPN is interrupted after epoch 1, then resumes on a "fresh runtime".
    real_select = ns["select_validation_weights"]
    epochs_seen = []

    def interrupt_second_epoch(*args, **kwargs):
        epochs_seen.append(1)
        if len(epochs_seen) == 2:
            raise KeyboardInterrupt("simulated disconnect during epoch 2")
        return real_select(*args, **kwargs)

    ns["select_validation_weights"] = interrupt_second_epoch
    try:
        ns["run_architecture"](RFPN)
        raise AssertionError("the simulated disconnect did not propagate")
    except KeyboardInterrupt:
        pass
    finally:
        ns["select_validation_weights"] = real_select
    first = torch.load(drive_dir / RFPN / "last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1 and first["ema_num_updates"] == steps, (first["ema_num_updates"], steps)
    assert any(key.startswith("neck.") for key in first["model"]) and "fuse_fc.weight" in first["model_ema"]
    shutil.rmtree(run_dir / RFPN)
    summary = ns["run_architecture"](RFPN)
    assert summary is not None, ns["ARCH_FAILURES"]
    for name in ("best.pt", "last.pt", "inference.pt", "summary.json", "result.json", "report.md",
                 "test_bootstrap_ci.csv", "test_bootstrap_draws.npy", "evaluation_complete.json",
                 "test_bootstrap_ci_clean.csv", "test_bootstrap_draws_clean.npy", "per_view_metrics.csv",
                 "test_bootstrap_ci_augmented.csv", "test_bootstrap_draws_augmented.npy",
                 "training_history.csv", "drive_sync_manifest.json", "benchmark.json", "curves.png"):
        assert (drive_dir / RFPN / name).is_file(), name
    history = pd.read_csv(run_dir / RFPN / "training_history.csv")
    assert list(history.epoch) == [1, 2] and set(history.micro_batch_size) == {2}
    final = torch.load(run_dir / RFPN / "last.pt", map_location="cpu", weights_only=False)
    # 32 training sheets (both views of 16 records): 8 optimizer steps per epoch.
    assert final["ema_num_updates"] == 2 * steps == 16 and final["scheduler"]["last_epoch"] == 2 * steps
    assert final["config"]["runtime"]["split_image_counts_by_view"]["train"] == {"clean": 16, "augmented": 16}
    identity = final["config"]["identity"]
    assert identity["fusion"] == "rfpn" and identity["fusion_spec"]["head"] == "residual"
    assert identity["base_lr"] == 2e-4 and identity["layer_decay"] == 0.86
    assert identity["rare_label_sampling"]["enabled"] is False
    assert summary["fusion"] == "rfpn" and summary["fusion_parameters_m"] > 1.9
    assert identity["train_variants"] == ["clean", "augmented"] and set(identity["dataset_archive_sha256"]) == set(VIEWS)
    report_text = (drive_dir / RFPN / "report.md").read_text()
    assert "- Fusion: `rfpn`" in report_text and "augmented view:" in report_text
    # Per-view test metrics are the pooled predictions restricted to one view, scored alone.
    predictions = pd.read_csv(drive_dir / RFPN / "test_predictions.csv")
    assert predictions["variant_id"].value_counts().to_dict() == {"clean": 12, "augmented": 12}
    labels = ns["LABELS"]
    for view in VIEWS:
        rows = predictions[predictions["variant_id"].eq(view)]
        metrics = ns["evaluate_predictions"](rows[[f"true_{label}" for label in labels]].to_numpy(),
                                             rows[[f"prob_{label}" for label in labels]].to_numpy(), [0.5] * len(labels))
        assert np.isclose(metrics["pathology_only_macro_average_precision"],
                          summary[f"test_{view}_pathology_macro_ap"]), view
        assert summary[f"test_{view}_patients_sha256"] == summary["test_patients_sha256"], view
        ci = summary[f"test_{view}_pathology_macro_ap_ci_low"], summary[f"test_{view}_pathology_macro_ap_ci_high"]
        assert ci[0] <= ci[1], (view, ci)
    exported, rebuilt = load_exported(ns, drive_dir / RFPN / "inference.pt")
    assert (exported["fusion"], exported["fusion_neck"], exported["fusion_head"]) == ("rfpn", "fpn", "residual")
    assert exported["fusion_channels"] == 256 and exported["fusion_stage_indices"] == [1, 2, 3]
    assert rebuilt.fuse_fc.weight.count_nonzero() > 0, "the fusion classifier never trained"
    print(f"PASS: {RFPN} on both views interrupted in epoch 2, resumed from Drive last.pt on a wiped runtime, "
          f"finished ({steps} steps/epoch, export -> timm + ConvNeXtMultiScaleFusion, strict load); per-view test "
          "metrics == the pooled predictions restricted to that view; per-view bootstraps on the same patients")

    # 2. A wiped runtime skips a finished model using Drive only.
    shutil.rmtree(run_dir / RFPN)
    calls = []
    with patch.dict(ns, {"train_architecture": lambda ctx: calls.append(ctx.arch)}):
        again = ns["run_architecture"](RFPN)
    assert again == summary and not calls and not (run_dir / RFPN / "last.pt").exists()
    print("PASS: finished model skipped after the local copy is wiped (small files restored from Drive only)")

    # 3. The plain baseline fails first; the failure is recorded and the notebook continues.
    real_probe = ns["probe_micro_batch"]

    def failing_probe(ctx):
        raise RuntimeError("simulated: no micro-batch passed the VRAM preflight")

    ns["probe_micro_batch"] = failing_probe
    try:
        assert ns["run_architecture"](PLAIN) is None
    finally:
        ns["probe_micro_batch"] = real_probe
    failure = json.loads((drive_dir / PLAIN / "failure.json").read_text())
    assert failure["error_type"] == "RuntimeError" and PLAIN in ns["ARCH_FAILURES"]
    ns["ARCH_FAILURES"].pop(PLAIN)
    print("PASS: a failing model writes failure.json (on Drive) and does not stop the notebook")

    # 4. The plain V1 Tiny baseline end to end (re-run after its failure).
    summary_plain = ns["run_architecture"](PLAIN)
    assert summary_plain is not None, ns["ARCH_FAILURES"]
    assert not (run_dir / PLAIN / "failure.json").exists() and summary_plain["fusion"] == "none"
    exported, _ = load_exported(ns, drive_dir / PLAIN / "inference.pt")
    timm.create_model(exported["model_name"], pretrained=False, num_classes=12).load_state_dict(exported["model"])
    identity = torch.load(run_dir / PLAIN / "last.pt", map_location="cpu", weights_only=False)["config"]["identity"]
    assert identity["fusion"] == "none" and identity["base_lr"] == 2e-4
    print(f"PASS: {PLAIN} baseline full run; its export loads into a plain timm model")

    # 5. Comparison cell with clean-only references: a nested group/arch path holding
    # the RFPN run's clean-view predictions (pairs with [clean] at exactly 0), one whose
    # test records differ (skipped, never mixed in), and a missing one.
    runs_root = ns["DRIVE_RUNS_ROOT"]
    good, bad = runs_root / "tiny_group" / "convnext_tiny", runs_root / "base_bad"
    good.mkdir(parents=True)
    bad.mkdir()
    clean_predictions = predictions[predictions["variant_id"].eq("clean")]
    clean_predictions.to_csv(good / "test_predictions.csv", index=False)
    (good / "result.json").write_text(json.dumps({"summary": {
        "model_name": "convnext_tiny.check", "parameters_m": 27.8, "epochs_run": 11, "best_epoch": 6}}))
    clean_predictions.iloc[1:].to_csv(bad / "test_predictions.csv", index=False)
    with patch.dict(os.environ, {"CS_REFERENCE_RUNS": "tiny_group/convnext_tiny,base_bad,base_missing"}):
        exec(full_cells["comparison"], ns)
    reference = "ref:tiny_group/convnext_tiny [clean]"
    rows = [name for arch in (RFPN, PLAIN) for name in (arch, f"{arch} [clean]", f"{arch} [augmented]")]
    comparison = pd.read_csv(drive_dir / "comparison.csv")
    assert list(comparison.architecture) == rows + [reference], comparison.architecture.tolist()
    reference_row = comparison.set_index("architecture").loc[reference]
    rfpn_clean_row = comparison.set_index("architecture").loc[f"{RFPN} [clean]"]
    assert np.isclose(reference_row.test_pathology_macro_ap, rfpn_clean_row.test_pathology_macro_ap)
    assert np.isclose(reference_row.ci_low, rfpn_clean_row.ci_low) and reference_row.epochs_run == 11
    paired = pd.read_csv(drive_dir / "comparison_paired_bootstrap.csv")
    # Like-for-like only: pooled R-P; clean R-P, R-ref, P-ref; augmented R-P; each model's clean - augmented.
    assert len(paired) == 7 and paired["rounds"].eq(ns["BOOTSTRAP_ROUNDS"]).all(), paired[["first", "second"]]
    assert not ((paired["second"] == reference) & ~paired["first"].str.endswith("[clean]")).any()
    pair = paired[(paired["first"] == f"{RFPN} [clean]") & (paired["second"] == reference)].iloc[0]
    assert pair.mean_difference_pathology_macro_ap == 0.0 and pair.ci_low == 0.0 and pair.ci_high == 0.0
    pathology = [labels.index(label) for label in ns["PATHOLOGY_LABELS"]]
    for first_name, second_name, first_file, second_file in (
            (RFPN, PLAIN, drive_dir / RFPN / "test_bootstrap_draws.npy", run_dir / PLAIN / "test_bootstrap_draws.npy"),
            (f"{RFPN} [clean]", f"{RFPN} [augmented]", drive_dir / RFPN / "test_bootstrap_draws_clean.npy",
             drive_dir / RFPN / "test_bootstrap_draws_augmented.npy")):
        a, b = np.load(first_file), np.load(second_file)
        expected = float((np.nanmean(a[:, pathology], axis=1) - np.nanmean(b[:, pathology], axis=1)).mean())
        row = paired[(paired["first"] == first_name) & (paired["second"] == second_name)].iloc[0]
        assert np.isclose(row.mean_difference_pathology_macro_ap, expected), (first_name, second_name)
    report = (drive_dir / "comparison.md").read_text()
    assert reference in report and "base_bad" not in report and "fusion rfpn" in report
    assert (drive_dir / "comparison_curves.png").is_file()
    print("PASS: comparison cell (pooled + per-view rows, per-label AP, like-for-like paired bootstrap); a clean-only "
          "reference pairs with the clean view only, exactly 0 on a copy; a reference with other test records and "
          "a missing one are skipped")

    # 6. A changed recipe under the same run group is refused, never mixed.
    ns["ARCHITECTURE_SPECS"][RFPN]["recipe"]["base_lr"] *= 2
    ns["CONTINUE_ON_ERROR"] = False
    try:
        ns["run_architecture"](RFPN)
        raise AssertionError("a fingerprint change was accepted")
    except RuntimeError as error:
        assert "different configuration" in str(error)
    finally:
        ns["ARCHITECTURE_SPECS"][RFPN]["recipe"]["base_lr"] /= 2
        ns["CONTINUE_ON_ERROR"] = True
    print("PASS: a changed recipe under the same CS_RUN_GROUP is refused")


def check_benchmark_only(ns, full_cells, folder):
    """CS_BENCHMARK_ONLY=1: every model probes and benchmarks, nothing trains, 10c writes the table."""
    saved = {name: ns[name] for name in ("RUN_DIR", "DRIVE_RUN_DIR", "RUN_LOG_PATH", "BENCHMARK_ONLY")}
    run_dir, drive_dir = folder / "bench_local", folder / "bench_drive"
    run_dir.mkdir()
    drive_dir.mkdir()
    ns.update(RUN_DIR=run_dir, DRIVE_RUN_DIR=drive_dir, RUN_LOG_PATH=run_dir / "run.log",
              BENCHMARK_ONLY=True, COMPUTE_UNITS_PER_HOUR=2.0, copy_seconds=1800.0)
    ns["BENCHMARK_RESULTS"].clear()
    count = len(ns["ARCHITECTURES"])
    try:
        for arch in ns["ARCHITECTURES"]:
            row = ns["run_architecture"](arch)
            assert row is not None and row["architecture"] == arch, ns["ARCH_FAILURES"]
            assert not (run_dir / arch / "last.pt").exists() and not (run_dir / arch / "summary.json").exists()
            assert (drive_dir / arch / "benchmark.json").is_file()
            assert (drive_dir / arch / "benchmark_summary.json").is_file()
        exec(full_cells["benchmark_table"], ns)
        table = pd.read_csv(drive_dir / "benchmark_comparison.csv")
        assert list(table.architecture[:count]) == ns["ARCHITECTURES"]
        assert table.architecture.iloc[count].startswith("TOTAL")
        total = table.iloc[count]
        assert np.isclose(total.hours_at_max_epochs, table.hours_at_max_epochs[:count].sum() + 0.5)
        assert np.isclose(total.units_at_max_epochs, total.hours_at_max_epochs * 2.0)
        extra_epochs = ns["MAX_EPOCHS"] - ns["EARLY_STOPPING_MIN_EPOCHS"]
        assert np.allclose(table.hours_at_max_epochs[:count] - table.hours_at_min_epochs[:count],
                           extra_epochs * table.epoch_minutes[:count] / 60)
        exec(full_cells["comparison"], ns)  # prints the skip note, writes nothing
        assert not (run_dir / "comparison.csv").exists()
        print("PASS: CS_BENCHMARK_ONLY=1 benchmarks both models without training; 10c cost table "
              "(hours, units, TOTAL incl. staging) saved to Drive; comparison skipped")
    finally:
        ns.update(saved)
        ns.pop("copy_seconds", None)


def check_shared_folder(ns, folder):
    """Config cell: find Account A's shared folder on Account B's mount, then the access test."""
    import io
    import tarfile
    import warnings

    resolve, verify = ns["resolve_shared_folder"], ns["verify_shared_folder_access"]
    mount = folder / "shared_mount"
    targets = mount / ".shortcut-targets-by-id"
    (mount / "MyDrive").mkdir(parents=True)
    # 1. No shortcut yet: a clear instruction, not a silent fallback.
    try:
        resolve(mount, "CardioSentry")
        raise AssertionError("a missing shortcut was accepted")
    except RuntimeError as error:
        assert "Add shortcut" in str(error)
    # 2. Account B owns a same-named folder AND has a shortcut to the shared one:
    # the shortcut target wins (by name and by id).
    (mount / "MyDrive" / "CardioSentry").mkdir()
    shared = targets / "A_FOLDER_ID" / "CardioSentry"
    shared.mkdir(parents=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        root, folder_id, how = resolve(mount, "CardioSentry")
    assert (root, folder_id) == (shared, "A_FOLDER_ID") and any("Account B itself" in str(w.message) for w in caught)
    assert resolve(mount, "CardioSentry", "A_FOLDER_ID")[:2] == (shared, "A_FOLDER_ID")
    try:
        resolve(mount, "CardioSentry", "WRONG_ID")
        raise AssertionError("an unknown folder id was accepted")
    except RuntimeError as error:
        assert "WRONG_ID" in str(error)
    # 3. Two shared folders with the same name: refuse and ask for the id.
    (targets / "OTHER_ID" / "CardioSentry").mkdir(parents=True)
    try:
        resolve(mount, "CardioSentry")
        raise AssertionError("an ambiguous shared folder name was accepted")
    except RuntimeError as error:
        assert "CS_SHARED_FOLDER_ID" in str(error)
    assert resolve(mount, "CardioSentry", "OTHER_ID")[1] == "OTHER_ID"
    assert resolve(mount, "x", explicit_root=str(shared))[:2] == (shared, "A_FOLDER_ID")

    # 4. Access test: reads both tar headers + checksums, writes/renames/reads/deletes a probe.
    dataset_dir, runs_root = shared / "new_dataset", shared / "new_runs"
    dataset_dir.mkdir()
    archives = []
    for name in ("clean.tar", "augmented.tar"):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(f"{name[:-4]}/manifest.csv")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"x\n"))
        (dataset_dir / name).write_bytes(buffer.getvalue())
        (dataset_dir / f"{name}.sha256").write_text("0" * 64 + "\n")
        archives.append((dataset_dir / name, dataset_dir / f"{name}.sha256"))
    result = verify(archives, runs_root)
    assert result["dataset_bytes"] == {path.name: path.stat().st_size for path, _ in archives}
    assert list(runs_root.iterdir()) == []
    clean, augmented = archives
    for bad, expected_error, needle in (
        ((dataset_dir / "missing.tar", augmented[1]), FileNotFoundError, "do not copy"),
        ((augmented[1], augmented[1]), RuntimeError, "not readable as a tar"),
        ((augmented[0], dataset_dir / "missing.sha256"), FileNotFoundError, "CS_REQUIRE_TAR_SHA256"),
    ):
        try:
            verify([clean, bad], runs_root)
            raise AssertionError(f"{bad[0].name} passed the access test")
        except expected_error as error:
            assert needle in str(error), error
    runs_root.chmod(0o555)  # a Viewer-only share: reads work, writes fail
    try:
        verify(archives, runs_root)
        raise AssertionError("a read-only shared folder passed the write test")
    except RuntimeError as error:
        assert "Editor" in str(error)
    finally:
        runs_root.chmod(0o755)
    print("PASS: shared folder found through Account B's shortcut target (by name or CS_SHARED_FOLDER_ID; "
          "Account B's own same-named folder ignored; duplicates and a missing shortcut refused with "
          "instructions); access test reads both tar headers + checksums, writes/renames/reads/deletes a probe, "
          "and rejects a missing second archive or sidecar, a non-tar file and a read-only folder")


def png_bytes(rng, height, width):
    import io
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def build_archive(topdir, files, sheets, extra_members=()):
    """A tar in the layout of the real archives: <topdir>/ files + images/<site>/... sheets."""
    import io
    import tarfile
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory in (topdir, f"{topdir}/images"):
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        members = {f"{topdir}/{name}": payload for name, payload in files.items()}
        members.update({f"{topdir}/images/{name}": payload for name, payload in sheets.items()})
        members.update(dict(extra_members))
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue(), members


def check_dataset_staging(ns, cell, notebook_archives, folder):
    """Section 3 end to end: the clean and the augmented archive in a fake Account-A folder shared
    with Account B (reached through a My Drive shortcut), read via the API and via the mount,
    stored as PNG or as lossless WebP."""
    import hashlib
    import re
    import requests
    from torchvision.io import ImageReadMode, decode_png, decode_webp

    mount = folder / "mnt"
    # Colab lists a shortcut's target under .shortcut-targets-by-id/<folder id>/<name>;
    # the shortcut in Account B's My Drive points there.
    shared = mount / ".shortcut-targets-by-id" / "SHARED" / "CardioSentry"
    (shared / "new_dataset").mkdir(parents=True)
    (mount / "MyDrive").mkdir()
    (mount / "MyDrive" / "CardioSentry").symlink_to(shared, target_is_directory=True)
    root, shared_id, _ = ns["resolve_shared_folder"](mount, "CardioSentry")
    assert shared_id == "SHARED" and root == mount / "MyDrive" / "CardioSentry"
    drive_dir = root / "new_dataset"
    rng = np.random.default_rng(1)
    small = {"clean": ("manifest.csv", "record_manifest.csv", "final_counts.csv", "summary.json"),
             "augmented": ("manifest.csv", "record_manifest.csv", "summary.json", "config.json")}
    archives, members, data, file_ids = {}, {}, {}, {"clean": "T", "augmented": "U"}
    for view in VIEWS:
        template = notebook_archives[view]
        archive = SimpleNamespace(**{key: getattr(template, key) for key in (
            "view", "tar_name", "topdir", "webp_method", "webp_quality", "webp_size_ratio")}, drive_file_id="")
        archive.drive_path, archive.sidecar_path = drive_dir / archive.tar_name, drive_dir / f"{archive.tar_name}.sha256"
        data[view], members[view] = build_archive(
            archive.topdir, {name: f"{view},{name}\n".encode() for name in small[view]},
            {"I0001/WFDB/a.png": png_bytes(rng, 48, 64), "I0006/WFDB/b.png": png_bytes(rng, 40, 56)})
        archive.drive_path.write_bytes(data[view])
        archive.sidecar_path.write_text(f"{hashlib.sha256(data[view]).hexdigest()}  {archive.tar_name}\n")
        archives[view] = archive
    good_sha = {view: hashlib.sha256(data[view]).hexdigest() for view in VIEWS}

    extract_root, scratch_tmp = folder / "extract", folder / "scratch_tmp"
    extract_root.mkdir()
    scratch_tmp.mkdir()
    # API view: "new_dataset" is itself a shortcut, to check that shortcuts are followed.
    ids = {("SHARED", "new_dataset"): "N", **{("N", archives[v].tar_name): file_ids[v] for v in VIEWS}}
    view_of_id = {file_ids[v]: v for v in VIEWS}
    shortcut_names = {"new_dataset"}
    calls = {"media": 0, "list": 0, "meta": 0, "refresh": 0}
    drive_meta = {file_ids[v]: {"id": file_ids[v], "name": archives[v].tar_name, "size": str(len(data[v])),
                                "sha256Checksum": good_sha[v]} for v in VIEWS}
    flaky = np.random.default_rng(2)

    class Reply:
        def __init__(self, status_code, content=b"", payload=None):
            self.status_code, self.content, self._payload = status_code, content, payload
            self.text = content.decode(errors="replace")

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    def fake_get(url, params=None, timeout=None, headers=None):
        assert headers["Authorization"].startswith("Bearer token-")
        params = params or {}
        if url.endswith("/files"):
            calls["list"] += 1
            assert params["supportsAllDrives"] == "true" and params["includeItemsFromAllDrives"] == "true"
            name, parent = re.fullmatch(r"name = '(.+)' and '(.+)' in parents and trashed = false",
                                        params["q"]).groups()
            found = ids.get((parent, name))
            if found and name in shortcut_names:
                return Reply(200, payload={"files": [{
                    "id": "shortcut-" + found, "name": name, "mimeType": "application/vnd.google-apps.shortcut",
                    "shortcutDetails": {"targetId": found}}]})
            return Reply(200, payload={"files": [{"id": found, "name": name}] if found else []})
        file_id = url.rsplit("/", 1)[1]
        assert file_id in view_of_id, url
        if params.get("alt") != "media":
            calls["meta"] += 1
            return Reply(200, payload=dict(drive_meta[file_id]))
        calls["media"] += 1
        payload = data[view_of_id[file_id]]
        roll = flaky.random()
        if roll < 0.08:
            return Reply(503)
        if roll < 0.12:
            return Reply(429)
        if roll < 0.16:
            raise requests.ConnectionError("dropped")
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        if roll < 0.20:
            return Reply(206, payload[start:end])  # short body: must be retried
        return Reply(206, payload[start:end + 1])

    class FakeCredentials:
        valid, token = False, None

        def refresh(self, request):
            calls["refresh"] += 1
            self.valid, self.token = True, f"token-{calls['refresh']}"

    fake_requests = SimpleNamespace(get=fake_get, RequestException=requests.RequestException,
                                    HTTPError=requests.HTTPError, ConnectionError=requests.ConnectionError)
    ns.update(
        IN_COLAB=False, DRIVE_MOUNT_POINT=mount, CARDIOSENTRY_DRIVE_ROOT=root, SHARED_FOLDER_DRIVE_ID=shared_id,
        DATASET_DRIVE_DIR=drive_dir, DATASET_ARCHIVES=archives,
        REQUIRE_TAR_SHA256=True, DATASET_READ_MODE="api", DATASET_DOWNLOAD_WORKERS=4,
        DATASET_DOWNLOAD_CHUNK_BYTES=8192, EXTRACT_ROOT=extract_root, SCRATCH_TMP=scratch_tmp,
        SCRATCH_MARGIN_GIB=0.0, IMAGE_STORAGE="png", TRANSCODE_WORKERS=2, requests=fake_requests,
    )
    trees = {view: extract_root / archives[view].topdir for view in VIEWS}
    marker = {view: extract_root / f".{view}_extract_complete.json" for view in VIEWS}

    def sheets(view):
        return {name: payload for name, payload in members[view].items() if name.endswith(".png") and "/images/" in name}

    def assert_extracted(view, storage):
        for name, payload in members[view].items():
            if name in sheets(view) and storage == "webp":
                stored = extract_root / Path(name).with_suffix(".webp")
                assert not (extract_root / name).exists(), name
                expected = decode_png(torch.frombuffer(bytearray(payload), dtype=torch.uint8), mode=ImageReadMode.RGB)
                actual = decode_webp(torch.frombuffer(bytearray(stored.read_bytes()), dtype=torch.uint8),
                                     mode=ImageReadMode.RGB)
                assert torch.equal(actual, expected), name
            else:
                assert (extract_root / name).read_bytes() == payload, name
        state = json.loads(marker[view].read_text())
        assert state["sha256"] == good_sha[view] and state["storage"] == storage and state["sheets"] == 2, state
        assert not list(trees[view].rglob("*.part"))

    # Stand-ins for google-auth (on Colab, not necessarily in the local venv).
    fake_google = SimpleNamespace(auth=SimpleNamespace(
        default=lambda scopes=None: (FakeCredentials(), None),
        transport=SimpleNamespace(requests=SimpleNamespace(Request=lambda: None))))
    fake_modules = {"google": fake_google, "google.auth": fake_google.auth,
                    "google.auth.transport": fake_google.auth.transport,
                    "google.auth.transport.requests": fake_google.auth.transport.requests}
    real_disk_usage = shutil.disk_usage
    with patch.dict("sys.modules", fake_modules), patch("time.sleep", lambda seconds: None):
        # 1. API mode, PNG storage: each file id is resolved inside the shared folder (from
        # its id, following the shortcut), flaky ranges retried, both archives verified.
        exec(cell, ns)
        for view in VIEWS:
            assert_extracted(view, "png")
        assert calls["list"] == 4 and calls["meta"] == 2 and calls["media"] > sum(map(len, data.values())) // 8192
        print(f"PASS: section 3 api mode resolves both archives inside the shared folder (via its shortcut "
              f"target id, following a shortcut) and streams Drive -> tar ({calls['media']} ranged GETs incl. "
              "injected 503/429/drops/short reads), verifying each SHA-256; PNG storage is byte-exact")

        # 2. Re-running on the same runtime skips without touching Drive (no sign-in); "auto"
        # keeps the PNGs when the disk holds both archives.
        before = dict(calls)
        ns["IMAGE_STORAGE"] = "auto"
        exec(cell, ns)
        assert calls == before and ns["IMAGE_STORAGE_SELECTED"] == "png" and ns["STORED_IMAGE_SUFFIX"] == ".png"
        print("PASS: section 3 skips verified extractions without any Drive API call; auto keeps PNG on a large disk")

        # 3. "auto" on a disk too small for both archives as PNG: both are re-staged as
        # lossless WebP (mount mode), pixel-identical to the archived PNGs.
        free_values = iter([1])

        def small_then_large(path):
            usage = real_disk_usage(path)
            return usage._replace(free=next(free_values, 10 * 2**40))

        ns["DATASET_READ_MODE"] = "mount"
        with patch("shutil.disk_usage", small_then_large):
            exec(cell, ns)
        assert calls == before and ns["IMAGE_STORAGE_SELECTED"] == "webp" and ns["STORED_IMAGE_SUFFIX"] == ".webp"
        for view in VIEWS:
            assert_extracted(view, "webp")
        stamp = {view: marker[view].stat().st_mtime_ns for view in VIEWS}
        ns["IMAGE_STORAGE"] = "webp"
        exec(cell, ns)
        assert all(marker[view].stat().st_mtime_ns == stamp[view] for view in VIEWS)
        print("PASS: section 3 auto switches to lossless WebP when both archives do not fit as PNG (mount mode); "
              "every stored sheet decodes to its PNG's pixels, other files byte-exact; a WebP re-run skips")

        # 4. A sidecar mismatch on the augmented archive deletes only its tree and raises;
        # the verified clean extraction is kept.
        archives["augmented"].sidecar_path.write_text(f"{'0' * 64}  x\n")
        try:
            exec(cell, ns)
            raise AssertionError("a SHA-256 mismatch was accepted")
        except RuntimeError as error:
            assert "SHA-256" in str(error) and not trees["augmented"].exists()
        assert_extracted("clean", "webp")
        assert marker["clean"].stat().st_mtime_ns == stamp["clean"]
        print("PASS: section 3 rejects a SHA-256 mismatch on one archive, deletes its partial tree, keeps the other")

        # 5. A member outside the archive's top folder is refused.
        good_data = data["augmented"]
        data["augmented"], _ = build_archive(archives["augmented"].topdir, {"manifest.csv": b"x\n"}, {},
                                             extra_members=[("elsewhere/evil.png", b"x")])
        archives["augmented"].drive_path.write_bytes(data["augmented"])
        archives["augmented"].sidecar_path.write_text(hashlib.sha256(data["augmented"]).hexdigest() + "\n")
        try:
            exec(cell, ns)
            raise AssertionError("a member outside the top folder was accepted")
        except RuntimeError as error:
            assert "unexpected member" in str(error) and not trees["augmented"].exists()
            assert not (extract_root / "elsewhere").exists()
        data["augmented"] = good_data
        archives["augmented"].drive_path.write_bytes(good_data)
        print("PASS: section 3 refuses a tar member outside the archive's top folder")

        # 6. API mode refuses a damaged Drive copy before downloading anything.
        archives["augmented"].sidecar_path.write_text(f"{good_sha['augmented']}  x\n")
        drive_meta["U"]["sha256Checksum"] = "f" * 64
        ns["DATASET_READ_MODE"] = "api"
        media_before = calls["media"]
        try:
            exec(cell, ns)
            raise AssertionError("a damaged Drive copy was accepted")
        except RuntimeError as error:
            assert "damaged" in str(error) and calls["media"] == media_before
        print("PASS: section 3 api mode refuses a Drive copy whose SHA-256 disagrees, before downloading")

        # 7. Repaired: the augmented archive streams again (API, WebP); the clean one is kept.
        drive_meta["U"]["sha256Checksum"] = good_sha["augmented"]
        exec(cell, ns)
        for view in VIEWS:
            assert_extracted(view, "webp")
        assert marker["clean"].stat().st_mtime_ns == stamp["clean"]
        print("PASS: section 3 api mode + WebP restages only the archive that was missing")


def main():
    import copy
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ns, full_cells = load_notebook()
    notebook_archives = copy.deepcopy(ns["DATASET_ARCHIVES"])
    with tempfile.TemporaryDirectory(prefix="cardiosentry-tiny-rfpn-aug-check-") as directory:
        folder = Path(directory)
        check_webp_transcoder(ns, folder)
        check_view_cells(ns, full_cells, folder)
        configure_namespace(ns, folder)
        check_decoding(ns, folder)
        check_shared_folder(ns, folder)
        check_dataset_staging(ns, full_cells["dataset"], notebook_archives, folder)
        with cpu_stubs():
            check_shared_cells(ns, full_cells)
            check_native_size(ns)
            check_full_runs(ns, full_cells)
            check_benchmark_only(ns, full_cells, folder)
    print("All CPU checks passed. GPU speed/memory, torch.compile, real Drive I/O, and full-corpus "
          "training are not exercised here.")


if __name__ == "__main__":
    main()
