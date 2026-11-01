"""CPU checks of the generated ConvNeXt Tiny (V1, V2) clean12 Colab notebook; never mounts Drive or contacts Google.

Run: ../../../.venv/bin/python validate_notebook.py

The notebook's own definitions are executed. Sections 3 (Drive -> tar
streaming, against a fake Drive API and a fake mount), 6 (architectures), 7
(worker tuning) and 11 (comparison) run as whole cells, and
run_architecture() runs end to end for both real timm architectures (random
weights, no download) on tiny synthetic PNG sheets. "Drive" is a temporary
folder. CUDA-only calls are stubbed, so this checks the pipeline logic, not GPU
speed or memory.

Checked: the per-model recipes (layer decay, GRN swap, sampling plan), a full
run per model, skipping a finished model after the local copy is wiped,
resuming an interrupted model from Drive, continuing past a failing model, a
recipe change being refused, and the paired comparison including a Base
reference re-scored from its test_predictions.csv.
CS_CHECK_NATIVE=1 (default) also builds both models at 768x1024 for MACs.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import ast
import contextlib
import functools
import json
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
NOTEBOOK = HERE / "CardioSentry_ConvNeXt_Tiny_V1_V2_Clean12_Colab.ipynb"
RUN_CELL_PREFIXES = ("# 10a.", "# 10b.", "# 12.")
V1, V2 = "convnext_tiny", "convnextv2_tiny"
FULL_CELL_MARKERS = {
    "dataset": "def stream_extract_archive(",
    "architectures": "ARCHITECTURE_INFO = {}",
    "workers": "def tune_decoder_workers(",
    "comparison": "comparison_rows, per_label_rows, histories, draws = [], [], {}, {}",
    "benchmark_table": "# 10c. Benchmark-only cost table",
}
SMALL_SIZE = (32, 48)
CONSTANTS = {
    "LABELS", "NORMAL_LABEL", "PATHOLOGY_LABELS", "LABEL_NAMES", "SUBSETS", "SITES", "TRAIN_VARIANTS",
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
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        source = "".join(cell["source"])
        if "%pip" in source:
            assert '"timm==1.0.28"' in source
            continue
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
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
            )
            if keep:
                exec(compile(ast.Module(body=[node], type_ignores=[]), NOTEBOOK.name, "exec"), namespace)
    assert set(full_cells) == set(FULL_CELL_MARKERS), sorted(full_cells)
    assert [cell.splitlines()[1] for cell in run_cells[:2]] == [
        f'run_architecture("{V1}")', f'run_architecture("{V2}")']
    assert namespace["ARCHITECTURES"] == [V1, V2]
    assert namespace["TORCH_COMPILE_MODE"] == "default" and namespace["BENCHMARK_ONLY"] is False
    specs = namespace["ARCHITECTURE_SPECS"]
    # Each Tiny model carries its Base run's recipe; only layer decay and drop-path change.
    assert specs[V1]["model_name"] == "convnext_tiny.fb_in22k_ft_in1k_384"
    assert specs[V2]["model_name"] == "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
    assert specs[V1]["recipe"] == {
        "base_lr": 2e-4, "weight_decay": 0.02, "warmup_epochs": 2, "layer_decay": 0.86, "drop_path_rate": 0.1,
        "head_init_scale": 0.001, "rare_sampling": False, "mixed_precision_grn": False}, specs[V1]["recipe"]
    assert specs[V2]["recipe"] == {
        "base_lr": 1e-4, "weight_decay": 0.05, "warmup_epochs": 3, "layer_decay": 0.86, "drop_path_rate": 0.1,
        "head_init_scale": 0.001, "rare_sampling": True, "mixed_precision_grn": True}, specs[V2]["recipe"]
    with patch.dict(os.environ, {"CS_BASE_LR_CONVNEXTV2_TINY": "3e-4", "CS_RARE_SAMPLING_CONVNEXTV2_TINY": "0"}):
        override = namespace["architecture_recipe"](V2, base_lr=1e-4, weight_decay=0.05, warmup_epochs=3,
                                                    rare_sampling=True, mixed_precision_grn=True)
    assert override["base_lr"] == 3e-4 and override["rare_sampling"] is False
    for spec in specs.values():
        architecture, tag = spec["model_name"].split(".")
        assert spec["model_name"] in timm.list_pretrained(architecture + "*"), spec["model_name"]
    namespace["tqdm"] = functools.partial(tqdm, disable=True)
    print("PASS: notebook structure; every code cell compiles; per-model recipes and overrides; "
          "both pretrained tags exist in timm", timm.__version__)
    return namespace, full_cells


def write_synthetic_dataset(ns, folder):
    labels = ns["LABELS"]
    rng = np.random.default_rng(0)
    rows = []
    sizes = {"train": 16, "validation": 12, "test": 12}
    for split, count in sizes.items():
        for index in range(count):
            targets = np.zeros(len(labels), dtype=np.int8)
            if index % 3 == 0:
                targets[0] = 1  # NORM, exclusive
            else:
                targets[1 + (index % (len(labels) - 1))] = 1
                targets[1 + ((index * 5) % (len(labels) - 1))] = 1
            path = folder / "images" / f"{split}_{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (SMALL_SIZE[0], SMALL_SIZE[1], 3), dtype=np.uint8)).save(path)
            rows.append({"split": split, "subset": split, "record_id": f"{split}/{index}",
                         "patient_id": f"p-{split}-{index // 2}", "source_dataset": ("I0001", "I0006")[index % 2],
                         "variant_id": "clean", "primary_metric": 1, "local_path": str(path),
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
    run_dir, drive_dir = folder / "local_run", runs_root / "tiny_check"
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
        RUN_NAME="tiny_check", RUN_DIR=run_dir, DRIVE_RUN_DIR=drive_dir, DRIVE_RUNS_ROOT=runs_root,
        RUN_LOG_PATH=run_dir / "run.log", DRIVE_SYNC_ENABLED=True, HF_HOME=folder / "hf",
        gpu=SimpleNamespace(name="CPU check", total_memory=80 * 2**30),
        PACKAGE_VERSIONS={"timm": timm.__version__, "torch": torch.__version__, "torchvision": "check"},
        METADATA_SHA256="m" * 64, MANIFEST_SHA256="n" * 64, EXPECTED_TAR_SHA256="t" * 64,
        DATASET_TAR_NAME="clean.tar", DATASET_MODE="clean-only", SPLIT_SHA256="s" * 64,
        POS_WEIGHT_MODE="sqrt",
        # A high threshold so the synthetic labels are "rare" and the V2 sampler really repeats rows.
        RARE_SAMPLING_THRESHOLD=0.5,
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
    # Section 4c's sampling plans and section 5's samplers, from the synthetic training rows.
    train_positive = frames["train"][labels].sum(axis=0).astype(np.int64)
    ns.update(train_positive=train_positive, train_label_frequency=train_positive / len(frames["train"]),
              _train_targets=frames["train"][labels].to_numpy(dtype=np.float64))
    ns["SAMPLING_PLANS"] = {flag: ns["build_sampling_plan"](flag) for flag in (False, True)}
    ns["TRAIN_SAMPLERS"] = {flag: ns["RepeatFactorSampler"](plan.repeat_factors, ns["SEED"])
                            for flag, plan in ns["SAMPLING_PLANS"].items()}
    off, on = ns["SAMPLING_PLANS"][False], ns["SAMPLING_PLANS"][True]
    assert len(ns["TRAIN_SAMPLERS"][False]) == len(frames["train"])
    assert len(ns["TRAIN_SAMPLERS"][True]) > len(frames["train"])
    assert not torch.equal(off.pos_weight, on.pos_weight) and off.constant_predictor_loss > 0


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


def check_shared_cells(ns, full_cells):
    exec(full_cells["architectures"], ns)
    info = ns["ARCHITECTURE_INFO"]
    assert list(info) == ns["ARCHITECTURES"]
    for arch, row in info.items():
        assert row["normalization_mean"] == list(ns["IMAGENET_MEAN"]), arch
        assert row["gmacs_per_sheet"] > 0 and 25 < row["parameters_m"] < 31, row
        # 20 per-block levels; 0.86**19 puts the stem where Base's 0.925**37 put it (~0.056x).
        assert row["layer_decay_levels"] == 20 and abs(row["layer_lr_scale_min"] - 0.86 ** 19) < 1e-6, row
        assert row["layer_lr_scale_max"] == 1.0, row
    assert info[V1]["mixed_precision_grn_modules"] == 0 and info[V2]["mixed_precision_grn_modules"] == 18
    assert (ns["RUN_DIR"] / "architectures.json").is_file()
    assert (ns["DRIVE_RUN_DIR"] / "architectures.json").is_file()
    # Optimizer: per-block layer decay, decay on >=2-D weights only, every parameter once.
    for arch in ns["ARCHITECTURES"]:
        recipe = ns["ARCHITECTURE_SPECS"][arch]["recipe"]
        model = ns["build_model"](ns["ARCHITECTURE_SPECS"][arch], pretrained=False)
        optimizer = ns["build_optimizer"](model, recipe)
        rates = {group["lr"] for group in optimizer.param_groups}
        assert np.isclose(max(rates), recipe["base_lr"]) and np.isclose(min(rates), recipe["base_lr"] * 0.86 ** 19)
        for group in optimizer.param_groups:
            dims = {parameter.ndim for parameter in group["params"]}
            assert (group["weight_decay"] == 0.0) == (dims == {1}), (arch, group["weight_decay"], dims)
            assert group["weight_decay"] in (0.0, recipe["weight_decay"])
        assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))
        head = model.get_classifier().weight
        assert head.abs().max() < 0.01, "head_init_scale 0.001 was not applied"
    # MixedPrecisionGRN equals timm's GRN in FP32 and keeps its state-dict keys.
    reference = timm.create_model(ns["ARCHITECTURE_SPECS"][V2]["model_name"], pretrained=False, num_classes=12)
    swapped = ns["build_model"](ns["ARCHITECTURE_SPECS"][V2], pretrained=False)
    assert list(reference.state_dict()) == list(swapped.state_dict())
    for module in reference.modules():
        if hasattr(module, "wb_shape"):
            with torch.no_grad():
                module.weight.normal_()
                module.bias.normal_()
    swapped.load_state_dict(reference.state_dict())
    sample = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        assert torch.allclose(reference.eval()(sample), swapped.eval()(sample), atol=1e-4, rtol=1e-4)
    exec(full_cells["workers"], ns)
    assert ns["NUM_WORKERS"] == 0 and (ns["DRIVE_RUN_DIR"] / "worker_benchmark.json").is_file()
    print("PASS: section 6 (weights, MACs, layer decay 20 levels / stem 0.057x, head init, "
          "mixed-precision GRN = timm GRN with the same keys) and section 7 (worker tuning) cells")


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


def steps_per_epoch(ns, arch):
    sampler = ns["TRAIN_SAMPLERS"][ns["ARCHITECTURE_SPECS"][arch]["recipe"]["rare_sampling"]]
    batches = len(sampler) // 2          # micro-batch 2 (the probe picks it below)
    return (batches - batches % 2) // 2  # accumulation 2 -> effective batch 4


def check_full_runs(ns, full_cells):
    run_dir, drive_dir = ns["RUN_DIR"], ns["DRIVE_RUN_DIR"]

    # 1. ConvNeXt-V1 Tiny end to end.
    summary = ns["run_architecture"](V1)
    assert summary is not None, ns["ARCH_FAILURES"]
    for name in ("best.pt", "last.pt", "inference.pt", "summary.json", "result.json", "report.md",
                 "test_bootstrap_ci.csv", "test_bootstrap_draws.npy", "evaluation_complete.json",
                 "training_history.csv", "drive_sync_manifest.json", "benchmark.json", "curves.png"):
        assert (drive_dir / V1 / name).is_file(), name
    history = pd.read_csv(run_dir / V1 / "training_history.csv")
    assert list(history.epoch) == [1, 2] and set(history.micro_batch_size) == {2}
    last = torch.load(run_dir / V1 / "last.pt", map_location="cpu", weights_only=False)
    assert last["ema_num_updates"] == 2 * steps_per_epoch(ns, V1) == 8
    identity = last["config"]["identity"]
    assert identity["base_lr"] == 2e-4 and identity["layer_decay"] == 0.86
    assert identity["rare_label_sampling"]["enabled"] is False and identity["grn_implementation"] is None
    exported = torch.load(drive_dir / V1 / "inference.pt", map_location="cpu", weights_only=True)
    rebuilt = timm.create_model(exported["model_name"], pretrained=False, num_classes=len(exported["label_order"]))
    rebuilt.load_state_dict(exported["model"])
    print("PASS: convnext_tiny full run (probe -> micro-batch 2 x 2, train 2 epochs, evaluate, export, Drive sync)")

    # 2. A wiped runtime skips a finished model using Drive only.
    shutil.rmtree(run_dir / V1)
    calls = []
    with patch.dict(ns, {"train_architecture": lambda ctx: calls.append(ctx.arch)}):
        again = ns["run_architecture"](V1)
    assert again == summary and not calls and not (run_dir / V1 / "last.pt").exists()
    print("PASS: finished model skipped after the local copy is wiped (small files restored from Drive only)")

    # 3. ConvNeXt-V2 Tiny fails first; the failure is recorded and the notebook continues.
    real_probe = ns["probe_micro_batch"]

    def failing_probe(ctx):
        raise RuntimeError("simulated: no micro-batch passed the VRAM preflight")

    ns["probe_micro_batch"] = failing_probe
    try:
        assert ns["run_architecture"](V2) is None
    finally:
        ns["probe_micro_batch"] = real_probe
    failure = json.loads((drive_dir / V2 / "failure.json").read_text())
    assert failure["error_type"] == "RuntimeError" and V2 in ns["ARCH_FAILURES"]
    ns["ARCH_FAILURES"].pop(V2)
    print("PASS: a failing model writes failure.json (on Drive) and does not stop the notebook")

    # 4. ConvNeXt-V2 Tiny is interrupted after epoch 1, then resumes on a "fresh runtime".
    real_select = ns["select_validation_weights"]
    epochs_seen = []

    def interrupt_second_epoch(*args, **kwargs):
        epochs_seen.append(1)
        if len(epochs_seen) == 2:
            raise KeyboardInterrupt("simulated disconnect during epoch 2")
        return real_select(*args, **kwargs)

    ns["select_validation_weights"] = interrupt_second_epoch
    try:
        ns["run_architecture"](V2)
        raise AssertionError("the simulated disconnect did not propagate")
    except KeyboardInterrupt:
        pass
    finally:
        ns["select_validation_weights"] = real_select
    steps = steps_per_epoch(ns, V2)
    assert steps > steps_per_epoch(ns, V1), "the V2 rare-label sampler did not lengthen the epoch"
    first = torch.load(drive_dir / V2 / "last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1 and first["ema_num_updates"] == steps, (first["ema_num_updates"], steps)
    shutil.rmtree(run_dir / V2)
    summary = ns["run_architecture"](V2)
    assert summary is not None, ns["ARCH_FAILURES"]
    assert not (run_dir / V2 / "failure.json").exists()
    history = pd.read_csv(run_dir / V2 / "training_history.csv")
    assert list(history.epoch) == [1, 2], history.epoch.tolist()
    final = torch.load(run_dir / V2 / "last.pt", map_location="cpu", weights_only=False)
    assert final["ema_num_updates"] == 2 * steps and final["scheduler"]["last_epoch"] == 2 * steps
    identity = final["config"]["identity"]
    assert identity["base_lr"] == 1e-4 and identity["rare_label_sampling"]["enabled"] is True
    assert identity["rare_label_sampling"]["draws_per_epoch"] == len(ns["TRAIN_SAMPLERS"][True])
    assert identity["grn_implementation"].startswith("MixedPrecisionGRN")
    assert identity["train_pos_weight"] == dict(zip(ns["LABELS"], map(float, ns["SAMPLING_PLANS"][True].pos_weight.tolist())))
    exported = torch.load(drive_dir / V2 / "inference.pt", map_location="cpu", weights_only=True)
    timm.create_model(exported["model_name"], pretrained=False, num_classes=12).load_state_dict(exported["model"])
    print(f"PASS: convnextv2_tiny (rare-label sampler, {steps} steps/epoch) interrupted in epoch 2, resumed "
          "from Drive last.pt on a wiped runtime, finished; its export loads into a plain timm model")

    # 5. Comparison cell with two Base references: one valid (a copy of V1's
    # predictions, so its paired difference with V1 must be exactly 0) and one
    # whose test records differ (skipped, never mixed in).
    runs_root = ns["DRIVE_RUNS_ROOT"]
    good, bad = runs_root / "base_good", runs_root / "base_bad"
    good.mkdir()
    bad.mkdir()
    predictions = pd.read_csv(drive_dir / V1 / "test_predictions.csv")
    predictions.to_csv(good / "test_predictions.csv", index=False)
    (good / "result.json").write_text(json.dumps({"summary": {
        "model_name": "convnext_base.check", "parameters_m": 87.6, "epochs_run": 11, "best_epoch": 6}}))
    predictions.iloc[1:].to_csv(bad / "test_predictions.csv", index=False)
    with patch.dict(os.environ, {"CS_REFERENCE_RUNS": "base_good,base_bad,base_missing"}):
        exec(full_cells["comparison"], ns)
    comparison = pd.read_csv(drive_dir / "comparison.csv")
    assert list(comparison.architecture) == [V1, V2, "ref:base_good"], comparison.architecture.tolist()
    reference_row = comparison.set_index("architecture").loc["ref:base_good"]
    v1_row = comparison.set_index("architecture").loc[V1]
    assert np.isclose(reference_row.test_pathology_macro_ap, v1_row.test_pathology_macro_ap)
    assert np.isclose(reference_row.ci_low, v1_row.ci_low) and reference_row.epochs_run == 11
    paired = pd.read_csv(drive_dir / "comparison_paired_bootstrap.csv")
    assert len(paired) == 3 and paired["rounds"].eq(ns["BOOTSTRAP_ROUNDS"]).all()
    pair = paired[(paired["first"] == V1) & (paired["second"] == "ref:base_good")].iloc[0]
    assert pair.mean_difference_pathology_macro_ap == 0.0 and pair.ci_low == 0.0 and pair.ci_high == 0.0
    a = np.load(drive_dir / V1 / "test_bootstrap_draws.npy")
    b = np.load(run_dir / V2 / "test_bootstrap_draws.npy")
    pathology = [ns["LABELS"].index(label) for label in ns["PATHOLOGY_LABELS"]]
    expected = float((np.nanmean(a[:, pathology], axis=1) - np.nanmean(b[:, pathology], axis=1)).mean())
    first_pair = paired[(paired["first"] == V1) & (paired["second"] == V2)].iloc[0]
    assert np.isclose(first_pair.mean_difference_pathology_macro_ap, expected)
    report = (drive_dir / "comparison.md").read_text()
    assert "ref:base_good" in report and "base_bad" not in report
    assert (drive_dir / "comparison_curves.png").is_file()
    print("PASS: comparison cell (table, per-label AP, paired bootstrap); a Base reference re-scored from its "
          "predictions pairs exactly; a reference with other test records and a missing one are skipped")

    # 6. A changed recipe under the same run group is refused, never mixed.
    ns["ARCHITECTURE_SPECS"][V1]["recipe"]["base_lr"] *= 2
    ns["CONTINUE_ON_ERROR"] = False
    try:
        ns["run_architecture"](V1)
        raise AssertionError("a fingerprint change was accepted")
    except RuntimeError as error:
        assert "different configuration" in str(error)
    finally:
        ns["ARCHITECTURE_SPECS"][V1]["recipe"]["base_lr"] /= 2
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


def check_dataset_staging(ns, cell, folder):
    """Section 3 end to end: a small tar on a fake Drive, read via the API and via the mount."""
    import hashlib
    import io
    import re
    import requests
    import tarfile

    mount = folder / "mnt"
    drive_dir = mount / "MyDrive" / "CardioSentry" / "new_dataset"
    drive_dir.mkdir(parents=True)
    top = "clean_ecg_signal_images_clean12"
    buffer = io.BytesIO()
    rng = np.random.default_rng(1)
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        members = {f"{top}/{name}": b"x\n" for name in
                   ("manifest.csv", "record_manifest.csv", "final_counts.csv", "summary.json")}
        members[f"{top}/images/I0001/a.png"] = rng.integers(0, 256, 300_000, dtype=np.uint8).tobytes()
        members[f"{top}/images/I0006/b.png"] = rng.integers(0, 256, 70_001, dtype=np.uint8).tobytes()
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    data = buffer.getvalue()
    tar_path = drive_dir / "clean_ecg_signal_images_clean12.tar"
    tar_path.write_bytes(data)
    good_sha = hashlib.sha256(data).hexdigest()
    sidecar = drive_dir / (tar_path.name + ".sha256")
    sidecar.write_text(f"{good_sha}  {tar_path.name}\n")

    extract_root, scratch_tmp = folder / "extract", folder / "scratch_tmp"
    extract_root.mkdir()
    scratch_tmp.mkdir()
    ids = {("root", "CardioSentry"): "C", ("C", "new_dataset"): "N", ("N", tar_path.name): "T"}
    calls = {"media": 0, "list": 0, "meta": 0, "refresh": 0}
    drive_meta = {"id": "T", "name": tar_path.name, "size": str(len(data)), "sha256Checksum": good_sha}
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
            name, parent = re.fullmatch(r"name = '(.+)' and '(.+)' in parents and trashed = false",
                                        params["q"]).groups()
            found = ids.get((parent, name))
            return Reply(200, payload={"files": [{"id": found, "name": name}] if found else []})
        assert url.endswith("/files/T"), url
        if params.get("alt") != "media":
            calls["meta"] += 1
            return Reply(200, payload=dict(drive_meta))
        calls["media"] += 1
        roll = flaky.random()
        if roll < 0.08:
            return Reply(503)
        if roll < 0.12:
            return Reply(429)
        if roll < 0.16:
            raise requests.ConnectionError("dropped")
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        if roll < 0.20:
            return Reply(206, data[start:end])  # short body: must be retried
        return Reply(206, data[start:end + 1])

    class FakeCredentials:
        valid, token = False, None

        def refresh(self, request):
            calls["refresh"] += 1
            self.valid, self.token = True, f"token-{calls['refresh']}"

    fake_requests = SimpleNamespace(get=fake_get, RequestException=requests.RequestException,
                                    HTTPError=requests.HTTPError, ConnectionError=requests.ConnectionError)
    ns.update(
        IN_COLAB=False, DRIVE_MOUNT_POINT=mount, DATASET_DRIVE_DIR=drive_dir, DATASET_DRIVE_PATH=tar_path,
        DATASET_SIDECAR_PATH=sidecar, DATASET_TAR_NAME=tar_path.name, DATASET_DRIVE_FILE_ID="",
        REQUIRE_TAR_SHA256=True, DATASET_READ_MODE="api", DATASET_DOWNLOAD_WORKERS=4,
        DATASET_DOWNLOAD_CHUNK_BYTES=8192, EXTRACT_ROOT=extract_root, SCRATCH_TMP=scratch_tmp,
        SCRATCH_MARGIN_GIB=0.0, KIT_TOPDIR=top, requests=fake_requests,
    )
    kit = extract_root / top
    # Stand-ins for google-auth (on Colab, not necessarily in the local venv).
    fake_google = SimpleNamespace(auth=SimpleNamespace(
        default=lambda scopes=None: (FakeCredentials(), None),
        transport=SimpleNamespace(requests=SimpleNamespace(Request=lambda: None))))
    fake_modules = {"google": fake_google, "google.auth": fake_google.auth,
                    "google.auth.transport": fake_google.auth.transport,
                    "google.auth.transport.requests": fake_google.auth.transport.requests}
    with patch.dict("sys.modules", fake_modules), patch("time.sleep", lambda seconds: None):
        # 1. API mode: path resolved folder by folder, flaky ranges retried, archive verified.
        exec(cell, ns)
        assert (kit / "images" / "I0001" / "a.png").read_bytes() == members[f"{top}/images/I0001/a.png"]
        marker = json.loads((extract_root / ".clean12_extract_complete.json").read_text())
        assert marker["sha256"] == good_sha and marker["read_mode"] == "api"
        assert calls["list"] == 3 and calls["meta"] == 1 and calls["media"] > len(data) // 8192
        print(f"PASS: section 3 api mode streams Drive -> tar ({calls['media']} ranged GETs incl. "
              "injected 503/429/drops/short reads) and verifies the SHA-256")

        # 2. Re-running on the same runtime skips without touching Drive (no sign-in).
        before = dict(calls)
        exec(cell, ns)
        assert calls == before
        print("PASS: section 3 skips an already verified extraction without any Drive API call")

        # 3. Mount mode on a fresh runtime.
        shutil.rmtree(kit)
        (extract_root / ".clean12_extract_complete.json").unlink()
        ns["DATASET_READ_MODE"] = "mount"
        exec(cell, ns)
        assert calls == before and (kit / "images" / "I0006" / "b.png").stat().st_size == 70_001
        print("PASS: section 3 mount mode (positional reads through the mount) extracts and verifies")

        # 4. A sidecar mismatch deletes the extracted tree and raises.
        shutil.rmtree(kit)
        (extract_root / ".clean12_extract_complete.json").unlink()
        sidecar.write_text(f"{'0' * 64}  {tar_path.name}\n")
        try:
            exec(cell, ns)
            raise AssertionError("a SHA-256 mismatch was accepted")
        except RuntimeError as error:
            assert "SHA-256" in str(error) and not kit.exists()
        print("PASS: section 3 rejects a SHA-256 mismatch and deletes the partial tree")

        # 5. API mode refuses a damaged Drive copy before downloading anything.
        sidecar.write_text(f"{good_sha}  {tar_path.name}\n")
        drive_meta["sha256Checksum"] = "f" * 64
        ns["DATASET_READ_MODE"] = "api"
        media_before = calls["media"]
        try:
            exec(cell, ns)
            raise AssertionError("a damaged Drive copy was accepted")
        except RuntimeError as error:
            assert "damaged" in str(error) and calls["media"] == media_before
        print("PASS: section 3 api mode refuses a Drive copy whose SHA-256 disagrees, before downloading")


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ns, full_cells = load_notebook()
    with tempfile.TemporaryDirectory(prefix="cardiosentry-tiny-check-") as directory:
        folder = Path(directory)
        configure_namespace(ns, folder)
        check_dataset_staging(ns, full_cells["dataset"], folder)
        with cpu_stubs():
            check_shared_cells(ns, full_cells)
            check_native_size(ns)
            check_full_runs(ns, full_cells)
            check_benchmark_only(ns, full_cells, folder)
    print("All CPU checks passed. GPU speed/memory, torch.compile, real Drive I/O, and full-corpus "
          "training are not exercised here.")


if __name__ == "__main__":
    main()
