"""CPU checks of the generated notebook; never mounts disks or contacts S3.

Run: python3 validate_notebook.py
Uses definitions from the notebook itself and the packages in its install cell.
Native Base inference is real; training/resume checks use a small model so this
can run on a development machine without allocating Base's AdamW state.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from timm.optim import param_groups_layer_decay
from timm.utils import ModelEmaV3
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "CardioSentry_ConvNeXtV1_Base_Clean12_EC2.ipynb"
NUM_LABELS = 12
# Optional real clean12 manifest for the split check (skipped when absent).
REAL_MANIFEST = Path(os.environ.get(
    "CS_CHECK_MANIFEST",
    str(HERE.parents[1] / "dataset-files/HEEDB/extracted/ECG_500Hz_clean12/manifest.csv"),
))


def load_notebook():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    constants = {
        "LABELS", "PATHOLOGY_LABELS", "MODEL_NAME", "INPUT_SIZE", "MAX_EPOCHS",
        "EARLY_STOPPING_PATIENCE", "EARLY_STOPPING_MIN_EPOCHS", "WARMUP_EPOCHS",
        "BASE_LR", "MIN_LR_RATIO", "WEIGHT_DECAY", "LAYER_DECAY", "DROP_PATH_RATE",
        "HEAD_INIT_SCALE", "EFFECTIVE_BATCH_SIZE", "EMA_DECAY", "SEED",
        "TRAIN_BRIGHTNESS_JITTER", "TRAIN_CONTRAST_JITTER", "CONTRADICTION_LAMBDA",
        "GRAD_CLIP_NORM", "IMAGENET_MEAN", "IMAGENET_STD", "IMAGE_COLOR_MODE",
        "RESIZE_INTERPOLATION", "RESIZE_ANTIALIAS", "EVAL_METADATA_KEYS",
        "LOSS_COMPONENT_KEYS", "PREDICT_METADATA_KEYS", "COMBINATION_LOW_SUPPORT_THRESHOLD",
        "GPU_MEAN", "GPU_STD", "GPU_LUMINANCE", "TORCH_COMPILE_MODE",
        "DETERMINISTIC", "USE_FUSED_ADAMW", "PREFETCH_FACTOR", "DATALOADER_START_METHOD",
        "WORKER_BENCHMARK_BATCHES", "NORMAL_LABEL", "LABEL_NAMES", "SUBSETS", "SPLIT_RATIOS",
        "SPLIT_SEED", "SPLIT_VERSION", "SPLIT_RECORD_TOLERANCE", "SPLIT_LABEL_TOLERANCE",
    }
    namespace = {**globals(), "tqdm": functools.partial(tqdm, disable=True), "DEVICE": torch.device("cpu")}
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        source = "".join(cell["source"])
        if "%pip" in source:
            assert '"timm==1.0.28"' in source
            continue  # Only this installation cell requires IPython syntax.
        compile(source, f"{NOTEBOOK}:cell-{index}", "exec")
        sources.append(source)
        for node in ast.parse(source).body:
            is_constant = isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets
            )
            if is_constant or isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    namespace.update(DEVICE=torch.device("cpu"), USE_AMP=False, AMP_DTYPE=torch.bfloat16, COMPILE_SELECTED=False)
    namespace["image_transform"] = transforms.Compose([
        transforms.Resize((32, 48), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.PILToTensor(),
    ])
    assert len(namespace["LABELS"]) == NUM_LABELS and namespace["LABELS"][0] == namespace["NORMAL_LABEL"]
    print(f"PASS: notebook structure and {len(sources)} Python code cells compile")
    return namespace, sources


def check_base_model(ns):
    assert ns["MODEL_NAME"] in timm.list_models("convnext_base*", pretrained=True)
    torch.manual_seed(42)
    model = ns["build_model"](pretrained=False)
    count = sum(p.numel() for p in model.parameters())
    assert 85_000_000 < count < 90_000_000
    assert [len(stage.blocks) for stage in model.stages] == [3, 3, 27, 3]
    assert not any("GlobalResponseNorm" in type(module).__name__ for module in model.modules())
    assert model.get_classifier().out_features == NUM_LABELS
    assert model.get_classifier().weight.std().item() < 0.0001
    optimizer = ns["build_optimizer"](model)  # also checks complete, unique parameter coverage
    assert math.isclose(max(group["lr"] for group in optimizer.param_groups), ns["BASE_LR"])
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, *ns["INPUT_SIZE"]))
    assert output.shape == (1, NUM_LABELS) and torch.isfinite(output).all()
    model.train()
    logits = model(torch.randn(1, 3, 64, 96))
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(model.get_classifier().weight.grad).all()
    assert model.get_classifier().weight.grad.abs().sum() > 0
    print(f"PASS: real V1 Base ({count:,} parameters), native 768×1024 inference, backward, optimizer groups")


def check_dataset(ns, folder):
    pixels = np.random.default_rng(10).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    image_path = folder / "sheet.png"
    Image.fromarray(pixels).save(image_path)
    rows = []
    for index in range(4):
        rows.append({"local_path": str(image_path), "record_id": str(index), "source_dataset": "site",
                     "split": "validation", "subset": "validation", "variant_id": "clean",
                     "primary_metric": 1, **{label: int(i == 0) for i, label in enumerate(ns["LABELS"])}})
    frame = pd.DataFrame(rows)
    dataset = ns["ECGImageDataset"](frame, return_metadata=False, training=True)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True,
                        multiprocessing_context="fork", shuffle=False)
    dataset.set_epoch(1)
    epoch_one = torch.cat([ns["prepare_image_batch"](batch[0], batch[2]) for batch in loader])
    dataset.set_epoch(2)
    epoch_two = torch.cat([ns["prepare_image_batch"](batch[0], batch[2]) for batch in loader])
    expected = ns["prepare_image_batch"](
        torch.stack([dataset[i][0] for i in range(len(dataset))]),
        torch.stack([dataset[i][2] for i in range(len(dataset))]),
    )
    assert torch.equal(epoch_two, expected) and not torch.equal(epoch_one, epoch_two)
    replay = ns["ECGImageDataset"](frame, return_metadata=False, training=True)
    replay.set_epoch(2)
    replayed = ns["prepare_image_batch"](
        torch.stack([replay[i][0] for i in range(len(replay))]),
        torch.stack([replay[i][2] for i in range(len(replay))]),
    )
    assert torch.equal(epoch_two, replayed)
    validation = ns["ECGImageDataset"](frame, return_metadata=True)
    validation.set_epoch(100)
    assert torch.equal(validation[0][0], validation[0][0])
    assert validation[0][2]["source"] == "site"
    raw = validation[0][0].unsqueeze(0)
    assert raw.dtype == torch.uint8
    prepared = ns["prepare_image_batch"](raw)
    expected_normalization = TF.normalize(raw.float() / 255, ns["IMAGENET_MEAN"], ns["IMAGENET_STD"])
    assert torch.allclose(prepared, expected_normalization, atol=1e-6)
    factors = dataset[0][2]
    reference = TF.adjust_contrast(TF.adjust_brightness(raw.float() / 255, float(factors[0])), float(factors[1]))
    reference = TF.normalize(reference, ns["IMAGENET_MEAN"], ns["IMAGENET_STD"])
    assert torch.allclose(ns["prepare_image_batch"](raw, factors.unsqueeze(0)), reference, atol=1e-6)
    try:
        ns["prepare_image_batch"](prepared)
        raise AssertionError("Double normalization should be rejected")
    except TypeError:
        pass
    _, worker_rows = ns["tune_decoder_workers"](dataset, 2, [0, 2])
    assert (worker_rows.decoded_images_per_second > 0).all()
    del loader
    print("PASS: uint8 pipeline, normalization/augmentation parity, worker tuning, persistent-worker replay")


def check_compilation_and_selection(ns, folder):
    model = small_model().eval()
    inputs = torch.randn(2, 3, 32, 48)
    keys_before = list(model.state_dict())
    params_before = [id(parameter) for parameter in model.parameters()]
    output_before = model(inputs)
    ns["configure_model_compilation"](model, True, backend="eager")
    assert torch.equal(model(inputs), output_before)
    assert list(model.state_dict()) == keys_before
    assert [id(parameter) for parameter in model.parameters()] == params_before
    checkpoint = folder / "compiled_state.pt"
    torch.save(model.state_dict(), checkpoint)
    fresh = small_model().eval()
    fresh.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(fresh(inputs), output_before)
    ns["configure_model_compilation"](model, False)
    eager = {"mode": "eager", "status": "fits", "estimated_sec_per_epoch": 100}
    faster = {"mode": "compiled", "status": "fits", "estimated_sec_per_epoch": 90}
    slower = {"mode": "compiled", "status": "fits", "estimated_sec_per_epoch": 105}
    over_budget = {**faster, "status": "over_budget"}
    failed = {"mode": "compiled", "status": "failed"}
    assert ns["select_runtime_benchmark"]([eager, faster]) is faster
    for alternative in (slower, over_budget, failed):
        assert ns["select_runtime_benchmark"]([eager, alternative]) is eager
    print("PASS: compiled forward, ordinary checkpoint keys, parameter identities, runtime selection/fallback")


def check_sizing_and_benchmark_logic(ns):
    # Execute the actual probe and benchmark functions on a small CPU model.
    # CUDA memory readings/synchronization are simulated, not GPU measurements.
    overrides = {
        "build_model": lambda pretrained=False: small_model(), "model": small_model(),
        "INPUT_SIZE": (32, 48), "EFFECTIVE_BATCH_SIZE": 8, "POS_WEIGHT": torch.ones(NUM_LABELS),
        "BATCH_PROBE_TIMED_UPDATES": 2, "EVAL_PROBE_TIMED_BATCHES": 2,
        "VRAM_SAFETY_FRACTION": 0.9, "ACCUMULATION_STEPS": 2,
        "MICRO_BATCH_SIZE": 2, "EVAL_BATCH_SIZE": 4, "NUM_WORKERS": 0,
        "train_benchmark_batches": 4, "usable_train_batches": 8,
        "BENCHMARK_WARMUP_BATCHES": 3, "BENCHMARK_VAL_BATCHES": 2,
        "TRAIN_GENERATOR": torch.Generator().manual_seed(42),
    }
    absent = object()
    keys = set(overrides) | {"prepare_image_batch", "base_state", "train_generator_state",
                             "train_dataset", "val_dataset", "val_loader", "train_loader"}
    saved = {key: ns.get(key, absent) for key in keys}
    original_prepare = ns["prepare_image_batch"]
    def with_simulated_oom(pixels, factors=None):
        if len(pixels) == 4:
            raise torch.cuda.OutOfMemoryError("simulated probe OOM")
        return original_prepare(pixels, factors)
    try:
        ns.update(overrides)
        ns["prepare_image_batch"] = with_simulated_oom
        with contextlib.ExitStack() as stack:
            for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
                stack.enter_context(patch("torch.cuda." + name, lambda: None))
            class Properties:
                total_memory = 128 * 2**20
            stack.enter_context(patch("torch.cuda.get_device_properties", lambda index: Properties()))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", lambda: 64 * 2**20))
            stack.enter_context(patch("torch.cuda.max_memory_reserved", lambda: 64 * 2**20))
            selected, rows = ns["probe_micro_batch"]([4, 2, 1])
            assert selected in (1, 2) and rows.iloc[0].status == "OOM"
            assert selected == rows[rows.status.eq("fits")].sort_values("images_per_second").iloc[-1].micro_batch
            selected_eval, eval_rows = ns["probe_evaluation_batch"]([8, 4, 2], selected)
            assert selected_eval in (2, 8) and "OOM" in eval_rows.status.tolist()
            ns["prepare_image_batch"] = original_prepare
            images = torch.randint(0, 256, (24, 3, 32, 48), dtype=torch.uint8)
            labels = torch.zeros(24, NUM_LABELS)
            ns["train_dataset"] = TensorDataset(images, labels, torch.ones(24, 2))
            ns["val_dataset"] = ns["train_dataset"]
            ns["train_loader"] = DataLoader(ns["train_dataset"], batch_size=2)
            ns["val_loader"] = DataLoader(ns["val_dataset"], batch_size=4)
            ns["base_state"] = ns["model"].state_dict()
            ns["train_generator_state"] = ns["TRAIN_GENERATOR"].get_state().clone()
            result = ns["benchmark_execution_mode"](False)
            assert result["status"] == "fits" and result["train_images_per_second"] > 0
            assert result["eval_images_per_second"] > 0
            assert torch.equal(ns["TRAIN_GENERATOR"].get_state(), ns["train_generator_state"])
            assert ns["train_loader"]._iterator is None
    finally:
        for key, value in saved.items():
            if value is absent:
                ns.pop(key, None)
            else:
                ns[key] = value
    print("PASS: probe OOM fallback, fastest batch selection, benchmark execution and loader/RNG isolation (CPU simulation)")


def check_selection(ns):
    truth = np.zeros((4, NUM_LABELS), dtype=np.int8)
    truth[0, 0] = truth[2, 0] = 1
    truth[1, 1:] = truth[3, 1:] = 1
    perfect = truth * 0.8 + 0.1
    good, bad = object(), object()
    original = ns["predict_loader"]
    def fake_predict(model, loader, criterion):
        return {"y_true": truth, "y_probability": perfect if model is good else 1 - perfect,
                "metadata": {"primary_metric": [1] * 4}}
    ns["predict_loader"] = fake_predict
    try:
        class EmaCandidate:
            module = good
        selected, scores = ns["select_validation_weights"](bad, EmaCandidate(), None, None)
        assert selected["weight_source"] == "ema" and scores["ema"] == 1.0
        selected, _ = ns["select_validation_weights"](good, EmaCandidate(), None, None)
        assert selected["weight_source"] == "raw"  # ties favor raw
    finally:
        ns["predict_loader"] = original
    thresholds = ns["tune_thresholds_on_validation"](truth, perfect)
    metrics = ns["evaluate_predictions"](truth, perfect, thresholds)
    assert metrics["macro_f1"] == metrics["pathology_only_macro_average_precision"] == 1.0
    unsupported = ns["binary_metrics"](np.zeros(4), np.ones(4) * 0.1, 0.5)
    assert math.isnan(unsupported["auroc"]) and math.isnan(unsupported["average_precision"])
    print("PASS: validation-only raw/EMA selection, F1 thresholds, missing-support ranking metrics")


def check_patient_split(ns):
    split = ns["stratified_patient_split"]
    labels, ratios, seed = ns["LABELS"], ns["SPLIT_RATIOS"], ns["SPLIT_SEED"]
    # Synthetic: multi-record patients, a rare label, and a patient with 30 records.
    rng = np.random.default_rng(7)
    rows = []
    for patient in range(3000):
        for _ in range(30 if patient == 0 else int(rng.integers(1, 4))):
            hits = rng.random(len(labels)) < [0.3] + [0.08] * (len(labels) - 1)
            hits[labels.index("3AVB")] = rng.random() < 0.01
            if hits[0] or not hits.any():
                hits[:] = False
                hits[0] = True
            rows.append({"patient_key": f"p{patient:05d}", "site": "I0001",
                         **{label: int(hit) for label, hit in zip(labels, hits)}})
    frame = pd.DataFrame(rows)
    first = split(frame, labels, ratios, seed)
    shuffled = split(frame.sample(frac=1, random_state=3), labels, ratios, seed)
    assert first.equals(shuffled), "split depends on row order"
    assert ns["split_sha256"](first) == ns["split_sha256"](shuffled)
    assert not first.equals(split(frame, labels, ratios, seed + 1)), "seed has no effect"
    assigned = frame.assign(split=frame.patient_key.map(first))
    assert (assigned.groupby("patient_key").split.nunique() == 1).all()
    shares = assigned.split.value_counts(normalize=True)
    for name, ratio in ratios.items():
        assert abs(shares[name] - ratio) < 0.03, (name, shares[name])
    print("PASS: patient split is patient-disjoint, row-order independent, seed-dependent (synthetic)")
    if not REAL_MANIFEST.is_file():
        print(f"SKIP: real clean12 manifest not found at {REAL_MANIFEST}")
        return
    real = pd.read_csv(REAL_MANIFEST, dtype={"patient_key": str}, keep_default_na=False, low_memory=False)
    started = time.time()
    result = split(real, labels, ratios, seed)
    elapsed = time.time() - started
    real["split"] = real.patient_key.map(result)
    assert real.split.notna().all() and (real.groupby("patient_key").split.nunique() == 1).all()
    rows = ns["split_support"](real, result)
    worst_record = max(abs(rows[n]["record_share"] - ratios[n]) for n in ratios)
    worst_label = max(abs(rows[n]["label_share"][label] - ratios[n]) for n in ratios for label in labels)
    assert worst_record <= ns["SPLIT_RECORD_TOLERANCE"] and worst_label <= ns["SPLIT_LABEL_TOLERANCE"]
    print(f"PASS: real clean12 split in {elapsed:.1f}s: "
          + ", ".join(f"{n} {rows[n]['records']:,} rec / {rows[n]['patients']:,} pts" for n in ratios)
          + f"; worst record-share error {worst_record:.4f}, worst label-share error {worst_label:.4f}; "
          + f"3AVB {[rows[n]['positives']['3AVB'] for n in ratios]}, 2AVB {[rows[n]['positives']['2AVB'] for n in ratios]}; "
          + f"sha256 {ns['split_sha256'](result)[:16]}")


def small_model():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.GELU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, NUM_LABELS))


def reset_training_objects(ns):
    ns["model"] = small_model()
    ns["model_ema"] = ns["build_model_ema"](ns["model"])
    ns["optimizer"] = ns["build_optimizer"](ns["model"])
    ns["scheduler"] = LambdaLR(ns["optimizer"], ns["lr_multiplier"])
    ns["scaler"] = ns["make_grad_scaler"]()


def check_training_resume(ns, sources, folder):
    torch.manual_seed(42)
    ns.update(MAX_EPOCHS=2, EARLY_STOPPING_MIN_EPOCHS=2, EARLY_STOPPING_PATIENCE=10,
              WARMUP_EPOCHS=0, ACCUMULATION_STEPS=2, usable_train_batches=4,
              OPTIMIZER_STEPS_PER_EPOCH=2, PROGRESS_UPDATE_EVERY_N_BATCHES=10,
              RUN_DIR=folder, RUN_LOG_PATH=folder / "run.log", CONFIG_FINGERPRINT="smoke",
              CONFIG={"fingerprint": "smoke"}, start_epoch=1, history=[], bad_epochs=0,
              best_pathology_only_macro_ap=-float("inf"), INTRA_EPOCH_SYNC_MINUTES=15,
              S3_UPLOAD_ENABLED=False, S3_RUN_PREFIX="unused", S3_UPLOAD_EVERY_N_EPOCHS=1)
    ns["criterion"] = ns["CardioSentryLoss"](torch.ones(NUM_LABELS), 0)
    ns["TRAIN_GENERATOR"] = torch.Generator().manual_seed(42)
    images = torch.randint(0, 256, (8, 3, 32, 48), dtype=torch.uint8)
    labels = torch.zeros(8, NUM_LABELS)
    labels[::2, 0] = 1
    labels[1::2, 1:] = 1
    ns["train_loader"] = DataLoader(TensorDataset(images, labels, torch.ones(8, 2)), batch_size=2)
    metadata = [{"row_index": i, "record_id": str(i), "source": "site", "split": "validation",
                 "subset": "validation", "variant_id": "clean", "primary_metric": 1} for i in range(8)]
    ns["val_loader"] = DataLoader(list(zip(images, labels, metadata)), batch_size=2)
    class EpochDataset:
        def set_epoch(self, epoch):
            self.last_epoch = epoch
    ns["train_dataset"] = EpochDataset()
    for key, filename in {"best_path": "best.pt", "last_path": "last.pt", "intra_path": "last_intra.pt",
                          "history_path": "training_history.csv", "config_snapshot_path": "config_snapshot.json",
                          "completion_path": "training_complete.json"}.items():
        ns[key] = folder / filename
    ns.update(log_event=lambda *args, **kwargs: None, join_s3_uploads=lambda: None,
              upload_file_to_s3=lambda *args, **kwargs: None)
    def interrupt_after_epoch(paths, **kwargs):
        raise RuntimeError("simulated interruption after epoch boundary")
    ns["upload_run_files"] = interrupt_after_epoch
    reset_training_objects(ns)
    training_source = next(source for source in sources if source.startswith("# Long-running training cell"))
    resume_source = next(source for source in sources if "# Resume from the newest epoch-boundary checkpoint." in source)
    resume_source = resume_source[resume_source.index("checkpoint = None\nresume_source = None"):
                                  resume_source.index("# Only a fully validated compatible run")]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.reset_peak_memory_stats", lambda: None))
        stack.enter_context(patch("torch.cuda.max_memory_allocated", lambda: 0))
        try:
            exec(training_source, ns)
            raise AssertionError("Simulated interruption was not reached")
        except RuntimeError as error:
            assert "simulated interruption" in str(error)
        first = torch.load(ns["last_path"], map_location="cpu", weights_only=False)
        assert first["epoch"] == 1 and first["epoch_complete"] and first["ema_num_updates"] == 2
        assert not ns["completion_path"].exists()
        # Distract the RNG and initialize fresh objects, as on a replacement machine.
        random.random()
        np.random.rand()
        torch.randn(5)
        reset_training_objects(ns)
        exec(resume_source, ns)
        assert ns["start_epoch"] == 2 and ns["model_ema"].num_updates == 2
        for key, tensor in ns["model"].state_dict().items():
            assert torch.equal(tensor, first["model"][key])
        assert torch.equal(torch.get_rng_state(), first["torch_random_state"])
        ns["upload_run_files"] = lambda *args, **kwargs: None
        exec(training_source, ns)
    final = torch.load(ns["last_path"], map_location="cpu", weights_only=False)
    assert final["epoch"] == 2 and final["ema_num_updates"] == 4
    assert len(final["history"]) == 2 and ns["scheduler"].last_epoch == 4
    best = torch.load(ns["best_path"], map_location="cpu", weights_only=False)
    for weight_source in ("raw", "ema"):
        selected = best["model_ema" if weight_source == "ema" else "model"]
        ns["model"].load_state_dict(selected)
        export_path = folder / f"inference_{weight_source}.pt"
        ns["export_inference_checkpoint"](ns["model"], weight_source, best["epoch"], [0.6] * NUM_LABELS, export_path)
        exported = torch.load(export_path, map_location="cpu", weights_only=True)
        assert exported["thresholds"] == [0.6] * NUM_LABELS and "optimizer" not in exported
        assert exported["selected_weight_source"] == weight_source
        for key, tensor in selected.items():
            assert torch.equal(exported["model"][key], tensor)
    print("PASS: actual epoch loop, accumulation, raw + EMA checkpoint resume, RNG restore, selected inference export")


def check_skipped_updates(ns):
    class SkippingScaler:
        value, calls = 8.0, 0
        def get_scale(self): return self.value
        def is_enabled(self): return True
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer):
            if self.calls: optimizer.step()
            self.calls += 1
        def update(self):
            if self.calls == 1: self.value /= 2
    reset_training_objects(ns)
    values = ns["train_one_epoch"](ns["model"], ns["train_loader"], ns["criterion"],
                                    ns["optimizer"], SkippingScaler(), ns["scheduler"], ns["model_ema"])
    assert values["optimizer_updates"] == 1 and values["amp_skipped_optimizer_steps"] == 1
    assert ns["model_ema"].num_updates == ns["scheduler"].last_epoch == 1
    print("PASS: skipped FP16 optimizer updates do not advance the scheduler or EMA")


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ns, sources = load_notebook()
    check_patient_split(ns)
    check_base_model(ns)
    with tempfile.TemporaryDirectory(prefix="cardiosentry-v1-check-") as directory:
        folder = Path(directory)
        check_dataset(ns, folder)
        check_compilation_and_selection(ns, folder)
        check_sizing_and_benchmark_logic(ns)
        check_selection(ns)
        check_training_resume(ns, sources, folder)
        check_skipped_updates(ns)
    print("All CPU checks passed. GPU preflight, S3 staging, and full-corpus training are not exercised here.")


if __name__ == "__main__":
    main()
