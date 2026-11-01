"""CPU checks of the generated Colab notebook; never mounts Drive or contacts S3.

Run: python3 validate_notebook.py
Uses definitions from the notebook itself and the packages in its install cell.
Native V1 Base inference is real for every fusion variant; training/resume checks
use a tiny ConvNeXt with a fusion neck so this can run on a development machine
without allocating Base's AdamW state. S3 is a
fake in-memory client and "Drive" is a temporary folder.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import gc
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from botocore.exceptions import ClientError
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from timm.layers import trunc_normal_
from timm.optim import param_groups_layer_decay
from timm.utils import ModelEmaV3
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "CardioSentry_ConvNeXtV1_Base_MSFF_Clean12_Colab.ipynb"
FUSION_FILE = HERE / "fusion_necks.py"
NUM_LABELS = 12
# Optional real clean12 manifest for the split check (skipped when absent).
REAL_MANIFEST = Path(os.environ.get(
    "CS_CHECK_MANIFEST",
    str(HERE.parents[2] / "dataset-files/HEEDB/extracted/ECG_500Hz_clean12/manifest.csv"),
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
        "EXPECTED_SPLIT_SHA256", "COLLAPSE_GUARD_ENABLED",
        "FUSION", "FUSION_HEAD_OVERRIDE", "FUSION_CHANNELS", "BIFPN_REPEATS",
        "FUSION_CHOICES", "FUSION_HEAD_CHOICES", "FUSION_STAGE_INDICES",
        "COLLAPSE_LOSS_RATIO", "COLLAPSE_CHECK_EVERY_N_BATCHES", "COLLAPSE_PATIENCE_WINDOWS",
        "COLLAPSE_MIN_VAL_MACRO_AUROC", "RARE_SAMPLING", "RARE_SAMPLING_THRESHOLD",
        "BOOTSTRAP_ROUNDS", "BOOTSTRAP_CONFIDENCE", "KIT_TOPDIR", "SITES",
        # Drive copy bookkeeping, executed from the notebook's own source.
        "_sync_threads", "_sync_threads_lock", "_copy_key_threads", "_copy_key_lock",
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
    # The fusion module is fusion_necks.py copied verbatim into its own cell.
    fusion_cells = [s for s in sources if s == FUSION_FILE.read_text()]
    assert len(fusion_cells) == 1, "the notebook's fusion cell differs from fusion_necks.py; rerun make_notebook.py"
    assert namespace["FUSION"] == "rfpn" and namespace["RARE_SAMPLING"] is False
    neck, head = namespace["resolve_fusion"](namespace["FUSION"], namespace["FUSION_HEAD_OVERRIDE"])
    namespace.update(FUSION_NECK=neck, FUSION_HEAD=head, FUSION_SPEC=namespace["fusion_description"](
        neck, head, namespace["FUSION_CHANNELS"], namespace["BIFPN_REPEATS"]))
    namespace["image_transform"] = transforms.Compose([
        transforms.Resize((32, 48), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.PILToTensor(),
    ])
    assert len(namespace["LABELS"]) == NUM_LABELS and namespace["LABELS"][0] == namespace["NORMAL_LABEL"]
    print(f"PASS: notebook structure and {len(sources)} Python code cells compile")
    return namespace, sources


def use_fusion(ns, fusion, head=""):
    neck, head_mode = ns["resolve_fusion"](fusion, head)
    ns.update(FUSION=fusion, FUSION_HEAD_OVERRIDE=head, FUSION_NECK=neck, FUSION_HEAD=head_mode,
              FUSION_SPEC=ns["fusion_description"](neck, head_mode, ns["FUSION_CHANNELS"], ns["BIFPN_REPEATS"]))
    return neck, head_mode


def lr_profile(model, optimizer, prefix=""):
    names = {id(p): name for name, p in model.named_parameters()}
    return {names[id(p)][len(prefix):]: (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups for p in group["params"]
            if names[id(p)].startswith(prefix)}


def check_base_model(ns):
    assert ns["MODEL_NAME"] in timm.list_models("convnext_base*", pretrained=True)
    assert ns["MODEL_NAME"] == "convnext_base.fb_in22k_ft_in1k_384"
    assert (ns["BASE_LR"], ns["WARMUP_EPOCHS"], ns["WEIGHT_DECAY"], ns["LAYER_DECAY"]) == (2e-4, 2, 0.02, 0.925)
    saved = {key: ns[key] for key in ("FUSION", "FUSION_HEAD_OVERRIDE", "FUSION_NECK", "FUSION_HEAD", "FUSION_SPEC")}
    # Variant checks run at half resolution (same 2x alignment at every level);
    # one native 768x1024 forward follows when the machine has the RAM for it.
    half = torch.randn(1, 3, ns["INPUT_SIZE"][0] // 2, ns["INPUT_SIZE"][1] // 2)
    try:
        # Plain V1 Base: stock timm keys, outputs, and LR profile.
        use_fusion(ns, "none")
        torch.manual_seed(42)
        plain = ns["build_model"](pretrained=False).eval()
        count = sum(p.numel() for p in plain.parameters())
        assert 86_000_000 < count < 90_000_000
        assert [len(stage.blocks) for stage in plain.stages] == [3, 3, 27, 3]
        assert all(block.gamma is not None for stage in plain.stages for block in stage.blocks)  # LayerScale
        assert plain.get_classifier().weight.std().item() < 0.0001
        stock = timm.create_model(ns["MODEL_NAME"], pretrained=False, num_classes=NUM_LABELS)
        assert list(plain.state_dict()) == list(stock.state_dict())
        del stock
        plain_profile = lr_profile(plain, ns["build_optimizer"](plain))
        with torch.inference_mode():
            plain_half = plain(half)
        assert plain_half.shape == (1, NUM_LABELS) and torch.isfinite(plain_half).all()
        plain_state = {k: v.clone() for k, v in plain.state_dict().items()}
        del plain
        gc.collect()
        expected_new = {"rfpn": (2.0e6, 2.5e6), "fpn": (2.0e6, 2.5e6),
                        "pan": (4.3e6, 4.9e6), "bifpn": (0.9e6, 1.2e6),
                        "bifpn+residual": (0.9e6, 1.2e6)}
        summary = []
        for variant in ("rfpn", "fpn", "pan", "bifpn", "bifpn+residual"):
            fusion, _, head = variant.partition("+")
            neck, head_mode = use_fusion(ns, fusion, head)
            torch.manual_seed(42)
            model = ns["build_model"](pretrained=False)
            assert isinstance(model, ns["ConvNeXtMultiScaleFusion"])
            new = sum(p.numel() for m in model.new_modules() for p in m.parameters())
            low, high = expected_new[variant]
            assert low < new < high, (variant, new)
            # The backbone carries the plain model's weights under "backbone.".
            backbone_keys = [k[len("backbone."):] for k in model.state_dict() if k.startswith("backbone.")]
            expected_keys = [k for k in plain_state if head_mode == "residual" or not k.startswith("head.")]
            assert backbone_keys == expected_keys, variant
            model.backbone.load_state_dict(
                {k: v for k, v in plain_state.items() if k in expected_keys}, strict=True)
            # Every parameter is grouped once; backbone LRs equal the plain model's.
            optimizer = ns["build_optimizer"](model)
            profile = lr_profile(model, optimizer, "backbone.")
            assert set(profile) == set(expected_keys) & set(plain_profile)
            for key, value in profile.items():
                assert math.isclose(value[0], plain_profile[key][0], rel_tol=1e-9) and value[1] == plain_profile[key][1], (variant, key)
            fusion_groups = [g for g in optimizer.param_groups if g.get("fusion")]
            assert len(fusion_groups) == 2 and all(math.isclose(g["lr"], ns["BASE_LR"]) for g in fusion_groups)
            assert {g["weight_decay"] for g in fusion_groups} == {ns["WEIGHT_DECAY"], 0.0}
            assert math.isclose(max(g["lr"] for g in optimizer.param_groups), ns["BASE_LR"])
            assert model.get_classifier().out_features == NUM_LABELS
            model.eval()
            with torch.inference_mode():
                output = model(half)
            assert output.shape == (1, NUM_LABELS) and torch.isfinite(output).all()
            if head_mode == "residual":
                # Zero-initialised branch: step 0 is exactly the plain model.
                assert torch.equal(output, plain_half), variant
            else:
                assert model.fuse_fc.weight.std().item() < 0.0001 and not hasattr(model.backbone.head, "fc")
            model.train()
            logits = model(torch.randn(2, 3, 128, 192))
            F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), variant
            assert model.fuse_fc.weight.grad.abs().sum() > 0
            if head_mode == "replace":
                assert next(model.neck.parameters()).grad.abs().sum() > 0
            summary.append(f"{variant} {new / 1e6:.2f}M")
            del model, optimizer
            gc.collect()
        # Invalid combinations fail before anything is built.
        for fusion, head in (("bogus", ""), ("none", "residual"), ("rfpn", "replace"), ("fpn", "stacked")):
            try:
                ns["resolve_fusion"](fusion, head)
                raise AssertionError(f"accepted {fusion}/{head}")
            except ValueError:
                pass
        assert ns["resolve_fusion"]("fpn", "residual") == ("fpn", "residual")
        # One native-size forward through the widest neck (PAN), alone in memory.
        native_note = "native 768×1024 PAN forward SKIPPED (under 4 GiB RAM free; set CS_CHECK_NATIVE=1 to force)"
        available = psutil.virtual_memory().available
        if os.environ.get("CS_CHECK_NATIVE", "") == "1" or available > 4 * 2**30:
            use_fusion(ns, "pan")
            model = ns["build_model"](pretrained=False).eval()
            with torch.inference_mode():
                output = model(torch.randn(1, 3, *ns["INPUT_SIZE"]))
            assert output.shape == (1, NUM_LABELS) and torch.isfinite(output).all()
            del model
            gc.collect()
            native_note = "native 768×1024 PAN forward OK"
    finally:
        ns.update(saved)
    print(f"PASS: real V1 Base ({count:,} parameters, stock keys) and every fusion variant "
          f"({', '.join(summary)} new) at 384×512: residual = plain at step 0, backbone LR "
          "profile identical to plain, full-LR fusion groups, backward reaches every parameter; "
          + native_note)


def check_bf16_fusion(ns):
    # BF16 autocast (the A100 path): the fusion maps stay BF16, the pooled
    # features and the classifier input are FP32, and outputs track FP32.
    torch.manual_seed(3)
    backbone = timm.create_model("convnext_atto", pretrained=False, num_classes=NUM_LABELS)
    results = []
    for neck in ("fpn", "pan", "bifpn"):
        model = ns["ConvNeXtMultiScaleFusion"](backbone if neck == "fpn" else
                                               timm.create_model("convnext_atto", pretrained=False, num_classes=NUM_LABELS),
                                               neck, "replace", NUM_LABELS, channels=64, head_init_scale=1.0).eval()
        x = torch.randn(2, 3, 128, 192)
        with torch.inference_mode():
            reference = model(x)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                mixed = model(x)
        assert torch.isfinite(mixed).all()
        results.append(float((mixed.float() - reference).abs().max()))
        assert results[-1] < 0.25 * float(reference.abs().max()) + 0.05, (neck, results[-1])
    # Non-power-of-two inputs still align every level.
    model = ns["ConvNeXtMultiScaleFusion"](timm.create_model("convnext_atto", pretrained=False, num_classes=NUM_LABELS),
                                           "bifpn", "residual", NUM_LABELS, channels=64)
    assert model(torch.randn(1, 3, 100, 140)).shape == (1, NUM_LABELS)
    pan = ns["ConvNeXtMultiScaleFusion"](timm.create_model("convnext_atto", pretrained=False, num_classes=NUM_LABELS),
                                         "pan", "replace", NUM_LABELS, channels=64)
    assert pan(torch.randn(1, 3, 100, 140)).shape == (1, NUM_LABELS)
    print(f"PASS: fusion necks run under BF16 autocast (max |Δlogit| {max(results):.3f}) and on odd input sizes")


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


def small_fusion_model(ns):
    # A tiny ConvNeXt with the notebook's BiFPN neck, so the epoch loop, EMA,
    # optimizer groups, checkpoints, resume and export all run through the wrapper.
    torch.manual_seed(5)
    backbone = timm.create_model("convnext_atto", pretrained=False, num_classes=NUM_LABELS)
    return ns["ConvNeXtMultiScaleFusion"](backbone, "bifpn", "replace", NUM_LABELS,
                                          channels=32, head_init_scale=1.0)


def reset_training_objects(ns):
    ns["model"] = small_fusion_model(ns)
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
              DRIVE_SYNC_ENABLED=False, DRIVE_SYNC_EVERY_N_EPOCHS=1,
              # Guards stay enabled but cannot fire on this tiny smoke model.
              COLLAPSE_GUARD_ENABLED=True, COLLAPSE_MIN_VAL_MACRO_AUROC=0.0,
              CONSTANT_PREDICTOR_LOSS=1e9)
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
    ns.update(log_event=lambda *args, **kwargs: None, join_drive_copies=lambda: None,
              copy_to_drive=lambda *args, **kwargs: None)
    def interrupt_after_epoch(paths, **kwargs):
        raise RuntimeError("simulated interruption after epoch boundary")
    ns["copy_run_files"] = interrupt_after_epoch
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
        ns["copy_run_files"] = lambda *args, **kwargs: None
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


def check_collapse_guard(ns, sources, folder):
    # 1. The constant-predictor reference equals a brute-force optimum.
    section = next(source for source in sources if "CONSTANT_PREDICTOR_LOSS = float(" in source)
    snippet = section[section.index("_prevalence = "):section.index('print(f"constant-predictor')]
    rng = np.random.default_rng(3)
    targets = (rng.random((400, NUM_LABELS)) < rng.uniform(0.02, 0.5, NUM_LABELS)).astype(np.int64)
    train_positive = pd.Series(targets.sum(axis=0))
    weights = np.sqrt((len(targets) - train_positive) / train_positive)
    scope = {"np": np, "effective_positive": train_positive.astype(float),
             "effective_total": float(len(targets)), "pos_weight_series": weights}
    exec(snippet, scope)
    logits = torch.linspace(-12, 12, 24001, dtype=torch.float64).unsqueeze(1)
    brute = []
    for label in range(NUM_LABELS):
        y = torch.tensor(targets[:, label], dtype=torch.float64).unsqueeze(0).expand(len(logits), -1)
        losses = F.binary_cross_entropy_with_logits(
            logits.expand(-1, len(targets)), y, reduction="none",
            pos_weight=torch.tensor(float(weights[label]), dtype=torch.float64),
        ).mean(dim=1)
        brute.append(float(losses.min()))
    assert abs(scope["CONSTANT_PREDICTOR_LOSS"] - float(np.mean(brute))) < 1e-6

    def use_folder(name):
        target = folder / name
        target.mkdir()
        ns.update(RUN_DIR=target, RUN_LOG_PATH=target / "run.log", start_epoch=1, history=[],
                  bad_epochs=0, best_pathology_only_macro_ap=-float("inf"))
        for key, filename in {"best_path": "best.pt", "last_path": "last.pt", "intra_path": "last_intra.pt",
                              "history_path": "training_history.csv", "config_snapshot_path": "config_snapshot.json",
                              "completion_path": "training_complete.json"}.items():
            ns[key] = target / filename
        return target

    # 2. Rolling training-loss guard: inactive before one epoch of updates,
    #    then trips after the configured number of consecutive windows.
    target = use_folder("loss_guard")
    ns.update(COLLAPSE_CHECK_EVERY_N_BATCHES=1, COLLAPSE_PATIENCE_WINDOWS=2,
              OPTIMIZER_STEPS_PER_EPOCH=1, CONSTANT_PREDICTOR_LOSS=0.0)
    reset_training_objects(ns)
    try:
        ns["train_one_epoch"](ns["model"], ns["train_loader"], ns["criterion"], ns["optimizer"],
                              ns["scaler"], ns["scheduler"], ns["model_ema"])
        raise AssertionError("training-loss collapse guard did not trip")
    except ns["TrainingCollapseError"]:
        pass
    report = json.loads((target / "collapse_report.json").read_text())
    assert report["stage"] == "training_loss" and report["consecutive_windows"] == 2
    # Accumulation 2: step 1 lands at micro-batch 2 (first active window) and
    # micro-batch 3 is the second consecutive collapsed window.
    assert ns["scheduler"].last_epoch == 1 and report["optimizer_step"] == 1
    ns["CONSTANT_PREDICTOR_LOSS"] = 1e9
    reset_training_objects(ns)
    ns["train_one_epoch"](ns["model"], ns["train_loader"], ns["criterion"], ns["optimizer"],
                          ns["scaler"], ns["scheduler"], ns["model_ema"])
    ns["COLLAPSE_GUARD_ENABLED"] = False
    ns["CONSTANT_PREDICTOR_LOSS"] = 0.0
    reset_training_objects(ns)
    ns["train_one_epoch"](ns["model"], ns["train_loader"], ns["criterion"], ns["optimizer"],
                          ns["scaler"], ns["scheduler"], ns["model_ema"])
    ns.update(COLLAPSE_GUARD_ENABLED=True, CONSTANT_PREDICTOR_LOSS=1e9, OPTIMIZER_STEPS_PER_EPOCH=2,
              COLLAPSE_CHECK_EVERY_N_BATCHES=100, COLLAPSE_PATIENCE_WINDOWS=5)

    # 3. Validation guard: stops the real epoch loop before any checkpoint.
    target = use_folder("validation_guard")
    ns["COLLAPSE_MIN_VAL_MACRO_AUROC"] = 1.01
    reset_training_objects(ns)
    training_source = next(source for source in sources if source.startswith("# Long-running training cell"))
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.reset_peak_memory_stats", lambda: None))
        stack.enter_context(patch("torch.cuda.max_memory_allocated", lambda: 0))
        try:
            exec(training_source, ns)
            raise AssertionError("validation collapse guard did not trip")
        except ns["TrainingCollapseError"]:
            pass
    ns["COLLAPSE_MIN_VAL_MACRO_AUROC"] = 0.0
    report = json.loads((target / "collapse_report.json").read_text())
    assert report["stage"] == "validation" and report["epoch"] == 1
    assert not (target / "last.pt").exists() and not (target / "best.pt").exists()
    assert not (target / "training_complete.json").exists()
    print("PASS: constant-predictor reference, training-loss and validation collapse guards stop before checkpoints")


def check_rare_sampling(ns, folder):
    sampler_class = ns["RepeatFactorSampler"]
    # Real clean12 training label frequencies -> the documented factors.
    frequency = {"3AVB": 1230 / 90734, "2AVB": 2923 / 90734, "STEMI": 7297 / 90734, "AF": 0.1323}
    factors = {label: max(1.0, math.sqrt(ns["RARE_SAMPLING_THRESHOLD"] / f)) for label, f in frequency.items()}
    assert abs(factors["3AVB"] - 2.716) < 0.01 and abs(factors["2AVB"] - 1.762) < 0.01
    assert abs(factors["STEMI"] - 1.115) < 0.01 and factors["AF"] == 1.0

    repeat = np.array([1.0, 1.0, 2.72, 1.76, 1.12, 1.0, 3.0, 1.5])
    sampler = sampler_class(repeat, seed=42)
    rows = len(repeat)
    assert len(sampler) == int(np.floor(repeat).sum() + np.floor((repeat - np.floor(repeat)).sum()))
    sampler.set_epoch(3)
    first = list(sampler)
    assert len(first) == len(sampler) and first == list(sampler)          # pure function of the epoch
    sampler.set_epoch(4)
    assert list(sampler) != first                                          # new order per epoch
    assert len(set(first)) == len(first)                                   # occurrences are unique
    counts = np.bincount(np.asarray(first) % rows, minlength=rows)
    assert ((counts == np.floor(repeat)) | (counts == np.floor(repeat) + 1)).all()
    # Occurrence numbers per row are exactly 0..count-1.
    for row in range(rows):
        assert sorted(v // rows for v in first if v % rows == row) == list(range(counts[row]))
    # Over many epochs each row is drawn floor(r) + frac * floor(F)/F times on
    # average (F = sum of fractions; the < 1-draw shortfall is spread evenly).
    fraction = repeat - np.floor(repeat)
    expected = np.floor(repeat) + fraction * np.floor(fraction.sum()) / fraction.sum()
    mean = np.mean([np.bincount(sampler.epoch_indices(e) % rows, minlength=rows)
                    for e in range(1, 4001)], axis=0)
    assert np.allclose(mean, expected, atol=0.03), (mean, expected)
    # On a clean12-sized problem the shortfall is negligible: mean draws = r.
    big = np.ones(90_734)
    big[:1230] = 2.716
    big[1230:4153] = 1.762
    big[4153:11450] = 1.115
    big_sampler = sampler_class(big, seed=7)
    big_mean = np.mean([np.bincount(big_sampler.epoch_indices(e) % len(big), minlength=len(big))
                        for e in range(1, 201)], axis=0)
    for lo, hi, r in ((0, 1230, 2.716), (1230, 4153, 1.762), (4153, 11450, 1.115), (11450, 90_734, 1.0)):
        assert abs(big_mean[lo:hi].mean() - r) < 0.005, (r, big_mean[lo:hi].mean())
    # Off (all factors 1): exactly one shuffled pass, occurrence == row.
    plain = sampler_class(np.ones(rows), seed=42)
    assert sorted(plain.epoch_indices(1).tolist()) == list(range(rows))

    # Dataset decodes occurrence indices; repeats get different augmentation.
    pixels = np.random.default_rng(11).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    image_path = folder / "rare.png"
    Image.fromarray(pixels).save(image_path)
    frame = pd.DataFrame([{"local_path": str(image_path), "record_id": str(i),
                           **{label: int(j == (i % NUM_LABELS)) for j, label in enumerate(ns["LABELS"])}}
                          for i in range(rows)])
    dataset = ns["ECGImageDataset"](frame, return_metadata=False, training=True)
    dataset.set_epoch(3)
    image0, target0, factors0 = dataset[2]
    image1, target1, factors1 = dataset[rows + 2]
    assert torch.equal(image0, image1) and torch.equal(target0, target1)
    assert not torch.equal(factors0, factors1)
    loader = DataLoader(dataset, sampler=sampler, batch_size=4, num_workers=2,
                        multiprocessing_context="fork", drop_last=True)
    sampler.set_epoch(3)
    from_loader = torch.cat([batch[2] for batch in loader])
    direct = torch.stack([dataset[v][2] for v in first])[:len(from_loader)]
    assert torch.equal(from_loader, direct)
    print(f"PASS: repeat-factor sampler (fixed length, epoch-keyed, mean draws = r, occurrence decode, "
          f"distinct augmentation for repeats); clean12 factors 3AVB {factors['3AVB']:.2f}, "
          f"2AVB {factors['2AVB']:.2f}, STEMI {factors['STEMI']:.2f}")


def check_bootstrap(ns):
    bootstrap = ns["patient_bootstrap_average_precision"]
    rng = np.random.default_rng(5)
    n = 300
    patients = np.array([f"p{i // 2}" for i in range(n)])            # two ECGs per patient
    truth = (rng.random((n, NUM_LABELS)) < 0.2).astype(np.int8)
    truth[0, :] = 1
    truth[1, :] = 0
    probability = np.clip(truth * 0.6 + rng.random((n, NUM_LABELS)) * 0.5, 0, 1)
    draws = bootstrap(truth, probability, patients, 50, 42)
    assert draws.shape == (50, NUM_LABELS) and np.isfinite(draws).all()
    assert np.array_equal(draws, bootstrap(truth, probability, patients, 50, 42))
    # Round 0 equals AP on an explicitly resampled (duplicated) patient set.
    unique, inverse = np.unique(patients, return_inverse=True)
    replay = np.random.default_rng([42, 2026, 10, 2])
    chosen = replay.integers(0, len(unique), len(unique))
    rows = np.concatenate([np.flatnonzero(inverse == patient) for patient in chosen])
    for label in range(NUM_LABELS):
        explicit = average_precision_score(truth[rows, label], probability[rows, label])
        assert abs(explicit - draws[0, label]) < 1e-12
    perfect = bootstrap(truth, truth.astype(float), patients, 20, 1)
    assert np.allclose(perfect, 1.0)
    print("PASS: patient bootstrap is reproducible, equals explicit patient resampling, and is 1.0 for perfect scores")


def check_drive_persistence(ns, folder):
    run_dir, drive_dir = folder / "local_run", folder / "drive_run"
    run_dir.mkdir()
    drive_dir.mkdir()
    saved = {key: ns.get(key) for key in ("RUN_DIR", "DRIVE_RUN_DIR", "DRIVE_SYNC_ENABLED")}
    try:
        ns.update(RUN_DIR=run_dir, DRIVE_RUN_DIR=drive_dir, DRIVE_SYNC_ENABLED=True)
        (run_dir / "last.pt").write_bytes(os.urandom(4096))
        (run_dir / "sub").mkdir()
        (run_dir / "sub" / "table.csv").write_text("a,b\n1,2\n")
        assert ns["copy_to_drive"](run_dir / "last.pt", required=True) == drive_dir / "last.pt"
        assert (drive_dir / "last.pt").read_bytes() == (run_dir / "last.pt").read_bytes()
        assert not list(drive_dir.glob("*.partial"))
        (run_dir / "last.pt").write_bytes(b"newer")                           # overwrite in place
        ns["spawn_drive_copy"](run_dir / "last.pt")
        assert ns["join_drive_copies"]() == 0
        assert (drive_dir / "last.pt").read_bytes() == b"newer"
        try:
            ns["copy_to_drive"](run_dir / "missing.pt", required=True)
            raise AssertionError("missing required artifact was not reported")
        except FileNotFoundError:
            pass
        # Fresh runtime: local files gone, restore from Drive, never overwrite.
        (run_dir / "last.pt").unlink()
        (run_dir / "best.pt").write_bytes(b"local best")
        (drive_dir / "best.pt").write_bytes(b"drive best")
        restored = ns["restore_missing_run_files_from_drive"](["last.pt", "best.pt", "absent.pt"])
        assert restored == ["last.pt"] and (run_dir / "last.pt").read_bytes() == b"newer"
        assert (run_dir / "best.pt").read_bytes() == b"local best"
        manifest = ns["sync_complete_run_to_drive"](required=True)
        assert {row["relative_path"] for row in manifest["artifacts"]} == {"last.pt", "best.pt", "sub/table.csv"}
        assert (drive_dir / "sub" / "table.csv").read_text() == "a,b\n1,2\n"
        assert json.loads((drive_dir / "drive_sync_manifest.json").read_text())["artifacts"]
        # An unreachable Drive folder must never read as "fresh run".
        ns["DRIVE_RUN_DIR"] = folder / "not_mounted"
        try:
            ns["restore_from_drive"]("last.pt", run_dir / "x.pt")
            raise AssertionError("unreachable Drive read as a fresh run")
        except RuntimeError:
            pass
    finally:
        ns.update(saved)
    print("PASS: Drive copy (atomic rename, size check, background + join), restore-missing-only, sync manifest, unmounted Drive refused")


class FakeS3:
    """Ranged GetObject/HeadObject over in-memory objects, with IfMatch."""

    def __init__(self, objects, etag='"etag-1"'):
        self.objects, self.etag, self.range_calls = objects, etag, 0

    def _missing(self, key):
        return ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._missing(Key)
        return {"ContentLength": len(self.objects[Key]), "ETag": self.etag}

    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        if Key not in self.objects:
            raise self._missing(Key)
        if IfMatch is not None and IfMatch != self.etag:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "GetObject")
        data = self.objects[Key]
        if Range:
            start, end = map(int, Range.split("=")[1].split("-"))
            data = data[start:end + 1]
            self.range_calls += 1
        return {"Body": io.BytesIO(data)}


def check_streamed_extraction(ns, folder):
    topdir = ns["KIT_TOPDIR"]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in [("manifest.csv", b"source_header,image_path,status\n"),
                              ("images/I0001/WFDB/a.png", os.urandom(300_000)),
                              ("images/I0006/WFDB/b.png", os.urandom(200_000))]:
            info = tarfile.TarInfo(f"{topdir}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    blob = buffer.getvalue()
    digest = hashlib.sha256(blob).hexdigest()
    key = "dataset/test.tar"
    fake = FakeS3({key: blob, key + ".sha256": f"{digest}  test.tar\n".encode()})
    extract_root = folder / "extract"
    extract_root.mkdir()
    (folder / "scratch_tmp").mkdir()
    keys = ("S3_READ_CLIENT", "S3_BUCKET", "S3_DATASET_KEY", "EXTRACT_ROOT", "KIT_ROOT", "SCRATCH_TMP",
            "S3_DOWNLOAD_CHUNK_BYTES", "S3_DOWNLOAD_WORKERS", "REQUIRE_TAR_SHA256")
    saved = {name: ns.get(name) for name in keys}
    try:
        ns.update(S3_READ_CLIENT=fake, S3_BUCKET="bucket", S3_DATASET_KEY=key, EXTRACT_ROOT=extract_root,
                  KIT_ROOT=extract_root / topdir, SCRATCH_TMP=folder / "scratch_tmp",
                  S3_DOWNLOAD_CHUNK_BYTES=65_536, S3_DOWNLOAD_WORKERS=4, REQUIRE_TAR_SHA256=True)
        assert ns["expected_tar_sha256"]() == digest
        streamed = b"".join(ns["iter_s3_object"]("bucket", key, fake.etag, len(blob), 65_536, 4))
        assert streamed == blob and fake.range_calls == math.ceil(len(blob) / 65_536)
        actual, _ = ns["stream_extract_archive"](digest, len(blob), fake.etag)
        assert actual == digest
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            for member in archive.getmembers():
                assert (extract_root / member.name).read_bytes() == archive.extractfile(member).read()
        # Wrong expected hash: the tree is deleted and the cell raises.
        try:
            ns["stream_extract_archive"]("0" * 64, len(blob), fake.etag)
            raise AssertionError("hash mismatch was accepted")
        except RuntimeError as error:
            assert "SHA-256" in str(error)
        assert not (extract_root / topdir).exists()
        # The object changes mid-download (ETag no longer matches): refused, tree deleted.
        try:
            ns["stream_extract_archive"](digest, len(blob), '"stale-etag"')
            raise AssertionError("ETag change was accepted")
        except RuntimeError as error:
            assert "changed during the download" in str(error.__cause__ or error)
        assert not (extract_root / topdir).exists()
        # Missing sidecar with strict provenance: refused.
        del fake.objects[key + ".sha256"]
        try:
            ns["expected_tar_sha256"]()
            raise AssertionError("missing sidecar was accepted")
        except RuntimeError:
            pass
    finally:
        ns.update(saved)
    print("PASS: streamed S3 ranges reassemble in order, tar extraction is byte-exact, hash/ETag mismatches delete the tree, missing sidecar refused")


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ns, sources = load_notebook()
    check_patient_split(ns)
    check_base_model(ns)
    check_bf16_fusion(ns)
    with tempfile.TemporaryDirectory(prefix="cardiosentry-msff-c12-check-") as directory:
        folder = Path(directory)
        check_dataset(ns, folder)
        # Before check_training_resume, which stubs the Drive copy functions.
        check_drive_persistence(ns, folder)
        check_streamed_extraction(ns, folder)
        check_compilation_and_selection(ns, folder)
        check_sizing_and_benchmark_logic(ns)
        check_selection(ns)
        check_training_resume(ns, sources, folder)
        check_skipped_updates(ns)
        check_collapse_guard(ns, sources, folder)
        check_rare_sampling(ns, folder)
        check_bootstrap(ns)
    print("All CPU checks passed. GPU preflight, real S3/Drive I/O, and full-corpus training are not exercised here.")


if __name__ == "__main__":
    main()
