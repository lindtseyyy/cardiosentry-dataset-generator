"""Model runner: eager torch inference with the exact training-chain inputs.

Handles the two facts that make naive `torch.load` fail (plan §10.1):
  F2 — checkpoints are `{"model": state_dict, "thresholds": ...}` blobs; the
       architecture code (vendored under backend/arch/) must rebuild the module.
  F3 — round-2 models are called as `model({768: tensor})`, round-1 and the
       round-3 (hybrids_v2) geometry runs as `model(tensor)`; the descriptor's
       `architecture.call_style` selects the adapter.

Views are carried as (height, width) pairs throughout, because the round-3
models are trained on a RECTANGLE (768x1024, the sheet's own aspect) rather than
a square. A `view_dict` model keys its dict by a single integer, so that adapter
accepts square views only — the descriptor schema enforces it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from backend.errors import ApiError, NotFound
from backend.schemas.model_descriptor import ModelDescriptor
from backend.schemas.prediction import ModelRunInfo
from backend.services.preprocess import as_hw, resize_hw
from backend.services.registry import LoadedDescriptor
from backend.settings import settings


class ModelLoadError(Exception):
    pass


@dataclass
class LoadedModel:
    descriptor: ModelDescriptor
    model: torch.nn.Module
    arch_info: dict
    thresholds: list[float]                  # per-label, order = descriptor.labels
    checkpoint_sha256: str
    protocol_fingerprint: str | None
    epoch: int | None
    views: list[tuple[int, int]]             # per view, (height, width)
    num_threads: int
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def id(self) -> str:
        return self.descriptor.id

    def run_info(self) -> ModelRunInfo:
        return ModelRunInfo(
            id=self.descriptor.id,
            display_name=self.descriptor.display_name,
            version=self.descriptor.version,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_sha256_short=self.checkpoint_sha256[:8],
            protocol_fingerprint=self.protocol_fingerprint,
            epoch=self.epoch,
            views=[list(v) for v in self.views],
            input_size=list(self.descriptor.preprocess.input_hw),
            device=self.descriptor.runtime.device,
            dtype=self.descriptor.runtime.dtype,
            amp=False,
            torch_num_threads=self.num_threads)


def load_model(ld: LoadedDescriptor) -> LoadedModel:
    desc = ld.descriptor
    num_threads = desc.runtime.num_threads or settings.num_threads
    torch.set_num_threads(num_threads)

    try:
        model, info = ld.arch_module.build(desc.architecture.registry_key,
                                           desc.architecture.pretrained)
    except Exception as exc:
        raise ModelLoadError(
            f"{desc.id}: could not build architecture "
            f"{desc.architecture.registry_key!r} from {desc.architecture.module!r}:\n"
            f"  {exc}") from exc

    try:
        blob = torch.load(ld.weights_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ModelLoadError(f"{desc.id}: could not load {ld.weights_path}:\n  {exc}") from exc
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

    # strict=True: a silently partially-loaded model produces confident
    # nonsense (§10.3).
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ModelLoadError(
            f"{desc.id}: state_dict mismatch (strict). "
            f"missing={missing[:5]}{'…' if len(missing) > 5 else ''} "
            f"unexpected={unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    model.eval()

    # Thresholds (§10.3): per-class, from the checkpoint by default. A global
    # 0.5 default would silently change sensitivity relative to every
    # prototype number, so refuse instead.
    thresholds: list[float]
    if desc.thresholds.source == "checkpoint":
        thr = blob.get("thresholds") if isinstance(blob, dict) else None
        if not thr:
            raise ModelLoadError(
                f"{desc.id}: thresholds.source=checkpoint but the .pt carries no "
                "frozen thresholds. Set thresholds.source=explicit in the "
                "descriptor, or use a checkpoint that was saved with them.")
        thresholds = [float(t) for t in thr]
    else:
        thresholds = [float(t) for t in (desc.thresholds.explicit or [])]
    if len(thresholds) != len(desc.labels):
        raise ModelLoadError(
            f"{desc.id}: {len(thresholds)} thresholds for {len(desc.labels)} "
            "labels — the descriptor's labels must match the checkpoint's head.")

    fingerprint = None
    epoch = None
    if isinstance(blob, dict):
        fingerprint = blob.get("protocol_fingerprint") or blob.get("fingerprint")
        epoch = blob.get("epoch")
    if desc.expected_protocol_fingerprint and fingerprint != desc.expected_protocol_fingerprint:
        raise ModelLoadError(
            f"{desc.id}: protocol fingerprint mismatch — checkpoint says "
            f"{fingerprint!r}, descriptor expects {desc.expected_protocol_fingerprint!r}")

    views = (desc.architecture.view_sizes
             or [as_hw(v) for v in getattr(model, "views", ())]
             or [desc.preprocess.input_hw])
    if isinstance(blob, dict) and "views" in blob:
        views = [as_hw(v) for v in blob["views"]]   # the checkpoint is authoritative

    lm = LoadedModel(descriptor=desc, model=model,
                     arch_info=dict(info) if isinstance(info, dict) else {},
                     thresholds=thresholds, checkpoint_sha256=ld.checkpoint_sha256,
                     protocol_fingerprint=fingerprint, epoch=epoch, views=views,
                     num_threads=num_threads)

    # Warm-up forward (§7.4): the first torch forward on CPU costs seconds of
    # lazy init; without it the first real capture reports a wildly misleading
    # forward_ms.
    try:
        _forward_views(lm, _zeros_input(lm))
    except Exception as exc:
        raise ModelLoadError(
            f"{desc.id}: warm-up forward failed:\n  {exc}") from exc
    return lm


def _zeros_input(lm: LoadedModel) -> dict[tuple[int, int], torch.Tensor]:
    return {(h, w): torch.zeros(1, 3, h, w) for (h, w) in lm.views}


def call_model(lm: LoadedModel,
               views: dict[tuple[int, int], torch.Tensor]) -> torch.Tensor:
    """Call the model with the right F3 adapter, INSIDE whatever grad mode the
    caller established. Inference wraps this in `no_grad` (`_forward_views`);
    the explainability service does not, because every method it implements is
    defined by a gradient of this output."""
    if lm.descriptor.architecture.call_style == "tensor":
        if len(lm.views) != 1:
            raise ApiError(500, "INFERENCE_FAILED",
                           "call_style=tensor requires exactly one view")
        return lm.model(views[lm.views[0]])
    # view_dict models key their input by ONE integer (the square size the
    # research code used). The descriptor schema refuses a non-square view
    # on this call style, so h == w here by construction.
    return lm.model({h: t for (h, w), t in views.items()})


def _forward_views(lm: LoadedModel,
                   views: dict[tuple[int, int], torch.Tensor]) -> torch.Tensor:
    """Inference forward: the F3 adapter with gradients off."""
    with torch.no_grad():
        return call_model(lm, views)


def derive_views(lm: LoadedModel, model_input_path: Path,
                 mean, std) -> tuple[dict[tuple[int, int], torch.Tensor], float]:
    """Derive the per-view tensors from the stored model_input.png like the
    training loader did: ONE decode of the stored PNG, then a BICUBIC resize
    per view whose geometry differs from the stored one, ToTensor + Normalize
    each. The stored size is read from the file (not assumed), so a 384-input
    model can also score a capture whose model_input.png was stored at 768x1024
    — deterministically, from the same pixels every model shares."""
    t0 = time.perf_counter()
    with Image.open(model_input_path) as im:
        im.load()
        img = im.convert("RGB")
    out: dict[tuple[int, int], torch.Tensor] = {}
    tf_base = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    for hw in lm.views:
        out[hw] = tf_base(resize_hw(img, hw)).unsqueeze(0)
    ms = (time.perf_counter() - t0) * 1000.0
    return out, ms


@dataclass
class PredictResult:
    scores: dict[str, float]
    logits: dict[str, float]
    thresholds: dict[str, float]
    threshold_source: str
    positive: list[str]
    timing_ms: dict[str, float]
    model_input_sha256: str


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict(lm: LoadedModel, model_input_path: Path,
            thresholds_override: dict[str, float] | None = None) -> PredictResult:
    """Run one inference on the stored model_input.png.

    Serialized by the model lock: inference is CPU-bound and requests are
    single-user-sequential; concurrent forwards only queue (§16 case 16).
    """
    if not model_input_path.is_file():
        raise NotFound("MISSING_MODEL_INPUT",
                       "model_input.png not found — rectify this capture first.")
    desc = lm.descriptor
    pre = desc.preprocess
    sha = _sha256_file(model_input_path)     # proves which pixels were scored

    with lm.lock:
        views, prep_ms = derive_views(lm, model_input_path,
                                      pre.normalize.mean, pre.normalize.std)
        t0 = time.perf_counter()
        try:
            logits = _forward_views(lm, views)
        except Exception as exc:
            raise ApiError(500, "INFERENCE_FAILED",
                           "Inference failed — the capture was saved; you can "
                           "retry.") from exc
        forward_ms = (time.perf_counter() - t0) * 1000.0

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    logits = logits[0].float().detach().cpu().numpy()
    if logits.shape[0] != len(desc.labels):
        raise ApiError(500, "INFERENCE_FAILED",
                       f"model returned {logits.shape[0]} logits for "
                       f"{len(desc.labels)} labels — descriptor labels mismatch "
                       "the checkpoint's head.")

    scores = 1.0 / (1.0 + np.exp(-logits))    # torch.sigmoid on fp32 logits

    if thresholds_override:
        threshold_source = "user_override"
        thresholds = {l: float(thresholds_override[l]) for l in desc.labels}
    elif desc.thresholds.source == "explicit":
        threshold_source = "descriptor"
        thresholds = dict(zip(desc.labels, lm.thresholds))
    else:
        threshold_source = "checkpoint"
        thresholds = dict(zip(desc.labels, lm.thresholds))

    positive = [l for l in desc.labels if scores[desc.labels.index(l)] >= thresholds[l]]

    return PredictResult(
        scores={l: float(scores[i]) for i, l in enumerate(desc.labels)},
        logits={l: float(logits[i]) for i, l in enumerate(desc.labels)},
        thresholds=thresholds, threshold_source=threshold_source,
        positive=positive,
        timing_ms={"preprocess": round(prep_ms, 1),
                   "forward": round(forward_ms, 1),
                   "total": round(prep_ms + forward_ms, 1)},
        model_input_sha256=sha)
