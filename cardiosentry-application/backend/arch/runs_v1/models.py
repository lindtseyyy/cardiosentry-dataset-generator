"""
The five architectures, built through one factory so nothing but the backbone
differs between runs.

PRETRAINING ASYMMETRY - declare this in the writeup.

A controlled comparison wants every backbone to start from the same pretraining
corpus. Four of these have clean ImageNet-1k weights. ViT-B/16 at 384 does not
realistically exist as in1k-only in timm; the usable checkpoints are ImageNet-21k
pretrained and then fine-tuned to 1k. ViT therefore enters with strictly more
pretraining data than the CNNs.

This is not something the harness can fix, so it does not pretend to. It records
the resolved pretrained tag for every run in result.json, and compare.py prints
the tags side by side. If ViT wins, the honest reading is "ViT-B/384 with in21k
pretraining beat these CNNs with in1k pretraining" - which is a real and
publishable finding, just not the same claim as "ViT is the better architecture".
"""
from __future__ import annotations
import torch.nn as nn

from . import config as C  # vendored-copy adjustment: was `import config as C` (see arch/README.md)

# micro_batch is the only per-model knob, and it is a VRAM accommodation, not an
# experimental variable: gradient accumulation restores the same EFFECTIVE batch
# (config.EFFECTIVE_BATCH) for every model, so the optimisation trajectory is
# unchanged. Values target a 16 GB T4 at 384px with AMP. Raise on an A100.
REGISTRY: dict[str, dict] = {
    "vgg16": {
        "timm": "vgg16.tv_in1k",
        "micro_batch": 8,
        "pretrain": "ImageNet-1k",
        "params_m": 138.4,
    },
    "resnet50": {
        "timm": "resnet50.tv_in1k",
        "micro_batch": 16,
        "pretrain": "ImageNet-1k",
        "params_m": 25.6,
    },
    "efficientnet_b3": {
        "timm": "efficientnet_b3.ra2_in1k",
        "micro_batch": 16,
        "pretrain": "ImageNet-1k",
        "params_m": 12.2,
    },
    "convnext_base": {
        "timm": "convnext_base.fb_in1k",
        "micro_batch": 8,
        "pretrain": "ImageNet-1k",
        "params_m": 88.6,
    },
    "vit_b_384": {
        "timm": "vit_base_patch16_384.augreg_in21k_ft_in1k",
        "micro_batch": 4,
        "pretrain": "ImageNet-21k -> 1k   (ASYMMETRIC - see module docstring)",
        "params_m": 86.9,
    },
}

MODEL_NAMES = list(REGISTRY)


def build(name: str, pretrained: bool = True, timm_tag: str | None = None):
    """Return (model, info). Head is 4 independent logits - sigmoid lives in the loss."""
    import timm

    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. choose from: {MODEL_NAMES}")
    spec = REGISTRY[name]
    tag = timm_tag or spec["timm"]

    try:
        model = timm.create_model(tag, pretrained=pretrained, num_classes=C.NUM_CLASSES)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not create '{tag}'.\n"
            f"the pretrained tag may have changed in your timm version "
            f"({getattr(timm, '__version__', '?')}).\n"
            f"list valid tags with:  python preflight.py --list-tags {name}\n"
            f"original error: {exc}"
        ) from exc

    info = dict(spec)
    info["resolved_tag"] = tag
    info["params_m_actual"] = sum(p.numel() for p in model.parameters()) / 1e6
    info["timm_version"] = getattr(timm, "__version__", "?")
    return model, info


def micro_batch(name: str, override: int | None = None) -> int:
    return override or REGISTRY[name]["micro_batch"]


def accumulation_steps(name: str, override: int | None = None) -> int:
    mb = micro_batch(name, override)
    if C.EFFECTIVE_BATCH % mb != 0:
        raise ValueError(
            f"EFFECTIVE_BATCH ({C.EFFECTIVE_BATCH}) must be divisible by "
            f"micro_batch ({mb}) or the effective batch differs between models "
            f"and the comparison stops being controlled."
        )
    return C.EFFECTIVE_BATCH // mb
