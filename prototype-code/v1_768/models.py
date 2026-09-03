"""
The five architectures, built through one factory so nothing but the backbone
differs between runs.

PRETRAINING ASYMMETRY - declare this in the writeup.

A controlled comparison wants every backbone to start from the same pretraining
corpus. Four of these have clean ImageNet-1k weights. ViT-B/16 does not
realistically exist as an in1k-only checkpoint in timm at either resolution; the
usable checkpoints are ImageNet-21k pretrained and then fine-tuned to 1k. ViT
therefore enters with strictly more pretraining data than the CNNs.

This is not something the harness can fix, so it does not pretend to. It records
the resolved pretrained tag for every run in result.json, and compare.py prints
the tags side by side. If ViT wins, the honest reading is "ViT-B/16 with in21k
pretraining beat these CNNs with in1k pretraining" - which is a real and
publishable finding, just not the same claim as "ViT is the better architecture".

v1_768 RESOLUTION ADAPTATION - declare this in the writeup too.

At 768 the four CNNs are unchanged: they are fully convolutional, so the same
pretrained weights run at any resolution. ViT-B/16 is not: its pretrained
checkpoint carries fixed 384-position embeddings. The 768 run therefore uses the
SAME checkpoint (vit_base_patch16_384) and lets timm interpolate the position
embeddings 384 -> 768 (bicubic) on load, which is the standard fine-tune-at-
higher-resolution procedure. Verified: pos_embed (1, 577, D) -> (1, 2305, D),
forward passes at 768.
"""
from __future__ import annotations
import torch.nn as nn

import config as C

# micro_batch is the only per-model knob, and it is a VRAM accommodation, not an
# experimental variable: gradient accumulation restores the same EFFECTIVE batch
# (config.EFFECTIVE_BATCH) for every model, so the optimisation trajectory is
# unchanged.
#
# v1_768: tuned for the Colab Pro L4 (22.5 GB) this batch will run on, NOT for
# the 16 GB T4 the v1 values targeted. Values are the v1 micro-batches divided
# by 2:
#
#   * resnet50 = 8 is PROVEN on this exact GPU at 768: the previous
#     cardiosentry_hybrids_v1 `resnet50_hires` run used micro 8 x accum 4 and
#     trained with only 4.7/22G reserved (see prototype-results/hybrid_v1).
#   * the others keep v1's relative ratios (vgg:resnet 1:2, convnext:resnet 1:2,
#     effb3:resnet 1:1, vit:resnet 1:4), anchored to resnet50 = 8.
#
# Activation memory scales with pixel count (768^2 = 4x 384^2), so these are
# NOT the v1 values: each model gets half its v1 micro-batch, which the 384
# T4 logs (max reserved 2.5-4.5 GB of 15) and the L4 hires runs both leave
# generous headroom for. preflight.py measures the real peak on the L4 and
# will suggest raising any of these further - that is a pure speedup, the
# effective batch of 32 stays untouched. On a 16 GB T4, halve these again
# (vgg16 2, resnet50 4, efficientnet_b3 4, convnext_base 2, vit_b_768 1).
REGISTRY: dict[str, dict] = {
    "vgg16": {
        "timm": "vgg16.tv_in1k",
        "micro_batch": 4,       # v1: 8  (x 1/2, L4)
        "pretrain": "ImageNet-1k",
        "params_m": 138.4,
    },
    "resnet50": {
        "timm": "resnet50.tv_in1k",
        "micro_batch": 8,       # v1: 16 (x 1/2, L4; PROVEN at 768 on L4)
        "pretrain": "ImageNet-1k",
        "params_m": 25.6,
    },
    "efficientnet_b3": {
        "timm": "efficientnet_b3.ra2_in1k",
        "micro_batch": 8,       # v1: 16 (x 1/2, L4)
        "pretrain": "ImageNet-1k",
        "params_m": 12.2,
    },
    "convnext_base": {
        "timm": "convnext_base.fb_in1k",
        "micro_batch": 4,       # v1: 8  (x 1/2, L4)
        "pretrain": "ImageNet-1k",
        "params_m": 88.6,
    },
    "vit_b_768": {
        "timm": "vit_base_patch16_384.augreg_in21k_ft_in1k",
        "micro_batch": 2,       # v1: 4  (x 1/2, L4; seq len 4x)
        "pretrain": "ImageNet-21k -> 1k   (ASYMMETRIC - see module docstring)",
        "params_m": 86.9,
        "use_config_img_size": True,   # pos-embed interpolated 384 -> 768 on load
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

    # Only the ViT needs the explicit input size: its tag bakes in 384 and its
    # position embeddings must be interpolated to config.IMG_SIZE. The CNNs are
    # fully convolutional; passing img_size to them would be a no-op at best.
    kwargs = {"img_size": C.IMG_SIZE} if spec.get("use_config_img_size") else {}

    try:
        model = timm.create_model(tag, pretrained=pretrained,
                                  num_classes=C.NUM_CLASSES, **kwargs)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not create '{tag}'.\n"
            f"the pretrained tag may have changed in your timm version "
            f"({getattr(timm, '__version__', '?')}).\n"
            f"list valid tags with:  python preflight.py --list-tags {name}\n"
            f"original error: {exc}"
        ) from exc

    info = {k: v for k, v in spec.items() if k != "use_config_img_size"}
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
