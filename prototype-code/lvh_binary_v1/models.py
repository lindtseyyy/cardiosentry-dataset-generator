"""
The binary-task model, built through the same factory shape as round 2.

WHAT IS IDENTICAL TO THE FOUR-HEAD REFERENCE (resnet50_hires), BY CONSTRUCTION

  * timm backbone `resnet50.tv_in1k`, ImageNet-1k pretrained, global_pool="avg"
  * the SAME 2-layer MLP head structure: per-group LayerNorm -> Dropout(0.1)
    -> Linear(2048, 512) -> GELU -> Dropout(0.1) -> Linear(512, K)
  * feature group name `r768`, head input 2048
  * view (768,), channels_last on

The ONLY structural difference is the final width: K=4 -> K=1. That is the
experiment - the shared classification burden (STEMI, AF, NORMAL) is removed
while the backbone, the head depth/width, and the feature path stay put.

The head's random init is seeded (config.SEED) in both runs, so the first
three head layers start from the same distributions under the same protocol.
"""
from __future__ import annotations
from collections import OrderedDict

import torch
import torch.nn as nn

import config as C

HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.1


# ---------------------------------------------------------------- shared head
class FusionHead(nn.Module):
    """Per-group LayerNorm -> concat -> MLP -> logits.

    Byte-identical to round 2's FusionHead except the final width. Over one
    group the LayerNorm and concat are structurally inert, which is deliberate:
    the feature path to the last layer is the same computation on the same
    tensor the four-head run performed.
    """

    def __init__(self, group_dims: "OrderedDict[str,int]",
                 num_classes: int = C.NUM_CLASSES):
        super().__init__()
        self.group_names = list(group_dims)
        self.norms = nn.ModuleDict({g: nn.LayerNorm(d) for g, d in group_dims.items()})
        total = sum(group_dims.values())
        self.mlp = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(total, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, num_classes),
        )
        self.in_features = total

    def forward(self, feats: "OrderedDict[str,torch.Tensor]"):
        parts = [self.norms[g](feats[g]) for g in self.group_names]
        return self.mlp(torch.cat(parts, dim=1))


class GroupedModel(nn.Module):
    """Single-stream backbone + shared head, mirroring round 2's base class."""

    views: tuple[int, ...] = (C.VIEW_SIZE,)

    def features(self, x: dict) -> "OrderedDict[str,torch.Tensor]":
        raise NotImplementedError

    def forward(self, x: dict):
        return self.head(self.features(x))

    @property
    def groups(self) -> list[str]:
        return self.head.group_names


class SingleStream(GroupedModel):
    """One ResNet-50 at 768. The binary twin of round 2's resnet50_hires."""

    def __init__(self, tag: str, size: int, pretrained: bool = True):
        super().__init__()
        import timm
        self.views = (size,)
        self.size = size
        self.backbone = timm.create_model(tag, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        dim = self.backbone.num_features
        self.head = FusionHead(OrderedDict([(f"r{size}", dim)]))

    def features(self, x):
        return OrderedDict([(f"r{self.size}", self.backbone(x[self.size]))])


# ---------------------------------------------------------------- registry
#
# MICRO-BATCH IS NOT A FREE PARAMETER. BatchNorm normalises over the
# micro-batch and gradient accumulation does not touch it, so the binary run
# must use the SAME micro-batch the four-head reference used (8, uniform on
# the L4) or the comparison inherits a BN-statistics confound.
REGISTRY: dict[str, dict] = {
    "resnet50_lvh_binary": {
        "build": lambda p: SingleStream("resnet50.tv_in1k", C.VIEW_SIZE, p),
        "views": (C.VIEW_SIZE,), "micro_batch": 8, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k"], "order": 1,
        "why": "ResNet-50@768 trained on LVH ONLY - the negative-transfer test",
    },
}

MODEL_NAMES = sorted(REGISTRY, key=lambda m: REGISTRY[m]["order"])


def build(name: str, pretrained: bool = True):
    """Return (model, info)."""
    import timm

    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. choose from: {MODEL_NAMES}")
    spec = REGISTRY[name]
    try:
        model = spec["build"](pretrained)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not build '{name}'.\n"
            f"a pretrained tag may have changed in your timm version "
            f"({getattr(timm, '__version__', '?')}).\n"
            f"tags used: {spec['tags']}\n"
            f"original error: {exc}"
        ) from exc

    info = {k: v for k, v in spec.items() if k != "build"}
    info["channels_last"] = bool(C.CHANNELS_LAST)
    info["views"] = list(model.views)
    info["groups"] = model.groups
    info["params_m_actual"] = sum(p.numel() for p in model.parameters()) / 1e6
    info["head_in_features"] = model.head.in_features
    info["num_classes"] = C.NUM_CLASSES
    info["timm_version"] = getattr(timm, "__version__", "?")
    info["input_fingerprint"] = C.input_fingerprint(tuple(model.views))
    return model, info


def views(name: str) -> tuple[int, ...]:
    return tuple(REGISTRY[name]["views"])


def micro_batch(name: str, override: int | None = None) -> int:
    return override or REGISTRY[name]["micro_batch"]


def accumulation_steps(name: str, override: int | None = None) -> int:
    mb = micro_batch(name, override)
    if C.EFFECTIVE_BATCH % mb != 0:
        raise ValueError(
            f"EFFECTIVE_BATCH ({C.EFFECTIVE_BATCH}) must be divisible by "
            f"micro_batch ({mb}) or the effective batch differs between runs "
            f"and the comparison stops being controlled."
        )
    return C.EFFECTIVE_BATCH // mb
