"""
The seven round-2 architectures, built through one factory.

WHAT EVERY MODEL HAS IN COMMON, SO THAT NONE OF IT IS A CONFOUND

Each model produces one or more NAMED FEATURE GROUPS, each a pooled vector.
Every group is LayerNorm'd independently, the groups are concatenated, and the
SAME two-layer head maps the result to four logits:

    groups -> [LayerNorm per group] -> concat -> Dropout -> Linear(.,512)
           -> GELU -> Dropout -> Linear(512,4)

Round 1 used timm's default single Linear head. This one is deeper, so
`resnet50_base` is NOT the same model as round 1's `resnet50` and must be
re-trained rather than copied across — that is exactly what it is for.

Per-group LayerNorm is not decoration. It is what makes STREAM ABLATION valid:
zeroing one group after its own LayerNorm leaves every other group's statistics
untouched, so "what does this stream contribute" is answerable by setting
`model.ablate = "<group>"` and running test again. Normalising AFTER the concat
would have made the ablation meaningless, because dropping half the vector would
shift the statistics of the half that remained.

THE SEVEN RUNS, AND WHAT EACH ONE IS FOR

  resnet50_base        384        the in-family baseline. Same head, same
                                  pipeline, same protocol as everything below.
  resnet50_hires       768        THE CONTROL. Identical to base except it sees
                                  4x the pixels. If LVH is resolution-limited,
                                  this alone fixes it, and none of the hybrids
                                  below are needed. Run it first.
  resnet50_cbam        384        channel+spatial attention. Adds no pixels;
                                  tests whether reweighting what is already
                                  there is enough.
  resnet50_fpn         384        top-down pyramid, P2..P5 pooled separately.
                                  Adds no pixels but restores stride-4 detail
                                  that C5 discarded. Level ablation says which
                                  scale LVH actually lives at.
  resnet50_dual        384+768    the headline hybrid. Global sheet view plus a
                                  detail view at 2x linear resolution.
  resnet50_vit         768+768    CNN local features + ViT global attention,
                                  both at 768. Round 1's ViT had the BEST LVH
                                  AUPRC (0.8822) despite placing 4th overall,
                                  and Sokolow-Lyon is a long-range V1-vs-V5
                                  comparison that self-attention is built for.
                                  Trained after the AF x LVH subgroup finding
                                  (resnet50_hires misses LVH on AF+LVH cases:
                                  recall 0.6429 vs 0.8722 on LVH-only): does a
                                  768 representation enriched with global
                                  attention preserve LVH when AF is present?
  resnet50_effb3_dual  384+768    heterogeneous dual-resolution. Runs LAST: in
                                  round 1 EfficientNet-B3 finished last with an
                                  epoch-1 val AUPRC of 0.4467 against 0.77-0.83
                                  for everything else, which is an LR mismatch
                                  signature, not a weak backbone. Fusing a
                                  possibly-mis-optimised branch under the shared
                                  1e-4 risks re-measuring that confound.

PRETRAINING ASYMMETRY — unchanged from round 1, and it now applies to one hybrid.
ViT-B/16-384 has no realistic in1k-only checkpoint in timm; the usable ones are
in21k pretrained then fine-tuned to 1k. `resnet50_vit` therefore enters with
strictly more pretraining data than the other six. Recorded per run and printed
by compare.py. If it wins, the honest claim is "ResNet-50 + in21k ViT beat these
in1k models", not "cross-attention is better".
"""
from __future__ import annotations
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C  # vendored-copy adjustment: was `import config as C` (see arch/README.md)

HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.1


# ---------------------------------------------------------------- shared head
class FusionHead(nn.Module):
    """Per-group LayerNorm -> concat -> MLP -> 4 logits. Identical everywhere."""

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

    def forward(self, feats: "OrderedDict[str,torch.Tensor]", ablate: str | None = None):
        parts = []
        for g in self.group_names:
            v = self.norms[g](feats[g])
            if ablate == g:
                v = torch.zeros_like(v)
            parts.append(v)
        return self.mlp(torch.cat(parts, dim=1))


class GroupedModel(nn.Module):
    """Base class: subclasses fill `features()`, the head and ablation are free.

    `ablate` is an attribute rather than a forward argument so that the training
    and evaluation loops never need to know which models are multi-stream.
    """

    views: tuple[int, ...] = (C.BASE_SIZE,)
    ablate: str | None = None

    def features(self, x: dict) -> "OrderedDict[str,torch.Tensor]":
        raise NotImplementedError

    def forward(self, x: dict):
        return self.head(self.features(x), ablate=self.ablate)

    @property
    def groups(self) -> list[str]:
        return self.head.group_names


# ---------------------------------------------------------------- CBAM
class ChannelGate(nn.Module):
    """Avg- and max-pooled descriptors through one shared MLP (Woo et al. 2018)."""

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(ch // reduction, 8)
        self.mlp = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, ch))

    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3)))
        return torch.sigmoid(a).view(b, c, 1, 1)


class SpatialGate(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(s))


class CBAM(nn.Module):
    """CBAM with a ZERO-INITIALISED residual gate.

    Plain CBAM inserted into a pretrained backbone multiplies every feature map
    by a freshly-initialised sigmoid, whose expected value at init is 0.5. That
    halves the activations of a network whose downstream weights were tuned for
    the un-halved ones, so the first epochs are spent undoing the damage rather
    than learning. On a 4,810-image training set that cost is not recoverable.

    Wrapping it as  x + gamma * (cbam(x) - x)  with gamma initialised to zero
    makes the module EXACTLY the identity at step 0, so the ImageNet features
    arrive intact and the network learns how much attention it wants. The
    learned gamma per block is saved in result.json — a near-zero gamma is a
    real finding, and it means attention was not what LVH needed.
    """

    def __init__(self, ch: int, reduction: int = 16, kernel: int = 7):
        super().__init__()
        self.channel = ChannelGate(ch, reduction)
        self.spatial = SpatialGate(kernel)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = x * self.channel(x)
        y = y * self.spatial(y)
        return x + self.gamma * (y - x)


# ---------------------------------------------------------------- FPN
class FPN(nn.Module):
    """Standard top-down pyramid (Lin et al. 2017), pooled for classification.

    Detection FPNs predict per location. Here each level is global-average-pooled
    into its own feature group, so the head sees four scales side by side and the
    ablation can report which one carried the diagnosis.

    At 384 input the levels are P2 96x96, P3 48x48, P4 24x24, P5 12x12. P2 is the
    only one whose stride (4 px = 2.9 mm) is finer than the amplitude differences
    LVH criteria turn on; C5 alone, at stride 32, is 23 mm per cell.
    """

    def __init__(self, in_channels: list[int], out_ch: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1),
                           nn.GroupNorm(32, out_ch), nn.ReLU(inplace=True))
             for _ in in_channels])
        self.out_ch = out_ch

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 2, -1, -1):          # deepest -> shallowest
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:],
                                            mode="nearest")
        return [s(x) for s, x in zip(self.smooth, lat)]


# ---------------------------------------------------------------- the models
def _timm_backbone(tag: str, pretrained: bool):
    import timm
    return timm.create_model(tag, pretrained=pretrained, num_classes=0, global_pool="avg")


def _vit_backbone(tag: str, pretrained: bool):
    """ViT with dynamic image size, so the 384-trained checkpoint accepts 768.

    The checkpoint's pos-embed covers a 24x24 patch grid; feeding 768x768 input
    produces 48x48. `dynamic_img_size=True` (timm >= 0.9) interpolates the
    pos-embed on first forward. Older timm does not accept the kwarg, so it is
    tried and the plain path is the fallback - which then fails loudly in
    preflight with the standard error instead of silently misbehaving.
    """
    import timm
    try:
        return timm.create_model(tag, pretrained=pretrained, num_classes=0,
                                 global_pool="avg", dynamic_img_size=True)
    except TypeError:
        return timm.create_model(tag, pretrained=pretrained, num_classes=0,
                                 global_pool="avg")


class SingleStream(GroupedModel):
    """One backbone at one resolution. Covers resnet50_base and resnet50_hires."""

    def __init__(self, tag: str, size: int, pretrained: bool = True):
        super().__init__()
        self.views = (size,)
        self.size = size
        self.backbone = _timm_backbone(tag, pretrained)
        dim = self.backbone.num_features
        self.head = FusionHead(OrderedDict([(f"r{size}", dim)]))

    def features(self, x):
        return OrderedDict([(f"r{self.size}", self.backbone(x[self.size]))])


class ResNetCBAM(GroupedModel):
    """ResNet-50 with a gated CBAM after every bottleneck of layer3 and layer4.

    Stages 1 and 2 are left alone deliberately. Their features are edge and
    texture level, and on a rendered sheet that means gridlines, paper creases
    and JPEG ringing — the corpus's known source signature. Teaching the network
    to attend harder to those is the failure mode to avoid, not the goal.
    """

    def __init__(self, size: int = C.BASE_SIZE, pretrained: bool = True):
        import timm
        super().__init__()
        self.views = (size,)
        self.size = size
        self.backbone = timm.create_model("resnet50.tv_in1k", pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        for stage_name in ("layer3", "layer4"):
            stage = getattr(self.backbone, stage_name)
            wrapped = []
            for block in stage:
                ch = block.conv3.out_channels
                wrapped.append(nn.Sequential(OrderedDict([("block", block),
                                                          ("cbam", CBAM(ch))])))
            setattr(self.backbone, stage_name, nn.Sequential(*wrapped))
        dim = self.backbone.num_features
        self.head = FusionHead(OrderedDict([("cbam", dim)]))

    def features(self, x):
        return OrderedDict([("cbam", self.backbone(x[self.size]))])

    def gammas(self) -> dict:
        """Learned attention strength per block. Near zero = attention unused."""
        return {n: float(p.detach().cpu().item())
                for n, p in self.named_parameters() if n.endswith("cbam.gamma")}


class ResNetFPN(GroupedModel):
    def __init__(self, size: int = C.BASE_SIZE, pretrained: bool = True, out_ch: int = 256):
        import timm
        super().__init__()
        self.views = (size,)
        self.size = size
        self.backbone = timm.create_model("resnet50.tv_in1k", pretrained=pretrained,
                                          features_only=True, out_indices=(1, 2, 3, 4))
        chs = self.backbone.feature_info.channels()
        self.fpn = FPN(chs, out_ch)
        self.levels = [f"P{i}" for i in range(2, 2 + len(chs))]
        self.head = FusionHead(OrderedDict((l, out_ch) for l in self.levels))

    def features(self, x):
        pyr = self.fpn(self.backbone(x[self.size]))
        return OrderedDict((l, p.mean(dim=(2, 3))) for l, p in zip(self.levels, pyr))


class DualStream(GroupedModel):
    """Two backbones, each on its own view, fused by concatenation.

    The two views come from ONE decoded image, so the streams differ only in the
    resolution they were resampled to — never in content, augmentation or JPEG
    generation. That is what makes the ablation a resolution measurement rather
    than an ensembling measurement.
    """

    def __init__(self, lo_tag: str, hi_tag: str, lo_size: int, hi_size: int,
                 pretrained: bool = True):
        super().__init__()
        self.views = tuple(sorted({lo_size, hi_size}))
        self.lo_size, self.hi_size = lo_size, hi_size
        self.lo = _timm_backbone(lo_tag, pretrained)
        self.hi = _timm_backbone(hi_tag, pretrained)
        self.head = FusionHead(OrderedDict([("lo", self.lo.num_features),
                                            ("hi", self.hi.num_features)]))

    def features(self, x):
        return OrderedDict([("lo", self.lo(x[self.lo_size])),
                            ("hi", self.hi(x[self.hi_size]))])


class ResNetViT(GroupedModel):
    """ResNet-50 + ViT-B/16, both branches on the SAME 768 view.

    Both streams see every pixel the round's best LVH model (resnet50_hires)
    saw, so any gain over that model is attributable to the second branch's
    attention, not to extra resolution. ViT-B/16 at 768 is a 48x48 grid of
    16 px patches (~3 mm per patch); every patch attends to every other in the
    first block, which is the property the AF x LVH finding motivates: the LVH
    head of resnet50_hires misses 10 of the 28 AF+LVH cases, and global
    attention over the fine view is the mechanism under test for recovering
    them.

    The ViT pos-embed is interpolated from its 24x24 (384px) pretrained grid
    to 48x48 at first forward - the checkpoint itself never trained at 768.
    That, plus the in21k pretraining asymmetry, is why a win here supports
    "ResNet-50 + in21k ViT at 768", not "the ViT architecture is better".
    """

    def __init__(self, size: int = C.HI_SIZE, pretrained: bool = True):
        super().__init__()
        self.views = (size,)
        self.size = size
        self.cnn = _timm_backbone("resnet50.tv_in1k", pretrained)
        self.vit = _vit_backbone("vit_base_patch16_384.augreg_in21k_ft_in1k",
                                 pretrained)
        self.head = FusionHead(OrderedDict([("cnn", self.cnn.num_features),
                                            ("vit", self.vit.num_features)]))

    def features(self, x):
        v = x[self.size]
        return OrderedDict([("cnn", self.cnn(v)), ("vit", self.vit(v))])


# ---------------------------------------------------------------- registry
#
# MICRO-BATCH IS NOT A FREE PARAMETER HERE. READ THIS BEFORE CHANGING ONE.
#
# Round 1's harness called micro_batch "a VRAM accommodation, not an
# experimental variable", on the grounds that gradient accumulation restores
# config.EFFECTIVE_BATCH = 32. That is true of the GRADIENT and false of
# BATCHNORM: BN normalises over whatever tensor it is handed, which is the
# MICRO-batch. Accumulation never touches it.
#
# Every backbone in this round contains BatchNorm (ResNet-50 everywhere,
# EfficientNet-B3 in one). So a run at micro 4 and a run at micro 16 differ in
# BN statistics - noisier estimates, different implicit regularisation - on top
# of whatever they were meant to differ in. For the control pair
# resnet50_base vs resnet50_hires, that would mean "only the resolution
# differs" is simply not true.
#
# THE DEFAULTS BELOW ARE THE LARGEST THAT FIT A 16 GB T4, so they are NOT
# uniform and the control pair is NOT clean on a T4. On a 24 GB card (L4, A100)
# pass one micro-batch to every run:
#
#     MICRO_BATCH=8 ./run_hybrids.sh <data> <out>
#
# compare.py warns when compared runs disagree on micro_batch. Prefer a uniform
# value even if it means a smaller one - a slower clean comparison beats a fast
# confounded one.
REGISTRY: dict[str, dict] = {
    "resnet50_base": {
        "build": lambda p: SingleStream("resnet50.tv_in1k", C.BASE_SIZE, p),
        "views": (C.BASE_SIZE,), "micro_batch": 16, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k"], "order": 1,
        "why": "in-family baseline at round-1 resolution with the round-2 head",
    },
    "resnet50_hires": {
        "build": lambda p: SingleStream("resnet50.tv_in1k", C.HI_SIZE, p),
        "views": (C.HI_SIZE,), "micro_batch": 4, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k"], "order": 2,
        "why": "THE CONTROL - resolution alone, no architectural change",
    },
    "resnet50_cbam": {
        "build": lambda p: ResNetCBAM(C.BASE_SIZE, p),
        "views": (C.BASE_SIZE,), "micro_batch": 16, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k"], "order": 3,
        "why": "attention reweighting, no extra pixels",
    },
    # Round A tested CBAM and FPN at 384 ONLY, and both lost to a plain ResNet-50
    # at 768. That is not a verdict on attention or on multi-scale fusion - it is
    # a verdict on them AT A RESOLUTION WHERE THE DETAIL THEY EXIST TO EXPLOIT HAS
    # ALREADY BEEN THROWN AWAY. At 384 an ECG sheet is 1.73 px/mm vertically, so a
    # 0.5 mV LVH voltage difference is under 9 px; attention cannot reweight
    # information the resampling deleted, and an FPN's stride-4 level is only
    # finer than its stride-8 level if the pixels underneath differ.
    #
    # These two rerun them at 768, where that detail survives. They are the only
    # way to tell "attention does not help" apart from "attention did not help at
    # 384", and without them the round's conclusion rests on a confound of its own.
    "resnet50_cbam_hires": {
        "build": lambda p: ResNetCBAM(C.HI_SIZE, p),
        "views": (C.HI_SIZE,), "micro_batch": 4, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k"], "order": 4,
        "why": "does attention become useful once fine detail is preserved? "
               "pairs with resnet50_cbam - only the resolution differs",
    },
    "resnet50_fpn": {
        "build": lambda p: ResNetFPN(C.BASE_SIZE, p),
        "views": (C.BASE_SIZE,), "micro_batch": 8, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k"], "order": 5,
        "why": "multi-scale fusion, restores stride-4 detail without extra pixels",
    },
    "resnet50_fpn_hires": {
        # micro 2 by default, not 4: P2 at a 768 input is 192x192x256 and it is
        # retained for the backward pass. Raise it on a 24 GB card - preflight
        # measures the real peak before the batch starts.
        "build": lambda p: ResNetFPN(C.HI_SIZE, p),
        "views": (C.HI_SIZE,), "micro_batch": 2, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k"], "order": 6,
        "why": "does multi-scale extraction benefit from higher-resolution input? "
               "pairs with resnet50_fpn - only the resolution differs",
    },
    "resnet50_dual": {
        "build": lambda p: DualStream("resnet50.tv_in1k", "resnet50.tv_in1k",
                                      C.BASE_SIZE, C.HI_SIZE, p),
        "views": (C.BASE_SIZE, C.HI_SIZE), "micro_batch": 4, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k", "resnet50.tv_in1k"], "order": 7,
        "why": "headline hybrid - global sheet + 2x-resolution detail stream",
    },
    "resnet50_vit": {
        "build": lambda p: ResNetViT(C.HI_SIZE, p),
        "views": (C.HI_SIZE,), "micro_batch": 4,
        "pretrain": "ImageNet-1k + ImageNet-21k->1k  (ASYMMETRIC - see docstring)",
        "tags": ["resnet50.tv_in1k", "vit_base_patch16_384.augreg_in21k_ft_in1k"],
        "order": 8,
        # a ViT is LayerNorm and attention; NHWC buys nothing and can add
        # transposes around the patch embedding
        "channels_last": False,
        "why": "CNN detail + global attention, both at 768 - does the enriched "
               "representation preserve LVH when AF is present? (AF x LVH "
               "subgroup finding: hires missed 10 of 28 AF+LVH cases)",
    },
    "resnet50_effb3_dual": {
        "build": lambda p: DualStream("resnet50.tv_in1k", "efficientnet_b3.ra2_in1k",
                                      C.BASE_SIZE, C.HI_SIZE, p),
        "views": (C.BASE_SIZE, C.HI_SIZE), "micro_batch": 4, "pretrain": "ImageNet-1k",
        "tags": ["resnet50.tv_in1k", "efficientnet_b3.ra2_in1k"], "order": 9,
        "why": "heterogeneous dual-res; LOWEST PRIORITY - round-1 EffNet-B3 carried "
               "an unresolved LR confound",
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
    info["channels_last"] = bool(C.CHANNELS_LAST and spec.get("channels_last", True))
    info["views"] = list(model.views)
    info["groups"] = model.groups
    info["params_m_actual"] = sum(p.numel() for p in model.parameters()) / 1e6
    info["head_in_features"] = model.head.in_features
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
            f"micro_batch ({mb}) or the effective batch differs between models "
            f"and the comparison stops being controlled."
        )
    return C.EFFECTIVE_BATCH // mb
