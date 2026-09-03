"""
The twenty hybrid architectures, built through one factory.

Three backbones - the top three of the nine in `cardiosentry_baselines_v2` - each
carrying the SAME ladder of attention and fusion components:

    VGG-16              CBAM  ECA  MSA  LT  LT+concat  LT+gated  MSA+LT+gated
    ResNet-50           CBAM  ECA  MSA  LT  LT+concat  LT+gated  MSA+LT+gated
    EfficientNetV2-S    CBAM  ECA  MSA  LT  LT+concat  LT+gated  MSA+LT+gated

The grid is COMPLETE: 3 backbones x 7 rungs = 21 runs, every cell filled. That
matters more than it looks. With one cell missing, "CBAM is the weakest attention
module" would have rested on two backbones and been an observation; across all
three it is a result that either replicates or does not, and replication across
backbones is the only evidence here that separates a real effect from one seed's
noise (see confound 1).

Read `hybrids.py` for what each component is and why it is built the way it is.
This file is the registry, the per-model VRAM accommodation, and the factory.

================================================================================
THESE RUNS COMPARE DIRECTLY AGAINST baselines_v2 - THAT IS THE POINT
================================================================================

`config.py` is byte-identical to baselines_v2's, so `fingerprint()` is identical,
so `compare.py` will put a hybrid run and a baseline run in ONE table without
being forced to. That is not a convenience; it is what makes "+0.006 macro AUPRC
over the ResNet-50 baseline" a statement about the component rather than about
two experiments that happened to be run near each other.

For it to hold, the baseline has to be genuinely nested inside the hybrid, and it
is: with no attention, no transformer and no fusion, `HybridModel` reduces to

    backbone.forward_features -> backbone.forward_head(pre_logits=True) -> Linear

which is exactly what `timm.create_model(tag, num_classes=4)` builds and exactly
what baselines_v2 trained. `selftest.py` asserts the parameter counts match
rather than trusting this paragraph.

So there are no `*_base` runs in this package. The baseline for every row here
already exists in `runs/baselines_v2/<backbone>@nat768/`, and cell 8 of the
notebook copies those three run directories in so the comparison table has its
reference rows. Re-training them would cost four GPU hours to reproduce numbers
you already have.

================================================================================
WHAT IS AND IS NOT A CONFOUND HERE
================================================================================

1. ONE SEED PER CELL. Twenty-one runs is about thirty-one hours; at three seeds
   it would be four days, so nothing here estimates run-to-run variance and no
   delta carries a confidence interval. The complete 3x7 grid is what stands in
   for that: a component that helps on one backbone and not the other two is one
   draw, while a component that helps on all three is a pattern. Rank on
   agreement ACROSS the three ladders, never on the ordering within one.

2. MICRO-BATCH IS CONSTANT WITHIN A BACKBONE. Every variant of one backbone uses
   that backbone's baselines_v2 micro-batch, so accumulation steps are constant
   too and the effective batch of 32 never moves - within a family or between
   families. The components add parameters but almost no activation memory
   (the transformer works on a 24x32 grid at width 256), so the baseline value
   still fits. preflight.py measures the real peak and will say if it does not.

3. PRETRAINING IS UNIFORM HERE, unlike baselines_v2. All three backbones are
   supervised ImageNet-1k, and every added component is trained from scratch.
   The in21k asymmetry that qualified ViT-B/16's baseline result does not apply
   to any run in this package.

4. THE ADDED PARAMETERS ARE NOT UNIFORM, and cannot be. ECA adds 7 parameters to
   ResNet-50; the multi-scale block adds ~4.3 M; the transformer adds ~1.6 M plus
   a wider head. `result.json` records a per-component parameter breakdown and
   `compare.py` prints the "added over backbone" column, because "won by 0.004
   AUPRC for +0.03% parameters" and "won by 0.004 AUPRC for +25% parameters" are
   different findings.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C   # vendored: was `import config as C` (arch/README.md)
from . import hybrids as H   # vendored: was `import hybrids as H` (arch/README.md)

# ---------------------------------------------------------------- backbones
#
# The three that placed top in baselines_v2. `micro_batch` is a VRAM
# accommodation, not an experimental variable: gradient accumulation restores
# config.EFFECTIVE_BATCH = 32 for every model, and steps_per_epoch (and therefore
# the whole cosine schedule) is 151 at any of 4 / 8 / 16.
#
# WHETHER IT IS SAFE TO CHANGE ONE DEPENDS ON BATCHNORM, AND ONLY ON BATCHNORM.
#
# BatchNorm normalises over the MICRO-batch. Accumulation cannot restore that -
# it sums gradients, it does not merge batch statistics - so changing
# micro_batch on a BN network changes what the network computes, not just how
# fast it computes it. Every hybrid here is scored as a delta against the
# baselines_v2 run of its backbone, so a hybrid trained at a different
# micro_batch than that baseline is measuring the normalisation as well as the
# component.
#
#     vgg16              0 BatchNorm layers   -> FREE to change
#     resnet50          53 BatchNorm layers   -> LOCKED to the baseline's 8
#     efficientnetv2_s 110 BatchNorm layers   -> LOCKED to the baseline's 8
#
# preflight.py counts them on the real model and refuses to suggest raising a
# BN backbone, so this is enforced rather than remembered.
#
# On a 16 GB T4, halve every value below; that is a legitimate trade for
# resnet50 and efficientnetv2_s ONLY if you also re-train their baselines here,
# because it breaks the nesting the delta depends on.
BACKBONES: dict[str, dict] = {
    "vgg16": {
        "head_modules": ["pre_logits", "head"],   # pre_logits is a 4096-d ConvMlp: 120 M parameters, the bulk of VGG-16
        "timm": "vgg16.tv_in1k",
        # BatchNorm layers in the backbone. Recorded as data because it decides
        # whether micro_batch is yours to change, and run_all.sh has to know that
        # without paying to construct a model. selftest.py --build asserts this
        # against the real network rather than trusting the number.
        "batchnorm": 0,
        # 16, not the 4 baselines_v2 used. That 4 was never measured at this
        # geometry - it was scaled by pixel count from the v1_768 runs - and
        # preflight on an L4 (22 GB) at nat768 measures a real peak of 4.8 GB at
        # micro_batch 4: 2.0 GB fixed (138 M params x 16 bytes of weights, grads
        # and AdamW state) plus 0.69 GB per sample. Four samples in a 22 GB card.
        #
        # `vgg16.tv_in1k` is the plain VGG - ZERO BatchNorm layers - so this is
        # the one backbone of the three where raising it is genuinely free: the
        # accumulated gradient is the same sum, steps_per_epoch stays 151, and
        # the cosine schedule is untouched. It buys back wall-clock on the
        # longest family. Re-run preflight to confirm the real peak at 16 before
        # a long batch; 8 (~7.5 GB) is the conservative fallback.
        "micro_batch": 16,
        "pretrain": "ImageNet-1k (supervised)",
        "params_m": 138.4,
        "note": "512-channel feature map, then timm's 4096-d ConvMlp pre-logits. "
                "The widest head of the three, and the reason its runs are the "
                "slowest in this package.",
    },
    "resnet50": {
        "head_modules": ["global_pool", "fc"],   # parameter-free pooling + the Linear
        "timm": "resnet50.tv_in1k",
        # BatchNorm layers in the backbone. Recorded as data because it decides
        # whether micro_batch is yours to change, and run_all.sh has to know that
        # without paying to construct a model. selftest.py --build asserts this
        # against the real network rather than trusting the number.
        "batchnorm": 53,
        "micro_batch": 8,
        "pretrain": "ImageNet-1k (supervised)",
        "params_m": 25.6,
        "note": "the reference. Won v1 at 384, tied v1_768, and every hybrid "
                "here is measured against its baselines_v2 row.",
    },
    "efficientnetv2_s": {
        "head_modules": ["global_pool", "classifier"],   # conv_head is NOT here - it lives in forward_features and both streams need it
        "timm": "tf_efficientnetv2_s.in1k",
        # BatchNorm layers in the backbone. Recorded as data because it decides
        # whether micro_batch is yours to change, and run_all.sh has to know that
        # without paying to construct a model. selftest.py --build asserts this
        # against the real network rather than trusting the number.
        "batchnorm": 110,
        "micro_batch": 8,
        "pretrain": "ImageNet-1k (supervised)",
        "params_m": 21.5,
        "note": "fused-MBConv early stages. Cheapest of the three per epoch.",
    },
}

# ---------------------------------------------------------------- the ladder
#
# (suffix, attention, cnn stream, transformer stream, fusion, human label),
# applied to each backbone. The order is the order they are trained in: cheapest
# and most interpretable first, so a night that gets cut short leaves the
# attention block finished rather than three transformers half-done.
#
# `cnn` is False for exactly one rung - the LT-only one - because there the
# transformer REPLACES the backbone's pooling rather than joining it. That is
# what makes `lt_concat` the control that tells a weak `lt` result apart from a
# discarded-pooling result; see the ladder section of hybrids.py.
LADDER: list[tuple[str, str | None, bool, bool, str, str]] = [
    ("cbam",         "cbam", True,  False, "none",   "Convolutional Block Attention Module"),
    ("eca",          "eca",  True,  False, "none",   "Efficient Channel Attention"),
    ("msa",          "msa",  True,  False, "none",   "Multi-scale (strip) attention"),
    ("lt",           None,   False, True,  "none",   "Lightweight Transformer, replacing the pooling"),
    ("lt_concat",    None,   True,  True,  "concat", "Lightweight Transformer + concatenation fusion"),
    ("lt_gated",     None,   True,  True,  "gated",  "Lightweight Transformer + gated fusion"),
    ("msa_lt_gated", "msa",  True,  True,  "gated",  "Multi-scale attention + Lightweight Transformer + gated fusion"),
]

# Nothing is held back: every backbone gets every rung. Kept as an explicit
# (empty) exclusion set rather than deleted, because the RUN_ORDER /
# RUN_ORDER_OPTIONAL split below is how you would stage a partial grid again -
# put a (backbone, suffix) pair in here and it drops out of the default batch
# while staying trainable by name.
_SKIP: set[tuple[str, str]] = set()


def _build_registry() -> dict[str, dict]:
    reg: dict[str, dict] = {}
    for bname, b in BACKBONES.items():
        for suffix, attn, use_cnn, use_lt, fusion, label in LADDER:
            name = f"{bname}_{suffix}"
            reg[name] = {
                "backbone": bname,
                "timm": b["timm"],
                "attn": attn,
                "use_cnn": use_cnn,
                "use_lt": use_lt,
                "fusion": fusion,
                "micro_batch": b["micro_batch"],
                "pretrain": b["pretrain"],
                "family": bname,
                "params_m": b["params_m"],
                "label": f"{bname} + {label}",
                "in_default_batch": (bname, suffix) not in _SKIP,
            }
    return reg


REGISTRY = _build_registry()
MODEL_NAMES = list(REGISTRY)

# The batch run_all.sh executes by default: all 21, family by family, ladder
# order within a family. ResNet-50 first because it is the project's reference
# point and the cheapest per epoch of the three; VGG-16 last because it is the
# most expensive, so a batch cut short by a disconnect leaves two COMPLETE
# families rather than three partial ones.
_FAMILY_ORDER = ["resnet50", "efficientnetv2_s", "vgg16"]
RUN_ORDER = [n for f in _FAMILY_ORDER for n in MODEL_NAMES
             if REGISTRY[n]["backbone"] == f and REGISTRY[n]["in_default_batch"]]
RUN_ORDER_OPTIONAL = [n for n in MODEL_NAMES if not REGISTRY[n]["in_default_batch"]]

# The baselines_v2 run directories these hybrids are measured against. Cell 8 of
# the notebook copies them in so compare.py has its reference rows.
BASELINE_RUNS = {b: f"{b}@{C.DIM_NAME}" for b in BACKBONES}


# --------------------------------------------------------------- geometry
def input_geometry(name: str) -> dict:
    """What this model is actually fed. Kept for interface parity with
    baselines_v2, where ViT resampled and Swin-V2 padded. Nothing in this
    package adapts the input: all three backbones are fully convolutional and
    the transformer tokenises whatever grid they produce."""
    _ = REGISTRY[name]
    return {
        "requested": tuple(C.IMG_SIZE),
        "fed": tuple(C.IMG_SIZE),
        "pad_h": 0, "pad_w": 0,
        "pixel_overhead": 1.0,
        "token_grid": (C.IMG_H // H.BACKBONE_STRIDE, C.IMG_W // H.BACKBONE_STRIDE),
        "adaptation": "none - fully convolutional backbone, any input size",
    }


# --------------------------------------------------------------- factory
def build(name: str, pretrained: bool = True, timm_tag: str | None = None):
    """Return (model, info). Head is 4 independent logits - sigmoid is in the loss."""
    import timm

    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"unknown model '{name}'. choose from:\n  "
                       + "\n  ".join(MODEL_NAMES))
    tag = timm_tag or spec["timm"]

    try:
        # num_classes=0 keeps timm's own pooling stage (VGG's 4096-d ConvMlp,
        # ResNet's GAP, EfficientNetV2's conv_head) and drops only the final
        # Linear, which HybridModel replaces with one sized to the fused vector.
        backbone = timm.create_model(tag, pretrained=pretrained, num_classes=0)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not create '{tag}'.\n"
            f"the pretrained tag may have changed in your timm version "
            f"({getattr(timm, '__version__', '?')}).\n"
            f"list valid tags with:  python preflight.py --list-tags {name}\n"
            f"original error: {exc}"
        ) from exc

    feat_ch = backbone.num_features                     # feature-map channels
    pooled_dim = getattr(backbone, "head_hidden_size", None) or feat_ch

    # On the LT-only rung the transformer REPLACES the backbone's pooling, so
    # the pooling stage is not merely unused - it is not part of the model. Left
    # in place it would still be constructed, counted in params_total and
    # size_mb, and written into every checkpoint: for VGG-16 that is a 4096-d
    # ConvMlp, 120 M dead parameters and ~480 MB of Drive per run, and a
    # "parameters" column that describes a network the run does not contain.
    # `head_modules` is declared per backbone rather than guessed, because
    # EfficientNetV2's conv_head looks like a head and is not one - it lives
    # inside forward_features and BOTH streams read its output.
    if not spec["use_cnn"]:
        for attr in BACKBONES[spec["backbone"]]["head_modules"]:
            if hasattr(backbone, attr):
                setattr(backbone, attr, nn.Identity())
        pooled_dim = feat_ch

    model = H.HybridModel(backbone, feat_ch=feat_ch, pooled_dim=pooled_dim,
                          attn=spec["attn"], use_cnn=spec["use_cnn"],
                          use_lt=spec["use_lt"], fusion=spec["fusion"])

    info = {k: v for k, v in spec.items() if k not in ("in_default_batch",)}
    info["resolved_tag"] = tag
    info["params_m_actual"] = sum(p.numel() for p in model.parameters()) / 1e6
    info["timm_version"] = getattr(timm, "__version__", "?")
    info["input_geometry"] = input_geometry(name)
    info["feature_channels"] = int(feat_ch)
    info["pooled_dim"] = int(pooled_dim)
    info["component_params"] = model.component_params()
    info["hybrid"] = model.stats()
    return model, info


def hybrid_stats(model) -> dict:
    """Learned gammas and gate means, for result.json. Empty for a plain model."""
    if hasattr(model, "stats"):
        return {**model.stats(), "component_params": model.component_params()}
    return {}


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


if __name__ == "__main__":
    print(f"geometry: {C.geometry_str()}")
    gh, gw = C.IMG_H // H.BACKBONE_STRIDE, C.IMG_W // H.BACKBONE_STRIDE
    print(f"token grid for the transformer: {gh}x{gw} = {gh*gw} tokens\n")
    print(f"{'model':32}{'micro':>7}{'accum':>7}  {'in batch':>9}  description")
    print("-" * 118)
    for f in _FAMILY_ORDER:
        for n in MODEL_NAMES:
            s = REGISTRY[n]
            if s["backbone"] != f:
                continue
            print(f"{n:32}{micro_batch(n):7d}{accumulation_steps(n):7d}"
                  f"  {'yes' if s['in_default_batch'] else 'OPTIONAL':>9}  {s['label']}")
    print()
    print(f"default batch: {len(RUN_ORDER)} runs, in this order:")
    for i, n in enumerate(RUN_ORDER, 1):
        print(f"  {i:2d}. {n}")
    if RUN_ORDER_OPTIONAL:
        print(f"\nregistered but NOT in the default batch: "
              f"{', '.join(RUN_ORDER_OPTIONAL)}")
    print(f"\nbaseline rows come from baselines_v2: "
          f"{', '.join(sorted(BASELINE_RUNS.values()))}")
