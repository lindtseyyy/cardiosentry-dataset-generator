"""
Five CBAM hybrids, built through one factory.

    DenseNet-121      + CBAM
    EfficientNet-B3   + CBAM
    ConvNeXt-Base     + CBAM
    ViT-B/16          + CBAM
    SwinV2-Base       + CBAM

ONE component, FIVE backbones. That is the opposite shape from
`cardiosentry_hybrids_v2`, which ran seven components over three backbones, and
it answers the question v2 cannot: **does CBAM's effect depend on the backbone's
inductive bias, or on the backbone at all?** v2 can only say "CBAM helped/hurt
ResNet-50, EfficientNetV2-S and VGG-16" - three convolutional backbones with
stride-32 NCHW feature maps and very similar receptive-field structure. Adding a
densely-connected CNN, a compound-scaled CNN, a modernised CNN, a plain ViT and a
hierarchical windowed transformer turns that into eight backbones spanning three
genuinely different feature formats.

The five here are exactly the five `cardiosentry_baselines_v2` backbones that
`hybrids_v2` did NOT cover, minus ConvNeXt-V2 (whose FCMAE pretraining is a
second asymmetry nobody needs on top of ViT's). So every run in this package has
a baseline row already sitting in `runs/baselines_v2/<backbone>@nat768/`, trained
under a byte-identical `config.py`, and every result is reported as a DELTA.

================================================================================
THE DESIGN RULE IS UNCHANGED FROM v2
================================================================================

    hybrid(backbone, attn=None)  IS the baseline.

    x -> backbone.forward_features(x)   -> F
      -> FORMAT.to_map(F)               -> (B, C, gh, gw)   NCHW, whatever F was
      -> CBAM(.)                        -> same shape, identity at step 0
      -> FORMAT.from_map(.)             -> F' in F's own layout
      -> backbone.forward_head(F', pre_logits=True)
      -> nn.Linear(D, 4)

With `attn=None` the wrapper is parameter-for-parameter the model baselines_v2
trained, so the subtraction "hybrid minus baseline" measures CBAM and nothing
else. `selftest.py --build` asserts the parameter counts rather than asking you
to believe it.

================================================================================
WHAT IS NEW HERE, AND WHY IT WAS NEEDED
================================================================================

v2's three backbones all hand `forward_features` back an NCHW feature map at
stride 32, so CBAM could be dropped straight in. Two of the five below do not:

    densenet121       (B, 1024, H/32, W/32)      NCHW
    efficientnet_b3   (B, 1536, H/32, W/32)      NCHW
    convnext_base     (B, 1024, H/32, W/32)      NCHW
    vit_b16           (B, 1 + H/16*W/16, 768)    TOKENS, with a class token
    swinv2_base       (B, H/32, W/32, 1024)      NHWC

so `hybrids.py` grew a small `FeatureFormat` layer that converts each of those to
NCHW, runs the attention, and converts back. Two things about it matter:

  * ViT's PREFIX (class) token is split off and passed through untouched. CBAM's
    spatial gate is a 7x7 convolution over a GRID; a class token has no position
    on that grid, and folding it in would either corrupt it or force a reshape
    that scrambles which token is which. timm's ViT head pools the class token,
    so corrupting it would quietly damage exactly the vector the classifier reads.
  * the round trip is EXACT. `to_map` then `from_map` returns the same values, so
    "CBAM is the identity at step 0" is still true bit for bit on all five, which
    `selftest.py` checks with `torch.equal` on real backbones.

ViT is stride 16, not 32, so BACKBONE_STRIDE is now per backbone. At nat768 that
gives ViT a 48x64 = 3072-cell grid where the others get 24x32 = 768 - CBAM's
spatial gate is therefore looking at ~4.5 mm cells on ViT and ~9 mm cells on the
rest. That is a property of the backbones being compared, not a flaw, but it is
the reason a CBAM win on ViT is not automatically the same finding as a CBAM win
on DenseNet.

================================================================================
WHAT IS AND IS NOT A CONFOUND HERE
================================================================================

1. ONE SEED PER CELL, as in v2. Five runs is not enough to estimate run-to-run
   variance. What stands in for it is the SHAPE of the result across five very
   different backbones: CBAM helping four of five is a pattern, CBAM helping one
   is a draw. A delta under ~0.01 macro AUPRC on ~660 test records is a tie.

2. PRETRAINING IS NOT UNIFORM - and unlike v2, that asymmetry is live here.
   ViT-B/16's usable timm checkpoint at 384 is in21k-pretrained then fine-tuned
   to in1k; the other four are supervised in1k. `pretrain` is recorded per model
   and printed by compare.py. It does NOT contaminate the delta - each hybrid is
   measured against a baseline with the SAME weights - but it does contaminate
   any cross-backbone statement about absolute scores.

3. MICRO-BATCH IS THE BASELINE'S, ALWAYS. Every value below is what
   baselines_v2 trained that backbone at, because the delta is only clean if the
   two runs computed the same thing. On the two BatchNorm backbones that is
   ENFORCED (see `batchnorm` below and check_micro_batch.py); on the other three
   it is merely the default, and raising it is genuinely free.

4. THE INPUT IS NOT PADDED AT nat768, BUT COULD BE ELSEWHERE. SwinV2's window
   attention needs H and W divisible by 256. nat768 is 768x1024 - both already
   multiples of 256 - so no padding happens and Swin sees exactly the same
   pixels as the other four. At sq384 or nat672 it would, and `input_geometry`
   reports the overhead so it lands in result.json instead of being forgotten.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import config as C
import hybrids as H

# ---------------------------------------------------------------- backbones
#
# `micro_batch` is a VRAM accommodation, not an experimental variable: gradient
# accumulation restores config.EFFECTIVE_BATCH = 32 for every model, so the
# effective batch, steps_per_epoch and the whole cosine schedule are identical
# whatever it is set to.
#
# WHETHER IT IS SAFE TO CHANGE ONE DEPENDS ON BATCHNORM, AND ONLY ON BATCHNORM.
#
# BatchNorm normalises over the MICRO-batch. Accumulation cannot restore that -
# it sums gradients, it does not merge batch statistics - so changing
# micro_batch on a BN network changes what the network COMPUTES. Every hybrid
# here is scored as a delta against the baselines_v2 run of its backbone, so a
# hybrid trained at a different micro_batch than that baseline is measuring the
# normalisation as well as CBAM, and no metric in the report would reveal it.
#
#     densenet121      121 BatchNorm layers  -> LOCKED to the baseline's 4
#     efficientnet_b3   78 BatchNorm layers  -> LOCKED to the baseline's 8
#     convnext_base      0 (LayerNorm)       -> free to change
#     vit_b16            0 (LayerNorm)       -> free to change
#     swinv2_base        0 (LayerNorm)       -> free to change
#
# The counts are measured, not remembered: `selftest.py --build` asserts each one
# against the real network, and check_micro_batch.py REFUSES an override on a BN
# backbone rather than warning about it.
#
# `feat_format` is what backbone.forward_features hands back, and it is the one
# field in this table that had to be added for v3. See hybrids.FORMATS.
BACKBONES: dict[str, dict] = {
    # ---------------------------------------------------------------- CNNs
    "densenet121": {
        "timm": "densenet121.ra_in1k",
        "feat_format": "nchw",
        "stride": 32,
        "feat_ch": 1024,
        "batchnorm": 121,
        "micro_batch": 4,
        "pretrain": "ImageNet-1k (supervised)",
        "params_m": 8.0,
        "note": "8 M parameters, but dense connectivity keeps every feature map "
                "alive - ACTIVATION memory, not parameters, is what caps its "
                "micro-batch at 4. The cheapest backbone here by weight and one "
                "of the more expensive by VRAM.",
    },
    "efficientnet_b3": {
        "timm": "efficientnet_b3.ra2_in1k",
        "feat_format": "nchw",
        "stride": 32,
        "feat_ch": 1536,
        "batchnorm": 78,
        "micro_batch": 8,
        "pretrain": "ImageNet-1k (supervised)",
        "params_m": 12.2,
        "note": "came last in v1 at 384 and recovered strongly at 768 "
                "(mAUPRC 0.857 -> 0.931) - a resolution-sensitive model, and so "
                "the one most likely to care about a spatial gate.",
    },
    "convnext_base": {
        "timm": "convnext_base.fb_in1k",
        "feat_format": "nchw",
        "stride": 32,
        "feat_ch": 1024,
        "batchnorm": 0,
        "micro_batch": 4,
        "pretrain": "ImageNet-1k (supervised)",
        "params_m": 88.6,
        "note": "second in v1, third in v1_768. 7x7 depthwise convolutions "
                "already give it a wide receptive field, which makes it the "
                "sharpest test of whether CBAM adds anything a modern CNN does "
                "not already have.",
    },

    # ------------------------------------------------------------ transformers
    "vit_b16": {
        "timm": "vit_base_patch16_384.augreg_in21k_ft_in1k",
        # (B, 1 + N, C): a class token followed by the patch grid. The class
        # token is split off and passed through - see hybrids.TokenFormat.
        "feat_format": "tokens",
        "stride": 16,
        "feat_ch": 768,
        "batchnorm": 0,
        "micro_batch": 2,
        "pass_img_size": True,     # pos-embed resampled 384x384 -> config.IMG_SIZE
        "pretrain": "ImageNet-21k -> 1k   (ASYMMETRIC - see the module docstring)",
        "params_m": 86.9,
        "note": "placed last in both v1 and v1_768 despite the extra pretraining "
                "data. Stride 16, so its token grid is 4x finer than everything "
                "else here and CBAM's spatial gate is looking at ~4.5 mm cells "
                "rather than ~9 mm.",
    },
    "swinv2_base": {
        "timm": "swinv2_base_window8_256.ms_in1k",
        # (B, H/32, W/32, C) - channels LAST. Permuted to NCHW for the attention
        # and permuted back, exactly.
        "feat_format": "nhwc",
        "stride": 32,
        "feat_ch": 1024,
        "batchnorm": 0,
        "micro_batch": 2,
        "pass_img_size": True,
        "size_multiple": 256,      # window 8 x patch 4 x 8 stages; no-op at nat768
        "pretrain": "ImageNet-1k (supervised)",
        "params_m": 87.9,
        "note": "hierarchical windowed attention. Needs H,W divisible by 256; "
                "nat768 (768x1024) already is, so nothing is padded and its "
                "pixels are identical to the other four's.",
    },
}

# ---------------------------------------------------------------- the ladder
#
# (suffix, attention, cnn stream, transformer stream, fusion, human label).
#
# ONE RUNG. This package is a single-component study across five backbones, so
# the ladder has exactly one entry and the "grid" is 5 x 1. The tuple shape is
# kept identical to hybrids_v2's so that hybrids.py, train.py, compare.py and
# ablate.py are shared code rather than forks - and so that adding ECA or MSA
# here later is one line, not a refactor.
LADDER: list[tuple[str, str | None, bool, bool, str, str]] = [
    ("cbam", "cbam", True, False, "none", "Convolutional Block Attention Module"),
]

# Nothing is held back. Kept as an explicit (empty) exclusion set because it is
# how you would stage a partial grid: put a (backbone, suffix) pair in here and
# it drops out of the default batch while staying trainable by name.
_SKIP: set[tuple[str, str]] = set()

# ---------------------------------------------------------------- the two groups
#
# Which notebook trains what. The split is on the FEATURE FORMAT, which is also
# the cost boundary: the three NCHW CNNs are 4-8 samples per micro-batch, the two
# transformers are 2, and the two groups are completely independent - nothing in
# one reads anything the other writes - so they run in two Colab sessions at the
# same time and the wall clock becomes the longer group rather than the sum.
#
# It is not an arbitrary halving. Each group is a self-contained claim: "CBAM
# across three convolutional backbones" and "CBAM across two attention
# backbones". A session that dies leaves one whole claim standing.
GROUPS: dict[str, list[str]] = {
    "cnn": ["densenet121", "efficientnet_b3", "convnext_base"],
    "transformer": ["vit_b16", "swinv2_base"],
}


def group_of(backbone: str) -> str:
    for g, members in GROUPS.items():
        if backbone in members:
            return g
    raise KeyError(f"backbone '{backbone}' is in no group - add it to GROUPS, or "
                   f"run_all.sh --group and the notebooks will not see it")


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
                "group": group_of(bname),
                "params_m": b["params_m"],
                "label": f"{bname} + {label}",
                "in_default_batch": (bname, suffix) not in _SKIP,
            }
    return reg


REGISTRY = _build_registry()
MODEL_NAMES = list(REGISTRY)

# The batch run_all.sh executes by default: all five, cheapest first WITHIN each
# group and CNNs before transformers, so a session cut short leaves finished runs
# rather than an expensive one half-done.
_BACKBONE_ORDER = ["densenet121", "efficientnet_b3", "convnext_base",
                   "swinv2_base", "vit_b16"]
RUN_ORDER = [n for b in _BACKBONE_ORDER for n in MODEL_NAMES
             if REGISTRY[n]["backbone"] == b and REGISTRY[n]["in_default_batch"]]
RUN_ORDER_OPTIONAL = [n for n in MODEL_NAMES if not REGISTRY[n]["in_default_batch"]]

# The baselines_v2 run directories these hybrids are measured against.
# import_baselines.py copies their result.json in, fingerprint-checked.
BASELINE_RUNS = {b: f"{b}@{C.DIM_NAME}" for b in BACKBONES}


def models_in_group(group: str) -> list[str]:
    if group not in GROUPS:
        raise KeyError(f"unknown group '{group}'. choose from: {sorted(GROUPS)}")
    return [n for n in RUN_ORDER if REGISTRY[n]["group"] == group]


# --------------------------------------------------------------- geometry
def pad_to_multiple(size: tuple[int, int], multiple: int | None) -> tuple[int, int]:
    """Round (H, W) UP to the next multiple. Identity when multiple is None."""
    if not multiple:
        return tuple(size)
    h, w = size
    return (-(-h // multiple) * multiple, -(-w // multiple) * multiple)


def input_geometry(name: str) -> dict:
    """What this model is ACTUALLY fed, after any per-backbone adaptation.

    Two of the five adapt, and both adaptations are reported rather than hidden,
    because they change what a latency or VRAM number means:

      ViT-B/16     position embeddings are baked in at 384x384 and timm
                   resamples them bicubically to (H/16, W/16) on load. Standard
                   fine-tune-at-a-new-resolution procedure; costs no pixels.
      SwinV2-Base  window attention needs H and W divisible by 256. At nat768
                   they already are, so NOTHING happens. At a geometry where
                   they are not, the input is REPLICATE-padded (never resized -
                   resizing would give Swin a different px/mm than the other
                   four, which is the one thing a controlled comparison cannot
                   allow) and `pixel_overhead` rises above 1.0.
    """
    spec = REGISTRY[name]
    b = BACKBONES[spec["backbone"]]
    padded = pad_to_multiple(C.IMG_SIZE, b.get("size_multiple"))
    stride = b["stride"]
    return {
        "requested": tuple(C.IMG_SIZE),
        "fed": padded,
        "pad_h": padded[0] - C.IMG_H,
        "pad_w": padded[1] - C.IMG_W,
        "pixel_overhead": (padded[0] * padded[1]) / (C.IMG_H * C.IMG_W),
        "token_grid": (padded[0] // stride, padded[1] // stride),
        "feat_format": b["feat_format"],
        "adaptation": _adaptation(spec["backbone"], b, padded),
    }


def _adaptation(backbone: str, b: dict, padded: tuple[int, int]) -> str:
    bits = []
    if b.get("pass_img_size") and b["timm"].startswith("vit_"):
        bits.append(f"pos-embed resampled 384x384 -> {padded[0]}x{padded[1]}")
    mult = b.get("size_multiple")
    if mult:
        if padded != tuple(C.IMG_SIZE):
            bits.append(f"replicate-padded {C.IMG_H}x{C.IMG_W} -> "
                        f"{padded[0]}x{padded[1]} "
                        f"({(padded[0]*padded[1])/(C.IMG_H*C.IMG_W):.2f}x pixels)")
        else:
            bits.append(f"window attention; {C.IMG_H}x{C.IMG_W} is already a "
                        f"multiple of {mult}, no padding needed")
    gh, gw = padded[0] // b["stride"], padded[1] // b["stride"]
    bits.append(f"stride {b['stride']} -> {gh}x{gw} = {gh*gw} cells, "
                f"{b['feat_format']} feature layout")
    return "; ".join(bits)


# --------------------------------------------------------------- factory
def build(name: str, pretrained: bool = True, timm_tag: str | None = None):
    """Return (model, info). Head is 4 independent sigmoid logits - the sigmoid
    lives in the loss."""
    import timm

    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"unknown model '{name}'. choose from:\n  "
                       + "\n  ".join(MODEL_NAMES))
    b = BACKBONES[spec["backbone"]]
    tag = timm_tag or b["timm"]
    geom = input_geometry(name)

    if not spec["use_cnn"]:
        # v2's LT-only rung deletes the backbone's pooling stage, which needs a
        # per-backbone list of head module names. Three of the five here do not
        # have one that is safe to blank (ViT's `global_pool` is a STRING, not a
        # module), and no rung in this package uses it - so refuse rather than
        # ship an untested list that would silently build the wrong network.
        raise NotImplementedError(
            f"'{name}' asks to drop the backbone's pooling stage. That is v2's "
            f"LT-only rung and it is not defined for the token / NHWC backbones "
            f"in this package - see the head_modules discussion in "
            f"cardiosentry_hybrids_v2/models.py before adding it.")

    # Only the two transformers take an explicit input size. Passing img_size to
    # a fully convolutional backbone is a no-op at best and an error at worst.
    kwargs = {"img_size": geom["fed"]} if b.get("pass_img_size") else {}

    try:
        # num_classes=0 keeps timm's own pooling stage and drops only the final
        # Linear, which HybridModel replaces with one sized to the fused vector.
        backbone = timm.create_model(tag, pretrained=pretrained, num_classes=0,
                                     **kwargs)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not create '{tag}' at {geom['fed']}.\n"
            f"the pretrained tag may have changed in your timm version "
            f"({getattr(timm, '__version__', '?')}).\n"
            f"list valid tags with:  python preflight.py --list-tags {name}\n"
            f"original error: {exc}"
        ) from exc

    feat_ch = backbone.num_features
    if feat_ch != b["feat_ch"]:
        raise RuntimeError(
            f"{name}: registry says {b['feat_ch']} feature channels, the real "
            f"model has {feat_ch}. CBAM's channel gate is sized from this, so a "
            f"stale number would build a different module than the one recorded.")
    pooled_dim = getattr(backbone, "head_hidden_size", None) or feat_ch

    fmt = H.make_format(b["feat_format"], channels=feat_ch,
                        grid=geom["token_grid"],
                        num_prefix_tokens=getattr(backbone, "num_prefix_tokens", 0))

    model = H.HybridModel(backbone, feat_ch=feat_ch, pooled_dim=pooled_dim,
                          attn=spec["attn"], use_cnn=spec["use_cnn"],
                          use_lt=spec["use_lt"], fusion=spec["fusion"],
                          fmt=fmt, grid=geom["token_grid"],
                          pad=(geom["pad_h"], geom["pad_w"]))

    info = {k: v for k, v in spec.items() if k not in ("in_default_batch",)}
    info["resolved_tag"] = tag
    info["params_m_actual"] = sum(p.numel() for p in model.parameters()) / 1e6
    info["timm_version"] = getattr(timm, "__version__", "?")
    info["input_geometry"] = geom
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
    print(f"geometry: {C.geometry_str()}\n")
    print(f"{'model':26}{'micro':>7}{'accum':>7}{'stride':>8}{'grid':>10}"
          f"{'format':>9}{'BN':>6}  pretraining")
    print("-" * 118)
    for b in _BACKBONE_ORDER:
        for n in MODEL_NAMES:
            s = REGISTRY[n]
            if s["backbone"] != b:
                continue
            g = input_geometry(n)
            bb = BACKBONES[b]
            gh, gw = g["token_grid"]
            print(f"{n:26}{micro_batch(n):7d}{accumulation_steps(n):7d}"
                  f"{bb['stride']:8d}{f'{gh}x{gw}':>10}{bb['feat_format']:>9}"
                  f"{bb['batchnorm']:6d}  {s['pretrain']}")
    print()
    for g, members in GROUPS.items():
        runs = models_in_group(g)
        print(f"group {g:12} {len(runs)} run(s): {', '.join(runs)}")
    print()
    print(f"default batch: {len(RUN_ORDER)} runs, in this order:")
    for i, n in enumerate(RUN_ORDER, 1):
        print(f"  {i:2d}. {n:26} {REGISTRY[n]['label']}")
    if RUN_ORDER_OPTIONAL:
        print(f"\nregistered but NOT in the default batch: "
              f"{', '.join(RUN_ORDER_OPTIONAL)}")
    print(f"\nbaseline rows come from baselines_v2: "
          f"{', '.join(sorted(BASELINE_RUNS.values()))}")
