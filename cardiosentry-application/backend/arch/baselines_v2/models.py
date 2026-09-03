"""
The nine architectures, built through one factory so nothing but the backbone
differs between runs.

    VGG-16            ResNet-50         DenseNet-121
    EfficientNet-B3   EfficientNetV2-S
    ConvNeXt-Base     ConvNeXt-V2-Base
    ViT-B/16          Swin Transformer V2-Base

================================================================================
TWO ASYMMETRIES YOU MUST DECLARE IN THE WRITEUP
================================================================================

1. PRETRAINING IS NOT UNIFORM.

   A controlled comparison wants every backbone to start from the same
   pretraining corpus. Seven of these have clean ImageNet-1k weights. Two do not,
   for reasons that are properties of the model families rather than of this
   harness:

     ViT-B/16       no in1k-only checkpoint exists at 384 in timm that is worth
                    using; the usable one is in21k pretrained then fine-tuned to
                    1k. (`vit_base_patch16_384.augreg_in1k` exists but is the
                    weaker of the pair and is not what anyone would cite.)
     ConvNeXt-V2    `convnextv2_base.fcmae_ft_in1k` IS in1k-finetuned, but the
                    FCMAE pretraining stage is self-supervised on in1k rather
                    than supervised. Different recipe, same corpus.

   REGISTRY records `pretrain` for every model and compare.py prints the column.
   If ViT wins, the honest reading is "ViT-B/16 with in21k pretraining beat these
   CNNs with in1k pretraining" - a real and publishable finding, but not the same
   claim as "ViT is the better architecture".

2. THE INPUT GEOMETRY IS NOT EQUALLY FREE FOR EVERY MODEL.

   config.IMG_SIZE is a rectangle. Six of the nine are fully convolutional and
   accept any (H, W) with no adaptation at all. Three need help:

     ViT-B/16       position embeddings are baked in at 384x384. timm resamples
                    them bicubically to (H/16, W/16) on load. This is the
                    standard fine-tune-at-a-new-resolution procedure and is
                    exactly what v1_768 already did for 768.

     Swin-V2-Base   window attention needs each stage's feature map to be
                    divisible by the window size. With window 8, that means H
                    and W must both be MULTIPLES OF 256, which none of the
                    aspect-corrected geometries are. Rather than distort the
                    geometry to suit one model - which would break the very
                    comparison being run - the wrapper below REPLICATE-PADS the
                    input to the next legal multiple and leaves the ECG itself
                    untouched at its true scale.

                    Padding, not resizing, is the right fix here: it costs
                    compute but preserves geometry, so Swin sees the waveform at
                    the same px/mm as every other model. The overhead is recorded
                    in result.json (`input_pad`) and printed by preflight and
                    compare, because it makes Swin's latency and VRAM figures not
                    directly comparable to the others'. Read the cost table with
                    that column in view.

                    If you would rather have no padding at all, swap the registry
                    entry to `swinv2_cr_small_ns_224.sw_in1k` - timm's SwinV2-CR
                    re-implementation takes arbitrary sizes natively. It is a
                    different (small, ~50 M) model with unofficial weights, so it
                    is not the default.

     ConvNeXt-V2    fully convolutional, no adaptation - listed here only because
                    its GRN layers make people assume otherwise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C   # vendored: was `import config as C` (arch/README.md)

# ---------------------------------------------------------------------- registry
#
# micro_batch is the only per-model knob, and it is a VRAM accommodation, not an
# experimental variable: gradient accumulation restores the same EFFECTIVE batch
# (config.EFFECTIVE_BATCH) for every model, so the optimisation trajectory is
# unchanged. It must divide EFFECTIVE_BATCH exactly or accumulation_steps raises.
#
# The values below are anchored to the measured v1_768 runs on a Colab L4
# (22.5 GB) at 768x768 = 0.59 Mpx, then scaled by pixel count for the default
# geometry (crop608 = 0.62 Mpx, i.e. ~1.06x). They are a STARTING POINT:
# preflight.py measures the real peak on your actual GPU and prints the largest
# micro_batch that still fits. Raising it is a pure speedup - accumulation steps
# fall to compensate and the effective batch of 32 never moves.
#
# On a 16 GB T4, halve every value below. On an A100 40/80 GB, run preflight and
# take its suggestions.
REGISTRY: dict[str, dict] = {
    # ---------------------------------------------------------------- CNNs
    "vgg16": {
        "timm": "vgg16.tv_in1k",
        "micro_batch": 4,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "cnn",
        "params_m": 138.4,
        "note": "the 2014 baseline. Included because reviewers ask for it.",
    },
    "resnet50": {
        "timm": "resnet50.tv_in1k",
        "micro_batch": 8,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "cnn",
        "params_m": 25.6,
        "note": "won v1 at 384 and tied v1_768. The one to beat.",
    },
    "densenet121": {
        "timm": "densenet121.ra_in1k",
        "micro_batch": 4,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "cnn",
        "params_m": 8.0,
        "note": "8 M params but dense connectivity keeps every feature map "
                "alive - activation memory, not parameters, is what limits it.",
    },
    "efficientnet_b3": {
        "timm": "efficientnet_b3.ra2_in1k",
        "micro_batch": 8,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "cnn",
        "params_m": 12.2,
        "note": "came last in v1 at 384, recovered strongly at 768 "
                "(mAUPRC 0.857 -> 0.931). A resolution-sensitive model.",
    },
    "efficientnetv2_s": {
        "timm": "tf_efficientnetv2_s.in1k",
        "micro_batch": 8,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "cnn",
        "params_m": 21.5,
        "note": "fused-MBConv early stages; trains faster than B3 at "
                "comparable accuracy.",
    },
    "convnext_base": {
        "timm": "convnext_base.fb_in1k",
        "micro_batch": 4,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "cnn",
        "params_m": 88.6,
        "note": "second in v1, third in v1_768.",
    },
    "convnextv2_base": {
        "timm": "convnextv2_base.fcmae_ft_in1k",
        "micro_batch": 4,
        "pretrain": "ImageNet-1k (FCMAE self-supervised -> in1k fine-tune)",
        "family": "cnn",
        "params_m": 88.7,
        "note": "GRN replaces LayerScale. Same corpus as convnext_base, "
                "different recipe - see asymmetry 1.",
    },

    # ------------------------------------------------------------ transformers
    "vit_b16": {
        "timm": "vit_base_patch16_384.augreg_in21k_ft_in1k",
        "micro_batch": 2,
        "pretrain": "ImageNet-21k -> 1k   (ASYMMETRIC - see module docstring)",
        "family": "transformer",
        "params_m": 86.9,
        "pass_img_size": True,        # pos-embed resampled 384 -> config.IMG_SIZE
        "note": "placed last in both v1 and v1_768 despite the extra "
                "pretraining data.",
    },
    "swinv2_base": {
        "timm": "swinv2_base_window8_256.ms_in1k",
        "micro_batch": 2,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "transformer",
        "params_m": 87.9,
        "pass_img_size": True,
        "size_multiple": 256,         # window 8 x 4 (patch) x 8 (stages) - see below
        "note": "hierarchical windowed attention. Needs H,W divisible by 256; "
                "the wrapper pads rather than resizes - see asymmetry 2.",
    },
}

MODEL_NAMES = list(REGISTRY)

# An alternative Swin entry for anyone who would rather not pay the padding.
# Not in REGISTRY by default: different scale, unofficial weights.
ALTERNATES = {
    "swinv2_cr_small": {
        "timm": "swinv2_cr_small_ns_224.sw_in1k",
        "micro_batch": 4,
        "pretrain": "ImageNet-1k (supervised)",
        "family": "transformer",
        "params_m": 49.7,
        "pass_img_size": True,
        "note": "timm's SwinV2-CR re-implementation. Accepts ANY (H,W) natively, "
                "so it needs no padding - but it is Small, not Base, and the "
                "weights are timm's rather than Microsoft's.",
    },
}


# ------------------------------------------------------------------- padding
def pad_to_multiple(size: tuple[int, int], multiple: int | None) -> tuple[int, int]:
    """Round (H, W) UP to the next multiple. Identity when multiple is None."""
    if not multiple:
        return tuple(size)
    h, w = size
    return (-(-h // multiple) * multiple, -(-w // multiple) * multiple)


class PadToMultiple(nn.Module):
    """Replicate-pad the input so a window-attention backbone will accept it.

    WHY REPLICATE AND NOT ZEROS: a zero pad is BLACK, and black is the colour of
    ECG ink. A band of black down the right edge of every sheet is a strong,
    perfectly label-independent feature, but it also sits in the same intensity
    range the model uses to find the trace, and at the sheet border it fabricates
    a hard edge that no real photograph contains. Replicating the last row and
    column extends the paper (and its grid) instead, which is closer to what the
    scanner would have seen and carries no new intensity.

    WHY PAD AND NOT RESIZE: resizing to a Swin-legal size would give Swin a
    different px/mm than the other eight models, which is precisely the variable
    this study is measuring. Padding keeps the waveform at its true scale and
    pays for it in compute instead. The overhead is reported, not hidden.
    """

    def __init__(self, model: nn.Module, size: tuple[int, int], multiple: int):
        super().__init__()
        self.model = model
        self.size = tuple(size)
        self.padded = pad_to_multiple(size, multiple)
        self.pad_h = self.padded[0] - self.size[0]
        self.pad_w = self.padded[1] - self.size[1]
        self.overhead = (self.padded[0] * self.padded[1]) / (self.size[0] * self.size[1])

    def forward(self, x):
        if self.pad_h or self.pad_w:
            # (left, right, top, bottom) - pad only right and bottom, so the
            # sheet stays anchored at the top-left corner it was rendered in.
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="replicate")
        return self.model(x)

    # so engine.model_complexity / state_dict / named_parameters see through it
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["model"], name)


def input_geometry(name: str) -> dict:
    """What this model will ACTUALLY be fed, after any per-model adaptation."""
    spec = REGISTRY.get(name) or ALTERNATES[name]
    padded = pad_to_multiple(C.IMG_SIZE, spec.get("size_multiple"))
    return {
        "requested": tuple(C.IMG_SIZE),
        "fed": padded,
        "pad_h": padded[0] - C.IMG_H,
        "pad_w": padded[1] - C.IMG_W,
        "pixel_overhead": (padded[0] * padded[1]) / (C.IMG_H * C.IMG_W),
        "adaptation": _adaptation(name, spec, padded),
    }


def _adaptation(name: str, spec: dict, padded: tuple[int, int]) -> str:
    bits = []
    if spec.get("pass_img_size") and spec["timm"].startswith("vit_"):
        bits.append(f"pos-embed resampled 384x384 -> {padded[0]}x{padded[1]}")
    mult = spec.get("size_multiple")
    if mult:
        if padded != tuple(C.IMG_SIZE):
            bits.append(f"replicate-padded {C.IMG_H}x{C.IMG_W} -> "
                        f"{padded[0]}x{padded[1]} "
                        f"({(padded[0]*padded[1])/(C.IMG_H*C.IMG_W):.2f}x pixels)")
        else:
            bits.append(f"window attention; {C.IMG_H}x{C.IMG_W} is already a "
                        f"multiple of {mult}, no padding needed")
    return "; ".join(bits) or "none - fully convolutional, any input size"


# --------------------------------------------------------------------- factory
def build(name: str, pretrained: bool = True, timm_tag: str | None = None):
    """Return (model, info). Head is 4 independent logits - sigmoid lives in the loss."""
    import timm

    spec = REGISTRY.get(name) or ALTERNATES.get(name)
    if spec is None:
        raise KeyError(f"unknown model '{name}'. choose from: {MODEL_NAMES} "
                       f"(or alternates: {list(ALTERNATES)})")
    tag = timm_tag or spec["timm"]

    geom = input_geometry(name)
    fed = geom["fed"]

    # Only the two transformers need the explicit input size. Passing img_size to
    # a fully convolutional backbone is a no-op at best and an error at worst.
    kwargs = {"img_size": fed} if spec.get("pass_img_size") else {}

    try:
        model = timm.create_model(tag, pretrained=pretrained,
                                  num_classes=C.NUM_CLASSES, **kwargs)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not create '{tag}' at {fed}.\n"
            f"the pretrained tag may have changed in your timm version "
            f"({getattr(timm, '__version__', '?')}).\n"
            f"list valid tags with:  python preflight.py --list-tags {name}\n"
            f"original error: {exc}"
        ) from exc

    if spec.get("size_multiple") and fed != tuple(C.IMG_SIZE):
        model = PadToMultiple(model, C.IMG_SIZE, spec["size_multiple"])

    info = {k: v for k, v in spec.items()
            if k not in ("pass_img_size", "size_multiple")}
    info["resolved_tag"] = tag
    info["params_m_actual"] = sum(p.numel() for p in model.parameters()) / 1e6
    info["timm_version"] = getattr(timm, "__version__", "?")
    info["input_geometry"] = geom
    return model, info


def micro_batch(name: str, override: int | None = None) -> int:
    spec = REGISTRY.get(name) or ALTERNATES[name]
    return override or spec["micro_batch"]


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
    print(f"{'model':18}{'params M':>10}{'micro':>7}{'accum':>7}  adaptation")
    print("-" * 100)
    for n in MODEL_NAMES:
        s = REGISTRY[n]
        g = input_geometry(n)
        print(f"{n:18}{s['params_m']:10.1f}{micro_batch(n):7d}"
              f"{accumulation_steps(n):7d}  {g['adaptation']}")
