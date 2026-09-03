"""
The hybrid components: three attention modules, one lightweight transformer,
two fusion rules, a feature-format adapter, and the wrapper that assembles them
onto a timm backbone.

This file is cardiosentry_hybrids_v2's, with ONE addition: `FeatureFormat`. v2's
three backbones all returned an NCHW feature map at stride 32, so an attention
module could be dropped straight in. v3 adds a ViT (a token sequence with a class
token, stride 16) and a SwinV2 (channels-LAST), and the adapter is what lets the
SAME CBAM module sit on all five without any of them getting a special case.
Everything else here - the gated-refinement wrapper, the transformer, the fusion
rules - is unchanged, so a v2 result and a v3 result are the same measurement.

================================================================================
THE ONE DESIGN RULE
================================================================================

    hybrid(backbone, attn=None, lt=False, fusion="none")  IS the baseline.

Every hybrid is the baselines_v2 model with components INSERTED, never with the
model rebuilt around them. Concretely:

    x -> backbone.forward_features(x)          -> F   (whatever layout it uses)
      -> FORMAT.to_map(F)                      -> (B, C, gh, gw)
      -> ATTENTION(.)                          -> same shape, optional
      -> FORMAT.from_map(.)                    -> F'  in F's own layout
      -> backbone.forward_head(F', pre_logits=True)   -> v_cnn   (B, D)
      -> LightweightTransformer(to_map(F'))           -> v_lt    (B, d)
      -> FUSION(v_cnn, v_lt)                          -> v
      -> nn.Linear(dim(v), 4)

`forward_head(..., pre_logits=True)` is timm's own pooling stage - DenseNet-121's
global average pool, ConvNeXt-Base's NormMlp head, ViT-B/16's class-token pool -
and the classifier is a single `nn.Linear`, which is what `timm.create_model(...,
num_classes=4)` builds. The FORMAT round trip is exact, so with no attention, no
transformer and no fusion, the weights, the pooling and the head are IDENTICAL to
the baselines_v2 run of the same backbone, and any difference in the numbers is
attributable to the inserted component and to nothing else.

That is the whole reason this package re-uses baselines_v2's config.py byte for
byte: the fingerprints match, so `compare.py` will rank the baseline runs and the
hybrid runs in ONE table without being forced.

================================================================================
WHY EVERY ATTENTION MODULE IS THE IDENTITY AT STEP 0
================================================================================

Dropping a freshly-initialised sigmoid gate into a pretrained backbone multiplies
every feature map by something whose expected value at init is 0.5. That halves
activations the downstream ImageNet weights were tuned for, and the first epochs
go on undoing the damage rather than learning. On 4,810 training images that cost
is not recoverable, and it would show up as "attention hurt" when what actually
happened is "attention was initialised badly".

So every module here - CBAM, ECA and multi-scale alike - is wrapped as

    x + gamma * (refine(x) - x)          gamma initialised to ZERO

which is exactly the identity at step 0 and lets the network learn how much
attention it wants. The learned gamma is saved into result.json per run. A gamma
that stays near zero is a real finding: it says the backbone did not want the
reweighting, and it is a far more honest answer than a noisy accuracy delta.

The same wrapper is used for all three modules so that "which attention is best"
is not secretly "which attention was initialised least destructively".

================================================================================
THE TRANSFORMER LADDER, AND WHAT EACH RUNG ISOLATES
================================================================================

  <backbone> + LT                  v_lt alone. The transformer REPLACES the
                                   backbone's pooling: tokens are attended, then
                                   mean-pooled. Cheap, but it also throws the
                                   backbone's own pooled vector away, so a loss
                                   here is ambiguous between "attention did not
                                   help" and "256 dims is not enough".

  <backbone> + LT + Concatenation  [LN(v_cnn) ; LN(v_lt)]. Both streams reach the
                                   head. This is the rung that DISAMBIGUATES the
                                   one above: if concat recovers what LT alone
                                   lost, the loss was the discarded pooling.

  <backbone> + LT + Gated Fusion   g*v_cnn' + (1-g)*v_lt', g learned per channel.
                                   Same information as concat, but the model has
                                   to commit to a mixture - and the mixture is
                                   readable. The mean gate is recorded per run.

  <backbone> + MSA + LT + Gated    the full stack: multi-scale attention refines
                                   the feature map that BOTH streams then read.

ONLY THE CBAM RUNG IS REGISTERED IN THIS PACKAGE (models.LADDER has one entry).
The transformer and the fusion rules below are kept because they are what make
this file shared code with cardiosentry_hybrids_v2 rather than a fork of it, and
because adding a rung here is then one line in models.LADDER rather than a port.
If you do add a two-stream rung, copy `ablate.py` back in from v2 as well - it is
absent here because every run in this package is single-stream and it would have
nothing to ablate.

Per-stream LayerNorm is not decoration. It is what makes stream ablation valid:
zeroing one stream after its own LayerNorm leaves the other stream's statistics
untouched, so `ablate.py` can answer "what did the transformer actually
contribute" on a finished run without retraining anything. Normalising after the
concat instead would make the ablation meaningless, because dropping half the
vector would shift the statistics of the half that remained.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

# The stride most backbones here report from forward_features, used only as a
# fallback when nobody passes an explicit grid. FOUR of the five backbones in
# this package are stride 32 and ViT-B/16 is stride 16, so the real value is a
# per-backbone field in models.BACKBONES and selftest.py asserts each one against
# the real network. Nothing in the build path relies on this constant.
BACKBONE_STRIDE = 32

# Lightweight transformer defaults. "Lightweight" is a claim, so it is worth
# being concrete about what it costs at nat768 (768x1024 -> a 24x32 grid, 768
# tokens): 2 blocks at width 256 is ~1.6 M parameters and well under 1 GFLOP,
# against ResNet-50's 25.6 M and ~34 GFLOPs at this input. It is a rounding
# error on the backbone, which is the point - if it helps, it helped cheaply.
LT_DIM = 256
LT_DEPTH = 2
LT_HEADS = 4
LT_MLP_RATIO = 2.0

# Common width both streams are projected to before the gate. Chosen so the gate
# is per-channel rather than a single scalar: a scalar gate can only say "trust
# the CNN this much overall", while 512 gates can say "trust the transformer for
# these features and the CNN for those", which is the question worth asking.
GATE_DIM = 512


# ============================================================== attention
class GatedRefinement(nn.Module):
    """Base class: `x + gamma * (refine(x) - x)` with gamma initialised to zero.

    Subclasses implement `refine`. See the module docstring for why this wrapper
    is applied to every attention module rather than to some of them.
    """

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def refine(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * (self.refine(x) - x)

    def stats(self) -> dict:
        return {"gamma": float(self.gamma.detach().cpu().item())}


# ---------------------------------------------------------------- CBAM
class _ChannelGate(nn.Module):
    """Avg- and max-pooled descriptors through ONE shared MLP (Woo et al. 2018).

    The max branch is the half that matters here. Average pooling over a sheet
    that is mostly white paper dilutes any single tall deflection into the mean;
    max pooling keeps it. LVH and STEMI are both defined by the extremes of the
    trace, not by its average.
    """

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(ch // reduction, 8)
        self.mlp = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, ch))

    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3)))
        return torch.sigmoid(a).view(b, c, 1, 1)


class _SpatialGate(nn.Module):
    """7x7 conv over the channel-wise mean and max -> one attention map.

    On a paper ECG this is asking "which REGION of the sheet matters", and the
    sheet has a fixed layout: twelve leads in a 3x4 grid plus a rhythm strip. A
    spatial gate can in principle learn "look at V1-V3 for anterior ST elevation"
    - which is a real clinical rule - and it can just as easily learn "look at
    the top-left corner", which is where the redacted ID block used to be. Read
    the per-source spread in the report before believing the first story.
    """

    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(s))


class CBAM(GatedRefinement):
    """Convolutional Block Attention Module - channel gate, then spatial gate."""

    kind = "cbam"

    def __init__(self, ch: int, reduction: int = 16, kernel: int = 7):
        super().__init__()
        self.channel = _ChannelGate(ch, reduction)
        self.spatial = _SpatialGate(kernel)

    def refine(self, x):
        y = x * self.channel(x)
        return y * self.spatial(y)

    def extra_repr(self):
        return f"kind=cbam"


# ---------------------------------------------------------------- ECA
def eca_kernel_size(ch: int, gamma: int = 2, b: int = 1) -> int:
    """Wang et al. 2020's adaptive kernel: k = |log2(C)/gamma + b/gamma|_odd.

    C=512 -> 5, C=1280 -> 5, C=2048 -> 7. Not a hyperparameter anyone tunes; it
    is the paper's rule, applied so every backbone gets the module the paper
    describes rather than three hand-picked kernels.
    """
    t = int(abs(math.log2(ch) / gamma + b / gamma))
    return t if t % 2 else t + 1


class ECA(GatedRefinement):
    """Efficient Channel Attention: no dimensionality reduction, k parameters.

    CBAM's channel gate squeezes C -> C/16 -> C and so learns channel
    interactions through a bottleneck. ECA's claim is that the bottleneck is
    harmful and unnecessary: a single 1-D convolution of width k over the pooled
    channel descriptor captures LOCAL cross-channel interaction directly, for k
    parameters instead of 2*C*C/16.

    At C=2048 that is 7 parameters against 524,288. If ECA matches CBAM here, the
    honest reading is that this task's channel reweighting is local and shallow,
    which is a more interesting result than either number alone.
    """

    kind = "eca"

    def __init__(self, ch: int, kernel: int | None = None):
        super().__init__()
        self.k = kernel or eca_kernel_size(ch)
        self.conv = nn.Conv1d(1, 1, self.k, padding=self.k // 2, bias=False)

    def refine(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3)).view(b, 1, c)      # (B, 1, C)
        y = torch.sigmoid(self.conv(y)).view(b, c, 1, 1)
        return x * y

    def extra_repr(self):
        return f"kind=eca, k={self.k}"

    def stats(self) -> dict:
        return {**super().stats(), "kernel": self.k}


# ---------------------------------------------------------------- multi-scale
class MultiScaleAttention(GatedRefinement):
    """Multi-scale strip attention (the MSCA construction, Guo et al. 2022).

    A depthwise 5x5 for local context, plus three parallel branches of separable
    strip convolutions (1xk then kx1) at k = 3, 7, 11, summed and mixed by a 1x1.
    Cheap - all depthwise - and the reason it belongs on an ECG sheet rather than
    being a generic multi-scale block is that its two axes mean different things.

    At the nat768 geometry one feature cell is 32 source pixels, which is
    8.9 mm of TIME (357 ms at 25 mm/s) and 9.3 mm of VOLTAGE (0.93 mV at
    10 mm/mV). So the HORIZONTAL strips span:

        k=3   ~1.1 s     about one cardiac cycle
        k=7   ~2.5 s     the width of one lead panel in the 3x4 layout
        k=11  ~3.9 s     several beats of the rhythm strip - the scale on which
                         irregularly irregular RR intervals become visible, i.e.
                         the scale AF is actually defined at

    ...and the VERTICAL strips span:

        k=3   ~2.8 mV    a tall R wave
        k=7   ~6.5 mV    a full lead row plus its neighbours
        k=11  ~10 mV     two to three lead rows - the span a Sokolow-Lyon
                         S(V1) + R(V5) comparison has to reach across

    A square kernel large enough to reach across three lead rows would cost k^2;
    the separable strips cost 2k and keep the two axes independent, which is the
    right prior when one axis is time and the other is voltage.
    """

    kind = "msa"
    SCALES = (3, 7, 11)

    def __init__(self, ch: int, scales: tuple[int, ...] = SCALES):
        super().__init__()
        self.scales = tuple(scales)
        self.local = nn.Conv2d(ch, ch, 5, padding=2, groups=ch)
        self.strips = nn.ModuleList()
        for k in self.scales:
            self.strips.append(nn.Sequential(
                nn.Conv2d(ch, ch, (1, k), padding=(0, k // 2), groups=ch),
                nn.Conv2d(ch, ch, (k, 1), padding=(k // 2, 0), groups=ch),
            ))
        self.mix = nn.Conv2d(ch, ch, 1)

    def refine(self, x):
        u = self.local(x)
        for s in self.strips:
            u = u + s(x)
        return x * self.mix(u)

    def extra_repr(self):
        return f"kind=msa, scales={self.scales}"

    def stats(self) -> dict:
        return {**super().stats(), "scales": list(self.scales)}


ATTENTION = {"cbam": CBAM, "eca": ECA, "msa": MultiScaleAttention}


# ============================================================== transformer
class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"LT dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t):
        b, n, d = t.shape
        qkv = self.qkv(t).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, heads, N, d/heads)
        # SDPA picks a memory-efficient kernel, which matters: at nat768 there
        # are 768 tokens, and a materialised 768x768 attention matrix per head
        # per layer is the difference between fitting and not fitting.
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(b, n, d))


class _Block(nn.Module):
    """Pre-norm transformer block. Pre-norm, not post-norm, because these blocks
    are trained from scratch on 4,810 images on top of a pretrained CNN, and
    post-norm needs a warmup schedule tuned for it to be stable. The protocol's
    2-epoch warmup is fixed by config.py and shared with every baseline run, so
    the block has to be stable under it rather than the other way round."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, t):
        t = t + self.attn(self.n1(t))
        return t + self.mlp(self.n2(t))


class LightweightTransformer(nn.Module):
    """Backbone feature map -> tokens -> self-attention -> one pooled vector.

    WHY A TRANSFORMER ON TOP OF A CNN AT ALL, ON THIS TASK

    Two of the four heads are long-range comparisons that a stride-32 CNN can
    only reach through depth:

      LVH    Sokolow-Lyon adds S in V1 to R in V5. Those two leads sit in
             different columns AND different rows of the 3x4 layout - the
             comparison is a diagonal across most of the sheet.
      AF     irregular RR intervals are a relationship between beats seconds
             apart, spread along the rhythm strip.

    Self-attention over the 24x32 token grid makes every such pair one hop apart
    instead of many convolutions apart. That is the hypothesis; whether it pays
    for itself is what these runs measure.

    THE POSITION EMBEDDING IS 2-D AND RESAMPLED, NOT FLAT

    Stored as (1, dim, gh, gw) and bicubically interpolated if the runtime grid
    differs from the configured one. A flat (1, N, dim) table would silently
    scramble if the geometry ever changed - the token at flat index 100 is not
    the same place on the sheet at 24x32 as at 19x32 - and this package is meant
    to survive someone re-running it at crop640.
    """

    def __init__(self, in_ch: int, dim: int = LT_DIM, depth: int = LT_DEPTH,
                 heads: int = LT_HEADS, mlp_ratio: float = LT_MLP_RATIO,
                 grid: tuple[int, int] | None = None):
        super().__init__()
        self.dim, self.depth, self.heads = dim, depth, heads
        gh, gw = grid or (C.IMG_H // BACKBONE_STRIDE, C.IMG_W // BACKBONE_STRIDE)
        self.grid = (gh, gw)
        self.proj = nn.Conv2d(in_ch, dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, dim, gh, gw))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def _pos_for(self, h: int, w: int) -> torch.Tensor:
        if (h, w) == self.grid:
            return self.pos
        return F.interpolate(self.pos, size=(h, w), mode="bicubic",
                             align_corners=False)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        t = self.proj(f)
        t = t + self._pos_for(t.shape[-2], t.shape[-1]).to(t.dtype)
        t = t.flatten(2).transpose(1, 2)              # (B, N, dim)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t).mean(dim=1)               # (B, dim)

    def extra_repr(self):
        return (f"dim={self.dim}, depth={self.depth}, heads={self.heads}, "
                f"grid={self.grid[0]}x{self.grid[1]}={self.grid[0]*self.grid[1]} tokens")


# ============================================================== feature format
#
# WHAT THIS IS FOR
#
# CBAM, ECA and the multi-scale block are all written against an NCHW feature
# map, because that is what a channel gate and a 7x7 spatial gate mean. Only
# three of this package's five backbones hand one back:
#
#     densenet121 / efficientnet_b3 / convnext_base   (B, C, gh, gw)   NCHW
#     swinv2_base                                     (B, gh, gw, C)   NHWC
#     vit_b16                                         (B, 1 + N, C)    tokens
#
# The alternative to an adapter would be a second copy of every attention module
# written for tokens - at which point "CBAM on ViT" and "CBAM on DenseNet" would
# be two different modules and the comparison this package exists for would be
# gone. So the FEATURES move, not the module.
#
# TWO PROPERTIES THAT ARE LOAD-BEARING, AND BOTH ARE ASSERTED IN selftest.py
#
#   1. THE ROUND TRIP IS EXACT. `from_map(to_map(f)) == f`, bit for bit, with no
#      interpolation, no reordering and no dtype change. That is what keeps
#      "every attention module is EXACTLY the identity at step 0" true on all
#      five backbones rather than on the three where it happens to be easy.
#
#   2. PREFIX TOKENS ARE NEVER TOUCHED. timm's ViT prepends a class token, and
#      its head POOLS THAT TOKEN - `forward_head` reads x[:, 0]. It has no
#      position on the patch grid, so folding it into a 7x7 spatial convolution
#      would be meaningless, and reshaping 1+N tokens into a grid would scramble
#      which token is which while still running. It is split off, held, and
#      concatenated back in the same position.
class FeatureFormat:
    """Converts one backbone's forward_features output to NCHW and back.

    Deliberately NOT an nn.Module: it holds no parameters and no buffers, and
    keeping it out of the module tree means it cannot appear in a state_dict,
    cannot shift a parameter count, and cannot make the componentless hybrid
    differ from the baselines_v2 model it has to equal.
    """

    kind = "nchw"

    def __init__(self, channels: int, grid: tuple[int, int] | None = None,
                 num_prefix_tokens: int = 0):
        self.channels = channels
        self.grid = tuple(grid) if grid else None
        self.num_prefix_tokens = int(num_prefix_tokens)

    def to_map(self, f: torch.Tensor):
        """-> (nchw_tensor, ctx). `ctx` carries whatever from_map needs back."""
        return f, None

    def from_map(self, x: torch.Tensor, ctx) -> torch.Tensor:
        return x

    def describe(self) -> str:
        return f"{self.kind} (B, C, gh, gw) - no conversion"


class NCHWFormat(FeatureFormat):
    """The identity. DenseNet-121, EfficientNet-B3, ConvNeXt-Base."""
    kind = "nchw"


class NHWCFormat(FeatureFormat):
    """SwinV2: channels last. A permute each way, and permutes are exact."""

    kind = "nhwc"

    def to_map(self, f):
        # (B, gh, gw, C) -> (B, C, gh, gw). No .contiguous(): the attention
        # modules are all convolutions and reductions that accept a strided
        # view, and forcing a copy here would double the peak activation memory
        # of the largest tensor in the network for no numerical difference.
        return f.permute(0, 3, 1, 2), None

    def from_map(self, x, ctx):
        return x.permute(0, 2, 3, 1)

    def describe(self) -> str:
        return "nhwc (B, gh, gw, C) - permuted to NCHW and back, exactly"


class TokenFormat(FeatureFormat):
    """ViT: (B, prefix + gh*gw, C), prefix tokens held out of the attention.

    The grid comes from the model's own patch embedding rather than being
    inferred from the token count, because inferring it needs a factorisation
    and the wrong factorisation of 3072 tokens (48x64 vs 64x48) transposes the
    ECG sheet: the spatial gate would then be reading a 7x7 window across TIME
    where it thinks it is reading across VOLTAGE, and nothing about the result
    would look wrong.
    """

    kind = "tokens"

    def __init__(self, channels: int, grid: tuple[int, int] | None = None,
                 num_prefix_tokens: int = 0):
        if not grid:
            raise ValueError("TokenFormat needs an explicit (gh, gw) grid - a "
                             "token count alone does not determine it")
        super().__init__(channels, grid, num_prefix_tokens)

    def to_map(self, f):
        b, n, c = f.shape
        p = self.num_prefix_tokens
        gh, gw = self.grid
        if n - p != gh * gw:
            raise RuntimeError(
                f"TokenFormat: {n} tokens with {p} prefix does not match the "
                f"{gh}x{gw}={gh*gw} grid this format was built for. The model is "
                f"being fed a different input size than models.input_geometry "
                f"reported.")
        prefix, patches = f[:, :p], f[:, p:]
        # (B, N, C) -> (B, C, gh, gw). transpose then reshape, so channel c of
        # token (i, j) lands at [:, c, i, j] and the inverse is the same two
        # operations in reverse.
        x = patches.transpose(1, 2).reshape(b, c, gh, gw)
        return x, prefix

    def from_map(self, x, ctx):
        b, c, gh, gw = x.shape
        patches = x.reshape(b, c, gh * gw).transpose(1, 2)
        return torch.cat([ctx, patches], dim=1) if ctx.shape[1] else patches

    def describe(self) -> str:
        gh, gw = self.grid
        return (f"tokens (B, {self.num_prefix_tokens}+{gh*gw}, C) - "
                f"{self.num_prefix_tokens} prefix token(s) held out, patches "
                f"reshaped to {gh}x{gw}")


FORMATS = {"nchw": NCHWFormat, "nhwc": NHWCFormat, "tokens": TokenFormat}


def make_format(kind: str, channels: int, grid: tuple[int, int] | None = None,
                num_prefix_tokens: int = 0) -> FeatureFormat:
    if kind not in FORMATS:
        raise KeyError(f"unknown feature format '{kind}'. known: {sorted(FORMATS)}")
    return FORMATS[kind](channels, grid, num_prefix_tokens)


# ============================================================== fusion
class SingleStream(nn.Module):
    """One stream, passed straight through. No LayerNorm - deliberately.

    This is the fusion used by the attention-only runs, and it is where the
    package's attribution claim is either true or false. A LayerNorm here would
    be internally tidy (every other fusion has one) and it would also mean that

        densenet121_cbam = baselines_v2 densenet121 + CBAM + a LayerNorm

    so a win could be either component and the run would not answer the question
    it was built to answer. Passing through instead makes the attention-only
    hybrids differ from their baselines_v2 counterparts by EXACTLY the inserted
    module, at the cost of a small asymmetry against the two-stream variants -
    which need their per-stream norms for ablation to be valid.

    The LT-only runs go through here too and lose nothing by it: the transformer
    already ends in its own LayerNorm before pooling.
    """

    kind = "none"

    def __init__(self, dims: "OrderedDict[str,int]"):
        super().__init__()
        (self.name, d), = dims.items()
        self.out_dim = d

    def forward(self, feats, ablate=None):
        v = feats[self.name]
        return torch.zeros_like(v) if ablate == self.name else v

    def stats(self):
        return {}


class ConcatFusion(nn.Module):
    """Per-stream LayerNorm, then concatenate. The head sees both streams whole.

    The simplest thing that can work, and the right control for gated fusion: if
    the gate does not beat plain concatenation, then what helped was having both
    streams, not the mixing rule.
    """

    kind = "concat"

    def __init__(self, dims: "OrderedDict[str,int]"):
        super().__init__()
        self.names = list(dims)
        self.norms = nn.ModuleDict({n: nn.LayerNorm(d) for n, d in dims.items()})
        self.out_dim = sum(dims.values())

    def forward(self, feats, ablate=None):
        parts = []
        for n in self.names:
            v = self.norms[n](feats[n])
            parts.append(torch.zeros_like(v) if ablate == n else v)
        return torch.cat(parts, dim=1)

    def stats(self):
        return {}


class GatedFusion(nn.Module):
    """Per-channel learned mixture of the two streams.

        h_cnn = W_cnn . LN(v_cnn)          both projected to GATE_DIM, so the
        h_lt  = W_lt  . LN(v_lt)           gate is a per-feature decision rather
        g     = sigmoid(W_g . [h_cnn ; h_lt])       than one global scalar
        out   = g * h_cnn + (1 - g) * h_lt

    THE GATE IS A RESULT, NOT JUST A MECHANISM. Its mean is recorded in
    result.json for every run. g -> 1 means the model learned to ignore the
    transformer, which is a clean negative answer to "does global attention help
    here"; g -> 0.5 with a win means it genuinely blended them. Either way it is
    something you can write down, unlike an accuracy delta of 0.004.

    Ablation happens on the POST-LayerNorm stream vectors, before the gate, so
    zeroing a stream removes it from the gate's input too - "what if this stream
    were not there" rather than "what if this stream were zeros but the gate
    still knew about it".
    """

    kind = "gated"

    def __init__(self, dims: "OrderedDict[str,int]", gate_dim: int = GATE_DIM):
        super().__init__()
        if len(dims) != 2:
            raise ValueError(f"gated fusion takes exactly 2 streams, got {list(dims)}")
        self.names = list(dims)
        self.norms = nn.ModuleDict({n: nn.LayerNorm(d) for n, d in dims.items()})
        self.proj = nn.ModuleDict({n: nn.Linear(d, gate_dim) for n, d in dims.items()})
        self.gate = nn.Linear(gate_dim * 2, gate_dim)
        self.out_dim = gate_dim
        # Running mean of the gate, updated in eval only, so the number reported
        # in result.json describes the finished model on held-out data rather
        # than an average over training-time transients.
        self.register_buffer("gate_mean", torch.zeros(1))
        self.register_buffer("gate_seen", torch.zeros(1))

    def forward(self, feats, ablate=None):
        h = {}
        for n in self.names:
            v = self.norms[n](feats[n])
            if ablate == n:
                v = torch.zeros_like(v)
            h[n] = self.proj[n](v)
        a, b = self.names
        g = torch.sigmoid(self.gate(torch.cat([h[a], h[b]], dim=1)))
        if not self.training:
            with torch.no_grad():
                n_new = self.gate_seen + g.shape[0]
                self.gate_mean.copy_((self.gate_mean * self.gate_seen
                                      + g.float().mean() * g.shape[0]) / n_new)
                self.gate_seen.copy_(n_new)
        return g * h[a] + (1.0 - g) * h[b]

    def reset_gate_stats(self):
        self.gate_mean.zero_()
        self.gate_seen.zero_()

    def stats(self):
        a, b = self.names
        m = float(self.gate_mean.item())
        return {"gate_mean": m, "gate_weight_on": {a: m, b: 1.0 - m},
                "gate_samples": int(self.gate_seen.item())}


FUSION = {"none": SingleStream, "concat": ConcatFusion, "gated": GatedFusion}


# ============================================================== the wrapper
class HybridModel(nn.Module):
    """timm backbone + optional attention + optional transformer + fusion + head.

    `ablate` is an ATTRIBUTE rather than a forward() argument, so engine.predict
    and the training loop never need to know a model is multi-stream. ablate.py
    sets it, runs the test loader, and sets it back.
    """

    def __init__(self, backbone: nn.Module, feat_ch: int, pooled_dim: int,
                 attn: str | None = None, use_cnn: bool = True,
                 use_lt: bool = False, fusion: str = "none",
                 num_classes: int = C.NUM_CLASSES,
                 fmt: FeatureFormat | None = None,
                 grid: tuple[int, int] | None = None,
                 pad: tuple[int, int] = (0, 0)):
        super().__init__()
        if not (use_cnn or use_lt):
            raise ValueError("a hybrid must keep at least one stream")
        self.backbone = backbone
        self.use_cnn, self.use_lt = use_cnn, use_lt
        self.attn_kind = attn
        self.attn = ATTENTION[attn](feat_ch) if attn else None
        # A plain NCHW backbone needs no conversion, so the default costs
        # nothing and keeps this constructor call-compatible with v2's.
        self.fmt = fmt or NCHWFormat(feat_ch, grid)
        # Replicate padding for a window-attention backbone whose input size is
        # not a legal multiple. ZERO would be black, and black is the colour of
        # ECG ink; replicating extends the paper instead. It is (0, 0) at nat768
        # for every backbone here, and models.input_geometry reports it when it
        # is not, because it changes what a latency or VRAM number means.
        self.pad_h, self.pad_w = int(pad[0]), int(pad[1])

        dims: "OrderedDict[str,int]" = OrderedDict()
        if use_cnn:
            dims["cnn"] = pooled_dim
        if use_lt:
            self.lt = LightweightTransformer(feat_ch, grid=grid)
            dims["lt"] = self.lt.out_dim

        if len(dims) == 1 and fusion != "none":
            raise ValueError(f"fusion='{fusion}' needs two streams, got {list(dims)}")
        self.fuse = FUSION[fusion](dims)
        self.classifier = nn.Linear(self.fuse.out_dim, num_classes)
        self.stream_names = list(dims)
        self.ablate: str | None = None

    # ---- forward -------------------------------------------------------
    def features(self, x) -> "OrderedDict[str,torch.Tensor]":
        if self.pad_h or self.pad_w:
            # right and bottom only, so the sheet stays anchored at the
            # top-left corner it was rendered in
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="replicate")
        f = self.backbone.forward_features(x)
        if self.attn is not None:
            # The attention module always sees NCHW; `f` always leaves in the
            # layout forward_head expects. With gamma at 0 this whole block is
            # the identity, exactly - selftest.py checks that with torch.equal
            # on the real backbones, not on a stand-in tensor.
            m, ctx = self.fmt.to_map(f)
            f = self.fmt.from_map(self.attn(m), ctx)
        out: "OrderedDict[str,torch.Tensor]" = OrderedDict()
        if self.use_cnn:
            out["cnn"] = self.backbone.forward_head(f, pre_logits=True)
        if self.use_lt:
            m, _ = self.fmt.to_map(f)
            out["lt"] = self.lt(m)
        return out

    def forward(self, x):
        return self.classifier(self.fuse(self.features(x), ablate=self.ablate))

    # ---- what the run should record ------------------------------------
    def stats(self) -> dict:
        """Learned quantities worth reading straight out of result.json.

        The gamma of an attention module and the mean of a fusion gate are the
        two numbers that say what the hybrid actually DID, as opposed to what it
        scored. They cost nothing to record and they are the difference between
        "CBAM gained 0.003 AUPRC" and "CBAM gained 0.003 AUPRC with gamma 0.02,
        i.e. it barely turned itself on".
        """
        s: dict = {"streams": list(self.stream_names),
                   "attention": self.attn_kind,
                   "fusion": self.fuse.kind,
                   "feat_format": self.fmt.kind,
                   "feat_grid": list(self.fmt.grid) if self.fmt.grid else None,
                   "prefix_tokens": self.fmt.num_prefix_tokens,
                   "input_pad": [self.pad_h, self.pad_w]}
        if self.attn is not None:
            s["attention_stats"] = self.attn.stats()
        if self.use_lt:
            s["lt"] = {"dim": self.lt.dim, "depth": self.lt.depth,
                       "heads": self.lt.heads,
                       "tokens": self.lt.grid[0] * self.lt.grid[1],
                       "grid": list(self.lt.grid)}
        s.update(self.fuse.stats())
        return s

    def reset_stats(self):
        if hasattr(self.fuse, "reset_gate_stats"):
            self.fuse.reset_gate_stats()

    def component_params(self) -> dict:
        """Parameters added ON TOP of the backbone, per component.

        Printed by preflight and stored per run, because "this hybrid won" and
        "this hybrid won for 0.9% more parameters" are different claims and the
        second one is the one worth making.
        """
        n = lambda m: sum(p.numel() for p in m.parameters())
        out = {"backbone": n(self.backbone), "classifier": n(self.classifier),
               "fusion": n(self.fuse)}
        if self.attn is not None:
            out["attention"] = n(self.attn)
        if self.use_lt:
            out["transformer"] = n(self.lt)
        total = sum(p.numel() for p in self.parameters())
        out["total"] = total
        out["added_over_backbone"] = total - out["backbone"]
        out["added_pct"] = 100.0 * (total - out["backbone"]) / max(out["backbone"], 1)
        return out
