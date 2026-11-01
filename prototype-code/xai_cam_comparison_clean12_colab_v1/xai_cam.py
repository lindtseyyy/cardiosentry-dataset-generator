"""Grad-CAM, Grad-CAM++ and Score-CAM, plus the deletion / insertion metrics.

One module, used two ways: `make_notebook.py` embeds it verbatim as a notebook
cell, and `cpu_timing.py` imports it, so the Colab evaluation and the
deployment-CPU timing run the same code.

THE CUT
-------
Every method explains one sigmoid head k of a timm ConvNeXt at its last stage,
`stages.3` (1024 x 24 x 32 at 768x1024), the layer the app's Grad-CAM uses.
`ModelSplit.features` runs stem -> stages under no_grad; `ModelSplit.head` runs
norm_pre -> head (global pool, LayerNorm, fc). The gradient methods record a
graph through the head only, from a detached copy of the activations.

FINAL MAP, ALL METHODS
----------------------
cam = ReLU(sum_c w_c * A_c) at feature resolution (24 x 32), then bilinear
upsampling (align_corners=False) to the input size. Only the channel weights
w_c differ between methods:

  Grad-CAM    w_c = mean over (i, j) of dS/dA_c                 (Selvaraju 2017)
  Grad-CAM++  w_c = sum_ij alpha_cij * ReLU(dS/dA_cij), with the closed-form
              alpha of Chattopadhay 2018, written as in pytorch-grad-cam
  Score-CAM   w_c = sigmoid(S_k(x * M_c)), M_c = A_c upsampled to the input and
              min-max scaled to [0, 1]; one masked forward per channel
              (Wang 2020, as in the authors' reference code, with the
              softmax probability replaced by this model's sigmoid)

S is the head-k LOGIT for the gradient methods (a sigmoid gradient vanishes at
confident predictions) and the head-k probability for Score-CAM.

DELETION / INSERTION (Petsiuk et al. 2018)
------------------------------------------
Pixels are ranked by the upsampled map, highest first; ties (Grad-CAM's
zero regions, a constant map) are broken by a seeded random permutation.
Deletion replaces the top fraction f of pixels with the baseline; insertion
starts from the baseline and restores the top fraction f. The score is
p_k(x_f) at f = 0, 1/steps, ..., 1, and the AUC is the trapezoid over f.
Lower deletion AUC and higher insertion AUC mean a more faithful map.
"""
import contextlib
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

METHODS = ("gradcam", "gradcampp", "scorecam")
GRADIENT_METHODS = ("gradcam", "gradcampp")
METHOD_LABELS = {"gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++",
                 "scorecam": "Score-CAM", "random": "Random map"}


class ModelSplit:
    """A timm ConvNeXt cut at the output of its last stage."""

    def __init__(self, model):
        for part in ("stem", "stages", "norm_pre", "forward_head"):
            if not hasattr(model, part):
                raise TypeError(f"{type(model).__name__} has no {part!r}; "
                                "ModelSplit expects a timm ConvNeXt")
        self.model = model

    def features(self, x):
        return self.model.stages(self.model.stem(x))

    def head(self, a):
        return self.model.forward_head(self.model.norm_pre(a))


def autocast_context(device, amp_dtype):
    """Autocast for `amp_dtype`, or a no-op for float32."""
    if amp_dtype is None or amp_dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# ================================================================= activations
def activations(split, x, amp_dtype=None):
    """stages.3 output of a normalized (1, 3, H, W) batch, as float32, no graph."""
    with torch.no_grad(), autocast_context(x.device, amp_dtype):
        a = split.features(x)
    return a.float()


# ============================================================ gradient methods
def gradient_cams(split, a, label_indices, methods=GRADIENT_METHODS):
    """Grad-CAM and/or Grad-CAM++ maps from precomputed activations.

    a: (1, C, h, w) float32 activations of one image. Only the head is
    differentiated, in float32, once per label.
    Returns ({method: {k: (h, w) tensor}}, logits (1, n_labels)).
    """
    unknown = set(methods) - set(GRADIENT_METHODS)
    if unknown:
        raise ValueError(f"not gradient methods: {sorted(unknown)}")
    leaf = a.detach().requires_grad_(True)
    with torch.enable_grad():
        logits = split.head(leaf).float()
        maps = {method: {} for method in methods}
        activ = leaf.detach()[0]
        for position, k in enumerate(label_indices):
            (grad,) = torch.autograd.grad(
                logits[0, int(k)], leaf,
                retain_graph=position < len(label_indices) - 1)
            grad = grad[0].float()
            if "gradcam" in methods:
                weights = grad.mean(dim=(1, 2))
                maps["gradcam"][int(k)] = F.relu((weights[:, None, None] * activ).sum(0))
            if "gradcampp" in methods:
                g2 = grad.pow(2)
                g3 = g2 * grad
                sum_activ = activ.sum(dim=(1, 2), keepdim=True)
                alpha = g2 / (2 * g2 + sum_activ * g3 + 1e-6)
                alpha = torch.where(grad != 0, alpha, torch.zeros_like(alpha))
                weights = (F.relu(grad) * alpha).sum(dim=(1, 2))
                maps["gradcampp"][int(k)] = F.relu((weights[:, None, None] * activ).sum(0))
    return maps, logits.detach()


# ==================================================================== Score-CAM
def channel_masks(a_chunk, size):
    """Upsampled, per-channel min-max scaled masks of (1, b, h, w) activations.

    Returns (masks (b, H, W) in [0, 1], valid (b,) bool). A constant channel
    has no spatial information and gets valid=False (the reference
    implementation skips it; it contributes nothing and costs no forward).
    """
    up = F.interpolate(a_chunk, size=size, mode="bilinear", align_corners=False)[0]
    flat = up.flatten(1)
    low = flat.min(dim=1).values
    span = flat.max(dim=1).values - low
    valid = span > 0
    masks = (up - low[:, None, None]) / torch.where(valid, span, torch.ones_like(span))[:, None, None]
    return masks, valid


@torch.no_grad()
def scorecam_channel_scores(split, x, a, batch_size, amp_dtype=None, forward=None):
    """sigmoid(logits) of x * M_c for every channel c: (C, n_labels) float32.

    Also returns the number of masked forwards actually run (constant channels
    are skipped). The scores cover every label, so one pass serves every label
    of the image. `forward` replaces split.model (e.g. a compiled FixedBatch).
    """
    forward = forward or split.model
    channels = a.shape[1]
    size = tuple(x.shape[-2:])
    scores = None
    forwards = 0
    for start in range(0, channels, batch_size):
        masks, valid = channel_masks(a[:, start:start + batch_size], size)
        keep = valid.nonzero().flatten()
        if keep.numel() == 0:
            continue
        inputs = x * masks[keep, None]
        with autocast_context(x.device, amp_dtype):
            logits = forward(inputs)
        probs = torch.sigmoid(logits.float())
        if scores is None:
            scores = torch.zeros(channels, probs.shape[1], device=x.device)
        scores[start + keep] = probs
        forwards += int(keep.numel())
    if scores is None:
        raise RuntimeError("every activation channel is constant; Score-CAM is undefined")
    return scores, forwards


def scorecam_maps(a, scores, label_indices):
    """{k: (h, w)} Score-CAM maps from activations and channel scores."""
    activ = a[0]
    return {int(k): F.relu(torch.einsum("c,chw->hw", scores[:, int(k)], activ))
            for k in label_indices}


class FixedBatch:
    """Run `fn` on batches zero-padded to exactly `size` rows.

    A compiled model then sees one input shape and never recompiles for the
    last, shorter chunk. ConvNeXt has no batch statistics (LayerNorm only), so
    padding rows cannot change the real rows' outputs.
    """

    def __init__(self, fn, size):
        self.fn, self.size = fn, int(size)

    def __call__(self, x):
        n = x.shape[0]
        if n > self.size:
            raise ValueError(f"batch of {n} exceeds the fixed size {self.size}")
        if n < self.size:
            x = torch.cat([x, x.new_zeros((self.size - n, *x.shape[1:]))])
        return self.fn(x.contiguous(memory_format=torch.channels_last))[:n]


# ================================================================== one method
def explain(split, x, k, method, amp_dtype=None, score_batch=32):
    """One explanation from scratch, as a deployed system would compute it.

    x: normalized (1, 3, H, W) on the target device. Returns the (H, W) map and
    the number of model forwards (backwards count as one forward each).
    """
    a = activations(split, x, amp_dtype)
    if method in GRADIENT_METHODS:
        maps, _ = gradient_cams(split, a, [k], (method,))
        cam, forwards = maps[method][int(k)], 2
    elif method == "scorecam":
        scores, masked = scorecam_channel_scores(split, x, a, score_batch, amp_dtype)
        cam, forwards = scorecam_maps(a, scores, [k])[int(k)], 1 + masked
    else:
        raise ValueError(f"unknown method {method!r}")
    return upsample(cam, tuple(x.shape[-2:])), forwards


def timed(fn, device):
    """(seconds, result) of fn(), synchronized on CUDA."""
    synchronize(device)
    started = time.perf_counter()
    result = fn()
    synchronize(device)
    return time.perf_counter() - started, result


# ============================================================ ranking & curves
def upsample(cam, size):
    """Bilinear upsampling of a (h, w) map to `size`, as float32."""
    return F.interpolate(cam[None, None].float(), size=size, mode="bilinear",
                         align_corners=False)[0, 0]


def random_map(shape, seed, device):
    """A seeded uniform map at feature resolution: the floor every method must beat."""
    generator = torch.Generator().manual_seed(int(seed))
    return torch.rand(shape, generator=generator).to(device)


def pixel_rank(saliency, seed):
    """rank[i, j] = position of pixel (i, j) in descending saliency order.

    Ties are broken by a permutation seeded with `seed`, so equal pixels
    (Grad-CAM's zero regions) are removed in a fixed random order rather than
    raster order, which would delete the top of the sheet first.
    """
    flat = saliency.flatten().float()
    n = flat.numel()
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).to(flat.device)
    order = perm[torch.argsort(flat[perm], descending=True, stable=True)]
    rank = torch.empty(n, dtype=torch.int64, device=flat.device)
    rank[order] = torch.arange(n, device=flat.device)
    return rank.view_as(saliency)


def step_counts(n_pixels, steps):
    """Pixels changed at each of the steps + 1 curve points (0 ... n_pixels)."""
    return np.rint(np.linspace(0, n_pixels, steps + 1)).astype(np.int64)


@torch.no_grad()
def deletion_insertion(model, image01, baseline01, rank, k, steps, batch_size,
                       mean, std, amp_dtype=None):
    """Deletion and insertion curves of head k for one ranking.

    image01, baseline01: (3, H, W) float in [0, 1]; mean/std: (3, 1, 1).
    Returns (fractions, deletion probs, insertion probs), each (steps + 1,).
    """
    counts = step_counts(rank.numel(), steps)
    counts_t = torch.as_tensor(counts, device=rank.device)
    curves = {}
    for mode in ("deletion", "insertion"):
        probs = []
        for start in range(0, len(counts), batch_size):
            chunk = counts_t[start:start + batch_size]
            changed = (rank[None] < chunk[:, None, None])[:, None]  # (b, 1, H, W)
            if mode == "deletion":
                batch = torch.where(changed, baseline01[None], image01[None])
            else:
                batch = torch.where(changed, image01[None], baseline01[None])
            batch = ((batch - mean) / std).contiguous(memory_format=torch.channels_last)
            with autocast_context(rank.device, amp_dtype):
                logits = model(batch)
            probs.append(torch.sigmoid(logits.float()[:, int(k)]).cpu())
        curves[mode] = torch.cat(probs).numpy().astype(np.float64)
    return counts / counts[-1], curves["deletion"], curves["insertion"]


def curve_auc(fractions, values):
    """Trapezoid area under values(fractions)."""
    fractions = np.asarray(fractions, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return float(np.sum((values[1:] + values[:-1]) * np.diff(fractions)) / 2.0)


# ==================================================================== baselines
def paper_colour(image01, pad_px):
    """Median colour of the left paper pad of a clean12 sheet: (3,) and its spread.

    The render pads the 994-px trace area with 15 px of plain paper on each
    side, so that strip is the sheet's own blank-paper colour with no grid.
    """
    strip = image01[:, :, :pad_px].reshape(3, -1)
    return strip.median(dim=1).values, float(strip.std(dim=1).max())


def make_baseline(image01, kind, pad_px=15, blur_sigma_px=10.0):
    """The image every deleted pixel turns into, and insertion starts from."""
    if kind == "paper":
        colour, _ = paper_colour(image01, pad_px)
        return colour[:, None, None].expand_as(image01).contiguous()
    if kind == "blur":
        from torchvision.transforms.functional import gaussian_blur
        radius = int(math.ceil(3 * blur_sigma_px))
        return gaussian_blur(image01[None], [2 * radius + 1] * 2, [blur_sigma_px] * 2)[0]
    if kind == "zero":
        return torch.zeros_like(image01)
    raise ValueError(f"unknown baseline {kind!r} (paper, blur, zero)")
