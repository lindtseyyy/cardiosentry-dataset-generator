# Multi-scale feature fusion on a timm ConvNeXt backbone (single source of truth:
# make_notebook.py inlines this file into the notebook's model cell, and
# validate_notebook.py checks the copy in the notebook is identical).
#
# Every variant fuses stages 2-4 of ConvNeXt (strides 8/16/32; 256/512/1024
# channels; 96x128, 48x64 and 24x32 maps for a 768x1024 sheet). Stage 1
# (stride 4, 192x256) is left out: it would roughly double the neck's cost for
# features that stage 2 already sees at 8 px per cell.
#
#   fpn    top-down FPN (Lin et al., 2017)
#   pan    FPN plus a bottom-up path (PANet, Liu et al., 2018)
#   bifpn  repeated bidirectional FPN with fast normalized fusion weights
#          (EfficientDet, Tan et al., 2020), on 3 levels instead of 5
#   rfpn   the colleague's Residual FPN: the FPN branch is added to the
#          unchanged ConvNeXt logits through a zero-initialised linear layer
#
# Head modes:
#   replace   classifier = LayerNorm + Linear over [GAP(P3), GAP(P4), GAP(P5)];
#             the timm head (norm + fc) is removed
#   residual  logits = ConvNeXt head(stage 4) + ZeroLinear(LayerNorm(pooled
#             neck)); at step 0 the model is exactly the plain backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.optim import param_groups_layer_decay

FUSION_CHOICES = ("none", "fpn", "pan", "bifpn", "rfpn")
FUSION_HEAD_CHOICES = ("replace", "residual")
FUSION_STAGE_INDICES = (1, 2, 3)        # timm stages.1..3 = strides 8/16/32


def resolve_fusion(fusion, head=""):
    '''Return (neck, head_mode) for CS_FUSION / CS_FUSION_HEAD; rfpn = fpn + residual.'''
    fusion = (fusion or "none").strip().lower()
    head = (head or "").strip().lower()
    if fusion not in FUSION_CHOICES:
        raise ValueError(f"CS_FUSION must be one of {FUSION_CHOICES}, got {fusion!r}")
    if fusion == "none":
        if head:
            raise ValueError("CS_FUSION_HEAD needs a fusion neck (CS_FUSION=fpn|pan|bifpn|rfpn)")
        return "none", "none"
    if fusion == "rfpn":
        if head not in ("", "residual"):
            raise ValueError("CS_FUSION=rfpn is the residual FPN; use CS_FUSION=fpn for a replace head")
        return "fpn", "residual"
    head = head or "replace"
    if head not in FUSION_HEAD_CHOICES:
        raise ValueError(f"CS_FUSION_HEAD must be one of {FUSION_HEAD_CHOICES}, got {head!r}")
    return fusion, head


def _gn(channels):
    return nn.GroupNorm(32 if channels % 32 == 0 else 1, channels)


def _conv_gn_act(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        _gn(out_channels), nn.GELU(),
    )


def _match(x, reference):
    '''Resize x to reference's spatial size (a no-op for the 768x1024 input).'''
    if x.shape[-2:] != reference.shape[-2:]:
        x = F.interpolate(x, size=reference.shape[-2:], mode="nearest")
    return x


class FPNNeck(nn.Module):
    '''Top-down FPN: 1x1 laterals, nearest upsampling + add, 3x3 conv-GN-GELU per level.'''

    def __init__(self, in_channels, channels):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_channels])
        self.smooth = nn.ModuleList([_conv_gn_act(channels, channels) for _ in in_channels])

    def top_down(self, features):
        lateral = [layer(f) for layer, f in zip(self.lateral, features)]
        outputs = [lateral[-1]]
        for i in range(len(lateral) - 2, -1, -1):
            outputs.insert(0, lateral[i] + F.interpolate(
                outputs[0], size=lateral[i].shape[-2:], mode="nearest"))
        return [smooth(o) for smooth, o in zip(self.smooth, outputs)]

    def forward(self, features):
        return self.top_down(features)


class PANNeck(FPNNeck):
    '''PANet: the FPN outputs, then a bottom-up path (stride-2 3x3 conv, add, 3x3 conv).'''

    def __init__(self, in_channels, channels):
        super().__init__(in_channels, channels)
        levels = len(in_channels)
        self.downsample = nn.ModuleList([
            _conv_gn_act(channels, channels, stride=2) for _ in range(levels - 1)])
        self.bottom_up = nn.ModuleList([
            _conv_gn_act(channels, channels) for _ in range(levels - 1)])

    def forward(self, features):
        p = self.top_down(features)
        outputs = [p[0]]
        for i in range(1, len(p)):
            down = _match(self.downsample[i - 1](outputs[-1]), p[i])
            outputs.append(self.bottom_up[i - 1](p[i] + down))
        return outputs


class FastNormalizedFusion(nn.Module):
    '''EfficientDet fast fusion: sum_i relu(w_i) x_i / (sum_i relu(w_i) + eps).'''

    def __init__(self, inputs, eps=1e-4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(inputs))
        self.eps = eps

    def forward(self, tensors):
        weight = F.relu(self.weight)
        weight = weight / (weight.sum() + self.eps)
        out = tensors[0] * weight[0].to(tensors[0].dtype)
        for i in range(1, len(tensors)):
            out = out + tensors[i] * weight[i].to(tensors[i].dtype)
        return out


class SeparableConvGN(nn.Module):
    '''EfficientDet node: SiLU, depthwise 3x3, pointwise 1x1, GroupNorm.'''

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = _gn(channels)

    def forward(self, x):
        return self.norm(self.pointwise(self.depthwise(F.silu(x))))


class BiFPNLayer(nn.Module):
    '''One bidirectional pass over N levels (top-down, then bottom-up with skip).'''

    def __init__(self, channels, levels):
        super().__init__()
        self.levels = levels
        # Top-down nodes for levels N-2..0; the finest one is already an output.
        self.td_fuse = nn.ModuleList([FastNormalizedFusion(2) for _ in range(levels - 1)])
        self.td_conv = nn.ModuleList([SeparableConvGN(channels) for _ in range(levels - 1)])
        # Bottom-up nodes for levels 1..N-1: the middle levels also take their
        # own input (the extra skip edge), the coarsest level has no td node.
        self.bu_fuse = nn.ModuleList([
            FastNormalizedFusion(3 if i < levels - 1 else 2) for i in range(1, levels)])
        self.bu_conv = nn.ModuleList([SeparableConvGN(channels) for _ in range(1, levels)])

    @staticmethod
    def _down(x, reference):
        return _match(F.max_pool2d(x, kernel_size=3, stride=2, padding=1), reference)

    def forward(self, features):
        n = self.levels
        td = [None] * n
        td[n - 1] = features[n - 1]
        for k, i in enumerate(range(n - 2, -1, -1)):
            up = F.interpolate(td[i + 1], size=features[i].shape[-2:], mode="nearest")
            td[i] = self.td_conv[k](self.td_fuse[k]([features[i], up]))
        outputs = [td[0]]
        for i in range(1, n):
            down = self._down(outputs[-1], features[i])
            inputs = [features[i], td[i], down] if i < n - 1 else [features[i], down]
            outputs.append(self.bu_conv[i - 1](self.bu_fuse[i - 1](inputs)))
        return outputs


class BiFPNNeck(nn.Module):
    def __init__(self, in_channels, channels, repeats=2):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), _gn(channels)) for c in in_channels])
        self.layers = nn.ModuleList([BiFPNLayer(channels, len(in_channels)) for _ in range(repeats)])

    def forward(self, features):
        x = [layer(f) for layer, f in zip(self.lateral, features)]
        for layer in self.layers:
            x = layer(x)
        return x


def build_neck(neck, in_channels, channels, bifpn_repeats=2):
    if neck == "fpn":
        return FPNNeck(in_channels, channels)
    if neck == "pan":
        return PANNeck(in_channels, channels)
    if neck == "bifpn":
        return BiFPNNeck(in_channels, channels, bifpn_repeats)
    raise ValueError(f"unknown fusion neck {neck!r}")


class ConvNeXtMultiScaleFusion(nn.Module):
    '''A timm ConvNeXt with a multi-scale fusion neck and a replace or residual head.

    The backbone keeps its pretrained parameter names under "backbone.", so the
    layer-decay groups are timm's own. Only the neck and the new head are fresh.
    '''

    def __init__(self, backbone, neck, head_mode, num_classes, channels=256,
                 bifpn_repeats=2, head_init_scale=1.0):
        super().__init__()
        if head_mode not in FUSION_HEAD_CHOICES:
            raise ValueError(f"unknown head mode {head_mode!r}")
        self.neck_name = neck
        self.head_mode = head_mode
        self.backbone = backbone
        in_channels = [backbone.feature_info[i]["num_chs"] for i in FUSION_STAGE_INDICES]
        self.neck = build_neck(neck, in_channels, channels, bifpn_repeats)
        fused = channels * len(in_channels)
        self.fuse_norm = nn.LayerNorm(fused)
        self.fuse_fc = nn.Linear(fused, num_classes)
        if head_mode == "residual":
            # Zero contribution at step 0: the model starts as the plain backbone.
            nn.init.zeros_(self.fuse_fc.weight)
            nn.init.zeros_(self.fuse_fc.bias)
        else:
            # The timm head is not used, so it is removed (no dead parameters);
            # the new classifier gets timm's ConvNeXt init and head_init_scale.
            backbone.head = nn.Identity()
            trunc_normal_(self.fuse_fc.weight, std=0.02)
            nn.init.zeros_(self.fuse_fc.bias)
            with torch.no_grad():
                self.fuse_fc.weight.mul_(head_init_scale)
        for module in self.neck.modules():
            if isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def default_cfg(self):
        return self.backbone.default_cfg

    @property
    def pretrained_cfg(self):
        return self.backbone.pretrained_cfg

    def get_classifier(self):
        return self.fuse_fc if self.head_mode == "replace" else self.backbone.get_classifier()

    def new_modules(self):
        return [self.neck, self.fuse_norm, self.fuse_fc]

    def forward(self, x):
        backbone = self.backbone
        x = backbone.stem(x)
        features = []
        for stage in backbone.stages:
            x = stage(x)
            features.append(x)
        maps = self.neck([features[i] for i in FUSION_STAGE_INDICES])
        pooled = torch.cat([torch.mean(m, dim=(2, 3), dtype=torch.float32) for m in maps], dim=1)
        fused_logits = self.fuse_fc(self.fuse_norm(pooled))
        if self.head_mode == "replace":
            return fused_logits
        return backbone.forward_head(backbone.norm_pre(x)) + fused_logits


def fusion_param_groups(model, weight_decay, layer_decay):
    '''timm per-block layer decay on the backbone; new fusion layers at the full LR.

    Weight decay on conv/linear weights only (not on norms, biases or the
    BiFPN fusion weights), as timm does for the backbone.
    '''
    if not isinstance(model, ConvNeXtMultiScaleFusion):
        return param_groups_layer_decay(model, weight_decay=weight_decay, layer_decay=layer_decay)
    groups = list(param_groups_layer_decay(
        model.backbone, weight_decay=weight_decay, layer_decay=layer_decay))
    if model.head_mode == "replace":
        # timm puts the deepest level at scale 1.0. Without the timm head that is
        # the last stage-4 block, so shift by one level: every backbone parameter
        # then gets the same LR as in the plain model (stem ~0.056x), and the new
        # neck + classifier take the head's place at 1.0.
        for group in groups:
            group["lr_scale"] = float(group["lr_scale"]) * layer_decay
    fresh = [p for module in model.new_modules() for p in module.parameters() if p.requires_grad]
    decay = [p for p in fresh if p.ndim > 1]
    no_decay = [p for p in fresh if p.ndim <= 1]
    groups.append({"params": decay, "weight_decay": weight_decay, "lr_scale": 1.0, "fusion": True})
    groups.append({"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0, "fusion": True})
    return groups


def fusion_description(neck, head_mode, channels, bifpn_repeats):
    if neck == "none":
        return {"neck": "none", "head": "timm ConvNeXt head (stage 4 only)"}
    return {
        "neck": neck,
        "head": head_mode,
        "levels": "timm stages.1-3 (strides 8/16/32)",
        "channels": int(channels),
        "bifpn_repeats": int(bifpn_repeats) if neck == "bifpn" else None,
        "norm": "GroupNorm(32)",
        "pooling": "global average per level (FP32), concatenated",
        "classifier": ("LayerNorm + zero-init Linear added to the ConvNeXt logits"
                       if head_mode == "residual" else
                       "LayerNorm + Linear (trunc-normal 0.02 x head_init_scale); timm head removed"),
        "new_module_lr_scale": 1.0,
    }
