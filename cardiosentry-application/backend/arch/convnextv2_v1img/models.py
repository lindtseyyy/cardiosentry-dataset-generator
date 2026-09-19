"""Exact model factory used by the ConvNeXt-V2 Tiny V1-image training run.

The training notebook builds a bare timm model with twelve logits and a 0.20
drop-path rate.  Keeping those arguments here (instead of adapting a four-head
package) makes ``load_state_dict(strict=True)`` a faithful architecture check.
"""
from __future__ import annotations

REGISTRY = {
    "convnextv2_tiny": {
        "timm": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        "num_classes": 12,
        "drop_path_rate": 0.2,
        "pretrain": "FCMAE ImageNet-1K -> supervised ImageNet-22K -> ImageNet-1K",
    }
}


def build(name: str, pretrained: bool = False):
    import timm

    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {list(REGISTRY)}")
    spec = REGISTRY[name]
    model = timm.create_model(
        spec["timm"],
        pretrained=pretrained,
        num_classes=spec["num_classes"],
        drop_path_rate=spec["drop_path_rate"],
    )
    info = {
        **spec,
        "resolved_tag": spec["timm"],
        "params_m_actual": sum(p.numel() for p in model.parameters()) / 1e6,
        "timm_version": getattr(timm, "__version__", "?"),
    }
    return model, info
