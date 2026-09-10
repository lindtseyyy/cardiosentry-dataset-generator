"""Grad-CAM on ConvNeXt-V2 / ConvNeXt for CardioSentry 12-lead ECG models.

Applies Class Activation Mapping (Grad-CAM) to visualize where the ConvNeXt
backbone attends when diagnosing each of the four clinical cardiac conditions:
STEMI, AF (Atrial Fibrillation), LVH (Left Ventricular Hypertrophy), and NORMAL.

Supports loading trained CardioSentry checkpoints (e.g. convnextv2_tiny@nat768.pt)
with automatic fallback to timm ImageNet pretrained weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
DISPLAY_NAMES = {
    "STEMI": "ST-Elevation MI (STEMI)",
    "AF": "Atrial Fibrillation",
    "LVH": "Left Ventricular Hypertrophy",
    "NORMAL": "Normal Rhythm",
}
DEFAULT_THRESHOLDS = [0.73, 0.65, 0.92, 0.88]  # from convnextv2_tiny@nat768 val tuning


def build_model(
    timm_tag: str = "convnextv2_tiny.fcmae_ft_in1k",
    weights_path: Path | None = None,
    device: str = "cpu",
) -> tuple[torch.nn.Module, list[float], dict]:
    """Instantiate ConvNeXt model and load weights if available."""
    import timm

    thresholds = list(DEFAULT_THRESHOLDS)
    meta = {"tag": timm_tag, "checkpoint": None}

    has_weights = weights_path is not None and weights_path.is_file()

    print(f"Building model {timm_tag!r} (pretrained fallback={not has_weights})...")
    model = timm.create_model(
        timm_tag,
        pretrained=not has_weights,
        num_classes=len(LABELS),
    )

    if has_weights:
        print(f"Loading checkpoint weights from: {weights_path}")
        blob = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            print(f"Warning: partial load mismatch. Missing={missing}, Unexpected={unexpected}")
        else:
            print("Successfully loaded checkpoint with strict state_dict match.")

        if isinstance(blob, dict):
            if "thresholds" in blob:
                thresholds = [float(t) for t in blob["thresholds"]]
            meta["epoch"] = blob.get("epoch")
            meta["fingerprint"] = blob.get("fingerprint")
            meta["val_metrics"] = blob.get("val")
    else:
        print(f"No checkpoint file found at {weights_path}. Running with timm pretrained weights.")
        thresholds = [0.5, 0.5, 0.5, 0.5]

    model.to(device)
    model.eval()
    return model, thresholds, meta


def preprocess_ecg(
    image_path: Path,
    input_hw: tuple[int, int] = (768, 1024),
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess ECG photograph to match training input geometry."""
    img_pil = Image.open(image_path).convert("RGB")
    h, w = input_hw

    # Bicubic resize to target geometry (height, width)
    resized_pil = img_pil.resize((w, h), Image.Resampling.BICUBIC)
    rgb_np = np.array(resized_pil, dtype=np.uint8)

    # Standard ImageNet normalization
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(resized_pil).unsqueeze(0)  # (1, 3, H, W)
    return tensor, rgb_np


def resolve_layer(model: torch.nn.Module, target_layer_name: str | None = None) -> tuple[str, torch.nn.Module]:
    """Locate the target layer for Grad-CAM."""
    named_mods = dict(model.named_modules())

    if target_layer_name and target_layer_name in named_mods:
        return target_layer_name, named_mods[target_layer_name]

    # Auto-detection default: search for stages.3 or stages[-1]
    candidates = ["stages.3", "stages.3.blocks.2", "stages.3.blocks.-1", "stages.2"]
    for c in candidates:
        if c in named_mods:
            return c, named_mods[c]

    # Fallback to the last 4D module
    for name, mod in reversed(list(named_mods.items())):
        if "conv" in name.lower() or "stages" in name.lower():
            return name, mod

    raise RuntimeError(f"Could not locate target layer {target_layer_name!r} in model.")


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer_name: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, np.ndarray], dict]:
    """Compute Grad-CAM for all 4 diagnostic classes.

    Uses late-graph autograd: forward pass runs with detached leaf at the
    target module, recording autograd only from the target activations forward.
    """
    layer_name, module = resolve_layer(model, target_layer_name)
    print(f"Grad-CAM target layer resolved to: {layer_name} ({type(module).__name__})")

    activation: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(t, torch.Tensor) and t.dim() == 4:
            leaf = t.detach().requires_grad_(True)
            activation["act"] = leaf
            torch.set_grad_enabled(True)
            return leaf
        return None

    handle = module.register_forward_hook(fwd_hook)

    try:
        with torch.no_grad():
            x = input_tensor.to(device)
            logits = model(x)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits)[0]

            if "act" not in activation:
                raise RuntimeError(
                    f"Target layer {layer_name!r} produced no 4-D feature map during forward pass."
                )

            acts = activation["act"]  # (1, C, H, W)
            _, channels, gh, gw = acts.shape

            cams: dict[str, np.ndarray] = {}

            for idx, label in enumerate(LABELS):
                score = logits[0, idx]
                grads = torch.autograd.grad(score, acts, retain_graph=(idx < len(LABELS) - 1))[0]

                # Channel weights = Global Average Pooling of gradients
                alpha = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

                # Positive weighted activation combination
                cam = torch.relu((alpha * acts).sum(dim=1, keepdim=True))  # (1, 1, gh, gw)
                cam = cam.squeeze().detach().cpu().numpy().astype(np.float32)

                # Normalize to [0, 1]
                c_min, c_max = cam.min(), cam.max()
                if c_max - c_min > 1e-8:
                    cam = (cam - c_min) / (c_max - c_min)
                else:
                    cam = np.zeros_like(cam)

                cams[label] = cam

    finally:
        handle.remove()

    meta = {
        "target_layer": layer_name,
        "feature_grid": (gh, gw),
        "channels": channels,
    }
    return probs.cpu(), cams, meta


def overlay_cam_on_image(
    rgb_img: np.ndarray,
    cam_2d: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Resize heatmap to image size and blend with RGB image."""
    h, w = rgb_img.shape[:2]
    cam_resized = cv2.resize(cam_2d, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * cam_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    blended = np.float32(colored_heatmap) * alpha + np.float32(rgb_img) * (1.0 - alpha)
    return np.uint8(np.clip(blended, 0, 255))


def save_per_condition_overlays(
    rgb_img: np.ndarray,
    probs: torch.Tensor,
    cams: dict[str, np.ndarray],
    thresholds: list[float],
    meta: dict,
    out_dir: Path,
    image_stem: str,
) -> dict[str, Path]:
    """Save high-resolution Grad-CAM overlay per clinical condition."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    h, w = rgb_img.shape[:2]

    for i, label in enumerate(LABELS):
        p = float(probs[i])
        thresh = float(thresholds[i])
        is_pos = p >= thresh

        overlay = overlay_cam_on_image(rgb_img, cams[label], alpha=0.45)

        # Draw a sleek professional diagnostic badge on top
        annotated = overlay.copy()
        badge_h = 60
        header = np.zeros((badge_h, w, 3), dtype=np.uint8)
        header[:] = (20, 24, 28)  # dark slate background

        status_text = "POSITIVE" if is_pos else "NEGATIVE"
        badge_color = (60, 220, 100) if (is_pos and label != "NORMAL") or (not is_pos and label == "NORMAL") else (
            (80, 80, 240) if is_pos else (180, 180, 180)
        )
        if is_pos and label != "NORMAL":
            badge_color = (60, 70, 240)  # red/crimson in BGR
        elif is_pos and label == "NORMAL":
            badge_color = (60, 220, 100) # green in BGR

        text_left = f"{DISPLAY_NAMES[label]}"
        text_right = f"Prob: {p:.4f}  |  Thresh: {thresh:.2f}  [{status_text}]"

        cv2.putText(header, text_left, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, text_right, (w - 520, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, badge_color, 2, cv2.LINE_AA)

        combined = np.vstack([header, annotated])
        cond_out_path = out_dir / f"{image_stem}_gradcam_{label}.png"
        cv2.imwrite(str(cond_out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        saved_paths[label] = cond_out_path

    return saved_paths


def render_composite_figure(
    rgb_img: np.ndarray,
    probs: torch.Tensor,
    cams: dict[str, np.ndarray],
    thresholds: list[float],
    meta: dict,
    out_path: Path,
):
    """Save high-resolution diagnostic composite plot."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 22), dpi=150)
    fig.patch.set_facecolor("#121417")

    # 1. Original ECG Sheet
    axes[0].imshow(rgb_img)
    axes[0].set_title(
        f"Input ECG Photograph ({rgb_img.shape[1]}x{rgb_img.shape[0]} px) | Target Layer: {meta['target_layer']} (Grid: {meta['feature_grid'][1]}x{meta['feature_grid'][0]}, Ch: {meta['channels']})",
        color="#FFFFFF",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    axes[0].axis("off")

    # 2-5. Grad-CAM heatmaps per diagnosis
    for i, label in enumerate(LABELS):
        ax = axes[i + 1]
        p = float(probs[i])
        thresh = float(thresholds[i])
        is_pos = p >= thresh

        overlay = overlay_cam_on_image(rgb_img, cams[label], alpha=0.45)
        ax.imshow(overlay)

        verdict_color = "#FF4444" if (is_pos and label != "NORMAL") else ("#00E676" if is_pos else "#888888")
        status_text = "POSITIVE" if is_pos else "NEGATIVE"

        title = f"{DISPLAY_NAMES[label]} | Probability: {p:.4f} vs Threshold: {thresh:.2f} [{status_text}]"
        ax.set_title(title, color=verdict_color, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"\nSaved composite Grad-CAM visualization to: {out_path.resolve()}")



def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM on ConvNeXt ECG Model.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("cardiosentry-application/tests/fixtures/corpus_sheet.jpg"),
        help="Path to ECG sheet image.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("cardiosentry-application/models/weights/convnextv2_tiny@nat768.pt"),
        help="Path to ConvNeXt trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--timm-model",
        type=str,
        default="convnextv2_tiny.fcmae_ft_in1k",
        help="timm architecture tag.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default="stages.3",
        help="Layer to target with Grad-CAM.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/convnext_gradcam/convnext_gradcam_overlay.png"),
        help="Output path for the generated composite visualization.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Input height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Input width.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    if not args.image.is_file():
        print(f"Error: image not found at {args.image}", file=sys.stderr)
        sys.exit(1)

    model, thresholds, meta = build_model(
        timm_tag=args.timm_model,
        weights_path=args.weights,
        device=device,
    )

    print(f"Preprocessing {args.image} to ({args.height}, {args.width})...")
    tensor, rgb_img = preprocess_ecg(args.image, input_hw=(args.height, args.width))

    print(f"Computing Grad-CAM with target layer: {args.target_layer}...")
    probs, cams, cam_meta = compute_gradcam(
        model=model,
        input_tensor=tensor,
        target_layer_name=args.target_layer,
        device=device,
    )

    print("\n--- Model Predictions ---")
    for i, label in enumerate(LABELS):
        p = float(probs[i])
        th = float(thresholds[i])
        flag = " [POSITIVE]" if p >= th else ""
        print(f"  {label:<8}: prob = {p:8.4f} (threshold = {th:.2f}){flag}")

    out_dir = args.out.parent if args.out.suffix else args.out
    stem = args.image.stem
    cond_paths = save_per_condition_overlays(
        rgb_img=rgb_img,
        probs=probs,
        cams=cams,
        thresholds=thresholds,
        meta=cam_meta,
        out_dir=out_dir,
        image_stem=stem,
    )

    composite_path = (out_dir / f"{stem}_gradcam_composite.png") if not args.out.suffix else args.out
    render_composite_figure(
        rgb_img=rgb_img,
        probs=probs,
        cams=cams,
        thresholds=thresholds,
        meta=cam_meta,
        out_path=composite_path,
    )

    print("\n--- Generated Grad-CAM Output Images ---")
    for label, path in cond_paths.items():
        print(f"  {label:<8} overlay: {path.resolve()}")
    print(f"  Composite figure : {composite_path.resolve()}")



if __name__ == "__main__":
    main()
