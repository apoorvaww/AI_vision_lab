"""
xai.py  —  Explainable AI utilities for AI VisionLab
=====================================================
Provides Grad-CAM and Grad-CAM++ heatmap generation for any supported
PyTorch model architecture.  Returns PNG bytes overlaid on the original
image so the Flask route can stream them directly without touching disk.

Supported model families
------------------------
  dl       : SmallCNN, DeepCNN, ResNet18, MobileNetV2, VGG16, EfficientNet-B0
  hybrid   : HybridAttentionCNN, HybridCNNTransformer

Classical ML models (family='ml', 'ensemble') do not support gradient-based
XAI and will raise ``XAINotSupported``.

Usage
-----
    from stage1_pipeline.xai import gradcam_heatmap_png, XAINotSupported

    png_bytes = gradcam_heatmap_png(
        model,           # nn.Module, already trained
        image_path,      # path to an image file
        class_idx,       # integer class index to explain
        image_size,      # (int) resize to this before feeding the model
        method="gradcam" # or "gradcampp"
    )
"""

import io
from typing import Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchvision import transforms as tv_transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False


class XAINotSupported(Exception):
    """Raised when the model family does not support gradient-based XAI."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _load_tensor(image_path: str, image_size: int) -> "torch.Tensor":
    """Load and pre-process a single image into a (1, 3, H, W) tensor."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")

    if HAS_TV:
        tfm = tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        tensor = tfm(img).unsqueeze(0)
    else:
        img = img.resize((image_size, image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return tensor


def _pick_target_layer(model: "nn.Module", model_family: str, model_key: str) -> "nn.Module":
    """Return the last convolutional layer to hook for grad-cam.

    Each architecture is handled explicitly so we never rely on heuristics
    that might break across PyTorch versions.
    """
    name_lower = type(model).__name__.lower()

    # --- SmallCNN / DeepCNN (from-scratch CNNs) ---
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        # Walk backwards to find the last Conv2d
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, nn.Conv2d):
                return layer
            if isinstance(layer, nn.Sequential):
                for sub in reversed(list(layer.children())):
                    if isinstance(sub, nn.Conv2d):
                        return sub

    # --- HybridAttentionCNN  (has self.b3 as last conv block) ---
    if hasattr(model, "b3"):
        for layer in reversed(list(model.b3.children())):
            if isinstance(layer, nn.Conv2d):
                return layer

    # --- HybridCNNTransformer  (has self.stem) ---
    if hasattr(model, "stem") and isinstance(model.stem, nn.Sequential):
        for layer in reversed(list(model.stem.children())):
            if isinstance(layer, nn.Conv2d):
                return layer

    # --- ResNet18 ---
    if hasattr(model, "layer4"):
        return model.layer4[-1].conv2

    # --- MobileNetV2 ---
    if hasattr(model, "features") and hasattr(model, "classifier"):
        # torchvision MobileNetV2 — last InvertedResidual conv
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, nn.Sequential):
                for sub in reversed(list(layer.children())):
                    if isinstance(sub, nn.Conv2d):
                        return sub
            elif isinstance(layer, nn.Conv2d):
                return layer

    # --- VGG16 ---
    if hasattr(model, "features") and not hasattr(model, "layer4"):
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, nn.Conv2d):
                return layer

    # --- EfficientNet-B0 ---
    if hasattr(model, "features") and "efficient" in name_lower:
        for block in reversed(list(model.features.children())):
            for sub in reversed(list(block.modules())):
                if isinstance(sub, nn.Conv2d):
                    return sub

    raise XAINotSupported(
        f"Cannot auto-detect target layer for model type '{type(model).__name__}'. "
        "Grad-CAM is not supported for this architecture."
    )


def _overlay_heatmap(original: Image.Image, cam: np.ndarray,
                     alpha: float = 0.45) -> bytes:
    """Blend the CAM heatmap onto the original image and return PNG bytes."""
    # Normalise to [0,1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Colourmap: blue→green→red (jet-like, no matplotlib dependency)
    h, w = cam.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Blue channel
    rgb[:, :, 2] = np.clip((1.5 - np.abs(cam * 4 - 3)) * 255, 0, 255).astype(np.uint8)
    # Green channel
    rgb[:, :, 1] = np.clip((1.5 - np.abs(cam * 4 - 2)) * 255, 0, 255).astype(np.uint8)
    # Red channel
    rgb[:, :, 0] = np.clip((1.5 - np.abs(cam * 4 - 1)) * 255, 0, 255).astype(np.uint8)

    heatmap_img = Image.fromarray(rgb).resize(original.size, Image.BILINEAR)
    blended = Image.blend(original.convert("RGB"), heatmap_img, alpha=alpha)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gradcam_heatmap_png(
    model: "nn.Module",
    image_path: str,
    class_idx: int,
    image_size: int = 96,
    method: str = "gradcam",       # "gradcam" or "gradcampp"
    model_family: str = "dl",
    model_key: str = "",
) -> bytes:
    """Compute a Grad-CAM or Grad-CAM++ heatmap and return PNG bytes.

    Parameters
    ----------
    model       : trained nn.Module (eval mode is set internally)
    image_path  : path to the input image file
    class_idx   : class index whose activation map to visualise
    image_size  : spatial size to resize the image before inference
    method      : "gradcam" or "gradcampp"
    model_family: registry family string — "ml"/"ensemble" raises XAINotSupported

    Returns
    -------
    PNG bytes of the heatmap overlaid on the original image.
    """
    if not HAS_TORCH:
        raise XAINotSupported("PyTorch is not available.")
    if model_family in ("ml", "ensemble"):
        raise XAINotSupported(
            f"Grad-CAM is not applicable to '{model_family}' models "
            "(no convolutional layers / no gradients)."
        )

    device = next(model.parameters()).device
    model.eval()

    # Identify target convolutional layer
    target_layer = _pick_target_layer(model, model_family, model_key)

    # Storage for forward activations and backward gradients
    activations: list = []
    gradients:   list = []

    def fwd_hook(module, inp, out):
        activations.append(out)
        out.register_hook(lambda g: gradients.append(g.detach()))

    fwd_handle = target_layer.register_forward_hook(fwd_hook)

    try:
        tensor = _load_tensor(image_path, image_size).to(device)
        tensor.requires_grad_(True)

        logits = model(tensor)                  # forward pass
        score  = logits[0, class_idx]
        model.zero_grad()
        score.backward()                        # backward pass

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        act  = activations[0].squeeze(0).detach()        # C × H × W
        grad = gradients[0].squeeze(0).detach()          # C × H × W

        if method == "gradcampp":
            # Grad-CAM++ weights: second-order gradient correction
            relu_grad = F.relu(grad)
            grad_sq   = relu_grad ** 2
            grad_cu   = relu_grad ** 3
            sum_act   = act.sum(dim=(1, 2), keepdim=True)           # C×1×1
            denom     = 2 * grad_sq + sum_act * grad_cu             # C×H×W
            denom     = torch.where(denom != 0, denom,
                                    torch.ones_like(denom))
            alpha     = grad_sq / denom                             # C×H×W
            weights   = (alpha * relu_grad).sum(dim=(1, 2))        # C
        else:
            # Standard Grad-CAM: global average of gradients
            weights = grad.mean(dim=(1, 2))                        # C

        # Weighted combination of activation maps
        cam = (weights[:, None, None] * act).sum(dim=0)           # H × W
        cam = F.relu(cam).cpu().numpy()                            # ReLU

        # Resize CAM to input image resolution
        cam_img = Image.fromarray(cam).resize(
            (image_size, image_size), Image.BILINEAR)
        cam_np  = np.array(cam_img, dtype=np.float32)

    finally:
        fwd_handle.remove()

    # Load original (un-normalised) image for overlay
    with Image.open(image_path) as orig:
        orig = orig.convert("RGB").resize((image_size, image_size))

    return _overlay_heatmap(orig, cam_np, alpha=0.45)
