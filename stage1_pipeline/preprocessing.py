"""AI Preprocessing and Recommendation System for AI VisionLab.

Inspects actual dataset characteristics and generates data-driven
preprocessing recommendations (resizing, normalization, data augmentation,
class balancing, and validation splitting) with human-readable rationales.
"""

import io
import os
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


def generate_recommendations(summary: Dict[str, Any], characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Generate recommendations derived from actual dataset statistics."""
    total_images = summary.get("num_valid_images", 0)
    num_classes = summary.get("num_classes", 0)
    has_test_split = summary.get("has_test_split", False)

    avg_w = characteristics.get("avg_width", 96)
    avg_h = characteristics.get("avg_height", 96)
    uniform = characteristics.get("uniform_size", True)
    imbalance_ratio = characteristics.get("imbalance_ratio", 1.0)
    is_imbalanced = characteristics.get("is_imbalanced", False)
    maj_class = characteristics.get("majority_class", "majority")
    min_class = characteristics.get("minority_class", "minority")

    # 1. Resizing Recommendation
    if avg_w >= 160 or avg_h >= 160:
        rec_size = 128
        size_reason = (
            f"Native images average {avg_w}×{avg_h} px. Resizing to 128×128 standardizes tensor dimensions, "
            f"reduces training memory footprint, and preserves key spatial textures."
        )
    elif avg_w <= 64 and avg_h <= 64:
        rec_size = 64
        size_reason = (
            f"Native images are low resolution ({avg_w}×{avg_h} px). Preserving 64×64 avoids blurry upsampling artifacts."
        )
    else:
        rec_size = 96
        size_reason = (
            f"Native images average {avg_w}×{avg_h} px. A 96×96 target provides an optimal balance between "
            f"representational capacity and rapid iteration speed on modern backbones."
        )

    # 2. Normalization Recommendation
    rec_norm = "imagenet"
    norm_reason = (
        "Z-score channel normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], "
        "std=[0.229, 0.224, 0.225]) ensures zero-centered feature activations and fast gradient convergence."
    )

    # 3. Augmentation Recommendation & Granular Techniques
    if total_images < 300:
        rec_aug = True
        aug_level = "High Priority"
        aug_reason = (
            f"High Priority: Dataset contains only {total_images} images. Extensive data augmentation is "
            f"essential to synthetically expand sample diversity, prevent memorization, and combat overfitting."
        )
        rec_rot_deg = 15
        rec_crop = True
        rec_jitter = True
    elif total_images < 1500:
        rec_aug = True
        aug_level = "Moderate"
        aug_reason = (
            f"Recommended: With {total_images} images, targeted augmentation introduces realistic geometric and "
            f"photometric perturbations, improving model generalization on unseen holdout test data."
        )
        rec_rot_deg = 10
        rec_crop = avg_w >= 96
        rec_jitter = True
    else:
        rec_aug = True
        aug_level = "Standard"
        aug_reason = (
            f"Standard: Dataset volume is sufficient ({total_images} images). Conservative augmentation "
            f"(pose & lighting invariance) is advised without over-distorting patterns."
        )
        rec_rot_deg = 10
        rec_crop = False
        rec_jitter = False

    # Techniques analysis
    techniques = {
        "horizontal_flip": {
            "key": "horizontal_flip",
            "name": "Horizontal Reflection (Flip)",
            "recommended": True,
            "badge": "Pose Invariance",
            "rationale": "Mirrors images horizontally. Teaches left-right pose invariance while preserving semantic label identity.",
        },
        "vertical_flip": {
            "key": "vertical_flip",
            "name": "Vertical Inversion (Flip)",
            "recommended": False,  # Keep conservative for upright real-world scenes
            "badge": "Domain Sensitive",
            "rationale": "Disabled by default to preserve gravitational uprightness. Enable if analyzing aerial, satellite, microscopic, or circular objects.",
        },
        "random_rotation": {
            "key": "random_rotation",
            "name": f"Random Rotation (±{rec_rot_deg}°)",
            "recommended": True,
            "degree": rec_rot_deg,
            "badge": "Orientation Invariance",
            "rationale": f"Applies subtle angular rotation up to ±{rec_rot_deg}° to simulate slight camera angle variations.",
        },
        "color_jitter": {
            "key": "color_jitter",
            "name": "Color Jitter (Brightness & Contrast)",
            "recommended": rec_jitter,
            "badge": "Photometric Invariance",
            "rationale": "Perturbs brightness and contrast (±15%) so classifiers do not overfit to specific scene illumination.",
        },
        "random_crop": {
            "key": "random_crop",
            "name": "Random Resized Crop (Scale 85–100%)",
            "recommended": rec_crop,
            "badge": "Scale Invariance",
            "rationale": "Crops a random sub-region and resizes to target scale, simulating differing camera distances.",
        },
        "gaussian_blur": {
            "key": "gaussian_blur",
            "name": "Gentle Gaussian Blur (3×3)",
            "recommended": False,
            "badge": "Denoising Robustness",
            "rationale": "Applies gentle spatial smoothing. Useful when raw input images contain high-frequency sensor noise.",
        },
    }

    # 4. Class Balancing Recommendation
    if is_imbalanced:
        rec_balance = True
        balance_level = "High"
        balance_reason = (
            f"High Priority: Severe class skew detected ({imbalance_ratio}:1 ratio between '{maj_class}' and '{min_class}'). "
            f"Weighted sampling and class-weighted Cross-Entropy loss are required to prevent majority bias."
        )
    else:
        rec_balance = False
        balance_level = "Normal"
        balance_reason = (
            f"Balanced: Class distribution is uniform ({imbalance_ratio}:1 ratio). Standard uniform sampling is appropriate."
        )

    # 5. Train/Test Split Recommendation
    if has_test_split:
        rec_split = 1.0
        split_label = "Predefined test set"
        split_reason = "The uploaded dataset contains an explicit holdout test split. Preserving it prevents data leakage."
    elif total_images < 250:
        rec_split = 0.7
        split_label = "70% Train / 30% Test"
        split_reason = (
            f"With a small sample size ({total_images} images), a 30% holdout guarantees enough test examples per class for reliable evaluation."
        )
    else:
        rec_split = 0.8
        split_label = "80% Train / 20% Test"
        split_reason = "Standard 80/20 stratified split maximizes training data while maintaining statistically significant evaluation."

    return {
        "resizing": {
            "recommended_size": rec_size,
            "uniform_native": uniform,
            "avg_native": f"{avg_w}×{avg_h}",
            "rationale": size_reason,
        },
        "normalization": {
            "recommended": rec_norm,
            "options": ["imagenet", "minmax", "none"],
            "rationale": norm_reason,
        },
        "augmentation": {
            "recommended": rec_aug,
            "level": aug_level,
            "flips": True,
            "rotations": True,
            "color_jitter": rec_jitter,
            "rationale": aug_reason,
            "techniques": techniques,
        },
        "class_balancing": {
            "recommended": rec_balance,
            "level": balance_level,
            "imbalance_ratio": imbalance_ratio,
            "rationale": balance_reason,
        },
        "split": {
            "recommended_ratio": rec_split,
            "label": split_label,
            "has_predefined_test": has_test_split,
            "rationale": split_reason,
        },
        "summary_verdict": (
            f"AI VisionLab inspected {total_images} images across {num_classes} classes. "
            f"{'Augmentation and class balancing are strongly recommended.' if (rec_aug and rec_balance) else 'Standard preprocessing pipeline recommended.'}"
        ),
    }


def generate_transformed_preview(
    image_path: str,
    target_size: int = 96,
    aug_config: Optional[Dict[str, Any]] = None,
    apply_aug: bool = True,
    apply_norm: bool = True,
) -> bytes:
    """Generate a side-by-side or processed preview image of a sample based on selected techniques."""
    cfg = aug_config or {}

    with Image.open(image_path) as img:
        orig = img.convert("RGB")
        processed = orig.copy()

        if apply_aug:
            # Random Resized Crop simulation
            if cfg.get("random_crop", False):
                w, h = processed.size
                crop_scale = 0.88
                new_w, new_h = int(w * crop_scale), int(h * crop_scale)
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                processed = processed.crop((left, top, left + new_w, top + new_h))

            # Resize to target
            processed = processed.resize((target_size, target_size), Image.Resampling.BILINEAR)

            # Horizontal Flip
            if cfg.get("horizontal_flip", True):
                processed = processed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Vertical Flip
            if cfg.get("vertical_flip", False):
                processed = processed.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            # Rotation
            rot_deg = int(cfg.get("rotation_deg", 12))
            if cfg.get("random_rotation", True) and rot_deg > 0:
                processed = processed.rotate(rot_deg, fillcolor=(128, 128, 128))

            # Color Jitter
            if cfg.get("color_jitter", True):
                enhancer = ImageEnhance.Color(processed)
                processed = enhancer.enhance(1.25)
                contrast = ImageEnhance.Contrast(processed)
                processed = contrast.enhance(1.15)
                bright = ImageEnhance.Brightness(processed)
                processed = bright.enhance(1.1)

            # Gaussian Blur
            if cfg.get("gaussian_blur", False):
                processed = processed.filter(ImageFilter.GaussianBlur(radius=1.2))
        else:
            processed = processed.resize((target_size, target_size), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
