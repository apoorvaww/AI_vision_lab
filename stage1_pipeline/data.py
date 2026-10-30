"""Dataset discovery and validation for AI VisionLab.

Supports both the normal ImageFolder layout::

    dataset/class_a/*.jpg
    dataset/class_b/*.jpg

and common split datasets such as Intel Image Classification::

    dataset/seg_train/class_a/*.jpg
    dataset/seg_test/class_a/*.jpg
    dataset/seg_pred/*.jpg

The important distinction is that ``seg_pred`` is an *unlabelled prediction
set*, not a class. Dataset discovery therefore happens before class
validation instead of assuming every top-level directory is a class.
"""

import os

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetReport:
    root: str
    classes: List[str] = field(default_factory=list)
    valid_files: List[Tuple[str, str]] = field(default_factory=list)
    test_files: List[Tuple[str, str]] = field(default_factory=list)
    corrupted_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)

    dataset_format: str = "ImageFolder"

    train_root: str = ""
    test_root: Optional[str] = None
    prediction_root: Optional[str] = None

    split_counts: Dict[str, int] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)
    test_class_counts: Dict[str, int] = field(default_factory=dict)

    prediction_count: int = 0

    duplicate_files_removed: int = 0
    duplicate_conflicts: List[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_valid(self) -> int:
        return len(self.valid_files)

    def summary(self) -> dict:
        return {
            "root": self.root,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "num_valid_images": self.num_valid,
            "num_corrupted": len(self.corrupted_files),
            "num_skipped_non_image": len(self.skipped_files),
            "dataset_format": self.dataset_format,
            "train_images": self.split_counts.get("train", self.num_valid),
            "test_images": self.split_counts.get("test", 0),
            "prediction_images": self.prediction_count,
            "has_test_split": bool(self.test_root),
            "has_prediction_split": bool(self.prediction_root),
            "class_counts": self.class_counts,
            "test_class_counts": self.test_class_counts,
            "duplicate_files_removed": self.duplicate_files_removed,
            "duplicate_conflicts": len(self.duplicate_conflicts),
        }


def _visible_dirs(root: str) -> List[str]:
    """Return visible directories directly under root."""
    if not os.path.isdir(root):
        return []

    return sorted(
        name
        for name in os.listdir(root)
        if not name.startswith(".")
        and os.path.isdir(os.path.join(root, name))
    )


def _find_dataset_root(root: str) -> str:
    """Remove harmless single wrapper folders created by ZIP extraction."""
    current = root

    for _ in range(4):
        dirs = _visible_dirs(current)

        files = [
            f
            for f in os.listdir(current)
            if not f.startswith(".")
            and os.path.isfile(os.path.join(current, f))
        ]

        if len(dirs) == 1 and not files:
            current = os.path.join(current, dirs[0])
        else:
            break

    return current


def _image_files(
    root: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Return valid, corrupted and skipped image paths directly under root."""
    valid = []
    corrupted = []
    skipped = []

    if not os.path.isdir(root):
        return valid, corrupted, skipped

    for fname in sorted(os.listdir(root)):
        fpath = os.path.join(root, fname)

        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()

        if ext not in VALID_EXTENSIONS:
            skipped.append(fpath)
            continue

        try:
            with Image.open(fpath) as img:
                img.verify()

            valid.append(fpath)

        except (UnidentifiedImageError, OSError):
            corrupted.append(fpath)

    return valid, corrupted, skipped


def _validate_labeled_root(
    root: str,
    min_images_per_class: int,
):
    """Validate root/class_name/image and return discovery data."""
    class_dirs = _visible_dirs(root)

    if len(class_dirs) < 2:
        raise ValueError(
            f"Could not detect a labeled dataset with at least 2 class "
            f"folders in '{root}'. Expected folders such as class_a/, "
            f"class_b/ or a split such as seg_train/class_a/."
        )

    valid_files: List[Tuple[str, str]] = []
    corrupted_files: List[str] = []
    skipped_files: List[str] = []
    counts: Dict[str, int] = {}

    for class_name in class_dirs:
        class_path = os.path.join(root, class_name)
        count = 0

        for fname in sorted(os.listdir(class_path)):
            fpath = os.path.join(class_path, fname)

            if not os.path.isfile(fpath):
                continue

            ext = os.path.splitext(fname)[1].lower()

            if ext not in VALID_EXTENSIONS:
                skipped_files.append(fpath)
                continue

            try:
                with Image.open(fpath) as img:
                    img.verify()

                valid_files.append((fpath, class_name))
                count += 1

            except (UnidentifiedImageError, OSError):
                corrupted_files.append(fpath)

        counts[class_name] = count

        if min_images_per_class and count < min_images_per_class:
            raise ValueError(
                f"Class '{class_name}' has only {count} valid images "
                f"(minimum required: {min_images_per_class})."
            )

    return (
        class_dirs,
        valid_files,
        corrupted_files,
        skipped_files,
        counts,
    )


def _has_valid_images_direct(root: str) -> bool:
    """Return True if root contains at least one valid image directly."""
    valid, _, _ = _image_files(root)
    return bool(valid)


def _looks_like_labeled_root(root: str) -> bool:
    """Return True when root contains at least two class-like directories."""
    if not os.path.isdir(root):
        return False

    class_like = 0

    for name in _visible_dirs(root):
        child = os.path.join(root, name)

        if _has_valid_images_direct(child):
            class_like += 1

            if class_like >= 2:
                return True

    return False


def _find_labeled_root(
    root: str,
    max_depth: int = 4,
) -> str:
    """Find the actual ImageFolder root inside common dataset wrappers.

    This intentionally ignores folders such as ``Labels``, ``metadata`` and
    other non-image directories. It also supports layouts like::

        dataset/Images/cat/*.jpg
        dataset/Images/dog/*.jpg

        dataset/Labels/...

    without treating ``Labels`` as a class.
    """
    if _looks_like_labeled_root(root):
        return root

    preferred = {
        "images",
        "image",
        "data",
        "dataset",
        "classes",
        "train",
        "training",
    }

    queue = [(root, 0)]
    candidates = []

    while queue:
        current, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        for name in _visible_dirs(current):
            child = os.path.join(current, name)

            if name.lower() in {
                "labels",
                "label",
                "annotations",
                "annotation",
                "masks",
                "mask",
                "__macosx",
            }:
                # These are metadata/annotation folders, not classes.
                continue

            if _looks_like_labeled_root(child):
                score = (
                    1 if name.lower() in preferred else 0,
                    -depth,
                )

                candidates.append((score, child))

            queue.append((child, depth + 1))

    if candidates:
        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return candidates[0][1]

    return root


def validate_and_load(
    dataset_root: str,
    min_images_per_class: int = 5,
) -> DatasetReport:
    """Automatically detect and validate common image-classification layouts.

    Supported examples include ImageFolder, Intel's ``seg_train`` /
    ``seg_test`` / ``seg_pred`` structure, and wrapped datasets where the
    labeled class folders live under an ``Images``/``Data``/``Dataset`` folder.

    Non-image folders such as ``Labels`` and ``Annotations`` are never
    interpreted as classes.
    """
    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"Dataset path does not exist: {dataset_root}"
        )

    root = _find_dataset_root(dataset_root)

    names = {
        name.lower(): name
        for name in _visible_dirs(root)
    }

    split_train = next(
        (
            names[n]
            for n in ("seg_train", "train", "training")
            if n in names
        ),
        None,
    )

    split_test = next(
        (
            names[n]
            for n in (
                "seg_test",
                "test",
                "testing",
                "val",
                "validation",
            )
            if n in names
        ),
        None,
    )

    split_pred = next(
        (
            names[n]
            for n in (
                "seg_pred",
                "pred",
                "prediction",
                "predict",
            )
            if n in names
        ),
        None,
    )

    if split_train:
        train_root = os.path.join(root, split_train)

        # Some datasets add an Images wrapper inside train/.
        if not _looks_like_labeled_root(train_root):
            train_root = _find_labeled_root(train_root)

        fmt = (
            "Split dataset (train/test/prediction)"
            if (split_test or split_pred)
            else "Split dataset"
        )

    else:
        train_root = _find_labeled_root(root)
        fmt = "ImageFolder"

        if os.path.realpath(train_root) != os.path.realpath(root):
            fmt = "ImageFolder (nested/wrapped)"

    try:
        (
            classes,
            valid_files,
            corrupted,
            skipped,
            class_counts,
        ) = _validate_labeled_root(
            train_root,
            min_images_per_class,
        )

    except ValueError as exc:
        # Give a structure-level error rather than falsely calling folders
        # such as Labels a class when no classification root can be detected.
        if (
            train_root == root
            and not _looks_like_labeled_root(root)
        ):
            dirs = _visible_dirs(root)

            raise ValueError(
                "Could not automatically detect class folders. "
                "VisionLab expects a classification layout such as "
                "'class_name/image.jpg', "
                "'Images/class_name/image.jpg', or "
                "'seg_train/class_name/image.jpg'. "
                f"Detected top-level folders: {dirs or 'none'}. "
                "If your dataset uses separate image/label files rather "
                "than class folders, that format is not yet supported "
                "for supervised image classification."
            ) from exc

        raise

    report = DatasetReport(
        root=root,
        classes=classes,
        valid_files=valid_files,
        corrupted_files=corrupted,
        skipped_files=skipped,
        dataset_format=fmt,
        train_root=train_root,
        class_counts=class_counts,
        split_counts={
            "train": len(valid_files),
        },
    )

    if split_test:
        test_root = os.path.join(root, split_test)

        if not _looks_like_labeled_root(test_root):
            test_root = _find_labeled_root(test_root)

        (
            test_classes,
            _,
            test_corrupt,
            test_skipped,
            test_counts,
        ) = _validate_labeled_root(
            test_root,
            min_images_per_class=1,
        )

        if set(test_classes) != set(classes):
            missing = sorted(
                set(classes) - set(test_classes)
            )

            extra = sorted(
                set(test_classes) - set(classes)
            )

            raise ValueError(
                "Test split classes do not match the training split. "
                f"Missing: {missing or 'none'}; "
                f"Extra: {extra or 'none'}."
            )

        report.test_root = test_root
        report.test_class_counts = test_counts

        report.split_counts["test"] = sum(
            test_counts.values()
        )

        report.corrupted_files.extend(test_corrupt)
        report.skipped_files.extend(test_skipped)

    if split_pred:
        pred_root = os.path.join(root, split_pred)

        # Prediction sets are intentionally unlabeled and may contain images
        # directly in the folder. Never run class validation on this root.
        pred_valid, pred_corrupt, pred_skipped = _image_files(
            pred_root
        )

        report.prediction_root = pred_root
        report.prediction_count = len(pred_valid)

        report.split_counts["prediction"] = len(pred_valid)

        report.corrupted_files.extend(pred_corrupt)
        report.skipped_files.extend(pred_skipped)

    return report


def discover_for_training(
    dataset_root: str,
) -> DatasetReport:
    """Discover a dataset without creating an upload-time validation gate.

    If the archive already contains a labeled test/validation split, that split
    is preserved and used by the training pipeline. Otherwise the pipeline will
    create a stratified holdout from the discovered training images.
    """
    if not os.path.isdir(dataset_root):
        raise ValueError(
            f"Dataset path does not exist: {dataset_root}"
        )

    root = _find_dataset_root(dataset_root)

    names = {
        name.lower(): name
        for name in _visible_dirs(root)
    }

    split_train = next(
        (
            names[n]
            for n in ("seg_train", "train", "training")
            if n in names
        ),
        None,
    )

    split_test = next(
        (
            names[n]
            for n in (
                "seg_test",
                "test",
                "testing",
                "val",
                "validation",
            )
            if n in names
        ),
        None,
    )

    # ---------------------------------------------------------------
    # Find the actual labeled training directory.
    # ---------------------------------------------------------------
    if split_train:
        candidate_train_root = os.path.join(
            root,
            split_train,
        )

        if _looks_like_labeled_root(candidate_train_root):
            train_root = candidate_train_root
        else:
            train_root = _find_labeled_root(
                candidate_train_root
            )

    else:
        if _looks_like_labeled_root(root):
            train_root = root
        else:
            train_root = _find_labeled_root(root)

    # ---------------------------------------------------------------
    # Find class directories.
    # ---------------------------------------------------------------
    class_dirs = _visible_dirs(train_root)

    if len(class_dirs) < 2:
        raise ValueError(
            "Could not find at least two class folders in the "
            "uploaded dataset. "
            f"Training directory detected: {train_root}. "
            f"Folders found: {class_dirs or 'none'}. "
            "Expected a structure such as "
            "dataset/class_a/image.jpg and "
            "dataset/class_b/image.jpg."
        )

    # ---------------------------------------------------------------
    # Collect training images.
    # ---------------------------------------------------------------
    valid_files: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}

    for cls in class_dirs:
        cls_path = os.path.join(
            train_root,
            cls,
        )

        count = 0

        for fname in sorted(os.listdir(cls_path)):
            path = os.path.join(
                cls_path,
                fname,
            )

            if not os.path.isfile(path):
                continue

            if (
                os.path.splitext(fname)[1].lower()
                not in VALID_EXTENSIONS
            ):
                continue

            try:
                with Image.open(path) as img:
                    img.verify()

                valid_files.append(
                    (path, cls)
                )

                count += 1

            except (UnidentifiedImageError, OSError):
                # Corrupted files are ignored here.
                # validate_and_load() provides the detailed report.
                pass

        counts[cls] = count

    if not valid_files:
        raise ValueError(
            "No supported image files were found in the dataset."
        )

    # ---------------------------------------------------------------
    # Remove exact duplicate files BEFORE creating the train/test
    # split. This prevents the same image from appearing on both sides
    # of the evaluation boundary.
    # ---------------------------------------------------------------
    from .models import deduplicate_files

    valid_files, removed, conflicts = deduplicate_files(
        valid_files
    )

    if not valid_files:
        raise ValueError(
            "No unique labeled images remain after removing "
            "duplicates."
        )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Recalculate class counts AFTER duplicate removal.
    #
    # A class may have lost all of its images because of duplicate
    # removal or cross-label conflicts. In that case valid_files is
    # still non-empty, but classification can no longer be performed.
    # ---------------------------------------------------------------
    remaining_class_counts = {
        cls: sum(
            1
            for _, label in valid_files
            if label == cls
        )
        for cls in class_dirs
    }

    remaining_classes = [
        cls
        for cls, count in remaining_class_counts.items()
        if count > 0
    ]

    if len(remaining_classes) < 2:
        raise ValueError(
            "Fewer than two classes remain after duplicate removal. "
            f"Original classes: {class_dirs}. "
            f"Remaining classes: "
            f"{remaining_classes or 'none'}. "
            f"Images removed: {removed}. "
            f"Cross-label duplicate conflicts: "
            f"{len(conflicts)}. "
            "Classification requires at least two classes with "
            "usable images."
        )

    # Only retain classes that actually have usable images.
    class_dirs = remaining_classes

    # ---------------------------------------------------------------
    # Build the dataset report.
    # ---------------------------------------------------------------
    report = DatasetReport(
        root=root,
        classes=class_dirs,
        valid_files=valid_files,
        dataset_format=(
            "Split dataset"
            if split_train
            else "ImageFolder"
        ),
        train_root=train_root,
        class_counts=remaining_class_counts,
        split_counts={
            "train": len(valid_files),
        },
        duplicate_files_removed=removed,
        duplicate_conflicts=conflicts,
    )

    # ---------------------------------------------------------------
    # If a labeled test split exists, load it separately.
    # ---------------------------------------------------------------
    if split_test:
        test_root = os.path.join(
            root,
            split_test,
        )

        test_files: List[Tuple[str, str]] = []
        test_counts: Dict[str, int] = {}

        for cls in class_dirs:
            cls_path = os.path.join(
                test_root,
                cls,
            )

            if not os.path.isdir(cls_path):
                continue

            for fname in sorted(
                os.listdir(cls_path)
            ):
                path = os.path.join(
                    cls_path,
                    fname,
                )

                if not os.path.isfile(path):
                    continue

                if (
                    os.path.splitext(fname)[1].lower()
                    not in VALID_EXTENSIONS
                ):
                    continue

                try:
                    with Image.open(path) as img:
                        img.verify()

                    test_files.append(
                        (path, cls)
                    )

                    test_counts[cls] = (
                        test_counts.get(cls, 0) + 1
                    )

                except (UnidentifiedImageError, OSError):
                    pass

        if os.path.isdir(test_root):
            if set(test_counts) != set(class_dirs):
                missing = sorted(
                    set(class_dirs) - set(test_counts)
                )

                extra = sorted(
                    set(test_counts) - set(class_dirs)
                )

                raise ValueError(
                    "Test split classes do not match training "
                    "classes. "
                    f"Missing: {missing or 'none'}; "
                    f"Extra: {extra or 'none'}."
                )

            if not test_files:
                raise ValueError(
                    "A labeled test split was found, but it "
                    "contains no usable images."
                )

            # -------------------------------------------------------
            # Remove exact duplicates that also occur in training.
            # -------------------------------------------------------
            train_hashes = set()

            from .models import _file_hash

            for path, _ in valid_files:
                train_hashes.add(
                    _file_hash(path)
                )

            clean_test = []

            for item in test_files:
                if (
                    _file_hash(item[0])
                    not in train_hashes
                ):
                    clean_test.append(item)

            if not clean_test:
                raise ValueError(
                    "All images in the supplied test split also "
                    "occur in training; a leakage-safe test set "
                    "is required."
                )

            report.test_files = clean_test
            report.test_root = test_root

            report.test_class_counts = {
                cls: sum(
                    1
                    for _, label in clean_test
                    if label == cls
                )
                for cls in class_dirs
            }

            report.split_counts["test"] = len(
                clean_test
            )

    return report


def inspect_dataset_characteristics(data, class_counts: Optional[Dict[str, int]] = None) -> dict:
    """Inspect dataset image dimensions, formats, color channels, and balance."""
    if isinstance(data, DatasetReport):
        valid_files = data.valid_files
        if class_counts is None:
            class_counts = data.class_counts
    elif hasattr(data, "valid_files"):
        valid_files = data.valid_files
        if class_counts is None and hasattr(data, "class_counts"):
            class_counts = data.class_counts
    else:
        valid_files = data

    if not valid_files:
        return {
            "total_images": 0,
            "avg_width": 0,
            "avg_height": 0,
            "min_width": 0,
            "max_width": 0,
            "min_height": 0,
            "max_height": 0,
            "uniform_size": True,
            "formats": {},
            "color_modes": {},
            "imbalance_ratio": 1.0,
            "is_imbalanced": False,
            "avg_file_size_kb": 0.0,
        }

    sample_size = min(len(valid_files), 120)
    # Stratified or evenly sampled subset for speed
    step = max(1, len(valid_files) // sample_size)
    sampled = valid_files[::step][:sample_size]

    widths = []
    heights = []
    formats = {}
    modes = {}
    total_bytes = 0

    for path, _ in sampled:
        try:
            sz = os.path.getsize(path)
            total_bytes += sz
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                fmt = img.format or os.path.splitext(path)[1].replace(".", "").upper()
                formats[fmt] = formats.get(fmt, 0) + 1
                modes[img.mode] = modes.get(img.mode, 0) + 1
        except Exception:
            continue

    if not widths:
        widths, heights = [96], [96]

    avg_w = int(sum(widths) / len(widths))
    avg_h = int(sum(heights) / len(heights))
    min_w, max_w = min(widths), max(widths)
    min_h, max_h = min(heights), max(heights)
    uniform = (min_w == max_w and min_h == max_h)
    avg_size_kb = round(total_bytes / (len(sampled) * 1024), 1) if sampled else 0.0

    counts = class_counts or {}
    if counts:
        max_c = max(counts.values())
        min_c = max(min(counts.values()), 1)
        imbalance_ratio = round(max_c / min_c, 2)
        maj_class = max(counts, key=counts.get)
        min_class = min(counts, key=counts.get)
    else:
        imbalance_ratio = 1.0
        maj_class = "N/A"
        min_class = "N/A"

    return {
        "total_images": len(valid_files),
        "sampled_count": len(sampled),
        "avg_width": avg_w,
        "avg_height": avg_h,
        "min_width": min_w,
        "max_width": max_w,
        "min_height": min_h,
        "max_height": max_h,
        "uniform_size": uniform,
        "formats": formats,
        "color_modes": modes,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio >= 1.35,
        "majority_class": maj_class,
        "minority_class": min_class,
        "avg_file_size_kb": avg_size_kb,
    }