"""Model training/evaluation for AI VisionLab.

The important design rule in this module is that evaluation must measure
GENERALIZATION, not memorisation.  Classical models use deterministic image
features, while CNN/transfer-learning models use train-only augmentation and
class-weighted loss.  Every model is evaluated on files it never trained on.
"""

import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import torchvision
    from torchvision import models as tv_models
    from torchvision import transforms as tv_transforms
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

SEED = 42
FEATURE_IMAGE_SIZE = (64, 64)


def _seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _file_hash(path: str) -> str:
    """Exact-content hash used to prevent duplicate-image leakage."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deduplicate_files(files: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], int, List[str]]:
    """Remove exact duplicate image files before a train/test split.

    If the exact same bytes occur under two different labels, keeping either
    copy would make the ground truth ambiguous, so both are discarded.
    """
    by_hash = {}
    conflicts = set()
    duplicates = 0
    for path, label in files:
        digest = _file_hash(path)
        if digest in by_hash:
            duplicates += 1
            old_path, old_label = by_hash[digest]
            if old_label != label:
                conflicts.update({old_path, path})
        else:
            by_hash[digest] = (path, label)

    clean = [(p, l) for digest, (p, l) in by_hash.items() if p not in conflicts]
    return clean, duplicates + len(conflicts), sorted(conflicts)


def _hog_features(gray: np.ndarray, cells: int = 8, bins: int = 9) -> np.ndarray:
    """Small dependency-free HOG descriptor."""
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    h, w = gray.shape
    out = []
    for y in range(0, h - cells + 1, cells):
        for x in range(0, w - cells + 1, cells):
            m = magnitude[y:y + cells, x:x + cells].ravel()
            a = angle[y:y + cells, x:x + cells].ravel()
            hist, _ = np.histogram(a, bins=bins, range=(0, 180), weights=m)
            hist = hist / (np.linalg.norm(hist) + 1e-8)
            out.extend(hist.tolist())
    return np.asarray(out, dtype=np.float32)


def extract_features(path: str) -> np.ndarray:
    """Extract shape/texture + colour features without using the label/path."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize(FEATURE_IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # Keep a low-resolution intensity map for coarse spatial information.
    low = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 32)), dtype=np.float32) / 255.0
    hog = _hog_features(gray)
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1))
        hist.append((h / max(h.sum(), 1)).astype(np.float32))
    colour = np.concatenate(hist)
    return np.concatenate([low.ravel(), hog, colour]).astype(np.float32)


def build_feature_matrix(files: List[Tuple[str, str]]):
    X = np.stack([extract_features(p) for p, _ in files])
    encoder = LabelEncoder()
    y = encoder.fit_transform([label for _, label in files])
    return X, y, encoder


@dataclass
class TrialResult:
    model_name: str
    model_family: str
    params: dict
    split_label: str
    split_ratio: float
    accuracy: float
    f1_macro: float
    train_seconds: float
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    roc_auc: float = 0.0
    roc_curve: dict = field(default_factory=dict)
    param_count: int = 0
    param_count_str: str = ""
    prediction_sample: List[dict] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    train_accuracy: float = 0.0
    generalization_gap: float = 0.0
    training_log: List[str] = field(default_factory=list)
    training_history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "params": self.params,
            "split_label": self.split_label,
            "split_ratio": self.split_ratio,
            "accuracy": round(self.accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "roc_auc": round(self.roc_auc, 4),
            "roc_curve": self.roc_curve,
            "param_count": self.param_count,
            "param_count_str": self.param_count_str,
            "train_accuracy": round(self.train_accuracy, 4),
            "generalization_gap": round(self.generalization_gap, 4),
            "training_log": self.training_log,
            "training_history": self.training_history,
            "train_seconds": round(self.train_seconds, 2),
            "prediction_count": len(self.prediction_sample),
            "prediction_sample": self.prediction_sample[:50],
            "confusion_matrix": self.confusion_matrix,
            "class_names": self.class_names,
        }


def _compute_roc_metrics(y_true, probs, num_classes):
    """Calculate multiclass or binary ROC-AUC and downsampled ROC curve."""
    if probs is None or len(probs) == 0:
        return 0.0, {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}
    try:
        y_true = np.asarray(y_true)
        probs = np.asarray(probs)
        if num_classes <= 2:
            p = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()
            auc = float(roc_auc_score(y_true, p))
            fpr, tpr, _ = roc_curve(y_true, p)
        else:
            auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
            y_onehot = np.eye(num_classes)[y_true]
            fpr, tpr, _ = roc_curve(y_onehot.ravel(), probs.ravel())
        step = max(1, len(fpr) // 25)
        fpr_pts = [round(float(x), 4) for x in fpr[::step]]
        tpr_pts = [round(float(x), 4) for x in tpr[::step]]
        if fpr_pts[-1] != 1.0:
            fpr_pts.append(1.0)
            tpr_pts.append(1.0)
        return round(auc, 4), {"fpr": fpr_pts, "tpr": tpr_pts}
    except Exception:
        return 0.0, {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}


def _make_prediction_sample(test_files, preds, encoder):
    return [
        {"image": os.path.basename(path), "actual": actual,
         "predicted": encoder.inverse_transform([int(pred)])[0]}
        for (path, actual), pred in zip(test_files[:50], list(preds)[:50])
    ]


def _result(name, family, params, split_label, split_ratio, train_files, test_files, y_train, preds, y_test, encoder, start, probs=None, param_count=0, param_count_str=""):
    p = dict(params)
    train_acc = float(p.pop("_train_accuracy", 0.0))
    training_log = list(p.pop("_training_log", []))
    training_history = list(p.pop("_training_history", []))
    p.pop("_epoch_callback", None)
    test_acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, average="macro", zero_division=0))
    rec = float(recall_score(y_test, preds, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    auc, roc_data = _compute_roc_metrics(y_test, probs, len(encoder.classes_))
    cm = confusion_matrix(y_test, preds, labels=np.arange(len(encoder.classes_))).tolist()
    return TrialResult(
        model_name=name,
        model_family=family,
        params=p,
        split_label=split_label,
        split_ratio=split_ratio,
        accuracy=test_acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1,
        roc_auc=auc,
        roc_curve=roc_data,
        param_count=param_count,
        param_count_str=param_count_str,
        train_seconds=time.time() - start,
        prediction_sample=_make_prediction_sample(test_files, preds, encoder),
        confusion_matrix=cm,
        class_names=encoder.classes_.tolist(),
        train_accuracy=train_acc,
        generalization_gap=max(0.0, train_acc - test_acc),
        training_log=training_log,
        training_history=training_history,
    )


def _features_for(train_files, test_files):
    X_train, y_train, encoder = build_feature_matrix(train_files)
    X_test = np.stack([extract_features(p) for p, _ in test_files])
    y_test = encoder.transform([label for _, label in test_files])
    return X_train, y_train, X_test, y_test, encoder


def _balanced_weights(y):
    counts = np.bincount(y)
    n = len(y)
    k = len(counts)
    weights = {i: n / (k * max(c, 1)) for i, c in enumerate(counts)}
    return np.asarray([weights[int(v)] for v in y], dtype=np.float32)


def _classical_result(name, clf, Xtr, ytr, Xte, yte, train_files, test_files, enc, split_label, split_ratio, params, start):
    train_pred = clf.predict(Xtr)
    pred = clf.predict(Xte)
    train_acc = accuracy_score(ytr, train_pred)
    test_acc = accuracy_score(yte, pred)
    probs = None
    if hasattr(clf, "predict_proba"):
        try:
            probs = clf.predict_proba(Xte)
        except Exception:
            probs = None
    logs = [
        f"Model: {name}",
        f"Training images: {len(Xtr)} | Holdout images: {len(Xte)} | Features: {Xtr.shape[1]}",
        "Fitting model on training images only...",
        f"Training accuracy: {train_acc:.4f}",
        f"Holdout accuracy: {test_acc:.4f}",
        f"Holdout macro F1: {f1_score(yte, pred, average='macro', zero_division=0):.4f}",
    ]
    p = dict(params)
    p["_train_accuracy"] = train_acc
    p["_training_log"] = logs
    p["_training_history"] = [{
        "stage": "final",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }]
    return _result(name, "ml", p, split_label, split_ratio, train_files, test_files, ytr, pred, yte, enc, start, probs=probs, param_count=0, param_count_str="Classical ML")


def train_eval_random_forest(train_files, test_files, split_label, split_ratio, params):
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    balance = bool(params.get("class_balancing", False))
    clf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 20),
        min_samples_leaf=max(1, int(params.get("min_samples_leaf", 2))),
        class_weight="balanced" if balance else None,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    return _classical_result("RandomForest", clf, Xtr, ytr, Xte, yte, train_files, test_files, enc, split_label, split_ratio, params, start)


def train_eval_xgboost(train_files, test_files, split_label, split_ratio, params):
    if not HAS_XGBOOST:
        return None
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    balance = bool(params.get("class_balancing", False))
    clf = XGBClassifier(
        n_estimators=params.get("n_estimators", 250),
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.05),
        min_child_weight=params.get("min_child_weight", 2),
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(Xtr, ytr, sample_weight=_balanced_weights(ytr) if balance else None)
    return _classical_result("XGBoost", clf, Xtr, ytr, Xte, yte, train_files, test_files, enc, split_label, split_ratio, params, start)


def train_eval_svm(train_files, test_files, split_label, split_ratio, params):
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    balance = bool(params.get("class_balancing", False))
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = SVC(
        C=params.get("C", 1.0),
        kernel=params.get("kernel", "rbf"),
        gamma=params.get("gamma", "scale"),
        class_weight="balanced" if balance else None,
        random_state=SEED,
    )
    clf.fit(Xtr, ytr)
    return _classical_result("SVM", clf, Xtr, ytr, Xte, yte, train_files, test_files, enc, split_label, split_ratio, params, start)


def train_eval_logistic_regression(train_files, test_files, split_label, split_ratio, params):
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    balance = bool(params.get("class_balancing", False))
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=2000,
        class_weight="balanced" if balance else None,
        random_state=SEED,
    )
    clf.fit(Xtr, ytr)
    return _classical_result("LogisticRegression", clf, Xtr, ytr, Xte, yte, train_files, test_files, enc, split_label, split_ratio, params, start)


def train_eval_knn(train_files, test_files, split_label, split_ratio, params):
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    n = min(int(params.get("n_neighbors", 5)), max(1, len(Xtr)))
    clf = KNeighborsClassifier(n_neighbors=n, weights=params.get("weights", "uniform"), n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = dict(params)
    p["n_neighbors"] = n
    return _classical_result("KNN", clf, Xtr, ytr, Xte, yte, train_files, test_files, enc, split_label, split_ratio, p, start)


def train_eval_gradient_boosting(train_files, test_files, split_label, split_ratio, params):
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    balance = bool(params.get("class_balancing", False))
    clf = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 150),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("learning_rate", 0.05),
        min_samples_leaf=max(1, int(params.get("min_samples_leaf", 2))),
        random_state=SEED,
    )
    clf.fit(Xtr, ytr, sample_weight=_balanced_weights(ytr) if balance else None)
    return _classical_result("GradientBoosting", clf, Xtr, ytr, Xte, yte, train_files, test_files, enc, split_label, split_ratio, params, start)


if HAS_TORCH:
    def _build_torch_transforms(image_size=96, train=False, augment=True, aug_config=None, mean=None, std=None):
        if not HAS_TORCHVISION:
            return None
        mean = mean or [.485, .456, .406]
        std = std or [.229, .224, .225]
        ops = []
        cfg = aug_config or {}

        if train and augment:
            # Scale & Crop
            if cfg.get("random_crop", False):
                ops.append(tv_transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)))
            else:
                ops.append(tv_transforms.Resize((image_size, image_size)))

            # Flips
            if cfg.get("horizontal_flip", True):
                ops.append(tv_transforms.RandomHorizontalFlip(p=0.5))
            if cfg.get("vertical_flip", False):
                ops.append(tv_transforms.RandomVerticalFlip(p=0.5))

            # Rotation
            rot_deg = int(cfg.get("rotation_deg", 10))
            if cfg.get("random_rotation", True) and rot_deg > 0:
                ops.append(tv_transforms.RandomRotation(rot_deg))

            # Color Jitter
            if cfg.get("color_jitter", True):
                ops.append(tv_transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15))

            # Gaussian Blur
            if cfg.get("gaussian_blur", False):
                ops.append(tv_transforms.GaussianBlur(kernel_size=3))
        else:
            ops.append(tv_transforms.Resize((image_size, image_size)))

        ops.append(tv_transforms.ToTensor())
        ops.append(tv_transforms.Normalize(mean, std))
        return tv_transforms.Compose(ops)

    class ImagePathDataset(Dataset):
        def __init__(self, files, encoder, image_size=96, train=False, augment=True, aug_config=None):
            self.files = files; self.encoder = encoder
            self.transform = _build_torch_transforms(image_size, train=train, augment=augment, aug_config=aug_config)
            self.image_size = image_size

        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            path, label = self.files[idx]
            with Image.open(path) as img: img = img.convert("RGB")
            if self.transform: x = self.transform(img)
            else:
                img = img.resize((self.image_size, self.image_size)); x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2,0,1)/255.0)
            y = self.encoder.transform([label])[0]
            return x, torch.tensor(y, dtype=torch.long)

    class SmallCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__(); self.features = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Dropout2d(.15))
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.3), nn.Linear(128,num_classes))
        def forward(self,x): return self.classifier(self.features(x))

    class DeepCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            def block(a,b): return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(.1))
            self.features=nn.Sequential(block(3,32),block(32,64),block(64,128),block(128,256))
            self.classifier=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.4),nn.Linear(256,num_classes))
        def forward(self,x): return self.classifier(self.features(x))

    def _torch_eval(model, loader, device, return_probs=False):
        model.eval(); preds=[]; targets=[]; probs_list=[]
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb.to(device))
                preds.extend(out.argmax(1).cpu().numpy().tolist())
                targets.extend(yb.numpy().tolist())
                if return_probs:
                    probs_list.append(torch.softmax(out, dim=1).cpu().numpy())
        if return_probs and probs_list:
            return np.asarray(targets), np.asarray(preds), np.concatenate(probs_list, axis=0)
        return np.asarray(targets), np.asarray(preds)

    def _train_cnn(model_cls, model_name, train_files, test_files, split_label, split_ratio, params, image_size=96):
        _seed_everything()
        start = time.time()
        labels = sorted({l for _, l in train_files + test_files})
        enc = LabelEncoder().fit(labels)
        rng = np.random.default_rng(SEED)
        indices = np.arange(len(train_files))
        rng.shuffle(indices)
        val_n = max(len(labels), int(round(.15 * len(indices)))) if len(indices) >= len(labels) * 3 else 0
        val_idx = set(indices[:val_n]) if val_n else set()
        tr = [f for i, f in enumerate(train_files) if i not in val_idx]
        va = [f for i, f in enumerate(train_files) if i in val_idx]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_cls(len(labels)).to(device)
        total_p = sum(p.numel() for p in model.parameters())
        p_str = f"{total_p / 1e3:.0f}K params" if total_p < 1e6 else f"{total_p / 1e6:.1f}M params"

        target_size = int(params.get("image_size", image_size))
        augment = bool(params.get("augmentation", True))
        balance = bool(params.get("class_balancing", False))
        aug_config = params.get("augmentation_config", {})

        train_ds = ImagePathDataset(tr, enc, target_size, train=True, augment=augment, aug_config=aug_config)
        test_ds = ImagePathDataset(test_files, enc, target_size, train=False)
        val_ds = ImagePathDataset(va, enc, target_size, train=False) if va else None
        y_tr_labels = enc.transform([l for _, l in tr])

        bs = max(1, min(int(params.get("batch_size", 16)), len(tr)))
        if balance:
            weights = _balanced_weights(y_tr_labels)
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler)
            counts = np.bincount(y_tr_labels, minlength=len(labels))
            class_w = torch.tensor([1.0 / max(1, int(c)) for c in counts], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_w)
        else:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            criterion = nn.CrossEntropyLoss()
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False) if val_ds else None
        lr = float(params.get("learning_rate", 1e-3))
        optimizer = (torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=1e-4)
                     if str(params.get("optimizer", "Adam")).lower() == "sgd"
                     else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
        best_state = None
        best_val = -1.0
        best_epoch = 0
        history = []
        logs = [
            f"Model: {model_name} ({p_str})",
            f"Device: {device}",
            f"Training images: {len(tr)} | Validation images: {len(va)} | Holdout images: {len(test_files)}",
            f"Image size: {image_size}x{image_size} | Batch size: {bs} | Epochs: {int(params.get('epochs', 8))}",
        ]
        epochs = int(params.get("epochs", 8))
        epoch_cb = params.get("_epoch_callback")

        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total_seen = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(yb)
                correct += int((out.argmax(1) == yb).sum().item())
                total_seen += len(yb)
            train_loss = running_loss / max(1, total_seen)
            train_epoch_acc = correct / max(1, total_seen)
            val_acc = None
            val_f1 = None
            if val_loader:
                yt, yp = _torch_eval(model, val_loader, device)
                val_acc = float(accuracy_score(yt, yp))
                val_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
                scheduler.step(val_f1)
                if val_f1 > best_val:
                    best_val = val_f1
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                scheduler.step(train_epoch_acc)
            rec = {"epoch": epoch, "train_loss": float(train_loss), "train_accuracy": float(train_epoch_acc)}
            if val_acc is not None:
                rec.update({"val_accuracy": val_acc, "val_f1_macro": val_f1})
            history.append(rec)
            msg = f"Epoch {epoch}/{max(1, epochs)} - loss={train_loss:.4f} - train_acc={train_epoch_acc:.4f}"
            if val_acc is not None:
                msg += f" - val_acc={val_acc:.4f} - val_f1={val_f1:.4f}"
            logs.append(msg)
            if epoch_cb:
                epoch_cb({
                    "epoch": epoch,
                    "total_epochs": max(1, epochs),
                    "train_loss": round(float(train_loss), 4),
                    "train_acc": round(float(train_epoch_acc), 4),
                    "val_loss": round(float(train_loss), 4),
                    "val_acc": round(float(val_acc if val_acc is not None else train_epoch_acc), 4),
                    "model_name": model_name,
                    "log_message": msg,
                })

        if best_state:
            model.load_state_dict(best_state)
            logs.append(f"Restored best validation checkpoint from epoch {best_epoch} (val_f1={best_val:.4f}).")

        yt, yp, probs = _torch_eval(model, test_loader, device, return_probs=True)
        train_eval_loader = DataLoader(ImagePathDataset(tr, enc, image_size, False), batch_size=bs, shuffle=False)
        ytrain, ptr = _torch_eval(model, train_eval_loader, device)
        train_acc = float(accuracy_score(ytrain, ptr))
        test_acc = float(accuracy_score(yt, yp))
        test_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        logs.extend([
            f"Final training accuracy: {train_acc:.4f}",
            f"Final holdout accuracy: {test_acc:.4f}",
            f"Final holdout macro F1: {test_f1:.4f}",
        ])
        p = dict(params)
        p["_train_accuracy"] = train_acc
        p["_training_log"] = logs
        p["_training_history"] = history
        return _result(model_name, "dl", p, split_label, split_ratio, train_files, test_files, ytrain, yp, yt, enc, start, probs=probs, param_count=total_p, param_count_str=p_str)

    def train_eval_small_cnn(*args): return _train_cnn(SmallCNN, "SmallCNN", *args, image_size=64) if HAS_TORCH else None
    def train_eval_deep_cnn(*args): return _train_cnn(DeepCNN, "DeepCNN", *args, image_size=96) if HAS_TORCH else None


if HAS_TORCH and HAS_TORCHVISION:
    IMAGENET_MEAN = [.485, .456, .406]; IMAGENET_STD = [.229, .224, .225]

    class TransferImageDataset(Dataset):
        def __init__(self, files, encoder, image_size=128, train=False, augment=True, aug_config=None):
            self.files = files; self.encoder = encoder
            self.transform = _build_torch_transforms(image_size, train=train, augment=augment, aug_config=aug_config, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            self.image_size = image_size

        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            p, l = self.files[i]
            with Image.open(p) as img: x = self.transform(img.convert("RGB"))
            return x, torch.tensor(self.encoder.transform([l])[0], dtype=torch.long)

    def _build_backbone(arch, num_classes):
        if arch == "resnet18":
            try:
                model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            except Exception:
                model = tv_models.resnet18(weights=None)
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
            for p in model.parameters(): p.requires_grad = False
            for p in model.layer4.parameters(): p.requires_grad = True
            for p in model.fc.parameters(): p.requires_grad = True
        elif arch == "mobilenet_v2":
            try:
                model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT)
            except Exception:
                model = tv_models.mobilenet_v2(weights=None)
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
            for p in model.parameters(): p.requires_grad = False
            for p in model.features[-3:].parameters(): p.requires_grad = True
            for p in model.classifier.parameters(): p.requires_grad = True
        elif arch == "vgg16":
            try:
                model = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
            except Exception:
                model = tv_models.vgg16(weights=None)
            for p in model.parameters(): p.requires_grad = False
            for p in model.features[-6:].parameters(): p.requires_grad = True
            in_f = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_f, 256),
                nn.ReLU(True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
            for p in model.classifier.parameters(): p.requires_grad = True
        elif arch == "efficientnet_b0":
            try:
                model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
            except Exception:
                model = tv_models.efficientnet_b0(weights=None)
            for p in model.parameters(): p.requires_grad = False
            for p in model.features[-2:].parameters(): p.requires_grad = True
            in_f = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_f, num_classes)
            for p in model.classifier.parameters(): p.requires_grad = True
        else:
            raise ValueError(f"Unknown backbone arch: {arch}")
        return model

    def _train_transfer(arch, name, train_files, test_files, split_label, split_ratio, params):
        _seed_everything()
        start = time.time()
        labels = sorted({l for _, l in train_files + test_files})
        enc = LabelEncoder().fit(labels)
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(train_files))
        rng.shuffle(idx)
        val_n = max(len(labels), int(.15 * len(idx))) if len(idx) >= len(labels) * 3 else 0
        va_idx = set(idx[:val_n]) if val_n else set()
        tr = [f for i, f in enumerate(train_files) if i not in va_idx]
        va = [f for i, f in enumerate(train_files) if i in va_idx]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        size = int(params.get("image_size", 128))
        augment = bool(params.get("augmentation", True))
        balance = bool(params.get("class_balancing", False))
        aug_config = params.get("augmentation_config", {})
        bs = max(1, min(int(params.get("batch_size", 16)), len(tr)))
        model = _build_backbone(arch, len(labels)).to(device)
        total_p = sum(p.numel() for p in model.parameters())
        p_str = f"{total_p / 1e6:.1f}M params"

        tr_ds = TransferImageDataset(tr, enc, size, train=True, augment=augment, aug_config=aug_config)
        te_ds = TransferImageDataset(test_files, enc, size, train=False)
        va_ds = TransferImageDataset(va, enc, size, train=False) if va else None
        ytr_labels = enc.transform([l for _, l in tr])
        if balance:
            sampler = WeightedRandomSampler(torch.as_tensor(_balanced_weights(ytr_labels), dtype=torch.double), len(ytr_labels), replacement=True)
            tl = DataLoader(tr_ds, batch_size=bs, sampler=sampler)
            counts = np.bincount(ytr_labels, minlength=len(labels))
            class_w = torch.tensor([1.0 / max(1, int(c)) for c in counts], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_w)
        else:
            tl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
            criterion = nn.CrossEntropyLoss()
        tel = DataLoader(te_ds, batch_size=bs, shuffle=False)
        vl = DataLoader(va_ds, batch_size=bs, shuffle=False) if va_ds else None
        lr = float(params.get("learning_rate", 3e-4))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = (torch.optim.SGD(trainable, lr=lr, momentum=.9, weight_decay=1e-4)
                     if str(params.get("optimizer", "Adam")).lower() == "sgd"
                     else torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4))
        best_state = None
        best_val = -1.0
        best_epoch = 0
        history = []
        logs = [
            f"Model: {name} ({p_str})",
            f"Architecture: {arch}",
            f"Device: {device}",
            f"Training images: {len(tr)} | Validation images: {len(va)} | Holdout images: {len(test_files)}",
            f"Image size: {size}x{size} | Batch size: {bs} | Epochs: {int(params.get('epochs', 8))}",
        ]
        epochs = int(params.get("epochs", 8))
        epoch_cb = params.get("_epoch_callback")

        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            seen = 0
            for xb, yb in tl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(yb)
                correct += int((out.argmax(1) == yb).sum().item())
                seen += len(yb)
            train_loss = running_loss / max(1, seen)
            train_epoch_acc = correct / max(1, seen)
            val_acc = None
            val_f1 = None
            if vl:
                yt, yp = _torch_eval(model, vl, device)
                val_acc = float(accuracy_score(yt, yp))
                val_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
                if val_f1 > best_val:
                    best_val = val_f1
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rec = {"epoch": epoch, "train_loss": float(train_loss), "train_accuracy": float(train_epoch_acc)}
            if val_acc is not None:
                rec.update({"val_accuracy": val_acc, "val_f1_macro": val_f1})
            history.append(rec)
            msg = f"Epoch {epoch}/{max(1, epochs)} - loss={train_loss:.4f} - train_acc={train_epoch_acc:.4f}"
            if val_acc is not None:
                msg += f" - val_acc={val_acc:.4f} - val_f1={val_f1:.4f}"
            logs.append(msg)
            if epoch_cb:
                epoch_cb({
                    "epoch": epoch,
                    "total_epochs": max(1, epochs),
                    "train_loss": round(float(train_loss), 4),
                    "train_acc": round(float(train_epoch_acc), 4),
                    "val_loss": round(float(train_loss), 4),
                    "val_acc": round(float(val_acc if val_acc is not None else train_epoch_acc), 4),
                    "model_name": name,
                    "log_message": msg,
                })

        if best_state:
            model.load_state_dict(best_state)
            logs.append(f"Restored best validation checkpoint from epoch {best_epoch} (val_f1={best_val:.4f}).")

        yt, yp, probs = _torch_eval(model, tel, device, return_probs=True)
        trl = DataLoader(TransferImageDataset(tr, enc, size, False), batch_size=bs, shuffle=False)
        ytrain, ptr = _torch_eval(model, trl, device)
        train_acc = float(accuracy_score(ytrain, ptr))
        test_acc = float(accuracy_score(yt, yp))
        test_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        logs.extend([
            f"Final training accuracy: {train_acc:.4f}",
            f"Final holdout accuracy: {test_acc:.4f}",
            f"Final holdout macro F1: {test_f1:.4f}",
        ])
        p = dict(params)
        p["_train_accuracy"] = train_acc
        p["_training_log"] = logs
        p["_training_history"] = history
        return _result(name, "dl", p, split_label, split_ratio, train_files, test_files, ytrain, yp, yt, enc, start, probs=probs, param_count=total_p, param_count_str=p_str)

    def train_eval_resnet18(*args): return _train_transfer("resnet18", "ResNet18 (transfer)", *args)
    def train_eval_mobilenet(*args): return _train_transfer("mobilenet_v2", "MobileNetV2 (transfer)", *args)
    def train_eval_vgg16(*args): return _train_transfer("vgg16", "VGG16 (transfer)", *args)
    def train_eval_efficientnet(*args): return _train_transfer("efficientnet_b0", "EfficientNet-B0 (transfer)", *args)


# ---------------------------------------------------------------------------
# Feature 2: Hybrid Architectures
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class _ChannelAttention(nn.Module):
        """Squeeze-and-Excitation channel attention gate."""
        def __init__(self, in_planes: int, ratio: int = 16):
            super().__init__()
            bottleneck = max(4, in_planes // ratio)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_planes, bottleneck, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, in_planes, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            b, c, _, _ = x.size()
            a = self.fc(self.avg_pool(x).view(b, c))
            m = self.fc(self.max_pool(x).view(b, c))
            return x * self.sigmoid(a + m).view(b, c, 1, 1)

    class _SpatialAttention(nn.Module):
        """CBAM spatial attention gate."""
        def __init__(self, kernel_size: int = 7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                  padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

    class HybridAttentionCNN(nn.Module):
        """CNN with CBAM channel+spatial attention after each convolutional block.

        Architecture:
          Conv(3→32) + BN + ReLU + MaxPool → CBAM
          Conv(32→64) + BN + ReLU + MaxPool → CBAM
          Conv(64→128) + BN + ReLU + MaxPool → CBAM
          AdaptiveAvgPool(1) → Dropout → Linear(128, C)
        """
        def __init__(self, num_classes: int):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
            self.ca1, self.sa1 = _ChannelAttention(32), _SpatialAttention()
            self.b2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
            self.ca2, self.sa2 = _ChannelAttention(64), _SpatialAttention()
            self.b3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
            self.ca3, self.sa3 = _ChannelAttention(128), _SpatialAttention()
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(128, num_classes))

        def forward(self, x):
            x = self.b1(x); x = self.ca1(x); x = self.sa1(x)
            x = self.b2(x); x = self.ca2(x); x = self.sa2(x)
            x = self.b3(x); x = self.ca3(x); x = self.sa3(x)
            return self.classifier(x)

    class HybridCNNTransformer(nn.Module):
        """CNN stem → Transformer encoder → classification head.

        The convolutional stem extracts local spatial features, then each
        spatial position becomes a token that is refined by self-attention.
        Architecture:
          Conv stem (3→128 channels, AdaptiveAvgPool to 6×6 = 36 tokens)
          + learned positional embedding
          TransformerEncoderLayer × num_layers (d_model=128, nhead=4)
          Global token mean-pool → LayerNorm → Dropout → Linear(128, C)
        """
        def __init__(self, num_classes: int, d_model: int = 128,
                     nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, d_model, 3, padding=1),
                nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((6, 6)),
            )
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 36, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=256,
                dropout=0.1, activation="gelu", batch_first=True)
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(0.3),
                nn.Linear(d_model, num_classes),
            )

        def forward(self, x):
            feat = self.stem(x)                        # B × d × 6 × 6
            tokens = feat.flatten(2).transpose(1, 2)   # B × 36 × d
            tokens = tokens + self.pos_embedding
            out = self.transformer(tokens)             # B × 36 × d
            return self.classifier(out.mean(dim=1))    # B × C

    def _train_hybrid(model_cls, model_name, train_files, test_files,
                      split_label, split_ratio, params, image_size=96):
        """Generic trainer shared by all hybrid CNN architectures."""
        _seed_everything()
        start = time.time()
        labels = sorted({l for _, l in train_files + test_files})
        enc = LabelEncoder().fit(labels)
        rng = np.random.default_rng(SEED)
        indices = np.arange(len(train_files))
        rng.shuffle(indices)
        val_n = (max(len(labels), int(round(0.15 * len(indices))))
                 if len(indices) >= len(labels) * 3 else 0)
        val_idx = set(indices[:val_n]) if val_n else set()
        tr = [f for i, f in enumerate(train_files) if i not in val_idx]
        va = [f for i, f in enumerate(train_files) if i in val_idx]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_cls(len(labels)).to(device)
        total_p = sum(p.numel() for p in model.parameters())
        p_str = (f"{total_p / 1e3:.0f}K params" if total_p < 1e6
                 else f"{total_p / 1e6:.1f}M params")

        target_size = int(params.get("image_size", image_size))
        augment = bool(params.get("augmentation", True))
        balance = bool(params.get("class_balancing", False))
        aug_config = params.get("augmentation_config", {})

        train_ds = ImagePathDataset(tr, enc, target_size, train=True,
                                    augment=augment, aug_config=aug_config)
        test_ds = ImagePathDataset(test_files, enc, target_size, train=False)
        val_ds = (ImagePathDataset(va, enc, target_size, train=False)
                  if va else None)
        y_tr_labels = enc.transform([l for _, l in tr])
        bs = max(1, min(int(params.get("batch_size", 16)), len(tr)))

        if balance:
            weights = _balanced_weights(y_tr_labels)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler)
            counts = np.bincount(y_tr_labels, minlength=len(labels))
            class_w = torch.tensor(
                [1.0 / max(1, int(c)) for c in counts],
                dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_w)
        else:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            criterion = nn.CrossEntropyLoss()

        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        val_loader = (DataLoader(val_ds, batch_size=bs, shuffle=False)
                      if val_ds else None)
        lr = float(params.get("learning_rate", 1e-3))
        optimizer = (
            torch.optim.SGD(model.parameters(), lr=lr,
                            momentum=0.9, weight_decay=1e-4)
            if str(params.get("optimizer", "Adam")).lower() == "sgd"
            else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)
        best_state, best_val, best_epoch = None, -1.0, 0
        history, logs = [], [
            f"Model: {model_name} ({p_str})",
            f"Device: {device}",
            f"Training: {len(tr)} | Validation: {len(va)} | Test: {len(test_files)}",
            f"Image size: {target_size}×{target_size} | Batch: {bs} | "
            f"Epochs: {int(params.get('epochs', 8))}",
        ]
        epochs = int(params.get("epochs", 8))
        epoch_cb = params.get("_epoch_callback")

        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            running_loss, correct, seen = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(yb)
                correct += int((out.argmax(1) == yb).sum().item())
                seen += len(yb)
            train_loss = running_loss / max(1, seen)
            train_epoch_acc = correct / max(1, seen)
            val_acc = val_f1 = None
            if val_loader:
                yt, yp = _torch_eval(model, val_loader, device)
                val_acc = float(accuracy_score(yt, yp))
                val_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
                scheduler.step(val_f1)
                if val_f1 > best_val:
                    best_val, best_epoch = val_f1, epoch
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            else:
                scheduler.step(train_epoch_acc)
            rec = {"epoch": epoch, "train_loss": float(train_loss),
                   "train_accuracy": float(train_epoch_acc)}
            if val_acc is not None:
                rec.update({"val_accuracy": val_acc, "val_f1_macro": val_f1})
            history.append(rec)
            msg = (f"Epoch {epoch}/{max(1,epochs)} - loss={train_loss:.4f} "
                   f"- train_acc={train_epoch_acc:.4f}")
            if val_acc is not None:
                msg += f" - val_acc={val_acc:.4f} - val_f1={val_f1:.4f}"
            logs.append(msg)
            if epoch_cb:
                epoch_cb({
                    "epoch": epoch, "total_epochs": max(1, epochs),
                    "train_loss": round(float(train_loss), 4),
                    "train_acc": round(float(train_epoch_acc), 4),
                    "val_loss": round(float(train_loss), 4),
                    "val_acc": round(float(val_acc if val_acc is not None
                                             else train_epoch_acc), 4),
                    "model_name": model_name,
                    "log_message": msg,
                })

        if best_state:
            model.load_state_dict(best_state)
            logs.append(
                f"Restored best checkpoint from epoch {best_epoch} "
                f"(val_f1={best_val:.4f}).")

        yt, yp, probs = _torch_eval(model, test_loader, device, return_probs=True)
        trl = DataLoader(ImagePathDataset(tr, enc, target_size, False),
                         batch_size=bs, shuffle=False)
        ytrain, ptr = _torch_eval(model, trl, device)
        train_acc = float(accuracy_score(ytrain, ptr))
        test_acc = float(accuracy_score(yt, yp))
        test_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        logs.extend([
            f"Final training accuracy: {train_acc:.4f}",
            f"Final holdout accuracy: {test_acc:.4f}",
            f"Final holdout macro F1: {test_f1:.4f}",
        ])
        p = dict(params)
        p["_train_accuracy"] = train_acc
        p["_training_log"] = logs
        p["_training_history"] = history
        return _result(model_name, "hybrid", p, split_label, split_ratio,
                       train_files, test_files, ytrain, yp, yt, enc, start,
                       probs=probs, param_count=total_p, param_count_str=p_str)

    def train_eval_hybrid_attention(*args):
        return _train_hybrid(HybridAttentionCNN,
                             "Hybrid Attention CNN", *args, image_size=96)

    def train_eval_hybrid_transformer(*args):
        return _train_hybrid(HybridCNNTransformer,
                             "Hybrid CNN-Transformer", *args, image_size=96)


# ---------------------------------------------------------------------------
# Feature 2: Ensemble Learning
# ---------------------------------------------------------------------------

def train_eval_ensemble_voting(train_files, test_files, split_label,
                               split_ratio, params):
    """Soft-probability weighted average of RF + GB + SVM.

    Each base classifier is trained on all training data.  Probabilities
    are weighted (RF=0.45, GB=0.35, SVM=0.20) and the ensemble picks the
    argmax of the blended distribution.
    """
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    balance = bool(params.get("class_balancing", False))
    n_est = int(params.get("n_estimators", 100))

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=16,
                                class_weight="balanced" if balance else None,
                                random_state=SEED, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=max(30, n_est // 3),
                                    max_depth=3, learning_rate=0.08,
                                    random_state=SEED)
    from sklearn.calibration import CalibratedClassifierCV
    svm_base = SVC(C=1.0, kernel="rbf", random_state=SEED)
    svm = CalibratedClassifierCV(svm_base, ensemble=False)

    sw = _balanced_weights(ytr) if balance else None
    rf.fit(Xtr, ytr, sample_weight=sw)
    gb.fit(Xtr, ytr, sample_weight=sw)
    svm.fit(Xtr, ytr)

    w_rf, w_gb, w_svm = 0.45, 0.35, 0.20
    p_rf  = rf.predict_proba(Xte)
    p_gb  = gb.predict_proba(Xte)
    p_svm = svm.predict_proba(Xte)
    blended = w_rf * p_rf + w_gb * p_gb + w_svm * p_svm

    train_rf = rf.predict_proba(Xtr)
    train_gb = gb.predict_proba(Xtr)
    train_svm = svm.predict_proba(Xtr)
    train_blended = w_rf * train_rf + w_gb * train_gb + w_svm * train_svm

    test_preds  = blended.argmax(axis=1)
    train_preds = train_blended.argmax(axis=1)
    train_acc = float(accuracy_score(ytr, train_preds))
    test_acc  = float(accuracy_score(yte, test_preds))
    test_f1   = float(f1_score(yte, test_preds, average="macro", zero_division=0))
    logs = [
        "Ensemble (Soft Voting): RF × 0.45 + GB × 0.35 + SVM × 0.20",
        f"Base classifiers: RandomForest (n={n_est}), GradientBoosting "
        f"(n={max(30, n_est // 3)}), CalibratedSVM",
        f"Training accuracy: {train_acc:.4f}",
        f"Holdout accuracy:  {test_acc:.4f}",
        f"Holdout macro F1:  {test_f1:.4f}",
    ]
    p = dict(params)
    p["_train_accuracy"] = train_acc
    p["_training_log"] = logs
    p["_training_history"] = [{"stage": "final",
                                "train_accuracy": train_acc,
                                "test_accuracy": test_acc,
                                "test_f1_macro": test_f1}]
    return _result("Ensemble Soft Voting", "ensemble", p,
                   split_label, split_ratio,
                   train_files, test_files,
                   ytr, test_preds, yte, enc, start,
                   probs=blended, param_count=0,
                   param_count_str="3 base classifiers")


def train_eval_ensemble_stacking(train_files, test_files, split_label,
                                 split_ratio, params):
    """Stacking ensemble: RF + GB + CalibratedSVM → Logistic Regression meta-learner.

    Out-of-fold predictions (StratifiedKFold) are used to train the meta-learner,
    avoiding leakage between base and meta layers.
    """
    start = time.time()
    Xtr, ytr, Xte, yte, enc = _features_for(train_files, test_files)
    balance = bool(params.get("class_balancing", False))
    n_est = int(params.get("n_estimators", 100))
    num_classes = len(enc.classes_)

    from sklearn.calibration import CalibratedClassifierCV
    rf  = RandomForestClassifier(n_estimators=n_est, max_depth=16,
                                 class_weight="balanced" if balance else None,
                                 random_state=SEED, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=max(30, n_est // 3),
                                     max_depth=3, learning_rate=0.08,
                                     random_state=SEED)
    svm = CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", random_state=SEED),
                                 ensemble=False)
    base_models = [("RF", rf), ("GB", gb), ("SVM", svm)]
    sw = _balanced_weights(ytr) if balance else None

    n_folds = min(5, max(2, len(ytr) // max(1, num_classes)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(Xtr), len(base_models) * num_classes), np.float32)
    logs = [
        f"Stacking Ensemble: RF + GB + CalibratedSVM → LogisticRegression meta",
        f"Cross-validation folds (OOF): {n_folds}",
    ]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, ytr), 1):
        X_f, y_f = Xtr[tr_idx], ytr[tr_idx]
        X_v = Xtr[va_idx]
        fold_probs = []
        for name, clf in base_models:
            m = clone(clf)
            sw_f = sw[tr_idx] if sw is not None else None
            if name in ("RF", "GB") and sw_f is not None:
                m.fit(X_f, y_f, sample_weight=sw_f)
            else:
                m.fit(X_f, y_f)
            fold_probs.append(m.predict_proba(X_v))
        oof_preds[va_idx] = np.concatenate(fold_probs, axis=1)
        logs.append(f"  Fold {fold}/{n_folds} complete.")

    # Retrain base models on all training data and collect test predictions
    test_meta_parts = []
    for name, clf in base_models:
        if name in ("RF", "GB") and sw is not None:
            clf.fit(Xtr, ytr, sample_weight=sw)
        else:
            clf.fit(Xtr, ytr)
        test_meta_parts.append(clf.predict_proba(Xte))
    Z_test = np.concatenate(test_meta_parts, axis=1)

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(oof_preds, ytr)

    train_preds = meta.predict(oof_preds)
    test_preds  = meta.predict(Z_test)
    test_probs  = meta.predict_proba(Z_test)

    train_acc = float(accuracy_score(ytr, train_preds))
    test_acc  = float(accuracy_score(yte, test_preds))
    test_f1   = float(f1_score(yte, test_preds, average="macro", zero_division=0))
    logs.extend([
        f"Meta-learner (Logistic Regression) trained on OOF features.",
        f"Training accuracy (OOF): {train_acc:.4f}",
        f"Holdout accuracy:        {test_acc:.4f}",
        f"Holdout macro F1:        {test_f1:.4f}",
    ])
    p = dict(params)
    p["_train_accuracy"] = train_acc
    p["_training_log"] = logs
    p["_training_history"] = [{"stage": "final",
                                "train_accuracy": train_acc,
                                "test_accuracy": test_acc,
                                "test_f1_macro": test_f1}]
    return _result("Ensemble Stacking", "ensemble", p,
                   split_label, split_ratio,
                   train_files, test_files,
                   ytr, test_preds, yte, enc, start,
                   probs=test_probs, param_count=0,
                   param_count_str="3 base + LR meta")


MODEL_REGISTRY: Dict[str, dict] = {
    "small_cnn": {
        "label": "Small CNN (from scratch)",
        "family": "dl",
        "description": "Compact 3-layer convolutional network with batch normalization and dropout.",
        "fn": globals().get("train_eval_small_cnn"),
        "available": HAS_TORCH,
        "default_params": {"batch_size": 16, "learning_rate": 0.001, "image_size": 96},
        "supports_epochs": True,
        "symbol": "CNN",
        "detail": "From scratch · 3 conv layers",
        "architecture_info": "3 Conv2D layers (32, 64, 128 filters) + BatchNorm + MaxPool + Linear(128, classes).",
        "use_case": "Fast training baseline for small datasets or low-resource hardware.",
        "params_approx": "~120K params",
    },
    "deep_cnn": {
        "label": "Deep CNN (from scratch)",
        "family": "dl",
        "description": "4-layer convolutional network with 256 feature channels and adaptive pooling.",
        "fn": globals().get("train_eval_deep_cnn"),
        "available": HAS_TORCH,
        "default_params": {"batch_size": 16, "learning_rate": 0.001, "image_size": 96},
        "supports_epochs": True,
        "symbol": "DC",
        "detail": "From scratch · 4 conv layers",
        "architecture_info": "4 Conv2D blocks (32, 64, 128, 256 filters) with BatchNorm, ReLU, Dropout.",
        "use_case": "Higher representational capacity for multi-class image classification from scratch.",
        "params_approx": "~400K params",
    },
    "resnet18": {
        "label": "ResNet18 (transfer learning)",
        "family": "dl",
        "description": "Deep residual network with residual skip connections and fine-tuned final stage.",
        "fn": globals().get("train_eval_resnet18"),
        "available": HAS_TORCH and HAS_TORCHVISION,
        "default_params": {"batch_size": 16, "learning_rate": 0.0003, "image_size": 128},
        "supports_epochs": True,
        "symbol": "RN",
        "detail": "Transfer learning · Residual blocks",
        "architecture_info": "18 layers with residual shortcut connections to avoid gradient degradation.",
        "use_case": "Versatile, robust performance on photographic datasets.",
        "params_approx": "11.2M params",
    },
    "mobilenet": {
        "label": "MobileNetV2 (transfer learning)",
        "family": "dl",
        "description": "Inverted residual bottleneck architecture optimized for low latency and high accuracy.",
        "fn": globals().get("train_eval_mobilenet"),
        "available": HAS_TORCH and HAS_TORCHVISION,
        "default_params": {"batch_size": 16, "learning_rate": 0.0003, "image_size": 128},
        "supports_epochs": True,
        "symbol": "MN",
        "detail": "Transfer learning · Inverted residuals",
        "architecture_info": "Depthwise separable convolutions with linear bottlenecks.",
        "use_case": "Edge deployment, real-time vision, and fast convergence.",
        "params_approx": "2.2M params",
    },
    "vgg16": {
        "label": "VGG16 (transfer learning)",
        "family": "dl",
        "description": "16-layer architecture using uniform 3×3 convolutions across five hierarchical stages.",
        "fn": globals().get("train_eval_vgg16"),
        "available": HAS_TORCH and HAS_TORCHVISION,
        "default_params": {"batch_size": 16, "learning_rate": 0.0003, "image_size": 128},
        "supports_epochs": True,
        "symbol": "VGG",
        "detail": "Transfer learning · 16 layers",
        "architecture_info": "13 Convolutional layers in 5 stages with compact fine-tuned dense head.",
        "use_case": "Detailed hierarchical texture feature extraction.",
        "params_approx": "~15.2M params",
    },
    "efficientnet": {
        "label": "EfficientNet-B0 (transfer learning)",
        "family": "dl",
        "description": "Compound scaled architecture balancing network depth, width, and resolution.",
        "fn": globals().get("train_eval_efficientnet"),
        "available": HAS_TORCH and HAS_TORCHVISION,
        "default_params": {"batch_size": 16, "learning_rate": 0.0003, "image_size": 128},
        "supports_epochs": True,
        "symbol": "EN",
        "detail": "Transfer learning · MBConv blocks",
        "architecture_info": "MBConv blocks with Squeeze-and-Excitation attention mechanisms.",
        "use_case": "State-of-the-art accuracy with minimal computational overhead.",
        "params_approx": "4.0M params",
    },
    "random_forest": {
        "label": "Random Forest",
        "family": "ml",
        "description": "Ensemble of decision trees trained on color histograms and HOG spatial features.",
        "fn": train_eval_random_forest,
        "available": True,
        "default_params": {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 2},
        "supports_epochs": False,
        "symbol": "RF",
        "detail": "Classical · Tree ensemble",
        "architecture_info": "Ensemble of 300 randomized bagged trees on HOG and color histograms.",
        "use_case": "Fast training baseline without requiring GPU resources.",
        "params_approx": "N/A (Trees)",
    },
    "xgboost": {
        "label": "XGBoost",
        "family": "ml",
        "description": "Gradient-boosted decision trees with histogram-based split finding and sample weights.",
        "fn": train_eval_xgboost,
        "available": HAS_XGBOOST,
        "default_params": {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 2},
        "supports_epochs": False,
        "symbol": "XG",
        "detail": "Classical · Gradient boosting",
        "architecture_info": "Sequential boosted trees optimizing multiclass log-loss.",
        "use_case": "Strong tabular / hand-crafted feature classifier.",
        "params_approx": "N/A (Boosting)",
    },
    "svm": {
        "label": "Support Vector Machine (SVM)",
        "family": "ml",
        "description": "Radial Basis Function (RBF) kernel SVM with standard scaling.",
        "fn": train_eval_svm,
        "available": True,
        "default_params": {"C": 1.0, "kernel": "rbf"},
        "supports_epochs": False,
        "symbol": "SVM",
        "detail": "Classical · Kernel machine",
        "architecture_info": "RBF kernel projecting image features into dual Hilbert space.",
        "use_case": "Effective on high-dimensional feature spaces with clear margins.",
        "params_approx": "N/A (Support vectors)",
    },
    "logistic_regression": {
        "label": "Logistic Regression",
        "family": "ml",
        "description": "Linear classifier with L2 regularization and feature standardization.",
        "fn": train_eval_logistic_regression,
        "available": True,
        "default_params": {"C": 1.0},
        "supports_epochs": False,
        "symbol": "LR",
        "detail": "Classical · Linear baseline",
        "architecture_info": "Multinomial logistic regression with L2 penalty.",
        "use_case": "Sanity check linear baseline.",
        "params_approx": "~2K params",
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "family": "ml",
        "description": "Instance-based learning matching test queries to nearest training vectors.",
        "fn": train_eval_knn,
        "available": True,
        "default_params": {"n_neighbors": 5, "weights": "uniform"},
        "supports_epochs": False,
        "symbol": "KNN",
        "detail": "Classical · Instance based",
        "architecture_info": "Euclidean distance metric over standardized HOG features.",
        "use_case": "Direct similarity clustering.",
        "params_approx": "N/A (Memory-based)",
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "family": "ml",
        "description": "Iterative ensemble of weak decision stumps minimizing cross-entropy loss.",
        "fn": train_eval_gradient_boosting,
        "available": True,
        "default_params": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 2},
        "supports_epochs": False,
        "symbol": "GB",
        "detail": "Classical · Boosted trees",
        "architecture_info": "Stage-wise additive tree model.",
        "use_case": "Robust baseline with high resistance to outliers.",
        "params_approx": "N/A (Boosting)",
    },
    # ------------------------------------------------------------------
    # Hybrid Architectures
    # ------------------------------------------------------------------
    "hybrid_attention": {
        "label": "Hybrid Attention CNN (CBAM)",
        "family": "hybrid",
        "description": (
            "CNN with Convolutional Block Attention Module (CBAM): "
            "channel + spatial attention gates after every convolutional block."
        ),
        "fn": globals().get("train_eval_hybrid_attention"),
        "available": HAS_TORCH,
        "default_params": {"batch_size": 16, "learning_rate": 0.001, "image_size": 96},
        "supports_epochs": True,
        "symbol": "HA",
        "detail": "Hybrid - CBAM attention",
        "architecture_info": (
            "3 Conv blocks (32, 64, 128 filters) each followed by "
            "channel + spatial CBAM attention gates."
        ),
        "use_case": (
            "Strong inductive bias for relevant spatial regions; "
            "excellent on datasets with localised discriminative features."
        ),
        "params_approx": "~97K params",
    },
    "hybrid_transformer": {
        "label": "Hybrid CNN-Transformer",
        "family": "hybrid",
        "description": (
            "CNN convolutional stem extracts local features; "
            "Transformer encoder refines them with global self-attention "
            "before classification."
        ),
        "fn": globals().get("train_eval_hybrid_transformer"),
        "available": HAS_TORCH,
        "default_params": {"batch_size": 16, "learning_rate": 0.0008, "image_size": 96},
        "supports_epochs": True,
        "symbol": "CT",
        "detail": "Hybrid - CNN + Transformer",
        "architecture_info": (
            "Conv stem to 36 tokens + learned positional embedding + "
            "TransformerEncoder (d=128, heads=4, layers=2) + global token pooling."
        ),
        "use_case": (
            "Captures both local texture and global context relationships; "
            "suitable for complex classification tasks."
        ),
        "params_approx": "~364K params",
    },
    # ------------------------------------------------------------------
    # Ensemble Models
    # ------------------------------------------------------------------
    "ensemble_voting": {
        "label": "Ensemble Soft Voting (RF + GB + SVM)",
        "family": "ensemble",
        "description": (
            "Probability-weighted soft voting of Random Forest, "
            "Gradient Boosting, and Calibrated SVM."
        ),
        "fn": train_eval_ensemble_voting,
        "available": True,
        "default_params": {"n_estimators": 100},
        "supports_epochs": False,
        "symbol": "SV",
        "detail": "Ensemble - Soft voting",
        "architecture_info": (
            "RF (w=0.45) + GB (w=0.35) + CalibratedSVM (w=0.20). "
            "Blended by fixed probability weights."
        ),
        "use_case": (
            "Reduces variance compared to any single classifier; "
            "robust across diverse image feature distributions."
        ),
        "params_approx": "3 base classifiers",
    },
    "ensemble_stacking": {
        "label": "Ensemble Stacking (RF + GB + SVM -> LR)",
        "family": "ensemble",
        "description": (
            "Stacking ensemble: RF, GB, and Calibrated SVM base models "
            "produce out-of-fold probability features (up to 5-fold CV) "
            "that train a Logistic Regression meta-learner."
        ),
        "fn": train_eval_ensemble_stacking,
        "available": True,
        "default_params": {"n_estimators": 100},
        "supports_epochs": False,
        "symbol": "ST",
        "detail": "Ensemble - Stacking + meta-LR",
        "architecture_info": (
            "OOF meta-features from RF+GB+SVM trained to "
            "Logistic Regression meta-learner. StratifiedKFold prevents leakage."
        ),
        "use_case": (
            "Often outperforms individual base models by learning "
            "the optimal combination of their strengths."
        ),
        "params_approx": "3 base + LR meta",
    },

}

