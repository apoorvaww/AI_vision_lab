"""Leakage-safe train/test split generation."""

from dataclasses import dataclass
from typing import List, Tuple
from sklearn.model_selection import train_test_split


@dataclass
class SplitCandidate:
    ratio: float
    train_files: List[Tuple[str, str]]
    test_files: List[Tuple[str, str]]

    @property
    def label(self) -> str:
        if self.ratio >= 0.999:
            return "Provided test split"
        pct = int(round(self.ratio * 100))
        return f"{pct}/{100 - pct}"


def generate_splits(valid_files, ratios=(0.8,), seed=42):
    """Create stratified holdouts.

    A split is rejected when stratification cannot produce at least one test
    example for every class. This avoids silently producing misleading scores.
    """
    labels=[l for _,l in valid_files]
    if len(set(labels)) < 2:
        raise ValueError("At least two classes are required for classification.")
    min_class=min(labels.count(c) for c in set(labels))
    candidates=[]
    for ratio in ratios:
        test_size=1.0-ratio
        if int(round(len(valid_files)*test_size)) < len(set(labels)) or min_class < 2:
            continue
        train_p,test_p,train_l,test_l=train_test_split(
            [p for p,_ in valid_files], labels, train_size=ratio, stratify=labels, random_state=seed
        )
        candidates.append(SplitCandidate(ratio,list(zip(train_p,train_l)),list(zip(test_p,test_l))))
    if not candidates:
        raise ValueError("The dataset is too small for the selected train/test split. Use more images per class or a larger test set.")
    return candidates
