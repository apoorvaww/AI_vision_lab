"""
search.py
---------
The actual "AutoML" loop: for every split candidate, try every selected
model with every hyperparameter combination in the search space, score
each trial, and return a sorted leaderboard.

This is intentionally a simple grid search (see the note in the project
README about swapping this for Optuna/Ray Tune once the search space
grows -- the loop structure stays the same, only how you pick the next
combination changes).

Model dispatch is driven entirely by `MODEL_REGISTRY` in `models.py`, so
adding a new model doesn't require touching this file.
"""

import itertools
from typing import Callable, Dict, List, Optional

from .models import MODEL_REGISTRY, TrialResult
from .splits import SplitCandidate


class SearchCancelled(Exception):
    """Raised internally when a `should_stop` callback returns True."""


def _param_grid(param_space: Dict[str, list]) -> List[dict]:
    """Turns {'a': [1,2], 'b': [10,20]} into every combination as dicts."""
    if not param_space:
        return [{}]
    keys = list(param_space.keys())
    combos = itertools.product(*param_space.values())
    return [dict(zip(keys, combo)) for combo in combos]


def run_search(
    splits: List[SplitCandidate],
    search_space: Dict[str, Dict[str, list]],
    metric: str = "f1_macro",
    progress_callback: Optional[Callable] = None,
    epoch_callback: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[TrialResult]:
    """
    search_space example:
        {
          "random_forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
          "xgboost":       {"n_estimators": [100], "max_depth": [3, 6], "learning_rate": [0.1]},
          "resnet18":      {"epochs": [5], "batch_size": [16], "learning_rate": [1e-3]},
          ...
        }
    Keys must match entries in `MODEL_REGISTRY` (see models.py).

    progress_callback(done, total, description) can be passed by the web
    backend to update a live progress bar for the user.

    should_stop() -- optional, called before each trial. If it returns
    True, the search stops early and returns whatever trials completed
    so far (used to support a "cancel" button in the web UI).
    """
    trials: List[TrialResult] = []

    jobs = []  # (model_key, split, params)
    for split in splits:
        for model_key, param_space in search_space.items():
            entry = MODEL_REGISTRY.get(model_key)
            if entry is None or not entry["available"]:
                continue
            for params in _param_grid(param_space):
                jobs.append((model_key, split, params))

    total = len(jobs)

    for i, (model_key, split, params) in enumerate(jobs, start=1):
        if should_stop is not None and should_stop():
            if progress_callback:
                progress_callback(i - 1, total, "Cancelled")
            raise SearchCancelled()

        entry = MODEL_REGISTRY[model_key]
        if progress_callback:
            progress_callback(
                i - 1, total, f"Training {entry['label']} on {split.label} split"
            )

        p_copy = dict(params)
        if epoch_callback:
            def _make_epoch_cb(step_idx, total_jobs_count, label):
                def _cb(info):
                    info["job_step"] = step_idx
                    info["total_jobs"] = total_jobs_count
                    if "model_name" not in info:
                        info["model_name"] = label
                    epoch_callback(info)
                return _cb
            p_copy["_epoch_callback"] = _make_epoch_cb(i, total, entry["label"])

        result = entry["fn"](split.train_files, split.test_files, split.label, split.ratio, p_copy)

        if result is not None:
            trials.append(result)
        if progress_callback:
            progress_callback(i, total, f"Completed {entry['label']} on {split.label} split")

    trials.sort(key=lambda t: getattr(t, metric), reverse=True)

    if progress_callback:
        progress_callback(total, total, "Search complete")

    return trials
