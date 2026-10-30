"""
pipeline.py
-----------
The single entry point your web backend calls after a user uploads a
dataset. Wraps: validate -> split -> search -> pick winner -> save JSON.

This is deliberately a plain function (not yet wrapped in an orchestrator
like Prefect/Airflow) so it's easy to read and test. When you're ready to
add retries / parallel execution, wrap the same steps as tasks in
whichever orchestrator you pick -- the logic below doesn't change, only
how it's *scheduled*.

The Flask app in `app.py` (project root) is a real, working example of
wiring this up behind a browser UI: file upload -> model selection ->
a live progress bar backed by `progress_callback` -> a results page.
"""

import json
import os
import time
from typing import Callable, Dict, Optional

from .data import discover_for_training
from .splits import generate_splits
from .search import run_search, SearchCancelled
from .models import MODEL_REGISTRY


# Only the fast, dependency-light classical models run by default so a
# plain `python run_pipeline.py` stays quick. The web UI lets the user
# explicitly opt into the heavier DL / transfer-learning models instead
# of always running every registered model.
DEFAULT_SEARCH_SPACE = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
    },
    "xgboost": {
        "n_estimators": [100],
        "max_depth": [3, 6],
        "learning_rate": [0.1],
    },
    "small_cnn": {
        "epochs": [3],
        "batch_size": [16],
        "learning_rate": [1e-3],
    },
}

DEFAULT_SPLIT_RATIOS = [0.8]


def run_stage1(
    dataset_root: str,
    output_json_path: str,
    search_space: Optional[Dict] = None,
    split_ratios: Optional[list] = None,
    metric: str = "f1_macro",
    min_images_per_class: int = 0,
    progress_callback: Optional[Callable] = None,
    epoch_callback: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Runs the full Stage 1 pipeline and writes results to `output_json_path`.

    Returns the same dict that gets written to disk, so callers (e.g. a
    Flask/FastAPI endpoint) can return it directly in an HTTP response too.

    The output JSON is exactly what gets handed to "Stage 2" of your app:
    the recommended model family/name, its best hyperparameters, and the
    best train/test split ratio -- plus the full leaderboard so you can
    show the user "here's what we tried" if you want that transparency.

    `should_stop` -- optional zero-arg callable checked between trials.
    If it returns True, the run stops early (see SearchCancelled) and
    whatever trials completed are still written out, so a "cancel" button
    in the UI doesn't lose partial progress.
    """
    started = time.time()
    search_space = search_space or DEFAULT_SEARCH_SPACE
    split_ratios = split_ratios or DEFAULT_SPLIT_RATIOS

    # Fail fast with a clear error if the caller asked for a model that
    # doesn't exist or isn't available in this environment (e.g. torch
    # not installed), instead of silently skipping it deep in search.py.
    unknown = [k for k in search_space if k not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model key(s) in search_space: {unknown}")
    unavailable = [k for k in search_space if not MODEL_REGISTRY[k]["available"]]
    if unavailable and len(unavailable) == len(search_space):
        raise ValueError(
            f"None of the requested models are available in this environment: {unavailable}. "
            "Check that the required optional dependency (xgboost / torch / torchvision) is installed."
        )

    # No upload-time validation gate: dataset structure is discovered only when training starts.
    if progress_callback:
        progress_callback(5, 100, "Reading dataset structure")
    report = discover_for_training(dataset_root)

    if progress_callback:
        progress_callback(10, 100, "Preparing leakage-safe train/test split")

    # Prefer an explicitly supplied labeled test set. Never mix it into the
    # training data and never create a random split from it.
    if report.test_files:
        from .splits import SplitCandidate
        splits = [SplitCandidate(
            ratio=1.0,
            train_files=report.valid_files,
            test_files=report.test_files,
        )]
    else:
        # For ImageFolder datasets, remove exact duplicates first (done by
        # discovery) and then create a stratified holdout from the remaining
        # images. The test set is never used for fitting.
        splits = generate_splits(report.valid_files, ratios=split_ratios)

    try:
        def search_progress(done, total, description):
            if progress_callback:
                if total <= 0:
                    progress_callback(100, 100, description)
                else:
                    mapped = 10 + int(90 * done / total)
                    progress_callback(min(mapped, 100), 100, description)

        trials = run_search(
            splits=splits,
            search_space=search_space,
            metric=metric,
            progress_callback=search_progress,
            epoch_callback=epoch_callback,
            should_stop=should_stop,
        )
        cancelled = False
    except SearchCancelled:
        trials = []
        cancelled = True

    if not trials:
        if cancelled:
            raise RuntimeError("Search was cancelled before any trial finished.")
        raise RuntimeError(
            "No trials completed successfully. Check that the selected model(s) "
            "are available (e.g. xgboost / torch / torchvision installed)."
        )

    best = trials[0]

    # Identify trial with XAI visualizations (in case best is classical ML)
    xai_trial = best
    if not getattr(best, "xai_visualizations", None):
        for t in trials:
            if getattr(t, "xai_visualizations", None):
                xai_trial = t
                break

    code_snapshot = {}
    source_files = [
        "stage1_pipeline/models.py",
        "stage1_pipeline/data.py",
        "stage1_pipeline/splits.py",
        "stage1_pipeline/search.py",
        "stage1_pipeline/pipeline.py",
    ]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in source_files:
        try:
            with open(os.path.join(project_root, rel), "r", encoding="utf-8") as fh:
                code_snapshot[rel] = fh.read()
        except OSError:
            code_snapshot[rel] = ""

    result = {
        "dataset_summary": report.summary(),
        "search_metric": metric,
        "cancelled": cancelled,
        "best_config": {
            "model_family": best.model_family,   # "ml" or "dl"
            "model_name": best.model_name,
            "params": best.params,
            "split_ratio": best.split_ratio,
            "split_label": best.split_label,
            "expected_accuracy": round(best.accuracy, 4),
            "expected_precision": round(best.precision_macro, 4),
            "expected_recall": round(best.recall_macro, 4),
            "expected_f1_macro": round(best.f1_macro, 4),
            "expected_roc_auc": round(best.roc_auc, 4),
            "roc_curve": best.roc_curve,
            "param_count": best.param_count,
            "param_count_str": best.param_count_str,
            "prediction_sample": best.prediction_sample[:50],
            "confusion_matrix": best.confusion_matrix,
            "class_names": best.class_names,
            "train_accuracy": best.train_accuracy,
            "generalization_gap": best.generalization_gap,
            "training_history": best.training_history,
            "xai_enabled": getattr(best, "xai_enabled", False),
            "xai_method": getattr(best, "xai_method", "gradcam"),
            "xai_visualizations": getattr(best, "xai_visualizations", []),
            "xai_note": getattr(best, "xai_note", ""),
        },
        "xai_enabled": getattr(xai_trial, "xai_enabled", False),
        "xai_method": getattr(xai_trial, "xai_method", "gradcam"),
        "xai_visualizations": getattr(xai_trial, "xai_visualizations", []),
        "xai_model_name": getattr(xai_trial, "model_name", best.model_name),
        "xai_note": getattr(best, "xai_note", ""),
        "leaderboard": [t.to_dict() for t in trials],
        "evaluation": {
            "test_set_source": "provided test split" if report.test_files else "stratified holdout",
            "duplicate_files_removed": report.duplicate_files_removed,
            "note": "Metrics are calculated only on images excluded from fitting. A 100% score is not artificially lowered; if it occurs on a clean holdout, it is a genuine result for that dataset/split."
        },
        "num_trials_run": len(trials),
        "training_outputs": [{"model_name": t.model_name, "split_label": t.split_label, "log": t.training_log, "history": t.training_history} for t in trials],
        "code_snapshot": code_snapshot,
        "total_seconds": round(time.time() - started, 2),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
