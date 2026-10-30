"""
Stage 1 AutoML Pipeline
------------------------
Given a folder of labeled images (one subfolder per class), this package:
  1. Validates and loads the dataset
  2. Tries multiple train/test splits
  3. Tries multiple candidate models (classical ML + a small CNN) with
     a few hyperparameter combinations each
  4. Scores every candidate and produces a leaderboard
  5. Outputs the single best (model, params, split) as JSON, ready to be
     handed to "Stage 2" (the user-facing training page).
"""

from .pipeline import run_stage1
from .models import MODEL_REGISTRY

__all__ = ["run_stage1", "MODEL_REGISTRY"]
