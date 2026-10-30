# Stage 1 AutoML Pipeline

This is a working implementation of **"Stage 1"** of the two-stage system we discussed:
a pipeline that takes a user's uploaded image dataset and searches for the best model,
hyperparameters, and train/test split. Its JSON output is exactly what you'd hand off
to "Stage 2" (the page where the user picks ML/DL/Transformer and training actually
happens).

It's a **real, runnable pipeline** — not a mockup — with:

- **Classical ML** (on extracted image features): Random Forest, XGBoost, SVM,
  Logistic Regression, K-Nearest Neighbors, Gradient Boosting.
- **Deep learning from scratch** (needs PyTorch): a Small CNN and a deeper CNN
  with batch norm/dropout.
- **Transfer learning** (needs PyTorch + torchvision): ResNet18 and MobileNetV2,
  both with an ImageNet-pretrained, frozen backbone and a freshly trained head —
  the option you'd actually reach for on real photos.

There are two ways to run it: a **browser-based web UI** (recommended — this is
where you upload a dataset, pick models, and watch it train) or the original
**CLI script**.

## Quick start — Web UI

```bash
pip install -r requirements.txt

# Generate a tiny synthetic dataset (colored shapes) to test with
python make_sample_data.py
cd sample_data && zip -r ../shapes_test.zip shapes && cd ..

# Launch the app
python app.py
```

Open **http://127.0.0.1:5000**, upload `shapes_test.zip` (or your own dataset zip),
tick the model(s) you want to run, set epochs, and hit "Start training". You'll land
on a live progress page, then a results/leaderboard page with a **"← Back to
upload"** button so you can immediately start another run.

Your own dataset should be zipped up with one subfolder per class:

```
my_dataset.zip
    class_a/
        img1.jpg
        img2.jpg
    class_b/
        img3.jpg
        ...
```

### Web UI pages

| Page | What it does |
|---|---|
| `/` | Upload the dataset zip, pick model(s), epochs, split ratio(s), ranking metric |
| `/processing/<job_id>` | Live progress bar (polls the backend every second); Cancel and Back buttons |
| `/results/<job_id>` | Leaderboard + best config, with buttons to go back and start another run |

Training runs in a background thread per job, so the progress page can poll it
without blocking the server — the same pattern you'd use with a job queue
(Celery/Ray) in production, just single-process here for simplicity.

## Quick start — CLI

```bash
pip install -r requirements.txt
python make_sample_data.py
python run_pipeline.py --data sample_data/shapes --out results.json --config example_config.yaml
```

## Project structure

```
stage1_automl/
├── app.py                    # Flask web UI: upload -> pick models -> progress -> results
├── templates/                 # upload.html, processing.html, results.html, base.html
├── static/                    # style.css, app.js (dropzone UX + progress polling)
├── webapp_data/                # per-job uploads/ and results/ written by the web UI (gitignored contents)
├── run_pipeline.py            # CLI entry point (what you run from the terminal)
├── make_sample_data.py        # Generates a synthetic test dataset
├── example_config.yaml        # Example search-space config (CLI only)
├── requirements.txt
└── stage1_pipeline/
    ├── data.py                 # Validates the uploaded dataset (catches corrupted
    │                           #   images, checks min images/class, etc.)
    ├── splits.py                # Generates stratified train/test splits (70/30, 80/20, 90/10)
    ├── models.py                # Candidate models + MODEL_REGISTRY (single source of truth
    │                           #   for what models exist -- both search.py and app.py read it)
    ├── search.py                 # The grid-search loop: tries every model x params x split;
    │                           #   supports cancellation via `should_stop`
    └── pipeline.py                # Ties it all together; this is what app.py (or your backend) calls
```

## How this maps onto your web app

`app.py` is a real, working example of this: it unpacks the uploaded zip, calls
`run_stage1()` in a background thread, and uses `progress_callback` to drive the
live progress bar. In a bigger production deployment you'd swap the background
thread for a job queue (Celery/Ray) and the in-memory `JOBS` dict for
Redis/Postgres, but call it the same way:

```python
from stage1_pipeline.pipeline import run_stage1

result = run_stage1(
    dataset_root="/uploads/user_123/dataset",
    output_json_path="/results/user_123/stage1.json",
    progress_callback=update_progress_in_db,  # e.g. writes % done to Postgres/Redis
                                               # so the frontend can poll/show a progress bar
)

# result["best_config"] is what you show the user and pre-fill on the
# Stage 2 model-selection page (they can also override it, e.g. pick
# "transformer" even though Stage 1's winner was classical ML)
```

The `progress_callback` is the hook for wiring this into a job queue (Celery/Ray)
or orchestrator (Prefect/Airflow) so it runs asynchronously instead of blocking
an HTTP request — see the note in `pipeline.py`.

## What's a stub vs. what's real

- **Real and functional**: dataset validation, stratified splitting, feature
  extraction, all ten registered models' training/eval, the web UI (upload,
  live progress, cancel, leaderboard), the search loop, JSON output.
- **Already includes real transfer learning**: ResNet18 and MobileNetV2 use
  actual ImageNet-pretrained weights from torchvision with a frozen backbone
  — no longer a "swap this in later" note, it's a selectable option today.
  If you want to go further, a HuggingFace ViT would slot in the same way
  (a new `train_eval_vit` function + one `MODEL_REGISTRY` entry).
- **Simplified on purpose**: the from-scratch CNNs are trained on small images
  for speed, which is why the pretrained options above usually win on real
  photos. The feature extractor for classical ML (resized pixels + color
  histogram) is intentionally simple — swap in something like HOG or CLIP
  embeddings if you want stronger classical-ML baselines.
- **Grid search, not Bayesian search**: `search.py` tries every combination in the
  search space. Fine for a handful of models/params; swap in Optuna or Ray Tune
  (same loop structure, smarter candidate selection) once your search space grows.

## Extending this

- **Add a model**: write a `train_eval_<name>(train_files, test_files, split_label, split_ratio, params)`
  function in `models.py` that returns a `TrialResult`, then add one entry to
  `MODEL_REGISTRY` at the bottom of that file (label, family, description, the
  function, an `available` flag, default params, and whether it uses epochs).
  That's it — both `search.py` and the web UI's checkbox list pick it up
  automatically; you don't need to touch either of them.
- **Change the ranking metric**: pass `metric="accuracy"` (or add your own, e.g.
  precision/recall) to `run_stage1()` — the web UI already exposes this as a
  dropdown.
- **Swap the job store**: `app.py` keeps job state in an in-memory dict
  (`JOBS`), which is fine for a single dev-server process. Move it to Redis or
  a DB table if you deploy with multiple gunicorn workers.
- **Async / distributed execution**: wrap the loop body in `search.py` as tasks in
  Prefect, Ray, or Celery so multiple model trainings run in parallel instead of
  sequentially — the function signatures don't need to change. `app.py`'s
  background thread is the simplest possible version of this.

## AI VisionLab web workflow
The web application now follows the product flow: Landing -> Dataset Upload -> Dataset Preview -> Model Selection -> Model Configuration -> Training Progress -> Results -> Model Comparison & Report. Authentication and AI-driven preprocessing recommendations are intentionally deferred to a later stage. The current backend performs training-time dataset discovery, leakage-safe train/test split generation, real model training/evaluation, leaderboard generation and exports a reproducible `.ipynb`, CSV and PDF report.

## Automatic dataset structure detection

AI VisionLab now detects common image dataset layouts automatically when a ZIP is uploaded.

### Supported layouts

**ImageFolder**
```text
dataset/
  class_a/
    image1.jpg
  class_b/
    image2.jpg
```

**Split datasets (including Intel Image Classification)**
```text
dataset/
  seg_train/
    class_a/
    class_b/
  seg_test/
    class_a/
    class_b/
  seg_pred/
    image1.jpg
    image2.jpg
```

For split datasets, `seg_train` is used as the labeled training source, `seg_test` is validated and reported as the test split, and `seg_pred` is recognized as an **unlabeled prediction set**. `seg_pred` is never treated as a class.
