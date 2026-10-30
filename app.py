#!/usr/bin/env python3
"""
app.py
------
The web UI for AI VisionLab.

Flow:
    1. "/"                          Home / Dashboard
    2. "/upload"                    Upload dataset zip
    3. "/start"           (POST)    Saves upload → redirect to preview
    4. "/preview/<job_id>"          Dataset characteristics preview
    5. "/preprocessing/<job_id>"    AI preprocessing recommendations
    6. "/preprocessing/<job_id>/save" (POST) Persist preprocessing config
    7. "/models/<job_id>"           Multi-model selection
    8. "/models/<job_id>/save" (POST) Save selected models → config
    9. "/config/<job_id>"           Hyperparameter configuration
   10. "/train/<job_id>"   (POST)   Start training → processing page
   11. "/processing/<job_id>"       Live training progress
   12. "/results/<job_id>"          Results, curves, confusion matrix
   13. "/compare/<job_id>"          Side-by-side model comparison
   14. "/report/<job_id>"           Full experiment report
   15. "/download/<job_id>/csv|pdf|notebook"  Export endpoints

Run with:
    python app.py
Then open http://127.0.0.1:5000
"""

import csv
import io
import json
import os
import shutil
import threading
import time
import uuid
import zipfile

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, flash, abort, send_file,
)
from werkzeug.utils import secure_filename

from stage1_pipeline.pipeline import run_stage1
from stage1_pipeline.models import MODEL_REGISTRY
from stage1_pipeline.data import discover_for_training, inspect_dataset_characteristics
from stage1_pipeline.preprocessing import generate_recommendations, generate_transformed_preview

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "webapp_data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_UPLOAD_MB = 500

app = Flask(__name__)
app.secret_key = os.environ.get("STAGE1_SECRET_KEY", "dev-secret-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# In-memory job store
JOBS: dict = {}
JOBS_LOCK = threading.Lock()

SPLIT_CHOICES = [
    (0.7, "70% train / 30% test"),
    (0.8, "80% train / 20% test"),
    (0.9, "90% train / 10% test"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _extract_dataset(zip_path: str, extract_to: str) -> str:
    """Extract zip and return the path that actually contains class subfolders."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)

    macosx = os.path.join(extract_to, "__MACOSX")
    if os.path.isdir(macosx):
        shutil.rmtree(macosx, ignore_errors=True)

    root = extract_to
    for _ in range(3):
        entries = [e for e in os.listdir(root) if not e.startswith(".")]
        if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
            root = os.path.join(root, entries[0])
        else:
            break
    return root


def _build_search_space(selected_models, epochs, split_choices_selected):
    """Build the search_space dict run_stage1() expects."""
    search_space = {}
    for key in selected_models:
        entry = MODEL_REGISTRY.get(key)
        if entry is None or not entry["available"]:
            continue
        params = {k: [v] for k, v in entry["default_params"].items()}
        if entry["supports_epochs"]:
            params["epochs"] = [epochs]
        search_space[key] = params
    return search_space


def _inspect_dataset_safe(dataset_root: str, report=None):
    """Inspect dataset characteristics, returning a safe dict even on failure."""
    try:
        if report is None:
            report = discover_for_training(dataset_root)
        return inspect_dataset_characteristics(report)
    except Exception:
        return {
            "avg_width": 96, "avg_height": 96,
            "uniform_size": True, "formats": {},
            "is_imbalanced": False, "imbalance_ratio": 1.0,
            "majority_class": "", "minority_class": "",
        }


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

def _run_job(job_id, dataset_root, search_space, split_ratios, metric):
    start_time = time.time()

    def progress_callback(done, total, description):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                return
            job["done"] = done
            job["total"] = max(total, 1)
            job["description"] = description
            elapsed = int(time.time() - start_time)
            job["elapsed_seconds"] = elapsed
            if description:
                job.setdefault("logs", []).append(f"[{elapsed}s] {description}")

    def epoch_callback(info):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                return
            ep       = info.get("epoch", 1)
            tot_ep   = max(info.get("total_epochs", 1), 1)
            model_nm = info.get("model_name", "")
            j_step   = info.get("job_step", 1) - 1
            t_jobs   = max(info.get("total_jobs", 1), 1)

            job["current_epoch"]  = ep
            job["total_epochs"]   = tot_ep
            job["train_loss"]     = info.get("train_loss")
            job["val_loss"]       = info.get("val_loss")
            job["train_acc"]      = info.get("train_acc")
            job["val_acc"]        = info.get("val_acc")
            job["current_model"]  = model_nm
            job["model_index"]    = j_step + 1

            elapsed = int(time.time() - start_time)
            job["elapsed_seconds"] = elapsed

            frac = (j_step + (ep / tot_ep)) / t_jobs
            if frac > 0.05:
                est_total = elapsed / frac
                job["remaining_seconds"] = max(1, int(est_total - elapsed))
            else:
                job["remaining_seconds"] = None

            pct = min(int(10 + 85 * frac), 98)
            job["done"]  = pct
            job["total"] = 100
            job["description"] = f"Training {model_nm} (Epoch {ep}/{tot_ep})"
            log_msg = info.get("log_message")
            if log_msg:
                job.setdefault("logs", []).append(f"[{elapsed}s] {log_msg}")

    def should_stop():
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            return bool(job and job.get("cancel_requested"))

    out_path = os.path.join(RESULTS_DIR, f"{job_id}.json")

    try:
        result = run_stage1(
            dataset_root=dataset_root,
            output_json_path=out_path,
            search_space=search_space,
            split_ratios=split_ratios,
            metric=metric,
            min_images_per_class=0,
            progress_callback=progress_callback,
            epoch_callback=epoch_callback,
            should_stop=should_stop,
        )
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"]           = "done"
                job["done"]             = 100
                job["total"]            = 100
                job["description"]      = "Training and evaluation complete"
                job["remaining_seconds"] = 0
                job["result"]           = result
                job.setdefault("logs", []).append(
                    f"[{int(time.time() - start_time)}s] Experiment completed. Leaderboard generated."
                )
    except Exception as exc:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["error"]  = str(exc)
                job.setdefault("logs", []).append(
                    f"[{int(time.time() - start_time)}s] ERROR: {str(exc)}"
                )


def _launch_job(job_id):
    """Build search space from current job state and kick off training thread."""
    job = _job(job_id)
    selected     = job["selected_keys"]
    search_space = _build_search_space(selected, job["epochs"], job["split_ratios"])

    # Apply UI hyperparameter overrides to every DL / Hybrid model
    for key in search_space:
        m = MODEL_REGISTRY[key]
        if m["family"] in ("dl", "hybrid"):
            search_space[key]["batch_size"]          = [job["batch_size"]]
            search_space[key]["learning_rate"]       = [job["learning_rate"]]
            search_space[key]["optimizer"]           = [job["optimizer"]]
            search_space[key]["loss"]                = [job["loss"]]
            search_space[key]["image_size"]          = [job.get("image_size", 96)]
            search_space[key]["augmentation"]        = [job.get("augmentation_enabled", True)]
            search_space[key]["augmentation_config"] = [job.get("augmentation_config", {})]
            search_space[key]["class_balancing"]     = [job.get("class_balancing_enabled", False)]
        else:
            search_space[key]["class_balancing"]     = [job.get("class_balancing_enabled", False)]

    with JOBS_LOCK:
        job["status"]           = "running"
        job["done"]             = 0
        job["total"]            = 1
        job["description"]      = "Starting…"
        job["result"]           = None
        job["error"]            = None
        job["cancel_requested"] = False

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, job["dataset_root"], search_space, job["split_ratios"], job["metric"]),
        daemon=True,
    )
    thread.start()
    return redirect(url_for("processing_page", job_id=job_id))


# ---------------------------------------------------------------------------
# Routes — Home & Upload
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html", active="dashboard", job_id=None)


@app.route("/upload")
def upload_page():
    return render_template("upload.html", active="upload", job_id=None)


@app.route("/start", methods=["POST"])
def start_job():
    upload = request.files.get("dataset")
    if not upload or upload.filename == "":
        flash("Please choose a dataset .zip file to upload.", "error")
        return redirect(url_for("upload_page"))
    if not upload.filename.lower().endswith(".zip"):
        flash("Dataset must be uploaded as a .zip file.", "error")
        return redirect(url_for("upload_page"))

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    zip_path = os.path.join(job_upload_dir, secure_filename(upload.filename))
    upload.save(zip_path)

    dataset_dir = os.path.join(job_upload_dir, "dataset")
    try:
        dataset_root = _extract_dataset(zip_path, dataset_dir)
    except zipfile.BadZipFile:
        flash("That file doesn't look like a valid .zip archive.", "error")
        return redirect(url_for("upload_page"))
    except Exception as exc:
        flash(f"Could not read the uploaded dataset: {exc}", "error")
        return redirect(url_for("upload_page"))

    # Inspect dataset now so preview/preprocessing pages have real data
    try:
        report   = discover_for_training(dataset_root)
        summary  = report.summary()
        chars    = inspect_dataset_characteristics(report)
    except Exception as exc:
        flash(f"Dataset structure could not be read: {exc}", "error")
        return redirect(url_for("upload_page"))

    # Compute AI preprocessing recommendations immediately
    try:
        recs = generate_recommendations(summary, chars)
    except Exception:
        recs = {}

    # Pick a sample image per class for the preview page
    samples = []
    class_counts = {}
    try:
        seen_classes: set = set()
        for item in report.valid_files:
            if isinstance(item, tuple):
                path, cls = item
            else:
                path, cls = getattr(item, "path", str(item)), getattr(item, "label", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in seen_classes:
                seen_classes.add(cls)
                relpath = os.path.relpath(path, dataset_root).replace("\\", "/")
                samples.append({"label": cls, "relpath": relpath})
    except Exception as exc:
        pass

    dataset_name = request.form.get("dataset_name", "").strip() or upload.filename

    with JOBS_LOCK:
        JOBS[job_id] = {
            # core status
            "status":           "draft",
            "done":             0,
            "total":            1,
            "description":      "Dataset uploaded",
            "result":           None,
            "error":            None,
            "cancel_requested": False,
            # dataset
            "dataset_name":    dataset_name,
            "dataset_root":    dataset_root,
            "dataset_summary": summary,
            "characteristics": chars,
            "recommendations": recs,
            "class_counts":    class_counts,
            "samples":         samples,
            # config defaults (overridden via preprocessing / config pages)
            "selected_models": [],
            "selected_keys":   [],
            "epochs":          5,
            "batch_size":      16,
            "learning_rate":   0.001,
            "image_size":      recs.get("resizing", {}).get("recommended_size", 96),
            "optimizer":       "Adam",
            "loss":            "Cross Entropy",
            "metric":          "f1_macro",
            "split_ratios":    [recs.get("split", {}).get("recommended_ratio", 0.8)],
            "normalization":   recs.get("normalization", {}).get("recommended", "imagenet"),
            "augmentation_enabled": recs.get("augmentation", {}).get("recommended", True),
            "augmentation_config": {
                "horizontal_flip": recs.get("augmentation", {}).get("techniques", {}).get("horizontal_flip", {}).get("recommended", True),
                "vertical_flip": recs.get("augmentation", {}).get("techniques", {}).get("vertical_flip", {}).get("recommended", False),
                "random_rotation": recs.get("augmentation", {}).get("techniques", {}).get("random_rotation", {}).get("recommended", True),
                "rotation_deg": recs.get("augmentation", {}).get("techniques", {}).get("random_rotation", {}).get("degree", 12),
                "color_jitter": recs.get("augmentation", {}).get("techniques", {}).get("color_jitter", {}).get("recommended", True),
                "random_crop": recs.get("augmentation", {}).get("techniques", {}).get("random_crop", {}).get("recommended", False),
                "gaussian_blur": recs.get("augmentation", {}).get("techniques", {}).get("gaussian_blur", {}).get("recommended", False),
            },
            "class_balancing_enabled": recs.get("class_balancing", {}).get("recommended", False),
            "created_at":      time.time(),
        }

    return redirect(url_for("preview_page", job_id=job_id))


# ---------------------------------------------------------------------------
# Routes — Dataset Preview
# ---------------------------------------------------------------------------

@app.route("/preview/<job_id>")
def preview_page(job_id):
    job = _job(job_id)
    if not job:
        flash("Job not found. Please upload a dataset.", "error")
        return redirect(url_for("upload_page"))

    summary  = job.get("dataset_summary", {})
    chars    = job.get("characteristics", {})
    samples  = job.get("samples", [])
    class_counts = job.get("class_counts", {})
    max_count = max(class_counts.values()) if class_counts else 1

    return render_template(
        "preview.html",
        active="preview",
        job_id=job_id,
        job=job,
        summary=summary,
        characteristics=chars,
        samples=samples,
        class_counts=class_counts,
        max_count=max_count,
    )


@app.route("/dataset-image/<job_id>/<path:filename>")
def dataset_image(job_id, filename):
    job = _job(job_id)
    if not job:
        abort(404)
    root   = os.path.realpath(job["dataset_root"])
    target = os.path.realpath(os.path.join(root, filename))
    if not target.startswith(root + os.sep) or not os.path.isfile(target):
        abort(404)
    return send_file(target)


# ---------------------------------------------------------------------------
# Routes — AI Preprocessing
# ---------------------------------------------------------------------------

@app.route("/preprocessing/<job_id>")
def preprocessing_page(job_id):
    job = _job(job_id)
    if not job:
        flash("Job not found.", "error")
        return redirect(url_for("upload_page"))

    summary  = job.get("dataset_summary", {})
    chars    = job.get("characteristics", {})
    recs     = job.get("recommendations", {})

    # Ensure recommendations exist (edge case: job created before inspection)
    if not recs:
        recs = generate_recommendations(summary, chars)
        with JOBS_LOCK:
            job["recommendations"] = recs

    # Pick a single sample image for Before/After preview
    sample_image = job.get("samples", [None])[0] if job.get("samples") else None

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        job_id=job_id,
        job=job,
        summary=summary,
        characteristics=chars,
        recs=recs,
        aug_config=job.get("augmentation_config", {}),
        sample_image=sample_image,
    )


@app.route("/preprocessing/<job_id>/save", methods=["POST"])
def save_preprocessing(job_id):
    job = _job(job_id)
    if not job:
        abort(404)

    image_size    = int(request.form.get("image_size", 96))
    normalization = request.form.get("normalization", "imagenet")
    aug_enabled   = request.form.get("augmentation_enabled") == "true"
    bal_enabled   = request.form.get("class_balancing_enabled") == "true"
    split_ratio   = float(request.form.get("split_ratio", 0.8))

    aug_config = {
        "horizontal_flip": request.form.get("aug_horizontal_flip") == "true",
        "vertical_flip": request.form.get("aug_vertical_flip") == "true",
        "random_rotation": request.form.get("aug_random_rotation") == "true",
        "rotation_deg": int(request.form.get("aug_rotation_deg", 12)),
        "color_jitter": request.form.get("aug_color_jitter") == "true",
        "random_crop": request.form.get("aug_random_crop") == "true",
        "gaussian_blur": request.form.get("aug_gaussian_blur") == "true",
    }

    with JOBS_LOCK:
        job["image_size"]              = image_size
        job["normalization"]           = normalization
        job["augmentation_enabled"]    = aug_enabled
        job["augmentation_config"]     = aug_config
        job["class_balancing_enabled"] = bal_enabled
        job["split_ratios"]            = [split_ratio]

    return redirect(url_for("models_page", job_id=job_id))


@app.route("/preview-transform/<job_id>/<path:filename>")
def preview_transform(job_id, filename):
    """Serve a transformed (resized + augmented) preview of a dataset image."""
    job = _job(job_id)
    if not job:
        abort(404)
    root   = os.path.realpath(job["dataset_root"])
    target = os.path.realpath(os.path.join(root, filename))
    if not target.startswith(root + os.sep) or not os.path.isfile(target):
        abort(404)

    target_size = int(request.args.get("size", job.get("image_size", 96)))
    aug_enabled = request.args.get("enabled", "true" if job.get("augmentation_enabled", True) else "false") == "true"

    if any(k in request.args for k in ["hflip", "vflip", "rot", "jitter", "crop", "blur"]):
        aug_config = {
            "horizontal_flip": request.args.get("hflip", "false") == "true",
            "vertical_flip": request.args.get("vflip", "false") == "true",
            "random_rotation": request.args.get("rot", "false") == "true",
            "rotation_deg": int(request.args.get("rot_deg", 12)),
            "color_jitter": request.args.get("jitter", "false") == "true",
            "random_crop": request.args.get("crop", "false") == "true",
            "gaussian_blur": request.args.get("blur", "false") == "true",
        }
    else:
        aug_config = job.get("augmentation_config", {})

    try:
        img_bytes = generate_transformed_preview(
            target, target_size=target_size, aug_config=aug_config, apply_aug=aug_enabled, apply_norm=True
        )
        return send_file(io.BytesIO(img_bytes), mimetype="image/png")
    except Exception:
        abort(500)


# ---------------------------------------------------------------------------
# Routes — Model Selection & Configuration
# ---------------------------------------------------------------------------

@app.route("/models/<job_id>")
def models_page(job_id):
    job = _job(job_id)
    if not job:
        abort(404)
    dl       = {k: v for k, v in MODEL_REGISTRY.items() if v["family"] == "dl"}
    ml       = {k: v for k, v in MODEL_REGISTRY.items() if v["family"] == "ml"}
    hybrid   = {k: v for k, v in MODEL_REGISTRY.items() if v["family"] == "hybrid"}
    ensemble = {k: v for k, v in MODEL_REGISTRY.items() if v["family"] == "ensemble"}
    return render_template(
        "models.html",
        active="models",
        job_id=job_id,
        job=job,
        model_groups={"dl": dl, "ml": ml, "hybrid": hybrid, "ensemble": ensemble},
        selected=job.get("selected_keys", []),
    )


@app.route("/models/<job_id>/save", methods=["POST"])
def save_models(job_id):
    job = _job(job_id)
    if not job:
        abort(404)

    selected = [
        k for k in request.form.getlist("models")
        if k in MODEL_REGISTRY and MODEL_REGISTRY[k]["available"]
    ]
    if not selected:
        flash("Select at least one available model.", "error")
        return redirect(url_for("models_page", job_id=job_id))

    with JOBS_LOCK:
        job["selected_keys"]   = selected
        job["selected_models"] = [MODEL_REGISTRY[k]["label"] for k in selected]

    return redirect(url_for("config_page", job_id=job_id))


@app.route("/config/<job_id>")
def config_page(job_id):
    job = _job(job_id)
    if not job:
        abort(404)
    if not job.get("selected_keys"):
        return redirect(url_for("models_page", job_id=job_id))
    return render_template(
        "config.html",
        active="config",
        job_id=job_id,
        job=job,
        selected=job["selected_keys"],
        selected_labels=job["selected_models"],
        split_choices=SPLIT_CHOICES,
    )


# ---------------------------------------------------------------------------
# Routes — Training
# ---------------------------------------------------------------------------

@app.route("/train/<job_id>", methods=["POST"])
def train_job(job_id):
    job = _job(job_id)
    if not job:
        abort(404)

    with JOBS_LOCK:
        job["epochs"]       = max(1, min(int(request.form.get("epochs", 5)), 100))
        job["batch_size"]   = max(1, min(int(request.form.get("batch_size", 16)), 256))
        job["learning_rate"] = float(request.form.get("learning_rate", 0.001))
        job["image_size"]   = int(request.form.get("image_size", 96))
        job["optimizer"]    = request.form.get("optimizer", "Adam")
        job["loss"]         = request.form.get("loss", "Cross Entropy")
        job["metric"]       = request.form.get("metric", "f1_macro")
        split_vals = request.form.getlist("split_ratios")
        job["split_ratios"] = [float(x) for x in split_vals] if split_vals else [0.8]

    return _launch_job(job_id)


# Keep /start-training alias for backwards compat
@app.route("/start-training/<job_id>", methods=["POST"])
def start_training_alias(job_id):
    return train_job(job_id)


@app.route("/processing/<job_id>")
def processing_page(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        flash("That job wasn't found — it may have expired. Start a new run.", "error")
        return redirect(url_for("upload_page"))
    if job["status"] == "done":
        return redirect(url_for("results_page", job_id=job_id))
    return render_template("processing.html", active="training", job_id=job_id, job=job)


@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"status": "not_found"}), 404

        done, total = job["done"], job["total"]
        pct = int(100 * done / total) if total else 0

        payload = {
            "status":      job["status"],
            "done":        done,
            "total":       total,
            "pct":         pct,
            "description": job["description"],
            "error":       job.get("error"),
            # Live epoch counters
            "current_epoch":    job.get("current_epoch"),
            "total_epochs":     job.get("total_epochs"),
            "current_model":    job.get("current_model"),
            "model_index":      job.get("model_index"),
            "train_loss":       job.get("train_loss"),
            "val_loss":         job.get("val_loss"),
            "train_acc":        job.get("train_acc"),
            "val_acc":          job.get("val_acc"),
            "elapsed_seconds":  job.get("elapsed_seconds", 0),
            "remaining_seconds": job.get("remaining_seconds"),
            # Execution logs (last 30 lines)
            "logs": job.get("logs", [])[-30:],
        }
    return jsonify(payload)


@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"ok": False}), 404
        job["cancel_requested"] = True
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — Results, Comparison, Report
# ---------------------------------------------------------------------------

@app.route("/results/<job_id>")
def results_page(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        flash("That job wasn't found — it may have expired.", "error")
        return redirect(url_for("upload_page"))
    if job["status"] == "error":
        flash(f"Training failed: {job['error']}", "error")
        return redirect(url_for("upload_page"))
    if job["status"] != "done":
        return redirect(url_for("processing_page", job_id=job_id))

    result = job["result"]
    return render_template(
        "results.html",
        active="results",
        job_id=job_id,
        job=job,
        result=result,
        leaderboard=result["leaderboard"],
        best=result["best_config"],
        dataset_summary=result["dataset_summary"],
    )


@app.route("/compare/<job_id>")
def comparison_page(job_id):
    job = _job(job_id)
    if not job or job.get("status") != "done":
        return redirect(url_for("results_page", job_id=job_id))
    result = job["result"]
    return render_template(
        "comparison.html",
        active="comparison",
        job_id=job_id,
        job=job,
        result=result,
        leaderboard=result["leaderboard"],
        best=result["best_config"],
    )


@app.route("/report/<job_id>")
def report_page(job_id):
    job = _job(job_id)
    if not job or job.get("status") != "done":
        return redirect(url_for("results_page", job_id=job_id))
    result = job["result"]
    return render_template(
        "report.html",
        active="report",
        job_id=job_id,
        job=job,
        result=result,
        leaderboard=result["leaderboard"],
        best=result["best_config"],
    )


# ---------------------------------------------------------------------------
# Download endpoints
# ---------------------------------------------------------------------------

@app.route("/download/<job_id>/csv")
def download_csv(job_id):
    job = _job(job_id)
    if not job or job.get("status") != "done":
        abort(404)

    rows = job["result"]["leaderboard"]
    out  = io.StringIO()
    w    = csv.writer(out)
    w.writerow([
        "model", "family", "split",
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
        "train_accuracy", "generalization_gap",
        "train_seconds", "param_count", "params",
    ])
    for r in rows:
        w.writerow([
            r["model_name"],
            r["model_family"],
            r["split_label"],
            r["accuracy"],
            r.get("precision_macro", ""),
            r.get("recall_macro", ""),
            r["f1_macro"],
            r.get("roc_auc", ""),
            r.get("train_accuracy", ""),
            r.get("generalization_gap", ""),
            r["train_seconds"],
            r.get("param_count_str", ""),
            json.dumps(r["params"]),
        ])

    data = io.BytesIO(out.getvalue().encode())
    data.seek(0)
    return send_file(
        data,
        as_attachment=True,
        download_name=f"{job_id}_results.csv",
        mimetype="text/csv",
    )


@app.route("/download/<job_id>/notebook")
def download_notebook(job_id):
    """Export a reproducible Jupyter Notebook with the real training outputs."""
    job = _job(job_id)
    if not job or job.get("status") != "done":
        abort(404)

    result    = job["result"]
    snapshots = result.get("code_snapshot", {})
    best      = result["best_config"]

    code_bundle = {p: s for p, s in snapshots.items() if s}
    experiment  = {
        "dataset_name":     job["dataset_name"],
        "selected_models":  job["selected_models"],
        "selected_keys":    job["selected_keys"],
        "epochs":           job["epochs"],
        "batch_size":       job["batch_size"],
        "learning_rate":    job["learning_rate"],
        "image_size":       job["image_size"],
        "optimizer":        job["optimizer"],
        "loss":             job["loss"],
        "metric":           job["metric"],
        "split_ratios":     job["split_ratios"],
        "dataset_summary":  result["dataset_summary"],
        "best_config":      best,
    }

    def src(text):
        return text.splitlines(keepends=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata":  {},
            "source":    src(
                "# AI VisionLab — Reproducible Training Run\n\n"
                f"**Dataset:** `{job['dataset_name']}`  \n"
                f"**Best Model:** `{best['model_name']}`  \n"
                f"**Evaluation:** `{best['split_label']}`  \n\n"
                "This notebook was auto-generated by AI VisionLab and contains the "
                "exact pipeline source and training outputs from this experiment run."
            ),
        },
        {
            "cell_type":       "code",
            "execution_count": 1,
            "metadata":        {},
            "outputs":         [],
            "source":          src(
                "from pathlib import Path\nimport json\n\n"
                f"DATASET_DIR = Path('dataset')\n"
                f"EXPERIMENT = {json.dumps(experiment, indent=2)}\n"
                "EXPERIMENT\n"
            ),
        },
    ]

    for idx, (rel, source) in enumerate(code_bundle.items(), start=2):
        cells.append({
            "cell_type": "markdown", "metadata": {},
            "source":    src(f"## Source: `{rel}`\n"),
        })
        cells.append({
            "cell_type": "code", "execution_count": idx,
            "metadata":  {}, "outputs": [], "source": src(source),
        })

    setup_code = (
        "# Recreate the exact pipeline package from the snapshot above.\n"
        "from pathlib import Path\nimport sys\n\n"
        "PROJECT = Path('visionlab_repro')\n"
        "PACKAGE = PROJECT / 'stage1_pipeline'\n"
        "PACKAGE.mkdir(parents=True, exist_ok=True)\n"
        "(PACKAGE / '__init__.py').write_text('', encoding='utf-8')\n\n"
        f"SOURCE_FILES = {list(code_bundle.keys())!r}\n"
        f"SOURCE_TEXT  = {code_bundle!r}\n\n"
        "for rel in SOURCE_FILES:\n"
        "    t = PROJECT / rel\n"
        "    t.parent.mkdir(parents=True, exist_ok=True)\n"
        "    t.write_text(SOURCE_TEXT[rel], encoding='utf-8')\n\n"
        "if str(PROJECT.resolve()) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT.resolve()))\n\n"
        f"print(f'Recreated {{len(SOURCE_FILES)}} source files under {{PROJECT}}')\n"
    )
    cells.append({
        "cell_type": "code", "execution_count": len(cells) + 1,
        "metadata":  {},
        "outputs":   [{"output_type": "stream", "name": "stdout",
                       "text": [f"Recreated {len(code_bundle)} source files under visionlab_repro\n"]}],
        "source":    src(setup_code),
    })

    rerun_code = (
        "# Re-run the experiment on the original dataset.\n"
        "from stage1_pipeline.pipeline import run_stage1\n\n"
        f"SEARCH_SPACE = {json.dumps({k: {pk: [pv] for pk, pv in ({**MODEL_REGISTRY[k]['default_params'], **({'epochs': job['epochs']} if MODEL_REGISTRY[k]['supports_epochs'] else {})}).items()} for k in job['selected_keys'] if k in MODEL_REGISTRY}, indent=2)}\n\n"
        "RESULT = run_stage1(\n"
        "    dataset_root=str(DATASET_DIR),\n"
        "    output_json_path='visionlab_reproduction_results.json',\n"
        f"    search_space=SEARCH_SPACE,\n"
        f"    split_ratios={job['split_ratios']!r},\n"
        f"    metric={job['metric']!r},\n"
        ")\nRESULT\n"
    )
    cells.append({
        "cell_type": "markdown", "metadata": {},
        "source":    src("## Reproduction code and recorded result\n"),
    })
    cells.append({
        "cell_type": "code", "execution_count": len(cells) + 1,
        "metadata":  {},
        "outputs":   [{
            "output_type":    "execute_result",
            "execution_count": len(cells) + 1,
            "data":           {"text/plain": [json.dumps(result, indent=2, default=str)]},
            "metadata":       {},
        }],
        "source":    src(rerun_code),
    })

    for trial in result.get("training_outputs", []):
        log_text = "\n".join(trial.get("log", [])) + "\n"
        cells.append({
            "cell_type": "markdown", "metadata": {},
            "source":    src(f"## Recorded output — {trial['model_name']} / {trial['split_label']}\n"),
        })
        cells.append({
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs":   [{"output_type": "stream", "name": "stdout", "text": src(log_text)}],
            "source":    src(
                f"# Training history:\n"
                f"{json.dumps(trial.get('history', []), indent=2)}\n"
                "print('Output captured during original training run.')\n"
            ),
        })

    # Final results summary
    cells.append({
        "cell_type": "markdown", "metadata": {},
        "source":    src("## Final holdout results\n"),
    })
    final_data = {
        "model":               best["model_name"],
        "split":               best["split_label"],
        "accuracy":            best["expected_accuracy"],
        "precision_macro":     best.get("expected_precision", 0),
        "recall_macro":        best.get("expected_recall", 0),
        "f1_macro":            best["expected_f1_macro"],
        "roc_auc":             best.get("expected_roc_auc", 0),
        "train_accuracy":      best.get("train_accuracy", 0),
        "generalization_gap":  best.get("generalization_gap", 0),
        "confusion_matrix":    best["confusion_matrix"],
        "class_names":         best["class_names"],
        "prediction_sample":   best["prediction_sample"],
    }
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs":   [{"output_type": "stream", "name": "stdout",
                       "text": src(json.dumps(final_data, indent=2, default=str) + "\n")}],
        "source":    src(
            f"final_results = {json.dumps(final_data, indent=2, default=str)}\n"
            "final_results\n"
        ),
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    data = io.BytesIO(json.dumps(nb, indent=2, default=str).encode("utf-8"))
    data.seek(0)
    return send_file(
        data,
        as_attachment=True,
        download_name=f"{job_id}_visionlab.ipynb",
        mimetype="application/x-ipynb+json",
    )


@app.route("/download/<job_id>/pdf")
def download_pdf(job_id):
    job = _job(job_id)
    if not job or job.get("status") != "done":
        abort(404)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib import colors
    except ImportError:
        flash("PDF export requires reportlab. Install it with: pip install reportlab", "error")
        return redirect(url_for("report_page", job_id=job_id))

    r   = job["result"]
    buf = io.BytesIO()
    c   = rl_canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    def new_page_if_needed(y_pos, needed=40):
        if y_pos < needed:
            c.showPage()
            return H - 55
        return y_pos

    # Header
    y = H - 50
    c.setFillColor(colors.HexColor("#151827"))
    c.rect(0, H - 70, W, 70, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H - 42, "AI VisionLab — Experiment Report")
    c.setFont("Helvetica", 9)
    c.drawString(40, H - 58, f"Dataset: {job['dataset_name']}  |  Generated by AI VisionLab")

    y = H - 90

    # Dataset summary
    c.setFillColor(colors.HexColor("#1e1e2e"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Dataset Summary")
    y -= 18
    c.setFillColor(colors.HexColor("#475569"))
    c.setFont("Helvetica", 10)
    ds = r["dataset_summary"]
    c.drawString(40, y, f"Valid images: {ds.get('num_valid_images', 'N/A')}   Classes: {ds.get('num_classes', 'N/A')}   Trials run: {r['num_trials_run']}")
    y -= 14
    c.drawString(40, y, f"Best model: {r['best_config']['model_name']}   Split: {r['best_config']['split_label']}")
    y -= 30

    # Leaderboard table
    c.setFillColor(colors.HexColor("#1e1e2e"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Model Leaderboard")
    y -= 20

    headers = ["Model", "Split", "Acc", "Prec", "Recall", "F1", "ROC-AUC", "Time"]
    col_x   = [40, 155, 245, 285, 330, 375, 420, 475]
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#6d5dfc"))
    for i, h in enumerate(headers):
        c.drawString(col_x[i], y, h)
    y -= 14

    c.setFont("Helvetica", 8)
    for idx, row in enumerate(r["leaderboard"]):
        y = new_page_if_needed(y, 20)
        bg = colors.HexColor("#f5f3ff") if idx % 2 == 0 else colors.white
        c.setFillColor(bg)
        c.rect(36, y - 4, W - 72, 14, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#1e293b"))
        vals = [
            row["model_name"][:18],
            row["split_label"][:12],
            f"{row['accuracy']:.1%}",
            f"{row.get('precision_macro', 0):.1%}",
            f"{row.get('recall_macro', 0):.1%}",
            f"{row['f1_macro']:.1%}",
            f"{row.get('roc_auc', 0):.3f}",
            f"{row['train_seconds']}s",
        ]
        for i, v in enumerate(vals):
            c.drawString(col_x[i], y, v)
        y -= 14

    y -= 20
    y = new_page_if_needed(y, 80)

    # Best config details
    bc = r["best_config"]
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.HexColor("#1e1e2e"))
    c.drawString(40, y, "Best Configuration Detail")
    y -= 18
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#475569"))
    lines = [
        f"Model: {bc['model_name']}   Parameters: {bc.get('param_count_str', 'N/A')}",
        f"Holdout Accuracy: {bc['expected_accuracy']:.2%}   Macro F1: {bc['expected_f1_macro']:.2%}",
        f"Precision (macro): {bc.get('expected_precision', 0):.2%}   Recall (macro): {bc.get('expected_recall', 0):.2%}",
        f"ROC-AUC: {bc.get('expected_roc_auc', 0):.4f}   Train Accuracy: {bc.get('train_accuracy', 0):.2%}",
        f"Generalization Gap: {bc.get('generalization_gap', 0):.2%}",
    ]
    for line in lines:
        y = new_page_if_needed(y, 15)
        c.drawString(40, y, line)
        y -= 14

    y -= 16
    # Confusion matrix
    y = new_page_if_needed(y, 80)
    cm   = bc.get("confusion_matrix", [])
    clsn = bc.get("class_names", [])
    if cm and clsn:
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.HexColor("#1e1e2e"))
        c.drawString(40, y, "Confusion Matrix (holdout predictions)")
        y -= 20
        cell_w = min(50, int((W - 80) / (len(clsn) + 1)))
        # Header row
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(colors.HexColor("#6d5dfc"))
        c.drawString(40, y, "True \\ Pred")
        for j, cn in enumerate(clsn):
            c.drawString(40 + (j + 1) * cell_w, y, cn[:8])
        y -= 14
        c.setFont("Helvetica", 8)
        for i, row_vals in enumerate(cm):
            y = new_page_if_needed(y, 15)
            c.setFillColor(colors.HexColor("#1e293b"))
            c.drawString(40, y, clsn[i][:10] if i < len(clsn) else str(i))
            for j, val in enumerate(row_vals):
                fill = colors.HexColor("#eeecff") if i == j else colors.white
                c.setFillColor(fill)
                c.rect(40 + (j + 1) * cell_w - 2, y - 3, cell_w - 2, 12, fill=True, stroke=False)
                c.setFillColor(colors.HexColor("#1e293b"))
                c.drawString(40 + (j + 1) * cell_w, y, str(val))
            y -= 14

    c.save()
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{job_id}_report.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
