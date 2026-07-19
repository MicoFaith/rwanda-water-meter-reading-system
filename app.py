#!/usr/bin/env python3
"""
app.py

Minimal Flask API for the Water Meter Reading System.

Endpoints:
    POST /predict   multipart/form-data field: "image"

Usage:
    python app.py
    curl -X POST http://localhost:5000/predict -F "image=@meter.jpg"
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import database

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("wmrs")
logger.setLevel(logging.INFO)
_file_handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)
logger.addHandler(logging.StreamHandler())

# ---------------------------------------------------------------------------
# Load 04_predict.py by file path (module name starts with a digit, so
# regular import syntax is not valid Python).
# ---------------------------------------------------------------------------
_predict_path = Path(__file__).parent / "04_predict.py"
_spec = importlib.util.spec_from_file_location("predict04", _predict_path)
_predict_module = importlib.util.module_from_spec(_spec)
sys.modules["predict04"] = _predict_module
_spec.loader.exec_module(_predict_module)

predict_single = _predict_module.predict_single  # callable: Path -> dict

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

database.init_database()

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Error handlers (always return JSON, never Flask's default HTML pages)
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def handle_too_large(_exc):
    return jsonify({"status": "error", "message": "Image is too large (max 10 MB)."}), 413


@app.errorhandler(404)
def handle_not_found(_exc):
    return jsonify({"status": "error", "message": "Not found."}), 404


@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    return jsonify({"status": "error", "message": exc.description}), exc.code


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    logger.exception("Unhandled server error")
    return jsonify({"status": "error", "message": "Internal server error."}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/api/readings")
def api_readings():
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 100))
    return jsonify(database.get_recent_readings(limit))


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image field in request."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({"status": "error", "message": "Unsupported file type."}), 415

    safe_name = secure_filename(file.filename) or "upload"
    suffix = Path(file.filename).suffix.lower()
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            file.save(tmp_path)

        info = predict_single(tmp_path)

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    status = info["status"]

    if status == "success":
        confidence = float(info["confidence"])
        database.save_reading(safe_name, info["reading"], confidence, status)
        logger.info("prediction success image=%s reading=%s confidence=%.2f", safe_name, info["reading"], confidence)
        return jsonify({
            "status": "success",
            "reading": info["reading"],
            "confidence": confidence,
        })

    if status == "uncertain":
        warning = info.get("message", "Unknown digit(s) detected. Please recapture the image.")
        confidence = float(info["confidence"])
        database.save_reading(safe_name, info["reading"], confidence, status, warning)
        logger.info("prediction uncertain image=%s reading=%s confidence=%.2f", safe_name, info["reading"], confidence)
        return jsonify({
            "status": "uncertain",
            "reading": info["reading"],
            "warning": warning,
        })

    logger.info("prediction error image=%s message=%s", safe_name, info.get("message"))
    return jsonify({
        "status": "error",
        "message": info.get("message", "Prediction failed."),
    }), 422


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
