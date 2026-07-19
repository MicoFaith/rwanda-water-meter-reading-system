#!/usr/bin/env python3
"""
04_predict.py

Run YOLO inference on test images and save visualized predictions.

Fully automatic:
- Finds latest best.pt from training_runs/
- Uses dataset/images/test
- Saves annotated images
- Saves prediction .txt files
- Saves confidence scores

Usage:
    python3 scripts/04_predict.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


TRAINING_ROOT = next(
    (p for p in [Path("runs/detect/training_runs"), Path("training_runs")] if list(p.glob("*/weights/best.pt"))),
    Path("runs/detect/training_runs"),
)
TEST_IMAGES_DIR = Path("dataset/images/test")
OUTPUT_PROJECT = Path("prediction_outputs")
OUTPUT_NAME = "test_predictions"

IMAGE_SIZE = 640
CONFIDENCE = 0.25

SAVE_TXT = True
SAVE_CONF = True

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


_model_cache: "YOLO | None" = None


def _get_model() -> "YOLO":
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(str(find_latest_best_model(TRAINING_ROOT)))
    return _model_cache


def predict_single(image_path: Path) -> dict:
    """Run inference on one image and return a reading dict (for API use)."""
    model = _get_model()
    try:
        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            save=False,
            save_txt=False,
            save_conf=False,
            verbose=False,
        )
    except Exception:
        return {"status": "error", "message": "Could not read image file. It may be corrupted or invalid."}
    if not results:
        return {"status": "error", "message": "No detections found."}
    return reconstruct_reading(results[0])


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))

    if not best_models:
        raise FileNotFoundError(
            f"No best.pt model found inside: {training_root}"
        )

    return max(best_models, key=lambda p: p.stat().st_mtime)


def iter_test_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def reconstruct_reading(result) -> dict:
    """Return meter reading reconstructed from a single YOLO result."""
    if result.boxes is None or len(result.boxes) == 0:
        return {"status": "error", "message": "No detections found."}

    names = result.names  # {int: str}
    boxes = result.boxes.xyxy.tolist()
    classes = [int(c) for c in result.boxes.cls.tolist()]
    confs = result.boxes.conf.tolist()

    window_cls = next((k for k, v in names.items() if v == "window"), None)
    unknown_cls = next((k for k, v in names.items() if v == "u"), None)
    digit_labels = {str(d) for d in range(10)}

    window_box = next(
        (box for box, cls in zip(boxes, classes) if cls == window_cls), None
    )
    if window_box is None:
        return {"status": "error", "message": "No window detected — cannot reconstruct reading."}

    wx1, wy1, wx2, wy2 = window_box

    digits = []
    for box, cls, conf in zip(boxes, classes, confs):
        label = names.get(cls, "")
        if label not in digit_labels and cls != unknown_cls:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
            digits.append((cx, label, conf, cls))

    if not digits:
        return {"status": "error", "message": "No digits found inside window."}

    digits.sort(key=lambda d: d[0])
    reading = "".join(d[1] for d in digits)
    avg_conf = sum(d[2] for d in digits) / len(digits)

    if any(d[3] == unknown_cls for d in digits):
        return {
            "status": "uncertain",
            "reading": reading,
            "confidence": f"{avg_conf:.2f}",
            "message": "Unknown digit(s) detected — please recapture the image.",
        }

    return {
        "status": "success",
        "reading": reading,
        "confidence": f"{avg_conf:.2f}",
    }


def main() -> None:
    model_path = find_latest_best_model(TRAINING_ROOT)

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Test images directory not found: {TEST_IMAGES_DIR}"
        )

    test_images = list(iter_test_images(TEST_IMAGES_DIR))

    if not test_images:
        raise RuntimeError(f"No supported test images found in: {TEST_IMAGES_DIR}")

    OUTPUT_PROJECT.mkdir(parents=True, exist_ok=True)

    print("========== YOLO TEST PREDICTION ==========")
    print(f"Model:       {model_path}")
    print(f"Test folder: {TEST_IMAGES_DIR}")
    print(f"Images:      {len(test_images)}")
    print(f"Output root: {OUTPUT_PROJECT}")
    print(f"Run name:    {OUTPUT_NAME}")
    print(f"Image size:  {IMAGE_SIZE}")
    print(f"Confidence:  {CONFIDENCE}")
    print()

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(TEST_IMAGES_DIR),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        save=True,
        save_txt=SAVE_TXT,
        save_conf=SAVE_CONF,
        show_labels=True,
        show_conf=False,
        project=str(OUTPUT_PROJECT),
        name=OUTPUT_NAME,
        exist_ok=False,
    )

    output_dir = OUTPUT_PROJECT / OUTPUT_NAME
    label_dir = output_dir / "labels"

    print("\nPrediction completed. [√]")
    print(f"Annotated images saved in: {output_dir}")
    print(f"Prediction txt files saved in: {label_dir}")

    print("\nPer-image summary:")
    for image_path, result in zip(test_images, results):
        num_boxes = 0 if result.boxes is None else len(result.boxes)
        print(f"  - {image_path.name}: {num_boxes} detection(s)")
        info = reconstruct_reading(result)
        if info["status"] == "success":
            print(f"    Reading: {info['reading']}  (avg conf: {info['confidence']})")
        elif info["status"] == "uncertain":
            print(f"    Reading: {info['reading']}  (avg conf: {info['confidence']}) [UNCERTAIN]")
            print(f"    Warning: {info['message']}")
        else:
            print(f"    Error:   {info['message']}")


if __name__ == "__main__":
    main()
