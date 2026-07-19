#!/usr/bin/env python3
"""
05_retrain.py

Retrain / continue training YOLO to reach ≥95% mAP@0.5.

Modes:
  --mode fresh   Start from yolov8s.pt (larger model, more capacity)
  --mode resume  Continue from the latest best.pt checkpoint (default)

Key improvements over baseline:
  - yolov8s replaces yolov8n: S-model has 11.2M vs 3.2M params → higher ceiling
  - No flips: digits 6/9 and 2/5 are orientation-sensitive; flipping destroys class identity
  - Larger image size (832): resolves individual digits more sharply
  - Cosine LR decay + low final LR: smoother convergence at the end
  - copy_paste augmentation: synthesises more small-object examples
  - patience=50: keeps training until the model genuinely stalls

Usage:
    python 05_retrain.py                        # resume from best, 200 epochs
    python 05_retrain.py --mode fresh           # start fresh with yolov8s
    python 05_retrain.py --epochs 300 --batch 16
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


TRAINING_ROOT = next(
    (p for p in [Path("runs/detect/training_runs"), Path("training_runs")] if list(p.glob("*/weights/best.pt"))),
    Path("runs/detect/training_runs"),
)
DATA_YAML = Path("dataset/data.yaml")
PROJECT_NAME = "training_runs"
RUN_PREFIX = "water_meter_retrain"

# yolov8n is listed first: on this project's local machine (CPU-only, ~8GB RAM),
# every yolov8s/yolov8m "fresh" attempt has failed before finishing epoch 0,
# while yolov8n has trained successfully for 60+ epochs. Prefer what works locally;
# use colab_train_95pct.ipynb (GPU) for yolov8s-class training instead.
FRESH_MODEL_CANDIDATES = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def parse_args():
    p = argparse.ArgumentParser(description="Retrain YOLO for ≥95% mAP@0.5")
    p.add_argument("--mode",    choices=["fresh", "resume"], default="resume",
                   help="'fresh' = start from yolov8s.pt; 'resume' = continue from latest best.pt")
    p.add_argument("--epochs",  type=int,   default=200,  help="Training epochs (default 200)")
    p.add_argument("--imgsz",   type=int,   default=640,  help="Image size (use 832+ only if training on GPU)")
    p.add_argument("--batch",   type=int,   default=8,    help="Batch size")
    p.add_argument("--patience",type=int,   default=50,   help="Early-stop patience (epochs without improvement)")
    p.add_argument("--lr0",     type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--lrf",     type=float, default=0.005,help="Final LR = lr0 * lrf (cosine target)")
    return p.parse_args()


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))
    if not best_models:
        raise FileNotFoundError(f"No best.pt found in {training_root}")
    return max(best_models, key=lambda p: p.stat().st_mtime)


def find_fresh_model() -> str:
    for name in FRESH_MODEL_CANDIDATES:
        if Path(name).exists():
            return name
    return FRESH_MODEL_CANDIDATES[0]  # ultralytics will download it


def make_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    args = parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    if args.mode == "fresh":
        model_path = find_fresh_model()
        mode_label = f"FRESH — {model_path}"
    else:
        model_path = str(find_latest_best_model(TRAINING_ROOT))
        mode_label = f"RESUME — {model_path}"

    run_name = make_run_name(RUN_PREFIX)

    if not torch.cuda.is_available() and (Path(model_path).name in ("yolov8s.pt", "yolov8m.pt") or args.imgsz > 640):
        print("WARNING: No GPU detected on this machine (torch.cuda.is_available() is False).")
        print(f"         {Path(model_path).name} at imgsz={args.imgsz} on CPU has previously failed here")
        print("         before completing a single epoch (memory/time exhaustion).")
        print("         Consider colab_train_95pct.ipynb for GPU training instead.")
        print()

    print("========== YOLO OPTIMISED RETRAINING ==========")
    print(f"Mode:          {mode_label}")
    print(f"Data YAML:     {DATA_YAML}")
    print(f"Run name:      {run_name}")
    print(f"Epochs:        {args.epochs}  (patience={args.patience})")
    print(f"Image size:    {args.imgsz}")
    print(f"Batch size:    {args.batch}")
    print(f"LR:            {args.lr0} → {args.lr0 * args.lrf:.5f} (cosine)")
    print()

    model = YOLO(model_path)

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=PROJECT_NAME,
        name=run_name,

        # --- Learning rate (cosine schedule) ---
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # --- Regularisation ---
        weight_decay=0.0005,
        momentum=0.937,

        # --- Early stopping ---
        patience=args.patience,

        # --- Augmentation tuned for digit detection ---
        # IMPORTANT: Disable flips — digits are orientation-sensitive.
        # A horizontally flipped "6" looks like "9"; a flipped "2" looks like "5".
        fliplr=0.0,
        flipud=0.0,

        # Allow mild geometric / colour variation (meter is upright, but lighting varies)
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Mosaic + copy-paste help with small digit detections
        mosaic=1.0,
        close_mosaic=15,       # disable mosaic in last 15 epochs for cleaner fine-tuning
        copy_paste=0.3,
        mixup=0.0,             # mixup blends classes — bad for digit ID

        # workers=0 is required on Windows — multiprocessing spawn causes hangs with workers>0
        workers=0,
        verbose=True,
    )

    print("\nRetraining completed.")
    print(f"New run saved under: {PROJECT_NAME}/{run_name}")
    print("\nNext step: run  python 04_predict.py  to evaluate the new model.")


if __name__ == "__main__":
    main()
