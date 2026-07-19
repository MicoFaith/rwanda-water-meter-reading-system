# Rwanda Water Meter Reading System

An AI-based computer vision project for automatic water meter detection and reading using YOLO object detection.

This project includes:

- dataset collection
- image labeling
- dataset preparation
- YOLO training
- retraining
- prediction
- reading reconstruction
- a Flask REST API for serving predictions
- a SQLite database recording every reading
- a lightweight web UI for field agents (upload + history)

The system is trained to detect:

- water meter
- reading window
- digits `0–9`
- unclear digits (`unknown`)

---

# Features

The project provides tools for:

- collecting water meter datasets
- labeling water meter images
- rotating and cleaning images
- validating YOLO datasets
- preparing train/val/test datasets
- training YOLO models
- retraining from latest best model
- predicting on test images
- reconstructing final meter readings

---

# Project Structure

```text
rwanda-water-meter-reading-system/
├── raw_dataset/
│   ├── images/
│   └── labels/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── reports/
│   └── data.yaml
│
├── 01_label_images.py       # labeling GUI (PySide6)
├── 02_prepare_dataset.py    # build train/val/test splits + data.yaml
├── 03_train.py              # train a YOLO model
├── 04_predict.py            # inference + reading reconstruction (single source of truth)
├── 05_retrain.py            # continue training from latest best.pt
│
├── app.py                   # Flask API + web UI
├── database.py              # SQLite persistence for readings
├── templates/                # index.html, history.html
├── static/                   # style.css, script.js, history.js
│
├── runs/detect/training_runs/  # YOLO training output (best.pt lives here)
├── prediction_outputs/
├── logs/                     # app.log (created at runtime)
├── wmrs.db                   # SQLite database (created at runtime)
├── requirements.txt
└── README.md
```

---

# Supported Classes

| Class ID | Class Name | Description |
|---|---|---|
| 0 | meter | Full physical water meter |
| 1 | window | Reading/display window |
| 2 | 0 | Digit 0 |
| 3 | 1 | Digit 1 |
| 4 | 2 | Digit 2 |
| 5 | 3 | Digit 3 |
| 6 | 4 | Digit 4 |
| 7 | 5 | Digit 5 |
| 8 | 6 | Digit 6 |
| 9 | 7 | Digit 7 |
| 10 | 8 | Digit 8 |
| 11 | 9 | Digit 9 |
| 12 | unknown | Unclear/unreadable digit |

---

# Installation

## Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

---

# Dataset Structure

Expected raw dataset structure:

```text
raw_dataset/
├── images/
└── labels/
```

Example:

```text
raw_dataset/
├── images/
│   ├── wm_0001.jpg
│   ├── wm_0002.jpg
│   └── ...
│
└── labels/
    ├── wm_0001.txt
    ├── wm_0002.txt
    └── ...
```

---

# YOLO Label Format

Each line inside a `.txt` file follows:

```text
class_id x_center y_center width height
```

Example:

```text
2 0.309783 0.450836 0.028986 0.030898
```

Meaning:

```text
class 2 → digit 0
```

Coordinates are normalized between `0` and `1`.

---

# Workflow

```text
Data Collection
 ↓
Data Labeling
 ↓
Dataset Preparation
 ↓
YOLO Training
 ↓
Retraining
 ↓
Prediction
 ↓
Reading Reconstruction
```

---

# How to Label Images

Run:

```bash
python 01_label_images.py
```

Features:

- draw bounding boxes
- resize boxes
- move boxes
- rotate images
- remove bad images
- save YOLO labels

---

# Prepare YOLO Dataset

Run:

```bash
python 02_prepare_dataset.py raw_dataset dataset
```

The script automatically:

- validates labels
- removes invalid pairs
- creates train/val/test splits
- generates `data.yaml`

---

# Train YOLO

Run:

```bash
python 03_train.py
```

Training results are saved in:

```text
training_runs/
```

---

# Retrain from Latest Best Model

Run:

```bash
python 05_retrain.py
```

The script automatically finds the latest:

```text
best.pt
```

and continues training from it.

---

# Predict on Test Images

Run:

```bash
python 04_predict.py
```

The script automatically:

- finds latest `best.pt`
- predicts on test images
- saves annotated predictions
- saves prediction `.txt` files

---

# Prediction Logic

```text
Image
 ↓
YOLO prediction
 ↓
Find reading window
 ↓
Ignore digits outside the reading window
 ↓
Keep digits inside the reading window only
 ↓
Sort digits left to right
 ↓
Create final meter reading
 ↓
If unknown digit exists:
Ask user to retake image
```

---

# Web Application

A minimal Flask app exposes the prediction pipeline to field agents through a browser and a REST API. It reuses `04_predict.py` directly — no inference logic is duplicated.

## Run it

```bash
python app.py
```

Then open `http://localhost:5000` in a browser.

## Pages

| Route | Description |
|---|---|
| `GET /` | Upload a meter photo and get a reading |
| `GET /history` | Browse the most recent readings |

## API

| Route | Method | Description |
|---|---|---|
| `/predict` | POST | `multipart/form-data` field `image` → runs inference, saves the result, returns JSON |
| `/api/readings` | GET | `?limit=` (default 20, max 100) → JSON array of recent readings |
| `/health` | GET | Liveness check, returns `{"status": "ok"}` |

Example:

```bash
curl -X POST http://localhost:5000/predict -F "image=@meter.jpg"
```

```json
{"status": "success", "reading": "002345", "confidence": 0.96}
```

Every successful or uncertain prediction is recorded in `wmrs.db` (SQLite, table `meter_readings`) via `database.py`. Application logs are written to `logs/app.log` (rotating, 1 MB x 3 backups).

---

# Labeling Guidelines

- Tight bounding boxes improve accuracy
- Poor labels reduce model quality
- Window labels must cover only the reading window
- More diverse images improve generalization
- Strong rotation may reduce accuracy
- Images close to horizontal usually perform better
- Remove blurry or unusable images
- If labeling a digit may confuse the model, it is better not to label it than to force an incorrect label
- If training metrics are poor, optimize the labeling before retraining

---

# Future Work

Planned improvements:

- segmentation-based reading
- webcam live inference
- mobile deployment
- MQTT/cloud integration
- real-time validation
- automatic reading reconstruction

---

# Author

Gabriel BAZIRAMWABO  
Rwanda Coding Academy | Benax Technologies  
May 2026
