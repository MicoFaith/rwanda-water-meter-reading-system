---
title: Rwanda Water Meter Reading System
emoji: 💧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Rwanda Water Meter Reading System — MVP

A YOLO-based demo that reads analog water meter displays from a photo.

## How it works

1. Upload (or pick a sample) photo of a water meter.
2. A YOLOv8 model detects the meter's digit window and the individual digits inside it.
3. Detected digits are ordered left-to-right and joined into a reading.
4. If any digit is detected as class `u` (unclear/illegible), the reading is
   flagged as uncertain instead of silently guessing.

## Model

`best.pt` — trained on a 500-image labeled dataset of Rwandan water meters,
13 classes (`meter`, `window`, digits `0`-`9`, `u`). Validation mAP50 ≈ 0.895
on the held-out val split.

## Try it

Use one of the example images below the upload box, or upload your own
water meter photo. Output shows the annotated detections, the reconstructed
reading, average confidence across digit boxes, and a status line.
