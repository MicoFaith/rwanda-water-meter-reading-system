"""
Rwanda Water Meter Reading System — MVP demo.

Upload a photo of a water meter. YOLO detects the meter window and the
digits inside it, then the digits are ordered left-to-right and joined
into a reading. Reading-reconstruction logic mirrors 04_predict.py.
"""

from __future__ import annotations

import os

import gradio as gr
from ultralytics import YOLO

MODEL_PATH = "best.pt"
IMAGE_SIZE = 640
CONFIDENCE = 0.25

model = YOLO(MODEL_PATH)


def reconstruct_reading(result) -> tuple[str | None, float | None, str]:
    """Return (reading, avg_confidence, message) for one YOLO result."""
    if result.boxes is None or len(result.boxes) == 0:
        return None, None, "No detections found."

    names = result.names
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
        return None, None, "No window detected — cannot reconstruct reading."

    wx1, wy1, wx2, wy2 = window_box

    digits = []
    for box, cls, conf in zip(boxes, classes, confs):
        label = names.get(cls, "")
        if label not in digit_labels and cls != unknown_cls:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
            digits.append((cx, label, conf))

    if not digits:
        return None, None, "No digits found inside window."

    digits.sort(key=lambda d: d[0])
    reading = "".join(d[1] for d in digits)
    avg_conf = sum(d[2] for d in digits) / len(digits)

    if "u" in reading:
        return reading, avg_conf, "Unknown digit(s) detected — recapture recommended."
    return reading, avg_conf, "OK"


def predict(image):
    if image is None:
        return None, "—", "—", "Upload an image to begin."

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False,
    )
    result = results[0]

    # result.plot() draws boxes with OpenCV and returns BGR; Gradio expects RGB.
    annotated = result.plot()[..., ::-1]

    reading, avg_conf, message = reconstruct_reading(result)
    if reading is None:
        return annotated, "—", "—", message

    reading_display = reading if message == "OK" else f"{reading}  ⚠"
    confidence_display = f"{avg_conf:.1%}"
    return annotated, reading_display, confidence_display, message


with gr.Blocks(title="Rwanda Water Meter Reading System") as demo:
    gr.Markdown(
        "# Rwanda Water Meter Reading System — MVP\n"
        "Upload a photo of a water meter. The YOLO model detects the meter "
        "window and the digits inside it, then reconstructs the reading "
        "left-to-right. Class `u` marks a digit the model can't confidently read."
    )

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Water meter image")
            run_btn = gr.Button("Read meter", variant="primary")
            gr.Examples(
                examples=[
                    "examples/wm_clean_000013.jpg",
                    "examples/wm_clean_000045.jpg",
                    "examples/wm_clean_000102.jpg",
                ],
                inputs=image_in,
                label="Try a sample image",
            )
        with gr.Column():
            image_out = gr.Image(label="Detections")
            reading_out = gr.Textbox(label="Meter reading")
            confidence_out = gr.Textbox(label="Confidence (avg. over digit boxes)")
            status_out = gr.Textbox(label="Status")

    run_btn.click(
        fn=predict,
        inputs=image_in,
        outputs=[image_out, reading_out, confidence_out, status_out],
    )
    image_in.change(
        fn=predict,
        inputs=image_in,
        outputs=[image_out, reading_out, confidence_out, status_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
