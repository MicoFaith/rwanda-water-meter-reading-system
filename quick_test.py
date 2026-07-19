#!/usr/bin/env python3
"""
quick_test.py
Run inference on 3 test images and save annotated results.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH   = "runs/detect/training_runs/water_meter_test/weights/best.pt"
TEST_DIR     = Path("dataset/images/test")
OUTPUT_DIR   = Path("quick_test_output")
CONF         = 0.25
IMGSZ        = 320
NUM_IMAGES   = 3

CLASS_NAMES = [
    "meter", "window",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "unknown"
]

CLASS_COLORS = [
    (0, 255, 0),    # meter      — green
    (255, 0, 0),    # window     — blue
    (0, 215, 255),  # 0          — gold
    (255, 165, 0),  # 1          — orange
    (0, 255, 255),  # 2          — yellow
    (255, 0, 255),  # 3          — magenta
    (128, 255, 0),  # 4
    (0, 128, 255),  # 5
    (255, 128, 0),  # 6
    (128, 0, 255),  # 7
    (200, 200, 255),# 8
    (255, 200, 200),# 9
    (180, 180, 180),# unknown    — grey
]

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    images = sorted(TEST_DIR.glob("*.jpg"))[:NUM_IMAGES]
    if not images:
        print("No test images found.")
        return

    model = YOLO(MODEL_PATH)

    print(f"\nModel : {MODEL_PATH}")
    print(f"Images: {len(images)}")
    print(f"Output: {OUTPUT_DIR}/\n")
    print(f"{'Image':<35} {'Detections':>12}  Labels found")
    print("-" * 75)

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False,
        )[0]

        # Draw on a copy of the original image (full resolution)
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        detected_labels = []

        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale from model imgsz back to original resolution
                color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (255,255,255)
                label = f"{CLASS_NAMES[cls_id]} {conf:.0%}"
                detected_labels.append(f"{CLASS_NAMES[cls_id]}({conf:.0%})")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), frame)

        n = len(results.boxes) if results.boxes is not None else 0
        print(f"{img_path.name:<35} {n:>12}  {', '.join(detected_labels) if detected_labels else '(nothing detected)'}")

    print(f"\nSaved annotated images to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
