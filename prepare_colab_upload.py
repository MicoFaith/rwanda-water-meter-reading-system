#!/usr/bin/env python3
"""
prepare_colab_upload.py

Checks the dataset folder is ready for Google Drive upload and prints instructions.
No zipping required — just upload the folder directly.

Usage:
    python prepare_colab_upload.py
"""

from pathlib import Path

DATASET_DIR = Path("dataset")


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError("dataset/ folder not found — run from the project root.")

    splits = {}
    for split in ["train", "val", "test"]:
        imgs = list((DATASET_DIR / "images" / split).glob("*")) if (DATASET_DIR / "images" / split).exists() else []
        lbls = list((DATASET_DIR / "labels" / split).glob("*.txt")) if (DATASET_DIR / "labels" / split).exists() else []
        splits[split] = (len(imgs), len(lbls))

    print("Dataset summary:")
    for split, (n_img, n_lbl) in splits.items():
        print(f"  {split:5s}: {n_img} images, {n_lbl} labels")

    total_files = sum(n + l for n, l in splits.values())
    print(f"\nTotal files to upload: ~{total_files}")
    print(f"Dataset folder path:   {DATASET_DIR.resolve()}")

    print("""
How to upload to Google Drive (no zip needed):
  1. Open drive.google.com in your browser
  2. Click New → Folder upload
  3. Select the  dataset  folder shown above
  4. Wait for the upload to finish
  5. Open colab_train_95pct.ipynb in Google Colab
     (Runtime → Change runtime type → T4 GPU → Save)
  6. Run all cells top to bottom
""")


if __name__ == "__main__":
    main()
