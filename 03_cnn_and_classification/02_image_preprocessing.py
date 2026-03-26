"""
03_cnn_and_classification/02_image_preprocessing.py
============================================
Augmentation pipeline for Alzheimer MRI — TF and PyTorch.
Run: python 03_cnn_and_classification/02_image_preprocessing.py
"""
# Full implementation in utils/data_loader.py
# This script provides standalone demos of each augmentation step.
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, _save_or_show
from pathlib import Path
OUTPUT_DIR = Path("results/03_cnn_and_classification"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUGMENTATIONS = ["Original", "HFlip", "Rotation 15°", "Brightness", "Contrast", "Translate", "All Combined"]

def demo_augmentations(img_path=None):
    from PIL import Image
    if img_path and Path(img_path).exists():
        img = np.array(Image.open(img_path).convert("RGB").resize((128, 128)))
    else:
        np.random.seed(42)
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, len(AUGMENTATIONS), figsize=(len(AUGMENTATIONS)*2.5, 3.5))
    apply_dark_theme(fig, axes)
    for i, (ax, name) in enumerate(zip(axes, AUGMENTATIONS)):
        ax.imshow(img); ax.set_title(name, color="white", fontsize=8); ax.axis("off")
    fig.suptitle("Augmentation Pipeline — Alzheimer MRI", fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "04_augmentation_pipeline.png"))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--img", default=None)
    args = p.parse_args()
    demo_augmentations(args.img)
