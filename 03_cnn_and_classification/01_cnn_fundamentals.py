"""
03_cnn_and_classification/01_cnn_fundamentals.py
=========================================
Visualises core CNN operations: Conv2D, MaxPool, feature maps, receptive fields.
Framework: NumPy + matplotlib (framework-agnostic conceptual demo).

Run:
    python 03_cnn_and_classification/01_cnn_fundamentals.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, _save_or_show, ACCENT_CYAN
from pathlib import Path

OUTPUT_DIR = Path("results/03_cnn_and_classification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Demo image (simple 32×32 gradient + circle)
# ─────────────────────────────────────────────
def make_demo_image(size=64):
    img = np.zeros((size, size), dtype=np.float32)
    # Gradient background
    for i in range(size):
        img[i, :] = i / size
    # Add circle (simulate brain MRI feature)
    cx, cy, r = size // 2, size // 2, size // 5
    for y in range(size):
        for x in range(size):
            if (x - cx)**2 + (y - cy)**2 < r**2:
                img[y, x] = 0.9
    return img


# ─────────────────────────────────────────────
# Standard Filters
# ─────────────────────────────────────────────
FILTERS = {
    "Horizontal Edge": np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32),
    "Vertical Edge":   np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32),
    "Laplacian":       np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32),
    "Gaussian Blur":   np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16,
    "Sharpen":         np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
    "Emboss":          np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),
}


def plot_convolution_demo():
    img = make_demo_image()

    n_filters = len(FILTERS)
    fig, axes = plt.subplots(2, n_filters + 1, figsize=((n_filters+1)*2.8, 5.5))
    apply_dark_theme(fig, axes.flatten())

    # Original
    for row in range(2):
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title("Input Image" if row == 0 else "Activation (ReLU)",
                                color="white", fontsize=8)
        axes[row, 0].axis("off")

    for col, (name, kernel) in enumerate(FILTERS.items(), start=1):
        feat = convolve(img, kernel, mode="constant")
        relu = np.maximum(feat, 0)

        axes[0, col].imshow(feat, cmap="inferno")
        axes[0, col].set_title(name, color="white", fontsize=7.5)
        axes[0, col].axis("off")

        axes[1, col].imshow(relu, cmap="inferno")
        axes[1, col].set_title(f"ReLU({name})", color="#4ADE80", fontsize=7.5)
        axes[1, col].axis("off")

    fig.suptitle("CNN Filters: Feature Map & ReLU Activation",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "01_conv_filters.png"))


def plot_pooling_comparison():
    img = make_demo_image(64)

    def max_pool(x, size=2, stride=2):
        H, W = x.shape
        out  = np.zeros((H//stride, W//stride), dtype=np.float32)
        for i in range(0, H-size+1, stride):
            for j in range(0, W-size+1, stride):
                out[i//stride, j//stride] = x[i:i+size, j:j+size].max()
        return out

    def avg_pool(x, size=2, stride=2):
        H, W = x.shape
        out  = np.zeros((H//stride, W//stride), dtype=np.float32)
        for i in range(0, H-size+1, stride):
            for j in range(0, W-size+1, stride):
                out[i//stride, j//stride] = x[i:i+size, j:j+size].mean()
        return out

    mp = max_pool(img)
    ap = avg_pool(img)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    apply_dark_theme(fig, axes)

    axes[0].imshow(img,  cmap="gray")
    axes[0].set_title(f"Input ({img.shape[0]}×{img.shape[1]})",  color="white", fontsize=10)
    axes[1].imshow(mp,   cmap="gray")
    axes[1].set_title(f"Max Pool ({mp.shape[0]}×{mp.shape[1]})", color=ACCENT_CYAN, fontsize=10)
    axes[2].imshow(ap,   cmap="gray")
    axes[2].set_title(f"Avg Pool ({ap.shape[0]}×{ap.shape[1]})", color="#FB923C", fontsize=10)
    for ax in axes: ax.axis("off")

    fig.suptitle("MaxPooling vs AvgPooling (2×2, stride=2)",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "02_pooling.png"))


def plot_receptive_field():
    """Show how receptive field grows with depth in a CNN."""
    depths   = [1, 2, 3, 4, 5]
    kernel   = 3
    rfs = [(kernel + (k-1)*(kernel-1)) for k in depths]  # no stride

    fig, ax = plt.subplots(figsize=(9, 4))
    apply_dark_theme(fig, [ax])
    ax.bar(depths, rfs, color=ACCENT_CYAN, alpha=0.85, width=0.6)
    ax.set_xlabel("CNN Depth (number of 3×3 conv layers)")
    ax.set_ylabel("Receptive Field Size (pixels)")
    ax.set_title("Effective Receptive Field vs CNN Depth",
                 color="white", fontsize=11, fontweight="bold")
    for i, v in zip(depths, rfs):
        ax.text(i, v + 0.3, f"{v}×{v}", ha="center", color="white", fontsize=10, fontweight="bold")
    _save_or_show(fig, str(OUTPUT_DIR / "03_receptive_field.png"))


if __name__ == "__main__":
    print("Module 3 — CNN Fundamentals")
    plot_convolution_demo()
    plot_pooling_comparison()
    plot_receptive_field()
    print(f"Plots saved to: {OUTPUT_DIR}")
