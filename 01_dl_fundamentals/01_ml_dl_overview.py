"""
01_dl_fundamentals/01_ml_dl_overview.py
========================================
ML vs Deep Learning landscape — conceptual overview with visualizations.

Run:
    python 01_dl_fundamentals/01_ml_dl_overview.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, apply_light_theme, _save_or_show
from pathlib import Path

OUTPUT_DIR = Path("results/01_dl_fundamentals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. ML Taxonomy Diagram
# ─────────────────────────────────────────────
def plot_ml_taxonomy():
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_dark_theme(fig, [ax])
    ax.axis("off")

    taxonomy = {
        "Artificial Intelligence": (0.5, 0.92),
        "Machine Learning": (0.5, 0.75),
        "Supervised": (0.2, 0.57),
        "Unsupervised": (0.5, 0.57),
        "Reinforcement\nLearning": (0.8, 0.57),
        "Deep Learning": (0.2, 0.38),
        "CNN": (0.07, 0.20),
        "RNN/LSTM": (0.20, 0.20),
        "Transformer": (0.33, 0.20),
        "GAN": (0.07, 0.05),
        "ResNet/VGG": (0.20, 0.05),
        "BERT/GPT": (0.33, 0.05),
    }

    box_styles = {
        "Artificial Intelligence": dict(fc="#1E40AF", ec="#3B82F6", lw=2),
        "Machine Learning":         dict(fc="#7C3AED", ec="#A78BFA", lw=2),
        "Supervised":               dict(fc="#065F46", ec="#34D399", lw=1.5),
        "Unsupervised":             dict(fc="#065F46", ec="#34D399", lw=1.5),
        "Reinforcement\nLearning":  dict(fc="#065F46", ec="#34D399", lw=1.5),
        "Deep Learning":            dict(fc="#92400E", ec="#FB923C", lw=2),
        "CNN":                      dict(fc="#1E3A5F", ec="#58D9F9", lw=1),
        "RNN/LSTM":                 dict(fc="#1E3A5F", ec="#58D9F9", lw=1),
        "Transformer":              dict(fc="#1E3A5F", ec="#58D9F9", lw=1),
        "GAN":                      dict(fc="#2D1F5E", ec="#C084FC", lw=1),
        "ResNet/VGG":               dict(fc="#2D1F5E", ec="#C084FC", lw=1),
        "BERT/GPT":                 dict(fc="#2D1F5E", ec="#C084FC", lw=1),
    }

    for label, (x, y) in taxonomy.items():
        style = box_styles.get(label, dict(fc="#111827", ec="gray"))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", **style),
                transform=ax.transAxes)

    # Arrows
    edges = [
        ("Artificial Intelligence", "Machine Learning"),
        ("Machine Learning", "Supervised"),
        ("Machine Learning", "Unsupervised"),
        ("Machine Learning", "Reinforcement\nLearning"),
        ("Supervised", "Deep Learning"),
        ("Deep Learning", "CNN"),
        ("Deep Learning", "RNN/LSTM"),
        ("Deep Learning", "Transformer"),
    ]
    for src, dst in edges:
        x1, y1 = taxonomy[src]
        x2, y2 = taxonomy[dst]
        ax.annotate("", xy=(x2, y2 + 0.06), xytext=(x1, y1 - 0.06),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.2))

    ax.set_title("AI / ML / Deep Learning Taxonomy",
                 fontsize=14, fontweight="bold", color="white", pad=10)
    _save_or_show(fig, str(OUTPUT_DIR / "01_ml_taxonomy.png"))


# ─────────────────────────────────────────────
# 2. Feature Engineering vs Representation Learning
# ─────────────────────────────────────────────
def plot_feature_learning_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    apply_dark_theme(fig, axes)

    # Traditional ML pipeline (left)
    steps_ml = ["Raw Data", "Feature\nEngineering", "Feature\nSelection", "ML Model", "Output"]
    colors_ml = ["#1E40AF", "#065F46", "#065F46", "#7C3AED", "#92400E"]
    for i, (step, col) in enumerate(zip(steps_ml, colors_ml)):
        axes[0].barh(i, 1, left=0, color=col, height=0.6, alpha=0.85)
        axes[0].text(0.5, i, step, ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")
        if i < len(steps_ml) - 1:
            axes[0].annotate("", xy=(0.5, i - 0.35), xytext=(0.5, i - 0.65),
                             arrowprops=dict(arrowstyle="->", color="#6B7280"))
    axes[0].set_xlim(0, 1); axes[0].set_ylim(-1, len(steps_ml))
    axes[0].axis("off")
    axes[0].set_title("Traditional ML Pipeline\n(manual features)", color="white", fontsize=11)

    # DL pipeline (right)
    steps_dl = ["Raw Data", "CNN/DNN\n(Auto Features)", "Dense Layers", "Softmax Output"]
    colors_dl = ["#1E40AF", "#92400E", "#7C3AED", "#065F46"]
    for i, (step, col) in enumerate(zip(steps_dl, colors_dl)):
        axes[1].barh(i, 1, left=0, color=col, height=0.6, alpha=0.85)
        axes[1].text(0.5, i, step, ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")
        if i < len(steps_dl) - 1:
            axes[1].annotate("", xy=(0.5, i - 0.35), xytext=(0.5, i - 0.65),
                             arrowprops=dict(arrowstyle="->", color="#6B7280"))
    axes[1].set_xlim(0, 1); axes[1].set_ylim(-1, len(steps_dl))
    axes[1].axis("off")
    axes[1].set_title("Deep Learning Pipeline\n(learned representations)", color="white", fontsize=11)

    fig.suptitle("Traditional ML vs Deep Learning", fontsize=14,
                 color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "02_ml_vs_dl.png"))


# ─────────────────────────────────────────────
# 3. Why Deep Learning for Medical Imaging?
# ─────────────────────────────────────────────
def plot_dl_advantages_chart():
    categories = ["Image\nUnderstanding", "Pattern\nRecognition",
                  "Scalability", "Feature\nLearning", "Accuracy"]
    traditional_ml = [3, 4, 5, 2, 6]
    deep_learning   = [9, 9, 8, 10, 9.5]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_theme(fig, [ax])

    ax.bar(x - width/2, traditional_ml, width, label="Traditional ML",
           color="#3B82F6", alpha=0.85)
    ax.bar(x + width/2, deep_learning,  width, label="Deep Learning",
           color="#F97316", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(categories, color="white")
    ax.set_ylabel("Relative Performance Score (1–10)", color="white")
    ax.set_title("Traditional ML vs Deep Learning — Medical Imaging",
                 fontsize=13, color="white", fontweight="bold")
    ax.set_ylim(0, 11)
    ax.legend(frameon=False, labelcolor="white")

    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, f"{bar.get_height():.0f}",
                ha="center", va="bottom", color="white", fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "03_dl_advantages.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Module 1 — ML/DL Overview")
    print("Generating visualizations...")
    plot_ml_taxonomy()
    plot_feature_learning_comparison()
    plot_dl_advantages_chart()
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
