"""
utils/visualization.py
========================
Mixed-theme plotting suite:
  - Dark theme for training curves, CAMs, feature maps (stunning on GitHub)
  - Light/white theme for report-style confusion matrices and comparison tables

Usage:
    from utils.visualization import (
        plot_training_history, plot_confusion_matrix,
        plot_model_comparison, plot_sample_grid, apply_dark_theme
    )
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# Color Palettes
# ─────────────────────────────────────────────
DARK_BG = "#0D1117"
DARK_SURFACE = "#161B22"
DARK_BORDER = "#30363D"
ACCENT_CYAN = "#58D9F9"
ACCENT_PURPLE = "#C084FC"
ACCENT_GREEN = "#4ADE80"
ACCENT_ORANGE = "#FB923C"
ACCENT_RED = "#F87171"
ACCENT_YELLOW = "#FDE68A"

MODEL_COLORS = {
    "AlexNet":    ACCENT_CYAN,
    "VGGNet":     ACCENT_PURPLE,
    "GoogLeNet":  ACCENT_GREEN,
    "ResNet50":   ACCENT_ORANGE,
    "Ensemble":   ACCENT_YELLOW,
    "Custom CNN": ACCENT_RED,
}

CLASS_COLORS = ["#60A5FA", "#A78BFA", "#34D399", "#F97316"]
CLASS_NAMES  = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


# ─────────────────────────────────────────────
# Theme Helpers
# ─────────────────────────────────────────────
def apply_dark_theme(fig, ax_list=None):
    """Apply dark GitHub-style theme to a figure."""
    fig.patch.set_facecolor(DARK_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(DARK_SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_BORDER)
        ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        if ax.get_title():
            ax.title.set_color("white")
        ax.grid(True, color=DARK_BORDER, linewidth=0.5, alpha=0.7)


def apply_light_theme(fig, ax_list=None):
    """Apply clean white report theme."""
    fig.patch.set_facecolor("white")
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor("#FAFAFA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#333333", labelsize=10)
        ax.grid(True, color="#E5E7EB", linewidth=0.5)


# ─────────────────────────────────────────────
# Training History — DARK
# ─────────────────────────────────────────────
def plot_training_history(history, model_name: str = "Model", save_path: str = None):
    """
    Plot accuracy and loss curves (dark theme).
    Accepts either a Keras History object or a dict with 'acc'/'loss' keys.
    """
    if hasattr(history, "history"):
        h = history.history
    else:
        h = history

    epochs = range(1, len(h.get("accuracy", h.get("acc", []))) + 1)
    train_acc = h.get("accuracy", h.get("acc", []))
    val_acc   = h.get("val_accuracy", h.get("val_acc", []))
    train_loss = h.get("loss", [])
    val_loss   = h.get("val_loss", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    apply_dark_theme(fig, [ax1, ax2])

    # Accuracy
    ax1.plot(epochs, train_acc, color=ACCENT_CYAN, lw=2, label="Train Acc")
    if val_acc: ax1.plot(epochs, val_acc,   color=ACCENT_ORANGE, lw=2, ls="--", label="Val Acc")
    ax1.fill_between(epochs, train_acc, alpha=0.08, color=ACCENT_CYAN)
    if val_acc: ax1.fill_between(epochs, val_acc,   alpha=0.08, color=ACCENT_ORANGE)
    ax1.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold", color="white", pad=12)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(frameon=False, labelcolor="white")

    # Loss
    ax2.plot(epochs, train_loss, color=ACCENT_PURPLE, lw=2, label="Train Loss")
    if val_loss: ax2.plot(epochs, val_loss,   color=ACCENT_RED,    lw=2, ls="--", label="Val Loss")
    ax2.fill_between(epochs, train_loss, alpha=0.08, color=ACCENT_PURPLE)
    if val_loss: ax2.fill_between(epochs, val_loss,   alpha=0.08, color=ACCENT_RED)
    ax2.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold", color="white", pad=12)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(frameon=False, labelcolor="white")

    fig.suptitle(f"Training Curves — {model_name}", fontsize=15, color="white",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Confusion Matrix — LIGHT
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Model",
                          normalize: bool = True, save_path: str = None):
    """Confusion matrix with white/light report theme."""
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title_suffix = "(Normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title_suffix = "(Counts)"

    fig, ax = plt.subplots(figsize=(8, 7))
    apply_light_theme(fig, [ax])

    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cmap="Blues", linewidths=0.5, linecolor="#E5E7EB",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"{model_name} — Confusion Matrix {title_suffix}",
                 fontsize=13, fontweight="bold", pad=14, color="#111827")
    ax.set_xlabel("Predicted Label", fontsize=11, color="#374151")
    ax.set_ylabel("True Label", fontsize=11, color="#374151")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Model Comparison Bar Chart — DARK
# ─────────────────────────────────────────────
def plot_model_comparison(results: dict, metric: str = "test_acc",
                          save_path: str = None):
    """
    results = {
        "AlexNet-TF": {"test_acc": 0.886, "f1": 0.883, "auc": 0.971},
        "ResNet50-TF": {"test_acc": 0.942, ...},
        ...
    }
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) * 100 for m in models]
    colors = [MODEL_COLORS.get(m.split("-")[0], ACCENT_CYAN) for m in models]

    fig, ax = plt.subplots(figsize=(13, 6))
    apply_dark_theme(fig, [ax])

    bars = ax.barh(models, values, color=colors, height=0.55, alpha=0.88)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color="white", fontsize=10, fontweight="bold")

    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (%)", color="white")
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}",
                 fontsize=14, fontweight="bold", color="white", pad=14)
    ax.set_xlim(0, max(values) + 6)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Sample Image Grid — DARK
# ─────────────────────────────────────────────
def plot_sample_grid(images: np.ndarray, labels: np.ndarray,
                     preds: np.ndarray = None, n_cols: int = 4,
                     save_path: str = None):
    """Show a grid of MRI samples with true labels (and optional predictions)."""
    n = min(len(images), 16)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.2))
    apply_dark_theme(fig, axes.flatten())

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            img = images[i]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            ax.imshow(img if img.ndim == 3 else img[:, :, 0], cmap="gray")
            true_cls = CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else str(labels[i])
            title = f"True: {true_cls}"
            color = "white"
            if preds is not None:
                pred_cls = CLASS_NAMES[preds[i]] if preds[i] < len(CLASS_NAMES) else str(preds[i])
                correct = preds[i] == labels[i]
                title += f"\nPred: {pred_cls}"
                color = ACCENT_GREEN if correct else ACCENT_RED
            ax.set_title(title, fontsize=7.5, color=color)
        ax.axis("off")

    fig.suptitle("Sample MRI Images", fontsize=14, color="white", fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Feature Map Visualization — DARK
# ─────────────────────────────────────────────
def plot_feature_maps(feature_maps: np.ndarray, layer_name: str = "conv_layer",
                      n_filters: int = 16, save_path: str = None):
    """Display CNN feature maps for a given layer output."""
    n = min(feature_maps.shape[-1] if feature_maps.ndim == 3 else feature_maps.shape[0], n_filters)
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    apply_dark_theme(fig, axes.flatten())

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            if feature_maps.ndim == 3:
                fmap = feature_maps[:, :, i]
            else:
                fmap = feature_maps[i]
            ax.imshow(fmap, cmap="inferno")
            ax.set_title(f"f{i}", fontsize=7, color="white")
        ax.axis("off")

    fig.suptitle(f"Feature Maps — {layer_name}", fontsize=13,
                 color="white", fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Multi-Model XAI Side-by-Side — DARK
# ─────────────────────────────────────────────
def plot_xai_comparison(original: np.ndarray, heatmaps: dict,
                        true_label: str = "", pred_label: str = "",
                        save_path: str = None):
    """
    heatmaps: {"AlexNet": np.array, "VGGNet": np.array, ...}
    """
    n_models = len(heatmaps)
    fig, axes = plt.subplots(1, n_models + 1, figsize=((n_models + 1) * 3.5, 4))
    apply_dark_theme(fig, axes)

    # Original
    axes[0].imshow(original, cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title(f"Original\nTrue: {true_label}\nPred: {pred_label}",
                      fontsize=8.5, color="white")
    axes[0].axis("off")

    # Heatmaps
    for idx, (model_name, hmap) in enumerate(heatmaps.items()):
        col = MODEL_COLORS.get(model_name, ACCENT_CYAN)
        axes[idx + 1].imshow(hmap, cmap="jet", alpha=0.85)
        axes[idx + 1].set_title(model_name, fontsize=9, color=col, fontweight="bold")
        axes[idx + 1].axis("off")

    fig.suptitle("XAI Heatmap Comparison Across Architectures",
                 fontsize=13, color="white", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Internal Helper
# ─────────────────────────────────────────────
def _save_or_show(fig, save_path: str):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
