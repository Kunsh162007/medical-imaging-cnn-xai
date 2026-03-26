"""
01_dl_fundamentals/03_mlp_activation_loss.py
=============================================
Visual survey of activation and loss functions used in deep learning.

Run:
    python 01_dl_fundamentals/03_mlp_activation_loss.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, _save_or_show, ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED, ACCENT_PURPLE, ACCENT_YELLOW
from pathlib import Path

OUTPUT_DIR = Path("results/01_dl_fundamentals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

x = np.linspace(-5, 5, 500)


# ─────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────
ACTIVATIONS = {
    "Sigmoid":     (lambda x: 1/(1+np.exp(-x)),            ACCENT_CYAN),
    "Tanh":        (np.tanh,                                ACCENT_ORANGE),
    "ReLU":        (lambda x: np.maximum(0, x),            ACCENT_GREEN),
    "Leaky ReLU":  (lambda x: np.where(x>0, x, 0.01*x),   ACCENT_PURPLE),
    "ELU":         (lambda x: np.where(x>0, x, np.exp(x)-1), ACCENT_RED),
    "Swish":       (lambda x: x/(1+np.exp(-x)),            ACCENT_YELLOW),
    "GELU":        (lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))), "#F472B6"),
    "Softplus":    (lambda x: np.log1p(np.exp(x)),         "#38BDF8"),
}


def plot_activation_functions():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    apply_dark_theme(fig, axes.flatten())

    for ax, (name, (fn, color)) in zip(axes.flatten(), ACTIVATIONS.items()):
        y = fn(x)
        ax.plot(x, y, color=color, lw=2.5)
        ax.axhline(0, color="#4B5563", lw=0.8, ls="--")
        ax.axvline(0, color="#4B5563", lw=0.8, ls="--")
        ax.set_title(name, color="white", fontsize=11, fontweight="bold")
        ax.set_xlim(-5, 5)
        y_min, y_max = y.min(), y.max()
        margin = max((y_max - y_min) * 0.15, 0.2)
        ax.set_ylim(y_min - margin, y_max + margin)

    fig.suptitle("Activation Functions — Comparison",
                 fontsize=15, color="white", fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "06_activations.png"))


def plot_activation_derivatives():
    """Show activation + its derivative side-by-side (gradient flow insight)."""
    selected = {
        "Sigmoid": (lambda x: 1/(1+np.exp(-x)), ACCENT_CYAN),
        "Tanh":    (np.tanh,                     ACCENT_ORANGE),
        "ReLU":    (lambda x: np.maximum(0, x),  ACCENT_GREEN),
        "Swish":   (lambda x: x/(1+np.exp(-x)),  ACCENT_YELLOW),
    }

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    apply_dark_theme(fig, axes.flatten())

    for col_idx, (name, (fn, color)) in enumerate(selected.items()):
        y  = fn(x)
        dy = np.gradient(y, x)

        axes[0, col_idx].plot(x, y,  color=color, lw=2)
        axes[0, col_idx].set_title(f"{name}", color="white", fontsize=10, fontweight="bold")
        axes[0, col_idx].axhline(0, color="#374151", lw=0.7)

        axes[1, col_idx].plot(x, dy, color=color, lw=2, ls="--", alpha=0.9)
        axes[1, col_idx].set_title(f"{name} — Derivative", color="white", fontsize=10)
        axes[1, col_idx].axhline(0, color="#374151", lw=0.7)

    axes[0, 0].set_ylabel("f(x)", color="white")
    axes[1, 0].set_ylabel("f'(x)", color="white")

    fig.suptitle("Activation Functions & Their Derivatives (Gradient Flow)",
                 fontsize=13, color="white", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "07_activation_derivatives.png"))


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────
def plot_loss_functions():
    y_true_bin = 1        # binary target
    p = np.linspace(0.01, 0.99, 500)

    bce_loss  = -(y_true_bin * np.log(p) + (1 - y_true_bin) * np.log(1 - p))
    mse_loss  = (p - y_true_bin) ** 2
    hinge     = np.maximum(0, 1 - y_true_bin * (2*p - 1))
    focal     = -(1-p)**2 * np.log(p + 1e-8)  # Focal loss (gamma=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    apply_dark_theme(fig, axes)

    # Binary losses
    axes[0].plot(p, bce_loss, color=ACCENT_CYAN,   lw=2.5, label="Binary Cross-Entropy")
    axes[0].plot(p, mse_loss, color=ACCENT_ORANGE,  lw=2.5, label="MSE")
    axes[0].plot(p, hinge,    color=ACCENT_GREEN,   lw=2.5, label="Hinge Loss")
    axes[0].plot(p, focal,    color=ACCENT_RED,     lw=2.5, label="Focal Loss (γ=2)")
    axes[0].set_title("Loss vs Predicted Probability (y=1)",
                      color="white", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Probability p"); axes[0].set_ylabel("Loss")
    axes[0].set_ylim(0, 4)
    axes[0].legend(frameon=False, labelcolor="white")

    # Categorical cross-entropy — softmax probabilities
    n_classes = 4
    # true label = class 2 (one-hot index)
    true_class = 2
    prob_range = np.linspace(0.01, 0.97, 500)
    # spread remaining prob uniformly
    cce_loss = np.array([
        -np.log(pc) for pc in prob_range
    ])
    axes[1].plot(prob_range, cce_loss, color=ACCENT_PURPLE, lw=2.5,
                 label="Categorical Cross-Entropy")
    axes[1].axvline(0.25, color="#4B5563", ls="--", lw=1, label="Random guess (4-class)")
    axes[1].set_title("CCE vs True-Class Probability\n(4-class Alzheimer's)",
                      color="white", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("P(true class)"); axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False, labelcolor="white")

    fig.suptitle("Loss Functions Survey", fontsize=14,
                 color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "08_loss_functions.png"))


# ─────────────────────────────────────────────
# Softmax Demo
# ─────────────────────────────────────────────
def plot_softmax_demo():
    """Visualise how softmax converts raw logits to probabilities."""
    class_names = ["NonDemented", "VeryMild", "Mild", "Moderate"]
    logits      = np.array([2.1, 0.3, 1.7, -0.5])
    softmax     = np.exp(logits) / np.exp(logits).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    apply_dark_theme(fig, [ax1, ax2])

    colors = [ACCENT_CYAN, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_ORANGE]
    ax1.bar(class_names, logits, color=colors, alpha=0.85)
    ax1.set_title("Raw Logits", color="white", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Logit value")
    ax1.set_xticklabels(class_names, rotation=20, ha="right")

    ax2.bar(class_names, softmax, color=colors, alpha=0.85)
    for i, v in enumerate(softmax):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", color="white", fontsize=10)
    ax2.set_title("Softmax Probabilities", color="white", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 0.8)
    ax2.set_xticklabels(class_names, rotation=20, ha="right")

    fig.suptitle("Softmax: Logits → Probabilities (4-class Alzheimer's)",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "09_softmax_demo.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Module 1 — Activation & Loss Functions")
    plot_activation_functions()
    plot_activation_derivatives()
    plot_loss_functions()
    plot_softmax_demo()
    print(f"\nPlots saved to: {OUTPUT_DIR}")
