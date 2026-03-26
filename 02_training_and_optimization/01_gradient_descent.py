"""
02_training_and_optimization/01_gradient_descent.py
==========================================
Visualize and compare SGD, Mini-batch SGD, Momentum, RMSProp, Adam, AdaGrad
on a 2D loss surface (Rosenbrock banana function).

Run:
    python 02_training_and_optimization/01_gradient_descent.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, _save_or_show, MODEL_COLORS, ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED, ACCENT_PURPLE, ACCENT_YELLOW
from pathlib import Path

OUTPUT_DIR = Path("results/02_training_and_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Loss Surface: Rosenbrock (a=1, b=100)
# ─────────────────────────────────────────────
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b*(y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return np.array([dx, dy])


# ─────────────────────────────────────────────
# Optimizer Implementations (from scratch)
# ─────────────────────────────────────────────
def run_sgd(start, lr=0.001, steps=1500, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    path = [start.copy()]
    p = start.copy()
    for _ in range(steps):
        g = rosenbrock_grad(*p) + rng.normal(0, noise, 2)
        p -= lr * g
        path.append(p.copy())
    return np.array(path)


def run_momentum(start, lr=0.001, beta=0.9, steps=1500, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    path = [start.copy()]
    p = start.copy()
    v = np.zeros(2)
    for _ in range(steps):
        g = rosenbrock_grad(*p) + rng.normal(0, noise, 2)
        v = beta * v + (1 - beta) * g
        p -= lr * v
        path.append(p.copy())
    return np.array(path)


def run_rmsprop(start, lr=0.001, beta=0.99, eps=1e-8, steps=1500, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    path = [start.copy()]
    p = start.copy()
    s = np.zeros(2)
    for _ in range(steps):
        g = rosenbrock_grad(*p) + rng.normal(0, noise, 2)
        s = beta * s + (1 - beta) * g**2
        p -= lr * g / (np.sqrt(s) + eps)
        path.append(p.copy())
    return np.array(path)


def run_adam(start, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, steps=1500, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    path = [start.copy()]
    p = start.copy()
    m = np.zeros(2); v = np.zeros(2)
    for t in range(1, steps + 1):
        g = rosenbrock_grad(*p) + rng.normal(0, noise, 2)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(p.copy())
    return np.array(path)


def run_adagrad(start, lr=0.1, eps=1e-8, steps=1500, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    path = [start.copy()]
    p = start.copy()
    G = np.zeros(2)
    for _ in range(steps):
        g = rosenbrock_grad(*p) + rng.normal(0, noise, 2)
        G += g**2
        p -= lr * g / (np.sqrt(G) + eps)
        path.append(p.copy())
    return np.array(path)


# ─────────────────────────────────────────────
# Plot: All Optimizers on Loss Surface
# ─────────────────────────────────────────────
def plot_optimizer_paths():
    start = np.array([-1.5, 2.0])
    optimizers = {
        "SGD":       (run_sgd(start),       ACCENT_CYAN),
        "Momentum":  (run_momentum(start),  ACCENT_ORANGE),
        "RMSProp":   (run_rmsprop(start),   ACCENT_GREEN),
        "Adam":      (run_adam(start),      ACCENT_RED),
        "AdaGrad":   (run_adagrad(start),   ACCENT_PURPLE),
    }

    # Loss surface
    x_range = np.linspace(-2, 2, 400)
    y_range = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock(X, Y)
    Z_log = np.log1p(Z)

    fig, ax = plt.subplots(figsize=(11, 7))
    apply_dark_theme(fig, [ax])

    ax.contourf(X, Y, Z_log, levels=30, cmap="inferno", alpha=0.75)
    ax.contour(X, Y, Z_log, levels=15, colors="white", alpha=0.15, linewidths=0.5)
    ax.plot(1, 1, "w*", markersize=16, label="Global Min (1,1)", zorder=10)

    for name, (path, color) in optimizers.items():
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.8, alpha=0.9, label=name)
        ax.plot(path[0, 0], path[0, 1], "o", color=color, markersize=7, zorder=5)

    ax.set_title("Optimizer Comparison on Rosenbrock Surface",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(frameon=False, labelcolor="white", loc="upper right")
    ax.set_xlim(-2, 2); ax.set_ylim(-1, 3)

    _save_or_show(fig, str(OUTPUT_DIR / "01_optimizer_paths.png"))


# ─────────────────────────────────────────────
# Plot: Convergence Speed Comparison
# ─────────────────────────────────────────────
def plot_convergence_comparison():
    start = np.array([-1.5, 2.0])

    paths = {
        "SGD":      run_sgd(start),
        "Momentum": run_momentum(start),
        "RMSProp":  run_rmsprop(start),
        "Adam":     run_adam(start),
        "AdaGrad":  run_adagrad(start),
    }
    colors = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED, ACCENT_PURPLE]

    fig, ax = plt.subplots(figsize=(11, 5))
    apply_dark_theme(fig, [ax])

    for (name, path), color in zip(paths.items(), colors):
        losses = [rosenbrock(p[0], p[1]) for p in path]
        ax.plot(np.log10(np.array(losses) + 1), color=color, lw=2, label=name)

    ax.set_title("Optimizer Convergence (log₁₀ Rosenbrock Loss)",
                 fontsize=13, color="white", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("log₁₀(Loss + 1)")
    ax.legend(frameon=False, labelcolor="white")

    _save_or_show(fig, str(OUTPUT_DIR / "02_optimizer_convergence.png"))


# ─────────────────────────────────────────────
# Learning Rate Sensitivity
# ─────────────────────────────────────────────
def plot_lr_sensitivity():
    """Show effect of different learning rates on Adam convergence."""
    start = np.array([-1.5, 2.0])
    lrs   = [0.001, 0.01, 0.05, 0.1]
    col   = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED]

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_theme(fig, [ax])

    for lr, c in zip(lrs, col):
        path   = run_adam(start, lr=lr)
        losses = [rosenbrock(p[0], p[1]) for p in path]
        ax.plot(np.log10(np.array(losses) + 1), color=c, lw=2, label=f"lr={lr}")

    ax.set_title("Adam — Learning Rate Sensitivity",
                 fontsize=12, color="white", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("log₁₀(Loss + 1)")
    ax.legend(frameon=False, labelcolor="white")

    _save_or_show(fig, str(OUTPUT_DIR / "03_lr_sensitivity.png"))


if __name__ == "__main__":
    print("Module 2 — Gradient Descent Variants")
    plot_optimizer_paths()
    plot_convergence_comparison()
    plot_lr_sensitivity()
    print(f"Plots saved to: {OUTPUT_DIR}")
