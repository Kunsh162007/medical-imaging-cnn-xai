"""
01_dl_fundamentals/02_perceptron.py
====================================
Single-layer and multi-layer perceptron built from scratch with NumPy.
Demonstrates on linearly separable data and the XOR problem.

Run:
    python 01_dl_fundamentals/02_perceptron.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, _save_or_show, ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED
from pathlib import Path

OUTPUT_DIR = Path("results/01_dl_fundamentals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Single-Layer Perceptron
# ─────────────────────────────────────────────
class Perceptron:
    """
    Single-layer perceptron using Heaviside step activation.
    Rosenblatt (1957) learning rule: w += lr * (y - ŷ) * x
    """
    def __init__(self, n_inputs: int, lr: float = 0.1, epochs: int = 50, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-0.5, 0.5, n_inputs)
        self.bias    = 0.0
        self.lr      = lr
        self.epochs  = epochs
        self.errors_per_epoch = []

    def _activate(self, x):
        return 1 if x >= 0 else 0

    def predict_single(self, x):
        return self._activate(np.dot(x, self.weights) + self.bias)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def fit(self, X, y):
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                yhat = self.predict_single(xi)
                delta = self.lr * (yi - yhat)
                self.weights += delta * xi
                self.bias    += delta
                errors += int(delta != 0)
            self.errors_per_epoch.append(errors)
        return self


# ─────────────────────────────────────────────
# Two-Layer MLP (solves XOR)
# ─────────────────────────────────────────────
class MLP:
    """
    Minimal 2-layer MLP: sigmoid activations, MSE loss, manual backprop.
    """
    def __init__(self, n_in, n_hidden, n_out, lr=0.5, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.5, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.lr  = lr
        self.losses = []

    @staticmethod
    def _sigmoid(x):    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    @staticmethod
    def _sigmoid_d(s):  return s * (1 - s)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m  = X.shape[0]
        d2 = (self.a2 - y) * self._sigmoid_d(self.a2)
        d1 = (d2 @ self.W2.T) * self._sigmoid_d(self.a1)
        self.W2 -= self.lr * (self.a1.T @ d2) / m
        self.b2 -= self.lr * d2.mean(axis=0)
        self.W1 -= self.lr * (X.T @ d1) / m
        self.b1 -= self.lr * d1.mean(axis=0)

    def fit(self, X, y, epochs=5000):
        for _ in range(epochs):
            out = self.forward(X)
            self.backward(X, y)
            loss = np.mean((out - y) ** 2)
            self.losses.append(loss)
        return self

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


# ─────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────
def demo_single_perceptron():
    """Linearly-separable AND gate."""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 0, 1])             # AND gate

    p = Perceptron(n_inputs=2, lr=0.1, epochs=30)
    p.fit(X, y)
    preds = p.predict(X)
    acc = (preds == y).mean()
    print(f"[Perceptron] AND gate accuracy: {acc*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    apply_dark_theme(fig, [ax1, ax2])

    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([p.predict_single(g) for g in grid]).reshape(xx.shape)
    ax1.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=120, edgecolors="white", zorder=3)
    for i, (xi, yi) in enumerate(zip(X, y)):
        ax1.text(xi[0]+0.05, xi[1]+0.05, f"({xi[0]:.0f},{xi[1]:.0f})→{yi}",
                 color="white", fontsize=9)
    ax1.set_title("AND Gate — Decision Boundary", color="white", fontsize=11)
    ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")

    # Error curve
    ax2.plot(p.errors_per_epoch, color=ACCENT_CYAN, lw=2)
    ax2.set_title("Misclassifications per Epoch", color="white", fontsize=11)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Errors")

    _save_or_show(fig, str(OUTPUT_DIR / "04_perceptron_and.png"))
    return p


def demo_xor_mlp():
    """XOR problem — impossible for single perceptron, solved by MLP."""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor = np.array([[0],[1],[1],[0]], dtype=float)
    y_and = np.array([0, 0, 0, 1])

    # Show single perceptron fails on XOR
    p = Perceptron(n_inputs=2, epochs=100)
    p.fit(X, y_and)  # note: XOR labels for confusion demo
    # train on XOR
    p_xor = Perceptron(n_inputs=2, epochs=100)
    p_xor.fit(X, np.array([0,1,1,0]))
    print(f"[Perceptron on XOR] best errors: {min(p_xor.errors_per_epoch)}/4 (never 0!)")

    # MLP solves it
    mlp = MLP(n_in=2, n_hidden=4, n_out=1, lr=0.5)
    mlp.fit(X, y_xor, epochs=10000)
    preds = mlp.predict(X).flatten()
    acc   = (preds == y_xor.flatten().astype(int)).mean()
    print(f"[MLP on XOR] accuracy: {acc*100:.1f}%  preds={preds}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    apply_dark_theme(fig, axes)

    # XOR data
    axes[0].scatter(X[:, 0], X[:, 1], c=y_xor.flatten(), cmap="bwr",
                    s=200, edgecolors="white", zorder=3)
    for i, (xi, yi) in enumerate(zip(X, y_xor.flatten())):
        axes[0].text(xi[0]+0.06, xi[1]+0.06, f"{yi:.0f}", color="white", fontsize=12)
    axes[0].set_title("XOR Problem\n(not linearly separable)", color="white")
    axes[0].set_xlim(-0.5, 1.5); axes[0].set_ylim(-0.5, 1.5)

    # MLP decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
    axes[1].scatter(X[:, 0], X[:, 1], c=y_xor.flatten(), cmap="bwr",
                    s=200, edgecolors="white", zorder=3)
    axes[1].set_title("MLP Decision Boundary\n(2 hidden neurons)", color="white")

    # Loss curve
    axes[2].plot(mlp.losses[::50], color=ACCENT_GREEN, lw=1.5)
    axes[2].set_title("MLP Training Loss (MSE)", color="white")
    axes[2].set_xlabel("Epoch (×50)"); axes[2].set_ylabel("Loss")

    fig.suptitle("XOR: Perceptron Fails → MLP Solves",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "05_xor_mlp.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 1 — Perceptron & MLP from Scratch")
    print("=" * 50)
    demo_single_perceptron()
    demo_xor_mlp()
    print(f"\nPlots saved to: {OUTPUT_DIR}")
