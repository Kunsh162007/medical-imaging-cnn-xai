"""
02_training_and_optimization/02_backpropagation.py
=========================================
Manual forward + backward pass with analytical gradient verification.
Architecture: Input(4) → Hidden(8, ReLU) → Hidden(4, ReLU) → Output(4, Softmax)
Trained on synthetic Alzheimer feature data.

Run:
    python 02_training_and_optimization/02_backpropagation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, _save_or_show, ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED
from pathlib import Path

OUTPUT_DIR = Path("results/02_training_and_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


# ─────────────────────────────────────────────
# Activation + Loss Functions
# ─────────────────────────────────────────────
def relu(x):      return np.maximum(0, x)
def relu_d(x):    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(y_hat, y_true):
    m = y_hat.shape[0]
    return -np.sum(y_true * np.log(y_hat + 1e-9)) / m

def cross_entropy_d(y_hat, y_true):
    return (y_hat - y_true) / y_hat.shape[0]


# ─────────────────────────────────────────────
# Manual Backprop Neural Network
# ─────────────────────────────────────────────
class ManualBackpropNet:
    """
    3-layer MLP with analytical gradient computation.
    Layers: [4 → 8 → 4 → 4(softmax)]
    """

    def __init__(self, sizes=(4, 8, 4, 4), lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.params = {}
        self.grads  = {}
        self.cache  = {}

        # He initialization
        for i in range(1, len(sizes)):
            fan_in = sizes[i - 1]
            self.params[f"W{i}"] = rng.normal(0, np.sqrt(2/fan_in), (fan_in, sizes[i]))
            self.params[f"b{i}"] = np.zeros((1, sizes[i]))

    # ── Forward Pass ──────────────────────────
    def forward(self, X):
        self.cache["A0"] = X
        A = X
        depth = len([k for k in self.params if k.startswith("W")])

        for i in range(1, depth):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = relu(Z)
            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = A

        # Output layer (softmax)
        Z_out = A @ self.params[f"W{depth}"] + self.params[f"b{depth}"]
        A_out = softmax(Z_out)
        self.cache[f"Z{depth}"] = Z_out
        self.cache[f"A{depth}"] = A_out
        return A_out

    # ── Backward Pass ─────────────────────────
    def backward(self, y_true):
        depth = len([k for k in self.params if k.startswith("W")])
        m = y_true.shape[0]

        # Output layer gradient (softmax + CCE combined)
        dA = cross_entropy_d(self.cache[f"A{depth}"], y_true)

        for i in reversed(range(1, depth + 1)):
            A_prev = self.cache[f"A{i-1}"]
            if i == depth:
                dZ = dA
            else:
                dZ = dA * relu_d(self.cache[f"Z{i}"])

            self.grads[f"dW{i}"] = A_prev.T @ dZ / m
            self.grads[f"db{i}"] = dZ.mean(axis=0, keepdims=True)
            dA = dZ @ self.params[f"W{i}"].T

    # ── Parameter Update ──────────────────────
    def update(self):
        for key in self.params:
            grad_key = f"d{key}"
            if grad_key in self.grads:
                self.params[key] -= self.lr * self.grads[grad_key]

    def fit(self, X, y, epochs=500):
        losses = []
        for ep in range(epochs):
            y_hat = self.forward(X)
            loss  = cross_entropy(y_hat, y)
            self.backward(y)
            self.update()
            losses.append(loss)
            if (ep + 1) % 100 == 0:
                preds = y_hat.argmax(axis=1)
                acc   = (preds == y.argmax(axis=1)).mean()
                print(f"  Epoch {ep+1:>4} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%")
        return losses

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


# ─────────────────────────────────────────────
# Gradient Check (Numerical ≈ Analytical)
# ─────────────────────────────────────────────
def numerical_gradient(net, X, y, param_key, eps=1e-5):
    """Compute numerical gradient via finite differences."""
    W = net.params[param_key]
    grad_num = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] += eps
            loss_plus = cross_entropy(net.forward(X), y)
            W[i, j] -= 2 * eps
            loss_minus = cross_entropy(net.forward(X), y)
            W[i, j] += eps  # restore
            grad_num[i, j] = (loss_plus - loss_minus) / (2 * eps)

    return grad_num


def gradient_check(net, X, y):
    net.forward(X)
    net.backward(y)

    print("\n[Gradient Check]")
    for key in ["W1", "W2"]:
        grad_anal = net.grads[f"d{key}"]
        grad_num  = numerical_gradient(net, X, y, key)
        diff = np.abs(grad_anal - grad_num).max()
        rel  = diff / (np.abs(grad_anal).max() + 1e-8)
        status = "✓ PASS" if rel < 1e-4 else "✗ FAIL"
        print(f"  {key}: max |analytical - numerical| = {diff:.2e}  rel = {rel:.2e}  {status}")


# ─────────────────────────────────────────────
# Synthetic Data
# ─────────────────────────────────────────────
def make_synthetic_alzheimer(n=400, seed=42):
    """4-class synthetic tabular features (simulate clinical + imaging features)."""
    rng = np.random.default_rng(seed)
    means = [[0,0,0,0], [1,0.5,-0.5,0.3], [2,1,-1,0.7], [3,1.5,-1.5,1.2]]
    X, y_labels = [], []
    for cls, m in enumerate(means):
        x = rng.normal(m, 0.6, (n//4, 4))
        X.append(x); y_labels += [cls] * (n//4)
    X = np.vstack(X)
    y = np.eye(4)[y_labels]
    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_training_curve(losses, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    apply_dark_theme(fig, [ax])
    ax.plot(losses, color=ACCENT_CYAN, lw=2)
    ax.fill_between(range(len(losses)), losses, alpha=0.1, color=ACCENT_CYAN)
    ax.set_title("Manual Backprop — Training Loss", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    _save_or_show(fig, save_path)


def plot_gradient_flow(net, save_path):
    """Bar plot of mean absolute gradient per layer."""
    layers = [k for k in net.grads if k.startswith("dW")]
    means  = [np.abs(net.grads[k]).mean() for k in layers]
    labels = [k.replace("d", "") for k in layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_dark_theme(fig, [ax])

    colors = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED][:len(layers)]
    ax.bar(labels, means, color=colors, alpha=0.85, width=0.5)
    ax.set_title("Mean |Gradient| per Layer — Gradient Flow Check",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean |∂L/∂W|")
    for i, v in enumerate(means):
        ax.text(i, v + 0.0001, f"{v:.4f}", ha="center", color="white", fontsize=9)
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" Module 2 — Manual Backpropagation Implementation")
    print("=" * 55)

    X, y = make_synthetic_alzheimer(n=400)
    X_train, y_train = X[:320], y[:320]
    X_test,  y_test  = X[320:], y[320:]

    net = ManualBackpropNet(sizes=(4, 8, 4, 4), lr=0.05)

    # Gradient check before training
    gradient_check(net, X_train[:10], y_train[:10])

    print("\nTraining (500 epochs):")
    losses = net.fit(X_train, y_train, epochs=500)

    # Eval
    preds = net.predict(X_test)
    acc   = (preds == y_test.argmax(axis=1)).mean()
    print(f"\nTest Accuracy: {acc*100:.1f}%")

    plot_training_curve(losses, str(OUTPUT_DIR / "04_backprop_loss.png"))
    plot_gradient_flow(net,     str(OUTPUT_DIR / "05_gradient_flow.png"))
    print(f"\nPlots saved to: {OUTPUT_DIR}")
