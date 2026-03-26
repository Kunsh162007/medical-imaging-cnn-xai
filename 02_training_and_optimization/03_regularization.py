"""
02_training_and_optimization/03_regularization.py
=========================================
Compare L1, L2, Dropout, BatchNorm, and EarlyStopping on the Alzheimer dataset.
Framework: TensorFlow/Keras (with optional PyTorch comparison).

Run:
    python 02_training_and_optimization/03_regularization.py [--framework tensorflow|pytorch]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from utils.visualization import apply_dark_theme, _save_or_show, ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED, ACCENT_PURPLE, ACCENT_YELLOW
from pathlib import Path

OUTPUT_DIR = Path("results/02_training_and_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# TensorFlow Experiments
# ─────────────────────────────────────────────
def run_tf_experiments(data_dir="data/train", epochs=30, batch_size=32):
    import tensorflow as tf
    from utils.data_loader import get_dataloader, split_train_val
    from utils.metrics import evaluate_model, print_report

    train_ds, val_ds = split_train_val(data_dir, val_ratio=0.2,
                                       framework="tensorflow", batch_size=batch_size)
    IMG = (128, 128, 3)

    def base_cnn(name="base"):
        inp = tf.keras.Input(shape=IMG)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        return inp, x

    configs = {
        "No Reg": lambda: _build_tf(base_cnn, None, None, False, False),
        "L2 Reg": lambda: _build_tf(base_cnn, "l2", None, False, False),
        "L1 Reg": lambda: _build_tf(base_cnn, "l1", None, False, False),
        "Dropout": lambda: _build_tf(base_cnn, None, 0.4, False, False),
        "BatchNorm": lambda: _build_tf(base_cnn, None, None, True, False),
        "All Combined": lambda: _build_tf(base_cnn, "l2", 0.4, True, False),
    }

    histories = {}
    for name, build_fn in configs.items():
        print(f"  Training: {name}")
        model = build_fn()
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                      metrics=["accuracy"])
        cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                      callbacks=cb, verbose=0)
        histories[name] = h.history
        print(f"    Val Acc: {max(h.history['val_accuracy'])*100:.1f}%")

    _plot_regularization_comparison(histories)


def _build_tf(base_fn, regularizer, dropout_rate, batchnorm, l1):
    import tensorflow as tf
    reg = None
    if regularizer == "l2": reg = tf.keras.regularizers.l2(1e-4)
    if regularizer == "l1": reg = tf.keras.regularizers.l1(1e-4)

    inp, x = base_fn()
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out)


# ─────────────────────────────────────────────
# PyTorch Experiment
# ─────────────────────────────────────────────
def run_pytorch_experiments(data_dir="data/train", epochs=30, batch_size=32):
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from utils.data_loader import get_dataloader, split_train_val
    from utils.model_utils import get_device

    device = get_device()
    train_loader, val_loader = split_train_val(data_dir, 0.2, "pytorch", batch_size)

    class RegCNN(nn.Module):
        def __init__(self, dropout=0.0, batchnorm=False, weight_decay_type=None):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            layers = [nn.Flatten(), nn.Linear(64*32*32, 128)]
            if batchnorm: layers.append(nn.BatchNorm1d(128))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(128, 4))
            self.head = nn.Sequential(*layers)

        def forward(self, x): return self.head(self.conv(x))

    configs = {
        "No Reg":    dict(dropout=0.0, batchnorm=False),
        "Dropout":   dict(dropout=0.4, batchnorm=False),
        "BatchNorm": dict(dropout=0.0, batchnorm=True),
        "Combined":  dict(dropout=0.4, batchnorm=True),
    }
    histories = {}

    for name, cfg in configs.items():
        print(f"  [PyTorch] Training: {name}")
        model = RegCNN(**cfg).to(device)
        opt   = Adam(model.parameters(), lr=1e-3,
                     weight_decay=1e-4 if "l2" in name.lower() else 0)
        criterion = nn.CrossEntropyLoss()
        h = {"accuracy": [], "val_accuracy": []}

        for ep in range(epochs):
            model.train()
            correct = 0; total = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward(); opt.step()
                correct += (out.argmax(1) == y).sum().item()
                total   += y.size(0)
            h["accuracy"].append(correct / total)

            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    correct += (out.argmax(1) == y).sum().item()
                    total   += y.size(0)
            h["val_accuracy"].append(correct / total)

        histories[name + " (PT)"] = h
        print(f"    Val Acc: {max(h['val_accuracy'])*100:.1f}%")

    _plot_regularization_comparison(histories, suffix="_pytorch")


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
def _plot_regularization_comparison(histories, suffix=""):
    colors = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED, ACCENT_PURPLE, ACCENT_YELLOW]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    apply_dark_theme(fig, [ax1, ax2])

    for (name, h), color in zip(histories.items(), colors):
        val_acc = h.get("val_accuracy", h.get("val_acc", []))
        tr_acc  = h.get("accuracy",     h.get("acc", []))
        ax1.plot(tr_acc,  color=color, lw=2,      label=name)
        ax2.plot(val_acc, color=color, lw=2, ls="--", label=name)

    ax1.set_title("Train Accuracy",     color="white", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Accuracy", color="white", fontsize=11, fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.legend(frameon=False, labelcolor="white", fontsize=8)

    fig.suptitle("Regularization Comparison", fontsize=13,
                 color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"06_regularization{suffix}.png"))


def plot_dropout_demo():
    """Visualise how dropout zeros random neurons."""
    np.random.seed(42)
    neurons = 16
    rate    = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    apply_dark_theme(fig, axes)

    activations = np.abs(np.random.randn(1, neurons))
    mask = np.random.binomial(1, 1 - rate, neurons)
    dropped = activations * mask / (1 - rate)

    for ax, acts, title in [
        (axes[0], activations[0], "Without Dropout"),
        (axes[1], dropped[0],     f"With Dropout (p={rate})"),
    ]:
        colors = [ACCENT_GREEN if a > 0 else ACCENT_RED for a in acts]
        ax.bar(range(neurons), acts, color=colors, alpha=0.85)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Neuron Index"); ax.set_ylabel("Activation")

    fig.suptitle("Dropout Regularization — Neuron Masking",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "07_dropout_demo.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="tensorflow",
                        choices=["tensorflow", "pytorch"])
    parser.add_argument("--data_dir", default="data/train")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print("Module 2 — Regularization Techniques")
    plot_dropout_demo()

    if args.framework == "tensorflow":
        run_tf_experiments(args.data_dir, args.epochs)
    else:
        run_pytorch_experiments(args.data_dir, args.epochs)

    print(f"\nPlots saved to: {OUTPUT_DIR}")
