"""
02_training_and_optimization/04_ann_handson.py
=====================================
Full ANN pipeline on Alzheimer dataset (tabular features extracted from MRI).
Demonstrates all Module 2 concepts together: backprop, Adam, BatchNorm, Dropout,
EarlyStopping, class weights, and learning rate scheduling.

Framework: TensorFlow/Keras (primary) + PyTorch comparison.

Run:
    python 02_training_and_optimization/04_ann_handson.py --framework tensorflow
    python 02_training_and_optimization/04_ann_handson.py --framework pytorch
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, ACCENT_CYAN, ACCENT_ORANGE
from utils.metrics import evaluate_model, print_report
from utils.model_utils import set_seed

OUTPUT_DIR = Path("results/02_training_and_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)


# ─────────────────────────────────────────────
# Synthetic tabular features from MRI
# ─────────────────────────────────────────────
def make_clinical_features(n=800, seed=42):
    """
    Simulate MRI-derived clinical features:
    [hippocampal_vol, cortical_thickness, WMH_volume, MMSE_score,
     entorhinal_vol, ventricle_ratio, age_norm, CDR_score]
    """
    rng = np.random.default_rng(seed)
    class_means = {
        0: [1.0,  2.5, 0.1, 0.95, 1.0,  0.2, 0.5, 0.0],  # NonDemented
        1: [0.8,  2.1, 0.3, 0.80, 0.85, 0.35, 0.6, 0.3],  # VeryMild
        2: [0.6,  1.8, 0.6, 0.65, 0.65, 0.50, 0.65, 0.7], # Mild
        3: [0.4,  1.4, 1.0, 0.45, 0.45, 0.70, 0.75, 1.0], # Moderate
    }
    X, y = [], []
    per_cls = n // 4
    for cls, means in class_means.items():
        feats = rng.normal(means, 0.15, (per_cls, 8)).clip(0, 1)
        X.append(feats)
        y.extend([cls] * per_cls)
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


FEATURE_NAMES = ["Hippocampal Vol", "Cortical Thickness", "WMH Volume", "MMSE",
                  "Entorhinal Vol", "Ventricle Ratio", "Age (norm)", "CDR Score"]


# ─────────────────────────────────────────────
# TensorFlow ANN
# ─────────────────────────────────────────────
def train_tf_ann(X_train, y_train, X_val, y_val, X_test, y_test, epochs=200):
    import tensorflow as tf

    print("\n[TF] Building ANN...")
    inp = tf.keras.Input(shape=(8,))
    x   = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Activation("relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Activation("relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, verbose=1),
    ]
    class_weight = {0: 1.0, 1: 1.2, 2: 2.0, 3: 5.0}

    h = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=64, callbacks=cbs,
                  class_weight=class_weight, verbose=0)

    y_proba = model.predict(X_test, verbose=0)
    y_pred  = y_proba.argmax(axis=1)
    results = evaluate_model(y_test, y_pred, y_proba)
    print_report(y_test, y_pred, "ANN-TF")
    return h.history, results


# ─────────────────────────────────────────────
# PyTorch ANN
# ─────────────────────────────────────────────
def train_pytorch_ann(X_train, y_train, X_val, y_val, X_test, y_test, epochs=200):
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from utils.model_utils import get_device

    device = get_device()

    class ClinicalANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 4),
            )
        def forward(self, x): return self.net(x)

    model     = ClinicalANN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2, 2.0, 5.0]).to(device))
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=8)

    def to_t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype).to(device)

    Xtr, ytr = to_t(X_train), to_t(y_train, torch.long)
    Xva, yva = to_t(X_val),   to_t(y_val,   torch.long)
    Xte, yte = to_t(X_test),  to_t(y_test,  torch.long)

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val, best_state = 0.0, None
    patience = 20; patience_ctr = 0

    for ep in range(1, epochs + 1):
        model.train()
        out  = model(Xtr)
        loss = criterion(out, ytr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tr_acc = (out.argmax(1) == ytr).float().mean().item()

        model.eval()
        with torch.no_grad():
            val_out  = model(Xva)
            val_loss = criterion(val_out, yva).item()
            val_acc  = (val_out.argmax(1) == yva).float().mean().item()
        scheduler.step(val_loss)

        history["accuracy"].append(tr_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(loss.item())
        history["val_loss"].append(val_loss)

        if val_acc > best_val:
            best_val  = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            print(f"  [EarlyStopping] at epoch {ep}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_proba = torch.softmax(model(Xte), 1).cpu().numpy()
    y_pred  = y_proba.argmax(1)
    results = evaluate_model(y_test, y_pred, y_proba)
    print_report(y_test, y_pred, "ANN-PyTorch")
    return history, results


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────
def plot_feature_importance(X, y, save_path):
    """Simple feature correlation with class label as proxy for importance."""
    from scipy.stats import f_oneway
    f_stats = []
    for fi in range(X.shape[1]):
        groups = [X[y == c, fi] for c in range(4)]
        f, _ = f_oneway(*groups)
        f_stats.append(f)

    fig, ax = plt.subplots(figsize=(9, 4))
    apply_dark_theme(fig, [ax])
    colors = ["#58D9F9" if f > np.median(f_stats) else "#6B7280" for f in f_stats]
    ax.barh(FEATURE_NAMES, f_stats, color=colors, alpha=0.85)
    ax.set_title("Feature Importance (ANOVA F-statistic)",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("F-statistic (higher = more discriminative)")
    for i, v in enumerate(f_stats):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", color="white", fontsize=9)
    _save_or_show(fig, save_path)


def compare_frameworks(tf_hist, pt_hist, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    apply_dark_theme(fig, axes)

    for hist, color, label in [
        (tf_hist, ACCENT_CYAN,   "TF"),
        (pt_hist, ACCENT_ORANGE, "PyTorch"),
    ]:
        acc  = hist.get("accuracy",     hist.get("acc", []))
        vacc = hist.get("val_accuracy", hist.get("val_acc", []))
        axes[0].plot(acc,  color=color, lw=2, label=f"Train {label}")
        axes[0].plot(vacc, color=color, lw=2, ls="--", alpha=0.7, label=f"Val {label}")
        axes[1].plot(hist.get("loss", []),     color=color, lw=2, label=f"Train {label}")
        axes[1].plot(hist.get("val_loss", []), color=color, lw=2, ls="--", alpha=0.7,
                     label=f"Val {label}")

    axes[0].set_title("Accuracy: TF vs PyTorch", color="white", fontsize=11, fontweight="bold")
    axes[1].set_title("Loss: TF vs PyTorch",     color="white", fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, labelcolor="white", fontsize=8)
    fig.suptitle("ANN — Framework Comparison", fontsize=13,
                 color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="both",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    print("Module 2 — ANN Hands-On (Clinical Features)")
    X, y = make_clinical_features(n=800)
    n = len(X)
    X_train, y_train = X[:int(n*0.7)], y[:int(n*0.7)]
    X_val,   y_val   = X[int(n*0.7):int(n*0.85)], y[int(n*0.7):int(n*0.85)]
    X_test,  y_test  = X[int(n*0.85):], y[int(n*0.85):]

    plot_feature_importance(X, y, str(OUTPUT_DIR / "08_feature_importance.png"))

    tf_hist = pt_hist = None

    if args.framework in ("tensorflow", "both"):
        tf_hist, tf_results = train_tf_ann(X_train, y_train, X_val, y_val,
                                            X_test, y_test, args.epochs)
        print(f"\n[TF] Test Acc: {tf_results['accuracy']*100:.2f}%")

    if args.framework in ("pytorch", "both"):
        pt_hist, pt_results = train_pytorch_ann(X_train, y_train, X_val, y_val,
                                                 X_test, y_test, args.epochs)
        print(f"\n[PT] Test Acc: {pt_results['accuracy']*100:.2f}%")

    if tf_hist and pt_hist:
        compare_frameworks(tf_hist, pt_hist,
                           str(OUTPUT_DIR / "09_ann_framework_comparison.png"))

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
