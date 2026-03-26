"""
03_cnn_and_classification/03_binary_classification.py  (FIXED)
"""

import argparse
import numpy as np
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.model_utils import set_seed, get_device
from utils.visualization import _save_or_show
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path("results/03_cnn_and_classification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)
BINARY_CLASSES = ["NonDemented", "Demented"]


def _plot_binary_cm(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#FAFAFA")
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                xticklabels=BINARY_CLASSES, yticklabels=BINARY_CLASSES,
                cmap="Blues", linewidths=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); _save_or_show(fig, save_path)


def _plot_history(history, title, save_path):
    from utils.visualization import apply_dark_theme, ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, DARK_BG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    apply_dark_theme(fig, [ax1, ax2])
    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["accuracy"], color=ACCENT_CYAN, lw=2, label="Train Acc")
    ax1.set_title(f"{title} — Accuracy", color="white", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(frameon=False, labelcolor="white")
    ax2.plot(epochs, history["loss"], color=ACCENT_ORANGE, lw=2, label="Train Loss")
    ax2.set_title(f"{title} — Loss", color="white", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(frameon=False, labelcolor="white")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def train_tf_binary(data_dir="data", epochs=20, batch_size=32):
    import tensorflow as tf
    print("\n[TF] Binary CNN — Loading data...")

    def load_binary(split, augment=False):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            f"{data_dir}/{split}", image_size=(128,128),
            batch_size=batch_size, label_mode="int", seed=42)
        non_dem_idx = ds.class_names.index("NonDemented")
        def remap(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            y = tf.cast(tf.not_equal(y, non_dem_idx), tf.int32)
            if augment:
                x = tf.image.random_flip_left_right(x)
                x = tf.image.random_brightness(x, 0.2)
            return x, y
        return ds.map(remap, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    train_ds = load_binary("train", augment=True)
    test_ds  = load_binary("test",  augment=False)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(128,128,3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)]
    h = model.fit(train_ds, epochs=epochs, callbacks=cbs, verbose=1)

    y_true, y_pred = [], []
    for Xb, yb in test_ds:
        probs = model.predict(Xb, verbose=0).flatten()
        y_pred.extend((probs >= 0.5).astype(int))
        y_true.extend(yb.numpy())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    print(f"\n[TF Binary] Test Accuracy: {acc*100:.2f}%")
    print(f"  NonDemented correct: {((y_true==0)&(y_pred==0)).sum()}/{(y_true==0).sum()}")
    print(f"  Demented correct:    {((y_true==1)&(y_pred==1)).sum()}/{(y_true==1).sum()}")

    hist = {"accuracy": h.history.get("accuracy", []),
            "loss":     h.history.get("loss", [])}
    _plot_history(hist, "Binary CNN-TF", str(OUTPUT_DIR / "05_binary_tf_history.png"))
    from sklearn.metrics import confusion_matrix
    _plot_binary_cm(confusion_matrix(y_true, y_pred), "Binary CNN-TF",
                    str(OUTPUT_DIR / "06_binary_tf_cm.png"))
    return acc


def train_pytorch_binary(data_dir="data", epochs=20, batch_size=32):
    import torch, torch.nn as nn
    from torch.optim import Adam
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    device = get_device()
    print(f"\n[PyTorch] Binary CNN — Device: {device}")

    tfm = transforms.Compose([
        transforms.Resize((128,128)), transforms.Grayscale(3),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_t = transforms.Compose([
        transforms.Resize((128,128)), transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def remap(ds):
        """Fix: update BOTH targets and samples so DataLoader sees correct labels."""
        idx = ds.class_to_idx.get("NonDemented", 0)
        ds.targets = [0 if t == idx else 1 for t in ds.targets]
        ds.samples = [(p, 0 if t == idx else 1) for p, t in ds.samples]
        return ds

    train_ds = remap(datasets.ImageFolder(f"{data_dir}/train", transform=tfm))
    test_ds  = remap(datasets.ImageFolder(f"{data_dir}/test",  transform=tfm_t))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    class BinaryCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256,1),
            )
        def forward(self, x):
            return self.head(self.features(x)).squeeze(1)

    model = BinaryCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=1e-3)
    history = {"accuracy":[], "loss":[]}

    for ep in range(1, epochs+1):
        model.train(); correct=total=0; ep_loss=0
        for X, y in train_loader:
            X, y = X.to(device), y.float().to(device)
            opt.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward(); opt.step()
            correct += ((out > 0).float() == y).sum().item()
            total   += y.size(0)
            ep_loss += loss.item()
        acc = correct / total
        history["accuracy"].append(acc)
        history["loss"].append(ep_loss / len(train_loader))
        if ep % 5 == 0:
            print(f"  Ep {ep}/{epochs} | Loss: {ep_loss/len(train_loader):.4f} | Acc: {acc*100:.1f}%")

    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            out = model(X.to(device))
            y_pred.extend((out > 0).cpu().numpy().astype(int))
            y_true.extend(y.numpy())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    print(f"\n[PyTorch Binary] Test Accuracy: {acc*100:.2f}%")
    print(f"  NonDemented correct: {((y_true==0)&(y_pred==0)).sum()}/{(y_true==0).sum()}")
    print(f"  Demented correct:    {((y_true==1)&(y_pred==1)).sum()}/{(y_true==1).sum()}")

    _plot_history(history, "Binary CNN-PT", str(OUTPUT_DIR / "07_binary_pt_history.png"))
    from sklearn.metrics import confusion_matrix
    _plot_binary_cm(confusion_matrix(y_true, y_pred), "Binary CNN-PT",
                    str(OUTPUT_DIR / "08_binary_pt_cm.png"))
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="both",
                        choices=["tensorflow","pytorch","both"])
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--batch_size",type=int, default=32)
    args = parser.parse_args()

    print("Module 3 — Binary Classification (Demented vs NonDemented)")
    tf_acc = pt_acc = None
    if args.framework in ("tensorflow","both"):
        tf_acc = train_tf_binary(args.data_dir, args.epochs, args.batch_size)
    if args.framework in ("pytorch","both"):
        pt_acc = train_pytorch_binary(args.data_dir, args.epochs, args.batch_size)

    print("\n" + "="*50)
    if tf_acc: print(f"  TensorFlow : {tf_acc*100:.2f}%")
    if pt_acc:  print(f"  PyTorch    : {pt_acc*100:.2f}%")
    print(f"Plots saved to: {OUTPUT_DIR}")
