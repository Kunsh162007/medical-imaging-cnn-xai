"""
03_cnn_and_classification/04_multiclass_classification.py
=================================================
4-class Alzheimer MRI CNN (shallow custom architecture).
Run: python 03_cnn_and_classification/04_multiclass_classification.py --framework both
"""

import argparse
import numpy as np
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.model_utils import set_seed, get_device
from utils.visualization import plot_training_history, plot_confusion_matrix
from utils.metrics import evaluate_model, print_report

OUTPUT_DIR = Path("results/03_cnn_and_classification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)


def train_tf_multiclass(data_dir="data", epochs=25, batch_size=32):
    import tensorflow as tf
    print("\n[TF] 4-Class CNN — Loading data...")

    CLASS_NAMES = ["MildDemented","ModerateDemented","NonDemented","VeryMildDemented"]

    def load_ds(split, augment=False):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            f"{data_dir}/{split}", image_size=(128,128),
            batch_size=batch_size, label_mode="categorical",
            class_names=CLASS_NAMES, seed=42)
        def preprocess(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            if augment:
                x = tf.image.random_flip_left_right(x)
                x = tf.image.random_brightness(x, 0.2)
                x = tf.image.random_contrast(x, 0.8, 1.2)
            return x, y
        return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    train_ds = load_ds("train", augment=True)
    test_ds  = load_ds("test",  augment=False)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(128,128,3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    cbs = [tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)]
    h = model.fit(train_ds, epochs=epochs, callbacks=cbs, verbose=1)

    y_true, y_pred, y_proba = [], [], []
    for Xb, yb in test_ds:
        probs = model.predict(Xb, verbose=0)
        y_proba.append(probs)
        y_pred.extend(probs.argmax(1))
        y_true.extend(yb.numpy().argmax(1))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    results = evaluate_model(y_true, y_pred, np.vstack(y_proba))
    print_report(y_true, y_pred, "4-Class CNN-TF")
    print(f"[TF 4-Class] Test Accuracy: {results['accuracy']*100:.2f}%  F1: {results['f1_macro']:.4f}")

    plot_training_history(h, "4-Class CNN-TF",
                          save_path=str(OUTPUT_DIR / "09_multiclass_tf_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "4-Class CNN-TF",
                          save_path=str(OUTPUT_DIR / "10_multiclass_tf_cm.png"))
    return results


def train_pytorch_multiclass(data_dir="data", epochs=25, batch_size=32):
    import torch, torch.nn as nn
    from torch.optim import Adam
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    device = get_device()
    print(f"\n[PyTorch] 4-Class CNN — Device: {device}")

    tfm = transforms.Compose([
        transforms.Resize((128,128)), transforms.Grayscale(3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_t = transforms.Compose([
        transforms.Resize((128,128)), transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_loader = DataLoader(datasets.ImageFolder(f"{data_dir}/train", transform=tfm),
                              batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(datasets.ImageFolder(f"{data_dir}/test",  transform=tfm_t),
                              batch_size=batch_size, shuffle=False, num_workers=2)

    class MultiCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256,512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512,4))
        def forward(self, x): return self.head(self.features(x))

    model = MultiCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-3)
    history = {"accuracy":[], "val_accuracy":[], "loss":[], "val_loss":[]}

    for ep in range(1, epochs+1):
        model.train(); correct=total=0; ep_loss=0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); out = model(X); loss = criterion(out, y)
            loss.backward(); opt.step()
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0); ep_loss += loss.item()
        history["accuracy"].append(correct/total)
        history["loss"].append(ep_loss/len(train_loader))
        if ep % 5 == 0:
            print(f"  Ep {ep}/{epochs} | Loss: {ep_loss/len(train_loader):.4f} | Acc: {correct/total*100:.1f}%")

    model.eval(); y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            probs = torch.softmax(model(X.to(device)),1).cpu().numpy()
            y_proba.append(probs); y_pred.extend(probs.argmax(1)); y_true.extend(y.numpy())

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    results = evaluate_model(y_true, y_pred, np.vstack(y_proba))
    print_report(y_true, y_pred, "4-Class CNN-PT")
    print(f"[PT 4-Class] Test Accuracy: {results['accuracy']*100:.2f}%  F1: {results['f1_macro']:.4f}")

    plot_training_history(history, "4-Class CNN-PT",
                          save_path=str(OUTPUT_DIR / "11_multiclass_pt_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "4-Class CNN-PT",
                          save_path=str(OUTPUT_DIR / "12_multiclass_pt_cm.png"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="both",
                        choices=["tensorflow","pytorch","both"])
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--epochs",    type=int, default=25)
    parser.add_argument("--batch_size",type=int, default=32)
    args = parser.parse_args()

    print("Module 3 — 4-Class Alzheimer MRI Classification")
    if args.framework in ("tensorflow","both"):
        train_tf_multiclass(args.data_dir, args.epochs, args.batch_size)
    if args.framework in ("pytorch","both"):
        train_pytorch_multiclass(args.data_dir, args.epochs, args.batch_size)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
