"""
04_advanced_architectures/01_alexnet.py
=================================
AlexNet for 4-class Alzheimer's MRI (TensorFlow + PyTorch).

Run:
    python 04_advanced_architectures/01_alexnet.py --framework tensorflow --epochs 30
    python 04_advanced_architectures/01_alexnet.py --framework pytorch    --epochs 30
"""

import argparse
import numpy as np
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.model_utils import set_seed, get_device, save_results_json, save_tf_model, save_torch_model
from utils.metrics import evaluate_model, print_report
from utils.visualization import plot_training_history, plot_confusion_matrix

OUTPUT_DIR = Path("results/04_advanced_architectures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)


def build_tf_alexnet(img_shape=(128, 128, 3)):
    import tensorflow as tf
    inp = tf.keras.Input(shape=img_shape)
    x   = tf.keras.layers.Conv2D(64, 11, strides=4, activation="relu", padding="same")(inp)
    x   = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
    x   = tf.keras.layers.Conv2D(192, 5, activation="relu", padding="same")(x)
    x   = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
    x   = tf.keras.layers.Conv2D(384, 3, activation="relu", padding="same")(x)
    x   = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x   = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(1024, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Dense(512, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inputs=inp, outputs=out)


def build_pt_alexnet():
    import torch.nn as nn
    from torchvision import models
    m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    m.classifier[-1] = nn.Linear(4096, 4)
    return m


def train_tf(data_dir="data", epochs=30, batch_size=32):
    import tensorflow as tf
    from utils.data_loader import get_dataloader

    train_ds = get_dataloader(f"{data_dir}/train", "tensorflow", batch_size, augment=True)
    val_ds   = get_dataloader(f"{data_dir}/train", "tensorflow", batch_size, augment=False,
                               validation_split=0.1, subset="validation")
    test_ds  = get_dataloader(f"{data_dir}/test",  "tensorflow", batch_size, augment=False)

    model = build_tf_alexnet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"[AlexNet-TF] Parameters: {model.count_params():,}")

    cbs = [tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)]
    h   = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=1)

    y_true, y_pred, y_proba = [], [], []
    for Xb, yb in test_ds:
        probs = model.predict(Xb, verbose=0)
        y_proba.append(probs); y_pred.extend(probs.argmax(1)); y_true.extend(yb.numpy().argmax(1))
    results = evaluate_model(np.array(y_true), np.array(y_pred), np.vstack(y_proba))
    print_report(np.array(y_true), np.array(y_pred), "AlexNet-TF")
    save_tf_model(model, str(OUTPUT_DIR / "alexnet_tf.h5"))
    save_results_json(results, str(OUTPUT_DIR / "alexnet_tf_results.json"))
    plot_training_history(h, "AlexNet-TF", save_path=str(OUTPUT_DIR / "alexnet_tf_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "AlexNet-TF",
                          save_path=str(OUTPUT_DIR / "alexnet_tf_cm.png"))
    return results


def train_pytorch(data_dir="data", epochs=30, batch_size=32):
    import torch, torch.nn as nn
    from torch.optim import Adam
    from utils.data_loader import get_dataloader

    device = get_device()
    train_loader = get_dataloader(f"{data_dir}/train", "pytorch", batch_size, augment=True)
    val_loader   = get_dataloader(f"{data_dir}/train", "pytorch", batch_size, augment=False)
    test_loader  = get_dataloader(f"{data_dir}/test",  "pytorch", batch_size, augment=False)

    model = build_pt_alexnet().to(device)
    opt   = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = tr_c = tr_n = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); out = model(X); loss = criterion(out, y)
            loss.backward(); opt.step()
            tr_loss += loss.item() * y.size(0); tr_c += (out.argmax(1)==y).sum().item(); tr_n += y.size(0)
        model.eval(); va_c = va_n = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                va_c += (model(X).argmax(1)==y).sum().item(); va_n += y.size(0)
        history["accuracy"].append(tr_c/tr_n); history["loss"].append(tr_loss/tr_n)
        history["val_accuracy"].append(va_c/va_n)
        if ep % 5 == 0: print(f"  Ep {ep}/{epochs} | Acc {tr_c/tr_n*100:.1f}% | Val {va_c/va_n*100:.1f}%")

    model.eval(); y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            probs = torch.softmax(model(X.to(device)), 1).cpu().numpy()
            y_proba.append(probs); y_pred.extend(probs.argmax(1)); y_true.extend(y.numpy())
    results = evaluate_model(np.array(y_true), np.array(y_pred), np.vstack(y_proba))
    print_report(np.array(y_true), np.array(y_pred), "AlexNet-PT")
    save_torch_model(model, str(OUTPUT_DIR / "alexnet_pt_best.pt"))
    save_results_json(results, str(OUTPUT_DIR / "alexnet_pt_results.json"))
    plot_training_history(history, "AlexNet-PyTorch", save_path=str(OUTPUT_DIR / "alexnet_pt_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "AlexNet-PyTorch",
                          save_path=str(OUTPUT_DIR / "alexnet_pt_cm.png"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="tensorflow", choices=["tensorflow","pytorch","both"])
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch_size",type=int, default=32)
    args = parser.parse_args()

    if args.framework in ("tensorflow", "both"):
        train_tf(args.data_dir, args.epochs, args.batch_size)
    if args.framework in ("pytorch", "both"):
        train_pytorch(args.data_dir, args.epochs, args.batch_size)
