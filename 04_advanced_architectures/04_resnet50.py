"""
04_advanced_architectures/04_resnet50.py
==================================
ResNet50 for 4-class Alzheimer's MRI classification.
Supports both TensorFlow/Keras and PyTorch.

Run:
    python 04_advanced_architectures/04_resnet50.py --framework tensorflow --epochs 50
    python 04_advanced_architectures/04_resnet50.py --framework pytorch    --epochs 50
"""

import argparse
import numpy as np
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.model_utils import set_seed, get_device, save_results_json, save_tf_model, save_torch_model
from utils.metrics import evaluate_model, print_report, compare_models
from utils.visualization import plot_training_history, plot_confusion_matrix

OUTPUT_DIR = Path("results/04_advanced_architectures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)


# ─────────────────────────────────────────────
# TensorFlow / Keras
# ─────────────────────────────────────────────
def train_tf(data_dir="data", epochs=50, batch_size=32, fine_tune=True):
    import tensorflow as tf
    from utils.data_loader import get_dataloader

    print("\n[TF] Loading dataset...")
    train_ds = get_dataloader(f"{data_dir}/train", "tensorflow", batch_size, augment=True)
    val_ds   = get_dataloader(f"{data_dir}/train", "tensorflow", batch_size, augment=False,
                              validation_split=0.1, subset="validation")
    test_ds  = get_dataloader(f"{data_dir}/test",  "tensorflow", batch_size, augment=False)

    # ── Build ResNet50 ──────────────────────────
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(128, 128, 3)
    )
    # Stage 1: freeze base
    base.trainable = False

    inp = tf.keras.Input(shape=(128, 128, 3))
    x   = base(inp, training=False)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dense(512, activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(OUTPUT_DIR / "resnet50_tf_best.h5"),
                                           save_best_only=True, verbose=0),
    ]

    print("[TF] Stage 1: Training head (frozen base)...")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=min(epochs, 15), callbacks=cbs, verbose=1)

    if fine_tune:
        print("[TF] Stage 2: Fine-tuning top 30 layers...")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy", metrics=["accuracy"],
        )
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=epochs - min(epochs, 15), callbacks=cbs, verbose=1)
        # Merge histories
        merged = {}
        for k in h1.history:
            merged[k] = h1.history[k] + h2.history.get(k, [])
        history = type("H", (), {"history": merged})()
    else:
        history = h1

    # ── Evaluate ───────────────────────────────
    print("[TF] Evaluating on test set...")
    y_true, y_pred, y_proba = [], [], []
    for X_batch, y_batch in test_ds:
        probs = model.predict(X_batch, verbose=0)
        y_proba.append(probs)
        y_pred.extend(probs.argmax(axis=1))
        y_true.extend(y_batch.numpy().argmax(axis=1))

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_proba = np.vstack(y_proba)

    results = evaluate_model(y_true, y_pred, y_proba)
    print_report(y_true, y_pred, "ResNet50-TF")
    print(f"  Test Acc: {results['accuracy']*100:.2f}%  F1: {results['f1_macro']:.4f}  AUC: {results.get('auc','N/A')}")

    # ── Save artefacts ─────────────────────────
    save_tf_model(model, str(OUTPUT_DIR / "resnet50_tf_final.h5"))
    save_results_json(results, str(OUTPUT_DIR / "resnet50_tf_results.json"))
    plot_training_history(history, "ResNet50-TF",
                          save_path=str(OUTPUT_DIR / "resnet50_tf_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "ResNet50-TF",
                          save_path=str(OUTPUT_DIR / "resnet50_tf_cm.png"))
    return results


# ─────────────────────────────────────────────
# PyTorch
# ─────────────────────────────────────────────
def train_pytorch(data_dir="data", epochs=50, batch_size=32, fine_tune=True):
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision import models
    from utils.data_loader import get_dataloader

    device = get_device()
    print(f"\n[PyTorch] Device: {device}")

    train_loader = get_dataloader(f"{data_dir}/train", "pytorch", batch_size, augment=True)
    val_loader   = get_dataloader(f"{data_dir}/train", "pytorch", batch_size, augment=False)
    test_loader  = get_dataloader(f"{data_dir}/test",  "pytorch", batch_size, augment=False)

    # ── Build ResNet50 ──────────────────────────
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all layers first
    for p in model.parameters():
        p.requires_grad = False

    # Replace head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 4),
    )
    model = model.to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val_acc = 0.0
    patience_counter = 0

    def _run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss = correct = total = 0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                out  = model(X)
                loss = criterion(out, y)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)
        return total_loss / total, correct / total

    print("[PyTorch] Stage 1: Training head (frozen backbone)...")
    stage1_epochs = min(epochs, 15)
    for ep in range(1, stage1_epochs + 1):
        tr_loss, tr_acc = _run_epoch(train_loader, train=True)
        vl_loss, vl_acc = _run_epoch(val_loader,   train=False)
        scheduler.step(vl_loss)
        history["accuracy"].append(tr_acc); history["loss"].append(tr_loss)
        history["val_accuracy"].append(vl_acc); history["val_loss"].append(vl_loss)
        print(f"  Ep {ep:>3}/{stage1_epochs} | Loss {tr_loss:.4f} | Acc {tr_acc*100:.1f}% "
              f"| Val {vl_acc*100:.1f}%")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_torch_model(model, str(OUTPUT_DIR / "resnet50_pt_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 7:
            print("  [EarlyStopping]"); break

    if fine_tune:
        print("[PyTorch] Stage 2: Unfreezing top layers for fine-tuning...")
        # Unfreeze layer4 + head
        for name, p in model.named_parameters():
            if "layer4" in name or "fc" in name:
                p.requires_grad = True
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=1e-5, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        patience_counter = 0

        for ep in range(1, epochs - stage1_epochs + 1):
            tr_loss, tr_acc = _run_epoch(train_loader, train=True)
            vl_loss, vl_acc = _run_epoch(val_loader,   train=False)
            scheduler.step(vl_loss)
            history["accuracy"].append(tr_acc); history["loss"].append(tr_loss)
            history["val_accuracy"].append(vl_acc); history["val_loss"].append(vl_loss)
            print(f"  FT {ep:>2} | Loss {tr_loss:.4f} | Acc {tr_acc*100:.1f}% "
                  f"| Val {vl_acc*100:.1f}%")
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                save_torch_model(model, str(OUTPUT_DIR / "resnet50_pt_best.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 7:
                print("  [EarlyStopping]"); break

    # ── Evaluate on test set ───────────────────
    print("[PyTorch] Evaluating on test set...")
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            probs = torch.softmax(model(X), dim=1).cpu().numpy()
            y_proba.append(probs)
            y_pred.extend(probs.argmax(axis=1))
            y_true.extend(y.numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_proba = np.vstack(y_proba)

    results = evaluate_model(y_true, y_pred, y_proba)
    print_report(y_true, y_pred, "ResNet50-PyTorch")
    print(f"  Test Acc: {results['accuracy']*100:.2f}%  F1: {results['f1_macro']:.4f}  AUC: {results.get('auc','N/A')}")

    save_torch_model(model, str(OUTPUT_DIR / "resnet50_pt_final.pt"))
    save_results_json(results, str(OUTPUT_DIR / "resnet50_pt_results.json"))
    plot_training_history(history, "ResNet50-PyTorch",
                          save_path=str(OUTPUT_DIR / "resnet50_pt_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "ResNet50-PyTorch",
                          save_path=str(OUTPUT_DIR / "resnet50_pt_cm.png"))
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 on Alzheimer MRI")
    parser.add_argument("--framework", default="tensorflow",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--no_finetune", action="store_true")
    args = parser.parse_args()

    all_results = {}
    fine_tune = not args.no_finetune

    if args.framework in ("tensorflow", "both"):
        tf_res = train_tf(args.data_dir, args.epochs, args.batch_size, fine_tune)
        all_results["ResNet50-TF"] = tf_res

    if args.framework in ("pytorch", "both"):
        pt_res = train_pytorch(args.data_dir, args.epochs, args.batch_size, fine_tune)
        all_results["ResNet50-PyTorch"] = pt_res

    if len(all_results) > 1:
        compare_models(all_results)
