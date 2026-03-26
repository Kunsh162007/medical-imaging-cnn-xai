"""
04_advanced_architectures/02_vggnet.py
================================
VGG16 for 4-class Alzheimer's MRI — PyTorch with transfer learning.

Run:
    python 04_advanced_architectures/02_vggnet.py --framework pytorch --epochs 30 --data_dir data
"""

import argparse
import numpy as np
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.model_utils import set_seed, get_device, save_results_json, save_torch_model
from utils.metrics import evaluate_model, print_report
from utils.visualization import plot_training_history, plot_confusion_matrix

OUTPUT_DIR = Path("results/04_advanced_architectures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)


def train_pytorch(data_dir="data", epochs=30, batch_size=32, fine_tune=True):
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision import models
    from utils.data_loader import get_dataloader

    device = get_device()
    print(f"\n[VGGNet-PyTorch] Device: {device}")

    train_loader = get_dataloader(f"{data_dir}/train", "pytorch", batch_size, augment=True)
    val_loader   = get_dataloader(f"{data_dir}/train", "pytorch", batch_size, augment=False)
    test_loader  = get_dataloader(f"{data_dir}/test",  "pytorch", batch_size, augment=False)

    # Build VGG16
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier
    model.classifier[-1] = nn.Linear(4096, 4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

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

    print("[VGGNet] Stage 1: Training classifier head (frozen backbone)...")
    stage1 = min(epochs, 15)
    for ep in range(1, stage1 + 1):
        tr_loss, tr_acc = _run_epoch(train_loader, train=True)
        vl_loss, vl_acc = _run_epoch(val_loader,   train=False)
        scheduler.step(vl_loss)
        history["accuracy"].append(tr_acc); history["loss"].append(tr_loss)
        history["val_accuracy"].append(vl_acc); history["val_loss"].append(vl_loss)
        if ep % 5 == 0:
            print(f"  Ep {ep:>3}/{stage1} | Loss {tr_loss:.4f} | Acc {tr_acc*100:.1f}% | Val {vl_acc*100:.1f}%")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_torch_model(model, str(OUTPUT_DIR / "vggnet_pt_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 7:
            print("  [EarlyStopping]"); break

    if fine_tune:
        print("[VGGNet] Stage 2: Fine-tuning top conv layers...")
        for name, p in model.named_parameters():
            if "features.24" in name or "features.26" in name or "features.28" in name or "classifier" in name:
                p.requires_grad = True
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=1e-5, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        patience_counter = 0

        for ep in range(1, epochs - stage1 + 1):
            tr_loss, tr_acc = _run_epoch(train_loader, train=True)
            vl_loss, vl_acc = _run_epoch(val_loader,   train=False)
            scheduler.step(vl_loss)
            history["accuracy"].append(tr_acc); history["loss"].append(tr_loss)
            history["val_accuracy"].append(vl_acc); history["val_loss"].append(vl_loss)
            if ep % 5 == 0:
                print(f"  FT {ep:>2} | Loss {tr_loss:.4f} | Acc {tr_acc*100:.1f}% | Val {vl_acc*100:.1f}%")
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                save_torch_model(model, str(OUTPUT_DIR / "vggnet_pt_best.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 7:
                print("  [EarlyStopping]"); break

    # Evaluate
    print("[VGGNet] Evaluating on test set...")
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
    print_report(y_true, y_pred, "VGGNet-PyTorch")
    print(f"  Test Acc: {results['accuracy']*100:.2f}%  F1: {results['f1_macro']:.4f}  AUC: {results.get('auc','N/A')}")

    save_torch_model(model, str(OUTPUT_DIR / "vggnet_pt_final.pt"))
    save_results_json(results, str(OUTPUT_DIR / "vggnet_pt_results.json"))
    plot_training_history(history, "VGGNet-PyTorch",
                          save_path=str(OUTPUT_DIR / "vggnet_pt_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "VGGNet-PyTorch",
                          save_path=str(OUTPUT_DIR / "vggnet_pt_cm.png"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework",   default="pytorch", choices=["pytorch"])
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--no_finetune", action="store_true")
    args = parser.parse_args()

    print("Module 4 — VGGNet (VGG16)")
    train_pytorch(args.data_dir, args.epochs, args.batch_size, not args.no_finetune)
