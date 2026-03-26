"""
04_advanced_architectures/03_googlenet_inception.py
=============================================
GoogLeNet / InceptionV3 for 4-class Alzheimer's MRI — PyTorch.

Run:
    python 04_advanced_architectures/03_googlenet_inception.py --epochs 30 --data_dir data
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
    from torchvision import models, transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    device = get_device()
    print(f"\n[GoogLeNet/InceptionV3-PyTorch] Device: {device}")

    # InceptionV3 requires 299x299
    tfm_train = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_test = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(ImageFolder(f"{data_dir}/train", transform=tfm_train),
                              batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(ImageFolder(f"{data_dir}/train", transform=tfm_test),
                              batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(ImageFolder(f"{data_dir}/test",  transform=tfm_test),
                              batch_size=batch_size, shuffle=False, num_workers=2)

    # Build InceptionV3 — must keep aux_logits=True when loading pretrained weights
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    # Disable aux logits after loading weights
    model.aux_logits = False
    model.AuxLogits  = None

    # Freeze all layers
    for p in model.parameters():
        p.requires_grad = False

    # Replace head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 4)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val_acc    = 0.0
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
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)
        return total_loss / total, correct / total

    print("[GoogLeNet] Stage 1: Training head (frozen backbone)...")
    stage1 = min(epochs, 15)
    for ep in range(1, stage1 + 1):
        tr_loss, tr_acc = _run_epoch(train_loader, train=True)
        vl_loss, vl_acc = _run_epoch(val_loader,   train=False)
        scheduler.step(vl_loss)
        history["accuracy"].append(tr_acc)
        history["loss"].append(tr_loss)
        history["val_accuracy"].append(vl_acc)
        history["val_loss"].append(vl_loss)
        if ep % 5 == 0:
            print(f"  Ep {ep:>3}/{stage1} | Loss {tr_loss:.4f} | Acc {tr_acc*100:.1f}% | Val {vl_acc*100:.1f}%")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_torch_model(model, str(OUTPUT_DIR / "googlenet_pt_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 7:
            print("  [EarlyStopping]")
            break

    if fine_tune:
        print("[GoogLeNet] Stage 2: Fine-tuning Mixed_7c + head...")
        for name, p in model.named_parameters():
            if "Mixed_7" in name or "fc" in name:
                p.requires_grad = True
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=1e-5, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        patience_counter = 0

        for ep in range(1, epochs - stage1 + 1):
            tr_loss, tr_acc = _run_epoch(train_loader, train=True)
            vl_loss, vl_acc = _run_epoch(val_loader,   train=False)
            scheduler.step(vl_loss)
            history["accuracy"].append(tr_acc)
            history["loss"].append(tr_loss)
            history["val_accuracy"].append(vl_acc)
            history["val_loss"].append(vl_loss)
            if ep % 5 == 0:
                print(f"  FT {ep:>2} | Loss {tr_loss:.4f} | Acc {tr_acc*100:.1f}% | Val {vl_acc*100:.1f}%")
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                save_torch_model(model, str(OUTPUT_DIR / "googlenet_pt_best.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 7:
                print("  [EarlyStopping]")
                break

    # Evaluate
    print("[GoogLeNet] Evaluating on test set...")
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
    print_report(y_true, y_pred, "GoogLeNet-PyTorch")
    print(f"  Test Acc: {results['accuracy']*100:.2f}%  F1: {results['f1_macro']:.4f}  AUC: {results.get('auc','N/A')}")

    save_torch_model(model, str(OUTPUT_DIR / "googlenet_pt_final.pt"))
    save_results_json(results, str(OUTPUT_DIR / "googlenet_pt_results.json"))
    plot_training_history(history, "GoogLeNet-PyTorch",
                          save_path=str(OUTPUT_DIR / "googlenet_pt_history.png"))
    plot_confusion_matrix(results["confusion_matrix"], "GoogLeNet-PyTorch",
                          save_path=str(OUTPUT_DIR / "googlenet_pt_cm.png"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework",   default="pytorch", choices=["pytorch"])
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--no_finetune", action="store_true")
    args = parser.parse_args()

    print("Module 4 — GoogLeNet / InceptionV3")
    train_pytorch(args.data_dir, args.epochs, args.batch_size, not args.no_finetune)
