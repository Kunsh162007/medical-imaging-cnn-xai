"""
04_advanced_architectures/run_ensemble.py
===================================
Loads all 4 pre-trained models and runs soft-voting ensemble evaluation.
No retraining — uses saved checkpoints only.

Run:
    python 04_advanced_architectures/run_ensemble.py --data_dir data
"""

import argparse
import numpy as np
import sys; sys.path.insert(0, "..")
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils.model_utils import set_seed, get_device, save_results_json
from utils.metrics import evaluate_model, print_report, compare_models
from utils.visualization import plot_confusion_matrix, apply_dark_theme, _save_or_show
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("results/04_advanced_architectures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)


def load_model(arch, ckpt_path, device):
    """Rebuild architecture and load saved weights."""
    if arch == "alexnet":
        m = models.alexnet(weights=None)
        m.classifier[-1] = nn.Linear(4096, 4)
    elif arch == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier[-1] = nn.Linear(4096, 4)
    elif arch == "inception":
        m = models.inception_v3(weights=None, aux_logits=True)
        m.aux_logits = False
        m.AuxLogits  = None
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 512),
                             nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 4))
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.BatchNorm1d(m.fc.in_features),
                             nn.Linear(m.fc.in_features, 512),
                             nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 4))

    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state.get("state_dict", state))
    return m.to(device).eval()


def run_ensemble(data_dir="data", batch_size=32):
    device = get_device()
    print(f"[Ensemble] Device: {device}")

    # Standard 128x128 transform
    tfm_128 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # 299x299 for GoogLeNet
    tfm_299 = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model_configs = {
        "AlexNet":   ("alexnet",   "alexnet_pt_best.pt",   tfm_128),
        "VGGNet":    ("vgg16",     "vggnet_pt_best.pt",    tfm_128),
        "GoogLeNet": ("inception", "googlenet_pt_best.pt", tfm_299),
        "ResNet50":  ("resnet50",  "resnet50_pt_best.pt",  tfm_128),
    }

    # Load all models
    loaded_models = {}
    for name, (arch, ckpt, _) in model_configs.items():
        ckpt_path = OUTPUT_DIR / ckpt
        if not ckpt_path.exists():
            print(f"  [SKIP] {name} — checkpoint not found: {ckpt_path}")
            continue
        print(f"  Loading {name} from {ckpt}...")
        loaded_models[name] = load_model(arch, str(ckpt_path), device)

    print(f"\n[Ensemble] Loaded {len(loaded_models)} models: {list(loaded_models.keys())}")

    # Get ground truth from test set (use any transform — just for labels)
    test_ds = ImageFolder(f"{data_dir}/test", transform=tfm_128)
    y_true_all = [label for _, label in test_ds.samples]
    y_true_all = np.array(y_true_all)

    # Get predictions from each model
    all_probas = {}
    individual_results = {}

    for name, (arch, ckpt, tfm) in model_configs.items():
        if name not in loaded_models:
            continue
        model = loaded_models[name]
        test_loader = DataLoader(ImageFolder(f"{data_dir}/test", transform=tfm),
                                 batch_size=batch_size, shuffle=False, num_workers=2)
        probas = []
        with torch.no_grad():
            for X, _ in test_loader:
                p = torch.softmax(model(X.to(device)), dim=1).cpu().numpy()
                probas.append(p)
        all_probas[name] = np.vstack(probas)

        # Individual results
        y_pred = all_probas[name].argmax(axis=1)
        individual_results[name] = evaluate_model(y_true_all, y_pred, all_probas[name])
        print(f"  {name}: {individual_results[name]['accuracy']*100:.2f}%")

    # Soft-voting ensemble
    print("\n[Ensemble] Computing soft-vote ensemble...")
    ensemble_proba = np.mean(list(all_probas.values()), axis=0)
    ensemble_pred  = ensemble_proba.argmax(axis=1)
    ensemble_results = evaluate_model(y_true_all, ensemble_pred, ensemble_proba)
    individual_results["Ensemble"] = ensemble_results

    print_report(y_true_all, ensemble_pred, "Soft-Vote Ensemble")
    print(f"\n  Ensemble Test Acc: {ensemble_results['accuracy']*100:.2f}%")
    print(f"  Ensemble F1:       {ensemble_results['f1_macro']:.4f}")
    print(f"  Ensemble AUC:      {ensemble_results.get('auc','N/A')}")

    # Save results
    save_results_json(individual_results, str(OUTPUT_DIR / "ensemble_pt_results.json"))
    compare_models(individual_results)

    # Plots
    plot_confusion_matrix(ensemble_results["confusion_matrix"], "Soft-Vote Ensemble",
                          save_path=str(OUTPUT_DIR / "ensemble_pt_cm.png"))
    _plot_comparison(individual_results)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return individual_results


def _plot_comparison(results):
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] * 100 for n in names]
    f1s    = [results[n]["f1_macro"] for n in names]

    colors = ["#58D9F9", "#A78BFA", "#4ADE80", "#FB923C", "#FDE68A"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    apply_dark_theme(fig, axes)

    for ax, vals, title in [(axes[0], accs, "Test Accuracy (%)"),
                             (axes[1], f1s,  "F1 Macro")]:
        bars = ax.bar(range(len(names)), vals,
                      color=colors[:len(names)], alpha=0.85, width=0.55)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", color="white")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.08)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                    f"{v:.2f}" if title == "Test Accuracy (%)" else f"{v:.4f}",
                    ha="center", color="white", fontsize=9, fontweight="bold")

    fig.suptitle("Module 4 — Model Comparison (PyTorch)",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / "ensemble_model_comparison.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    run_ensemble(args.data_dir, args.batch_size)
