"""
05_explainability_xai/07_pci_analysis.py
=============================
Pixel Contribution Index (PCI) — a custom XAI metric that measures each
pixel's relative contribution to the predicted class probability using
a combination of gradient magnitude and activation strength.

Formula:
    PCI(i,j) = |∂ŷ/∂x(i,j)| × σ(x(i,j))

where σ is a spatial softmax over the image.

Run:
    python 05_explainability_xai/07_pci_analysis.py --model resnet50 --framework tensorflow
    python 05_explainability_xai/07_pci_analysis.py --model all      --framework both
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, CLASS_NAMES

OUTPUT_DIR = Path("results/05_explainability_xai/pci")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["alexnet", "vggnet", "googlenet", "resnet50"]


def spatial_softmax(x: np.ndarray) -> np.ndarray:
    """Spatial softmax — normalise a 2D map so values sum to 1."""
    e = np.exp(x - x.max())
    return e / e.sum()


def compute_pci(gradient_map: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Pixel Contribution Index:
        PCI = |gradient| * spatial_softmax(image_intensity)

    Args:
        gradient_map : (H, W) absolute gradient magnitude
        image        : (H, W, C) float32 [0,1]
    Returns:
        (H, W) PCI map normalised to [0, 1]
    """
    intensity = image.mean(axis=-1)
    s_soft    = spatial_softmax(intensity)
    pci       = np.abs(gradient_map) * s_soft
    pci       = (pci - pci.min()) / (pci.max() - pci.min() + 1e-8)
    return pci


def load_samples(data_dir="data/test", n_per_class=1):
    from PIL import Image
    images, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*.jpg"))[:n_per_class]:
            img = np.array(Image.open(f).convert("RGB").resize((128, 128)),
                           dtype=np.float32) / 255.0
            images.append(img)
            labels.append(cls_idx)
    if not images:
        images = np.random.rand(4, 128, 128, 3).astype(np.float32)
        labels = list(range(4))
    return np.array(images), np.array(labels)


# ─────────────────────────────────────────────
# TensorFlow PCI
# ─────────────────────────────────────────────
def pci_tensorflow(model_name="resnet50", data_dir="data/test"):
    import tensorflow as tf

    model_paths = {
        "alexnet":   "results/04_advanced_architectures/alexnet_tf.h5",
        "vggnet":    "results/04_advanced_architectures/vggnet_tf.h5",
        "googlenet": "results/04_advanced_architectures/googlenet_tf.h5",
        "resnet50":  "results/04_advanced_architectures/resnet50_tf_final.h5",
    }
    names  = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)

    for name in names:
        path = model_paths.get(name, "")
        if not Path(path).exists():
            print(f"  [TF-PCI] Model not found: {path}. Skipping.")
            continue

        print(f"  [TF-PCI] PCI analysis for {name}...")
        model = tf.keras.models.load_model(path)
        results_list = []

        for img, lbl in zip(images, labels):
            img_norm   = ((img - mean) / std)[np.newaxis].astype(np.float32)
            img_tensor = tf.Variable(img_norm)

            with tf.GradientTape() as tape:
                preds = model(img_tensor, training=False)
                pred_cls = int(tf.argmax(preds[0]))
                score = preds[:, pred_cls]

            grads   = tape.gradient(score, img_tensor)[0].numpy()
            grad_mag = np.abs(grads).mean(axis=-1)
            pci_map  = compute_pci(grad_mag, img)

            # Statistics
            top10_mask = pci_map >= np.percentile(pci_map, 90)
            top10_contrib = pci_map[top10_mask].sum() / (pci_map.sum() + 1e-8)

            results_list.append({
                "image": img, "pci": pci_map, "label": lbl,
                "pred": pred_cls, "top10_contrib": top10_contrib,
            })

        _plot_pci_results(results_list, name, "TF")
        _plot_pci_statistics(results_list, name, "TF")

    print("  [TF-PCI] Done.")


# ─────────────────────────────────────────────
# PyTorch PCI
# ─────────────────────────────────────────────
def pci_pytorch(model_name="resnet50", data_dir="data/test"):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from utils.model_utils import get_device

    device = get_device()
    names  = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model_configs = {
        "alexnet":   ("results/04_advanced_architectures/alexnet_pt_best.pt",   "alexnet"),
        "vggnet":    ("results/04_advanced_architectures/vggnet_pt_best.pt",    "vgg16"),
        "googlenet": ("results/04_advanced_architectures/googlenet_pt_best.pt", "inception_v3"),
        "resnet50":  ("results/04_advanced_architectures/resnet50_pt_best.pt",  "resnet50"),
    }

    for name in names:
        if name not in model_configs:
            continue
        ckpt_path, arch = model_configs[name]
        if not Path(ckpt_path).exists():
            print(f"  [PT-PCI] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"  [PT-PCI] PCI analysis for {name}...")

        if arch == "alexnet":
            m = models.alexnet(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
        elif arch == "vgg16":
            m = models.vgg16(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
        elif arch == "inception_v3":
            m = models.inception_v3(weights=None, aux_logits=False)
            m.fc = nn.Sequential(
                    nn.Linear(m.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 4)
                    )
        elif arch == "resnet50":
            m = models.resnet50(weights=None)
            m.fc = nn.Sequential(
                    nn.BatchNorm1d(m.fc.in_features),
                    nn.Linear(m.fc.in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 4)
                    )

        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state.get("state_dict", state))
        m = m.to(device).eval()

        results_list = []

        for img, lbl in zip(images, labels):
            tensor = transform(img).unsqueeze(0).to(device).requires_grad_(True)
            out    = m(tensor)
            pred   = out.argmax(1).item()
            m.zero_grad()
            out[0, pred].backward()
            grads = tensor.grad.data.abs().squeeze().cpu().numpy()
            if grads.ndim == 3:
                grad_mag = grads.mean(axis=0)
            else:
                grad_mag = grads
            pci_map = compute_pci(grad_mag, img)
            top10_mask   = pci_map >= np.percentile(pci_map, 90)
            top10_contrib = pci_map[top10_mask].sum() / (pci_map.sum() + 1e-8)

            results_list.append({
                "image": img, "pci": pci_map, "label": lbl,
                "pred": pred, "top10_contrib": top10_contrib,
            })

        _plot_pci_results(results_list, name, "PT")
        _plot_pci_statistics(results_list, name, "PT")

    print("  [PT-PCI] Done.")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def _plot_pci_results(results, model_name, tag):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(14, n * 3.2))
    apply_dark_theme(fig, axes.flatten() if n > 1 else axes)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, r in enumerate(results):
        img     = r["image"]
        pci_map = r["pci"]
        true_nm = CLASS_NAMES[r["label"]] if r["label"] < 4 else str(r["label"])
        pred_nm = CLASS_NAMES[r["pred"]]  if r["pred"]  < 4 else str(r["pred"])

        # Column 0: Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {true_nm}", color="white", fontsize=8)
        axes[i, 0].axis("off")

        # Column 1: PCI heatmap
        im = axes[i, 1].imshow(pci_map, cmap="inferno")
        axes[i, 1].set_title("PCI Map", color="#FB923C", fontsize=8, fontweight="bold")
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

        # Column 2: Top-10% pixels highlighted
        top10_mask = (pci_map >= np.percentile(pci_map, 90)).astype(float)
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(top10_mask, alpha=0.5, cmap="Reds")
        axes[i, 2].set_title(f"Top 10% PCI\nPred: {pred_nm}", color="#F87171", fontsize=8)
        axes[i, 2].axis("off")

        # Column 3: PCI distribution histogram
        axes[i, 3].hist(pci_map.flatten(), bins=40, color="#58D9F9", alpha=0.8, density=True)
        axes[i, 3].axvline(np.percentile(pci_map, 90), color="#F87171",
                            lw=1.5, ls="--", label="P90")
        axes[i, 3].set_title(f"PCI Distribution\nTop10 mass={r['top10_contrib']:.2f}",
                              color="white", fontsize=8)
        axes[i, 3].legend(frameon=False, labelcolor="white", fontsize=8)

    fig.suptitle(f"Pixel Contribution Index (PCI) — {model_name.upper()} [{tag}]",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"pci_{model_name}_{tag.lower()}.png"))


def _plot_pci_statistics(results, model_name, tag):
    """Bar chart of mean PCI per quadrant to identify brain regions."""
    n = len(results)
    quadrant_labels = ["Top-Left\n(Frontal)", "Top-Right\n(Parietal)",
                       "Bot-Left\n(Temporal)", "Bot-Right\n(Occipital)"]

    fig, axes = plt.subplots(1, min(n, 4), figsize=(min(n, 4) * 3.5, 4))
    if n == 1:
        axes = [axes]
    apply_dark_theme(fig, axes)

    for i, (r, ax) in enumerate(zip(results, axes)):
        pci = r["pci"]
        H, W = pci.shape
        hH, hW = H // 2, W // 2
        quadrant_means = [
            pci[:hH, :hW].mean(),
            pci[:hH, hW:].mean(),
            pci[hH:, :hW].mean(),
            pci[hH:, hW:].mean(),
        ]
        colors = ["#58D9F9", "#A78BFA", "#4ADE80", "#FB923C"]
        ax.bar(range(4), quadrant_means, color=colors, alpha=0.85, width=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(quadrant_labels, fontsize=7, color="white")
        cls_name = CLASS_NAMES[r["label"]] if r["label"] < 4 else str(r["label"])
        ax.set_title(f"{cls_name}", color="white", fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean PCI")

    fig.suptitle(f"PCI by Brain Quadrant — {model_name.upper()} [{tag}]",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"pci_quadrant_{model_name}_{tag.lower()}.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="resnet50", choices=MODEL_NAMES + ["all"])
    parser.add_argument("--framework", default="tensorflow",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--data_dir",  default="data/test")
    args = parser.parse_args()

    print(f"\nModule 5 — PCI Analysis | Model: {args.model} | Framework: {args.framework}")

    if args.framework in ("tensorflow", "both"):
        pci_tensorflow(args.model, args.data_dir)

    if args.framework in ("pytorch", "both"):
        pci_pytorch(args.model, args.data_dir)

    print(f"\nPCI outputs saved to: {OUTPUT_DIR}")
