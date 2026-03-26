"""
05_explainability_xai/04_shap_xai.py
=========================
SHAP DeepExplainer applied to all CNN architectures on Alzheimer MRI.
Supports TensorFlow and PyTorch.

Run:
    python 05_explainability_xai/04_shap_xai.py --model resnet50 --framework tensorflow
    python 05_explainability_xai/04_shap_xai.py --all-models --framework both
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, CLASS_NAMES, ACCENT_CYAN, ACCENT_RED

OUTPUT_DIR = Path("results/05_explainability_xai/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["alexnet", "vggnet", "googlenet", "resnet50"]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_samples(data_dir="data/test", n=50):
    """Load n images as float32 arrays normalized to [0,1]."""
    from PIL import Image
    images, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.exists():
            continue
        files = sorted(cls_dir.glob("*.jpg"))[: n // len(CLASS_NAMES)]
        for f in files:
            img = np.array(Image.open(f).convert("RGB").resize((128, 128)),
                           dtype=np.float32) / 255.0
            images.append(img)
            labels.append(cls_idx)
    if not images:
        print("[warn] No images found — generating random demo data.")
        images = np.random.rand(n, 128, 128, 3).astype(np.float32)
        labels = np.random.randint(0, 4, n).tolist()
    return np.array(images), np.array(labels)


def normalize_shap(shap_vals):
    """Normalise SHAP values for display."""
    s = np.abs(shap_vals).max()
    return shap_vals / (s + 1e-8)


# ─────────────────────────────────────────────
# TensorFlow SHAP
# ─────────────────────────────────────────────
def shap_tensorflow(model_name="resnet50", data_dir="data/test"):
    import shap
    import tensorflow as tf

    model_paths = {
        "alexnet":   "results/04_advanced_architectures/alexnet_tf.h5",
        "vggnet":    "results/04_advanced_architectures/vggnet_tf.h5",
        "googlenet": "results/04_advanced_architectures/googlenet_tf.h5",
        "resnet50":  "results/04_advanced_architectures/resnet50_tf_final.h5",
    }
    names = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir, n=50)

    # ImageNet mean normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    images_norm = (images - mean) / std

    for name in names:
        path = model_paths.get(name, "")
        if not Path(path).exists():
            print(f"  [TF-SHAP] Model not found: {path}. Skipping.")
            continue

        print(f"  [TF-SHAP] Running SHAP for {name}...")
        model = tf.keras.models.load_model(path)

        # Background = 20 random training images
        background = images_norm[:20]
        explain_imgs = images_norm[20:24]  # 4 images to explain

        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(explain_imgs)  # list of [N,H,W,C] per class

        _plot_shap_summary(explain_imgs, shap_values, labels[20:24], name, "TF")
        _plot_shap_beeswarm(shap_values, name, "TF")

    print(f"  [TF-SHAP] Done.")


# ─────────────────────────────────────────────
# PyTorch SHAP
# ─────────────────────────────────────────────
def shap_pytorch(model_name="resnet50", data_dir="data/test"):
    import shap
    import torch
    from torchvision import models
    from utils.model_utils import get_device

    device  = get_device()
    names   = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir, n=50)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    images_norm = (images - mean) / std
    # Convert to CHW
    images_chw = np.transpose(images_norm, (0, 3, 1, 2))

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
            print(f"  [PT-SHAP] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"  [PT-SHAP] Running SHAP for {name}...")

        # Rebuild model
        if arch == "alexnet":
            m = models.alexnet(weights=None)
            m.classifier[-1] = torch.nn.Linear(4096, 4)
        elif arch == "vgg16":
            m = models.vgg16(weights=None)
            m.classifier[-1] = torch.nn.Linear(4096, 4)
        elif arch == "inception_v3":
            m = models.inception_v3(weights=None, aux_logits=False)
            m.fc = torch.nn.Sequential(
                        torch.nn.Linear(m.fc.in_features, 512),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.4),
                        torch.nn.Linear(512, 4)
                    )
        elif arch == "resnet50":
            m = models.resnet50(weights=None)
            m.fc = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(m.fc.in_features),
                    torch.nn.Linear(m.fc.in_features, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(512, 4)
                    )

        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state.get("state_dict", state))
        m = m.to(device).eval()
        background = torch.tensor(images_chw[:20], dtype=torch.float32).to(device)
        explain_t  = torch.tensor(images_chw[20:24], dtype=torch.float32).to(device)

        # GradientExplainer uses plain autograd — immune to all inplace-op
        # conflicts that plague DeepExplainer (F.relu inplace, residual +=,
        # BatchNorm, etc.).  No monkey-patching needed.
        explainer   = shap.GradientExplainer(m, background)
        shap_values = explainer.shap_values(explain_t)

        # Normalise SHAP output to list-of-classes format.
        # Older SHAP: list of N_classes arrays each [N, C, H, W]
        # Newer SHAP: single array [N, N_classes, C, H, W] or [N, C, H, W]
        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 5:          # [N, n_cls, C, H, W]
                shap_values = [shap_values[:, c] for c in range(shap_values.shape[1])]
            else:                              # [N, C, H, W] — treat as single class
                shap_values = [shap_values]
        # Already a list — keep as-is

        # Convert back to HWC for plotting
        imgs_display = images[20:24]
        _plot_shap_summary(imgs_display, shap_values, labels[20:24], name, "PT",
                           chw_format=True)
        _plot_shap_beeswarm(shap_values, name, "PT")

    print("  [PT-SHAP] Done.")


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────
def _plot_shap_summary(images, shap_values, labels, model_name, tag, chw_format=False):
    """
    Plot SHAP image overlay for each explained image × class.
    shap_values: list of arrays [N, H, W, C] or [N, C, H, W]
    """
    n_images  = min(len(images), 4)
    n_classes = min(len(shap_values), 4)  # shap_values list length = n_output_classes

    fig, axes = plt.subplots(n_images, n_classes + 1,
                             figsize=((n_classes + 1) * 3, n_images * 3))
    apply_dark_theme(fig, axes.flatten())

    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_images):
        # Original image
        axes[i, 0].imshow(images[i])
        cls_name = CLASS_NAMES[labels[i]] if labels[i] < 4 else str(labels[i])
        axes[i, 0].set_title(f"Original\n{cls_name}", color="white", fontsize=7.5)
        axes[i, 0].axis("off")

        for cls_idx in range(n_classes):
            sv = shap_values[cls_idx][i]
            if chw_format and sv.ndim == 3 and sv.shape[0] == 3:
                sv = np.transpose(sv, (1, 2, 0))  # CHW → HWC
            sv_agg = sv.sum(axis=-1)  # Sum across channels
            sv_norm = normalize_shap(sv_agg)

            ax = axes[i, cls_idx + 1]
            im = ax.imshow(sv_norm, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"SHAP: {CLASS_NAMES[cls_idx]}", color="#A5F3FC", fontsize=7)
            ax.axis("off")

    fig.suptitle(f"SHAP DeepExplainer — {model_name.upper()} [{tag}]",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"shap_{model_name}_{tag.lower()}.png"))


def _plot_shap_beeswarm(shap_values, model_name, tag):
    """Aggregate SHAP values per class and plot mean absolute importance."""
    means = []
    for sv in shap_values:  # per class
        if hasattr(sv, "shape"):
            means.append(np.abs(sv).mean())
        else:
            means.append(0.0)

    classes = CLASS_NAMES[:len(means)]
    fig, ax = plt.subplots(figsize=(8, 4))
    apply_dark_theme(fig, [ax])

    colors = [ACCENT_CYAN, "#A78BFA", "#4ADE80", ACCENT_RED][:len(means)]
    ax.barh(classes, means, color=colors, alpha=0.85, height=0.5)
    ax.set_title(f"Mean |SHAP| per Class — {model_name.upper()} [{tag}]",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    for i, v in enumerate(means):
        ax.text(v + 0.0001, i, f"{v:.4f}", va="center", color="white", fontsize=9)

    _save_or_show(fig, str(OUTPUT_DIR / f"shap_beeswarm_{model_name}_{tag.lower()}.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP XAI for Alzheimer CNN")
    parser.add_argument("--model",      default="resnet50",
                        choices=MODEL_NAMES + ["all"])
    parser.add_argument("--framework",  default="tensorflow",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--data_dir",   default="data/test")
    parser.add_argument("--all-models", action="store_true",
                        dest="all_models", help="Run all architectures")
    args = parser.parse_args()

    model = "all" if args.all_models else args.model
    print(f"\nModule 5 — SHAP | Model: {model} | Framework: {args.framework}")

    if args.framework in ("tensorflow", "both"):
        shap_tensorflow(model, args.data_dir)

    if args.framework in ("pytorch", "both"):
        shap_pytorch(model, args.data_dir)

    print(f"\nSHAP outputs saved to: {OUTPUT_DIR}")
