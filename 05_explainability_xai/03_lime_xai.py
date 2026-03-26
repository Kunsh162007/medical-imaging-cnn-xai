"""
05_explainability_xai/03_lime_xai.py
=========================
LIME (Local Interpretable Model-Agnostic Explanations) for Alzheimer MRI CNNs.
Works with any model that exposes a predict_proba-style function.

Run:
    python 05_explainability_xai/03_lime_xai.py --model resnet50 --framework tensorflow
    python 05_explainability_xai/03_lime_xai.py --model all      --framework both
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, CLASS_NAMES

OUTPUT_DIR = Path("results/05_explainability_xai/lime")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["alexnet", "vggnet", "googlenet", "resnet50"]


# ─────────────────────────────────────────────
# Shared image loader
# ─────────────────────────────────────────────
def load_samples(data_dir="data/test", n_per_class=1):
    from PIL import Image
    images, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*.jpg"))[:n_per_class]:
            img = np.array(Image.open(f).convert("RGB").resize((128, 128)))
            images.append(img)
            labels.append(cls_idx)
    if not images:
        print("[warn] No images found — using random demo data.")
        images = (np.random.rand(4, 128, 128, 3) * 255).astype(np.uint8)
        labels = list(range(4))
    return np.array(images), np.array(labels)


# ─────────────────────────────────────────────
# TensorFlow LIME
# ─────────────────────────────────────────────
def lime_tensorflow(model_name="resnet50", data_dir="data/test"):
    import tensorflow as tf
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    model_paths = {
        "alexnet":   "results/04_advanced_architectures/alexnet_tf.h5",
        "vggnet":    "results/04_advanced_architectures/vggnet_tf.h5",
        "googlenet": "results/04_advanced_architectures/googlenet_tf.h5",
        "resnet50":  "results/04_advanced_architectures/resnet50_tf_final.h5",
    }
    names  = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir, n_per_class=1)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for name in names:
        path = model_paths.get(name, "")
        if not Path(path).exists():
            print(f"  [TF-LIME] Model not found: {path}. Skipping.")
            continue

        print(f"  [TF-LIME] Running LIME for {name}...")
        model = tf.keras.models.load_model(path)

        def predict_fn(imgs):
            imgs_norm = (imgs.astype(np.float32) / 255.0 - mean) / std
            return model.predict(imgs_norm, verbose=0)

        explainer = lime_image.LimeImageExplainer()
        results_list = []

        for img, lbl in zip(images, labels):
            explanation = explainer.explain_instance(
                img, predict_fn, top_labels=4,
                num_samples=500, hide_color=0,
            )
            pred_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                pred_label, positive_only=False,
                num_features=10, hide_rest=False
            )
            results_list.append((img, temp, mask, lbl, pred_label))

        _plot_lime_results(results_list, name, "TF")
    print("  [TF-LIME] Done.")


# ─────────────────────────────────────────────
# PyTorch LIME
# ─────────────────────────────────────────────
def lime_pytorch(model_name="resnet50", data_dir="data/test"):
    import torch
    from torchvision import models, transforms
    from lime import lime_image
    from utils.model_utils import get_device

    device  = get_device()
    names   = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir, n_per_class=1)

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
            print(f"  [PT-LIME] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"  [PT-LIME] Running LIME for {name}...")

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

        def predict_fn(imgs):
            tensors = torch.stack([transform(img) for img in imgs]).to(device)
            with torch.no_grad():
                return torch.softmax(m(tensors), dim=1).cpu().numpy()

        explainer = lime_image.LimeImageExplainer()
        results_list = []

        for img, lbl in zip(images, labels):
            explanation = explainer.explain_instance(
                img, predict_fn, top_labels=4,
                num_samples=500, hide_color=0,
            )
            pred_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                pred_label, positive_only=False,
                num_features=10, hide_rest=False,
            )
            results_list.append((img, temp, mask, lbl, pred_label))

        _plot_lime_results(results_list, name, "PT")
    print("  [PT-LIME] Done.")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def _plot_lime_results(results_list, model_name, tag):
    from skimage.segmentation import mark_boundaries
    n = len(results_list)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    apply_dark_theme(fig, axes.flatten() if n > 1 else axes)

    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (orig, temp, mask, true_lbl, pred_lbl) in enumerate(results_list):
        true_name = CLASS_NAMES[true_lbl] if true_lbl < 4 else str(true_lbl)
        pred_name = CLASS_NAMES[pred_lbl] if pred_lbl < 4 else str(pred_lbl)

        # Original
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"True: {true_name}", color="white", fontsize=8)
        axes[i, 0].axis("off")

        # LIME segments
        boundaries = mark_boundaries(temp / 255.0, mask)
        axes[i, 1].imshow(boundaries)
        axes[i, 1].set_title(f"LIME Segments\nPred: {pred_name}", color="#58D9F9", fontsize=8)
        axes[i, 1].axis("off")

        # Positive superpixels only
        temp2, mask2 = temp, mask  # already computed
        axes[i, 2].imshow(mark_boundaries(temp / 255.0, mask > 0))
        axes[i, 2].set_title("Positive Superpixels", color="#4ADE80", fontsize=8)
        axes[i, 2].axis("off")

    fig.suptitle(f"LIME Explanations — {model_name.upper()} [{tag}]",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"lime_{model_name}_{tag.lower()}.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIME XAI for Alzheimer CNN")
    parser.add_argument("--model",     default="resnet50",
                        choices=MODEL_NAMES + ["all"])
    parser.add_argument("--framework", default="tensorflow",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--data_dir",  default="data/test")
    args = parser.parse_args()

    print(f"\nModule 5 — LIME | Model: {args.model} | Framework: {args.framework}")

    if args.framework in ("tensorflow", "both"):
        lime_tensorflow(args.model, args.data_dir)

    if args.framework in ("pytorch", "both"):
        lime_pytorch(args.model, args.data_dir)

    print(f"\nLIME outputs saved to: {OUTPUT_DIR}")
