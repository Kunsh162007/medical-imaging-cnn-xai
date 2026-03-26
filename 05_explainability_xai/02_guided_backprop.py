"""
05_explainability_xai/02_guided_backprop.py
================================
Guided Backpropagation for Alzheimer MRI CNNs.
Produces fine-grained edge-level attribution maps.

Run:
    python 05_explainability_xai/02_guided_backprop.py --model resnet50 --framework tensorflow
    python 05_explainability_xai/02_guided_backprop.py --model all      --framework pytorch
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, CLASS_NAMES

OUTPUT_DIR = Path("results/05_explainability_xai/guided_backprop")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["alexnet", "vggnet", "googlenet", "resnet50"]


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
# TensorFlow Guided Backprop
# ─────────────────────────────────────────────
def guided_backprop_tf(model_name="resnet50", data_dir="data/test"):
    import tensorflow as tf

    @tf.custom_gradient
    def guided_relu(x):
        def grad(dy):
            return dy * tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32)
        return tf.nn.relu(x), grad

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
            print(f"  [TF-GBP] Model not found: {path}. Skipping.")
            continue

        print(f"  [TF-GBP] Guided Backprop for {name}...")
        model = tf.keras.models.load_model(path)

        # Patch ReLUs with guided ReLU
        for layer in model.layers:
            if hasattr(layer, "activation") and layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu

        saliency_maps = []
        for img, lbl in zip(images, labels):
            img_norm = ((img - mean) / std)[np.newaxis].astype(np.float32)
            img_tensor = tf.Variable(img_norm)
            with tf.GradientTape() as tape:
                preds = model(img_tensor, training=False)
                target_class = tf.argmax(preds[0]).numpy()
                score = preds[:, target_class]
            grads = tape.gradient(score, img_tensor)[0].numpy()
            saliency = np.abs(grads).max(axis=-1)
            saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
            saliency_maps.append((img, saliency, lbl, target_class))

        _plot_guided_bp(saliency_maps, name, "TF")
    print("  [TF-GBP] Done.")


# ─────────────────────────────────────────────
# PyTorch Guided Backprop
# ─────────────────────────────────────────────
def guided_backprop_pytorch(model_name="resnet50", data_dir="data/test"):
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

    # Custom guided ReLU
    class GuidedReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.clamp(min=0)
        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            return grad_output * (grad_output > 0).float() * (x > 0).float()

    def patch_relu(model):
        for name_mod, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        return model

    for name in names:
        if name not in model_configs:
            continue
        ckpt_path, arch = model_configs[name]
        if not Path(ckpt_path).exists():
            print(f"  [PT-GBP] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"  [PT-GBP] Guided Backprop for {name}...")

        if arch == "alexnet":
            m = models.alexnet(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
        elif arch == "vgg16":
            m = models.vgg16(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
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
        m = patch_relu(m).to(device).eval()

        saliency_maps = []
        for img, lbl in zip(images, labels):
            tensor = transform(img).unsqueeze(0).to(device).requires_grad_(True)
            out    = m(tensor)
            pred   = out.argmax(1).item()
            m.zero_grad()
            out[0, pred].backward()
            grads = tensor.grad.data.abs().squeeze().cpu().numpy()
            if grads.ndim == 3:
                saliency = grads.max(axis=0)
            else:
                saliency = grads
            saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
            saliency_maps.append((img, saliency, lbl, pred))

        _plot_guided_bp(saliency_maps, name, "PT")
    print("  [PT-GBP] Done.")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def _plot_guided_bp(saliency_maps, model_name, tag):
    n = len(saliency_maps)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.0))
    apply_dark_theme(fig, axes.flatten() if n > 1 else axes)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (img, saliency, true_lbl, pred_lbl) in enumerate(saliency_maps):
        true_name = CLASS_NAMES[true_lbl] if true_lbl < 4 else str(true_lbl)
        pred_name = CLASS_NAMES[pred_lbl] if pred_lbl < 4 else str(pred_lbl)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {true_name}", color="white", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(saliency, cmap="hot")
        axes[i, 1].set_title(f"Guided BP\nPred: {pred_name}", color="#FB923C", fontsize=8)
        axes[i, 1].axis("off")

        # Overlay
        overlay = img.copy()
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + saliency * 0.5, 0, 1)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay", color="#4ADE80", fontsize=8)
        axes[i, 2].axis("off")

    fig.suptitle(f"Guided Backpropagation — {model_name.upper()} [{tag}]",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"guided_bp_{model_name}_{tag.lower()}.png"))


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

    print(f"\nModule 5 — Guided Backprop | Model: {args.model} | Framework: {args.framework}")

    if args.framework in ("tensorflow", "both"):
        guided_backprop_tf(args.model, args.data_dir)

    if args.framework in ("pytorch", "both"):
        guided_backprop_pytorch(args.model, args.data_dir)

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
