"""
05_explainability_xai/05_occlusion_sensitivity.py
=======================================
Occlusion Sensitivity Analysis — slides a patch across the MRI image,
blocking regions and measuring the drop in predicted class probability.

Framework-agnostic: works with TF and PyTorch models.

Run:
    python 05_explainability_xai/05_occlusion_sensitivity.py --model resnet50 --framework tensorflow
    python 05_explainability_xai/05_occlusion_sensitivity.py --model all      --framework both
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, CLASS_NAMES

OUTPUT_DIR = Path("results/05_explainability_xai/occlusion")
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
# Core Occlusion Algorithm (framework-agnostic)
# ─────────────────────────────────────────────
def compute_occlusion_map(image: np.ndarray, predict_fn, target_class: int,
                           patch_size: int = 16, stride: int = 8) -> np.ndarray:
    """
    Slide a grey patch across the image and record the target class prob drop.

    Args:
        image       : HWC float32 array [0,1]
        predict_fn  : callable(batch_of_images) -> probabilities [N, num_classes]
        target_class: class index to track
        patch_size  : size of occluding patch
        stride      : stride for sliding window

    Returns:
        Heatmap of same spatial size as image (H, W).
    """
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts  = np.zeros((H, W), dtype=np.float32)

    # Baseline probability
    baseline_prob = predict_fn(image[np.newaxis])[0, target_class]

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = image.copy()
            occluded[y:y+patch_size, x:x+patch_size] = 0.5  # grey patch
            prob = predict_fn(occluded[np.newaxis])[0, target_class]
            drop = baseline_prob - prob
            heatmap[y:y+patch_size, x:x+patch_size] += drop
            counts[y:y+patch_size, x:x+patch_size]  += 1

    counts[counts == 0] = 1
    heatmap /= counts
    return heatmap


# ─────────────────────────────────────────────
# TensorFlow
# ─────────────────────────────────────────────
def occlusion_tensorflow(model_name="resnet50", data_dir="data/test",
                          patch_size=16, stride=8):
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
            print(f"  [TF-OCC] Model not found: {path}. Skipping.")
            continue

        print(f"  [TF-OCC] Occlusion for {name} (patch={patch_size}, stride={stride})...")
        model = tf.keras.models.load_model(path)

        def predict_fn(imgs):
            imgs_norm = (imgs - mean) / std
            return model.predict(imgs_norm, verbose=0)

        for img, lbl in zip(images, labels):
            target = int(model.predict((img[np.newaxis] - mean) / std,
                                       verbose=0).argmax())
            hmap = compute_occlusion_map(img, predict_fn, target, patch_size, stride)
            _plot_occlusion(img, hmap, lbl, target, name, "TF")

    print("  [TF-OCC] Done.")


# ─────────────────────────────────────────────
# PyTorch
# ─────────────────────────────────────────────
def occlusion_pytorch(model_name="resnet50", data_dir="data/test",
                       patch_size=16, stride=8):
    import torch
    from torchvision import models, transforms
    from utils.model_utils import get_device

    device  = get_device()
    names   = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
            print(f"  [PT-OCC] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"  [PT-OCC] Occlusion for {name} (patch={patch_size}, stride={stride})...")

        if arch == "alexnet":
            import torch.nn as nn
            m = models.alexnet(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
        elif arch == "vgg16":
            import torch.nn as nn
            m = models.vgg16(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
        elif arch == "inception_v3":
            import torch.nn as nn
            m = models.inception_v3(weights=None, aux_logits=False)
            m.fc = torch.nn.Sequential(
                    torch.nn.Linear(m.fc.in_features, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(512, 4)
                    )
        elif arch == "resnet50":
            import torch.nn as nn
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
            # imgs: [N, H, W, C] float32
            tensors = torch.tensor(imgs.transpose(0, 3, 1, 2), dtype=torch.float32)
            tensors = torch.stack([normalize(t) for t in tensors]).to(device)
            with torch.no_grad():
                return torch.softmax(m(tensors), 1).cpu().numpy()

        for img, lbl in zip(images, labels):
            target = int(predict_fn(img[np.newaxis]).argmax())
            hmap   = compute_occlusion_map(img, predict_fn, target, patch_size, stride)
            _plot_occlusion(img, hmap, lbl, target, name, "PT")

    print("  [PT-OCC] Done.")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def _plot_occlusion(image, heatmap, true_lbl, pred_lbl, model_name, tag):
    import cv2
    true_name = CLASS_NAMES[true_lbl] if true_lbl < 4 else str(true_lbl)
    pred_name = CLASS_NAMES[pred_lbl] if pred_lbl < 4 else str(pred_lbl)

    heatmap_norm = ((heatmap - heatmap.min()) /
                    (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (image * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.6, heatmap_color, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    apply_dark_theme(fig, axes)

    axes[0].imshow(image)
    axes[0].set_title(f"Original\nTrue: {true_name}", color="white", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="hot")
    axes[1].set_title(f"Occlusion Sensitivity\nPred: {pred_name}", color="#FB923C", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", color="#4ADE80", fontsize=9)
    axes[2].axis("off")

    fig.suptitle(f"Occlusion Sensitivity — {model_name.upper()} [{tag}]",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"occlusion_{model_name}_{tag.lower()}_{true_lbl}.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="resnet50", choices=MODEL_NAMES + ["all"])
    parser.add_argument("--framework",  default="tensorflow",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--data_dir",   default="data/test")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride",     type=int, default=8)
    args = parser.parse_args()

    print(f"\nModule 5 — Occlusion Sensitivity | Model: {args.model} | "
          f"Framework: {args.framework} | Patch: {args.patch_size} | Stride: {args.stride}")

    if args.framework in ("tensorflow", "both"):
        occlusion_tensorflow(args.model, args.data_dir, args.patch_size, args.stride)

    if args.framework in ("pytorch", "both"):
        occlusion_pytorch(args.model, args.data_dir, args.patch_size, args.stride)

    print(f"\nOcclusion maps saved to: {OUTPUT_DIR}")
