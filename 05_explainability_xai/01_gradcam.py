"""
05_explainability_xai/01_gradcam.py
========================
Grad-CAM (Gradient-weighted Class Activation Mapping) applied to all CNN
architectures on the Alzheimer MRI dataset.

Supports TensorFlow (via tf-keras-vis) and PyTorch (via pytorch-grad-cam).

Run:
    python 05_explainability_xai/01_gradcam.py --model resnet50 --framework tensorflow
    python 05_explainability_xai/01_gradcam.py --model all     --framework pytorch
    python 05_explainability_xai/01_gradcam.py --model all     --framework both
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, plot_xai_comparison, CLASS_NAMES

OUTPUT_DIR = Path("results/05_explainability_xai/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["alexnet", "vggnet", "googlenet", "resnet50"]


# ─────────────────────────────────────────────
# Shared: Load & Preprocess Sample Images
# ─────────────────────────────────────────────
def load_sample_images(data_dir="data/test", n_per_class=2):
    """Load n_per_class images from each class folder."""
    from PIL import Image
    images, labels = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.exists():
            print(f"  [warn] {cls_dir} not found, skipping")
            continue
        files = sorted(cls_dir.glob("*.jpg"))[:n_per_class]
        for f in files:
            img = Image.open(f).convert("RGB").resize((128, 128))
            images.append(np.array(img))
            labels.append(cls_idx)
    return np.array(images), np.array(labels)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0, 255] uint8."""
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return (heatmap * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha=0.4) -> np.ndarray:
    """Overlay jet colormap heatmap on original image."""
    heatmap_color = cv2.applyColorMap(cv2.resize(heatmap, (128, 128)), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)


# ─────────────────────────────────────────────
# TensorFlow Grad-CAM
# ─────────────────────────────────────────────
def gradcam_tensorflow(model_name="resnet50", data_dir="data/test"):
    import tensorflow as tf

    target_layers = {
        "alexnet":  None,          # will use last conv layer
        "vggnet":   "block5_conv3",
        "googlenet":"mixed10",
        "resnet50": "conv5_block3_out",
    }

    model_paths = {
        "alexnet":  "results/04_advanced_architectures/alexnet_tf.h5",
        "vggnet":   "results/04_advanced_architectures/vggnet_tf.h5",
        "googlenet":"results/04_advanced_architectures/googlenet_tf.h5",
        "resnet50": "results/04_advanced_architectures/resnet50_tf_final.h5",
    }

    names = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_sample_images(data_dir, n_per_class=1)

    if len(images) == 0:
        print("[warn] No test images found. Generating random samples for demo.")
        images = (np.random.rand(4, 128, 128, 3) * 255).astype(np.uint8)
        labels = np.array([0, 1, 2, 3])

    all_heatmaps = {}

    for name in names:
        path = model_paths.get(name)
        if not Path(path).exists():
            print(f"  [TF] Model not found: {path}. Skipping.")
            continue

        print(f"  [TF] Grad-CAM for {name}...")
        model = tf.keras.models.load_model(path)

        # Get last conv layer
        layer_name = target_layers.get(name)
        if layer_name is None:
            conv_layers = [l.name for l in model.layers
                           if isinstance(l, tf.keras.layers.Conv2D)]
            layer_name = conv_layers[-1]

        # Build Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        heatmaps = []
        for img, lbl in zip(images, labels):
            img_tensor = tf.cast(img[np.newaxis], tf.float32) / 255.0

            with tf.GradientTape() as tape:
                conv_out, predictions = grad_model(img_tensor)
                pred_class = tf.argmax(predictions[0]).numpy()
                class_score = predictions[:, pred_class]

            grads = tape.gradient(class_score, conv_out)
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1).numpy()
            cam = normalize_heatmap(cam)
            overlay = overlay_heatmap(img, cam)
            heatmaps.append(overlay)
            all_heatmaps[name.capitalize()] = cam

        _plot_gradcam_grid(images, heatmaps, labels, name, "TF")

    return all_heatmaps


# ─────────────────────────────────────────────
# PyTorch Grad-CAM
# ─────────────────────────────────────────────
def gradcam_pytorch(model_name="resnet50", data_dir="data/test"):
    import torch
    from torchvision import models, transforms
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from utils.model_utils import get_device

    device = get_device()
    names  = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_sample_images(data_dir, n_per_class=1)

    if len(images) == 0:
        print("[warn] No test images found. Generating random samples for demo.")
        images = (np.random.rand(4, 128, 128, 3) * 255).astype(np.uint8)
        labels = np.array([0, 1, 2, 3])

    model_configs = {
        "alexnet":  (models.alexnet,     "features.10", "results/04_advanced_architectures/alexnet_pt_best.pt"),
        "vggnet":   (models.vgg16,       "features.28", "results/04_advanced_architectures/vggnet_pt_best.pt"),
        "googlenet":(models.inception_v3,"Mixed_7c",    "results/04_advanced_architectures/googlenet_pt_best.pt"),
        "resnet50": (models.resnet50,    "layer4",      "results/04_advanced_architectures/resnet50_pt_best.pt"),
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_heatmaps = {}

    for name in names:
        if name not in model_configs:
            continue
        arch_fn, layer_str, ckpt_path = model_configs[name]

        if not Path(ckpt_path).exists():
            print(f"  [PyTorch] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"  [PyTorch] Grad-CAM for {name}...")
        if name == "googlenet":
            model = arch_fn(weights=None, aux_logits=False)
            model.fc = torch.nn.Sequential(
                        torch.nn.Linear(model.fc.in_features, 512),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.4),
                        torch.nn.Linear(512, 4)
                        )
        else:
            model = arch_fn(weights=None)
            if name in ("alexnet", "vggnet"):
                model.classifier[-1] = torch.nn.Linear(4096, 4)
            elif name == "resnet50":
                model.fc = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(model.fc.in_features),
                            torch.nn.Linear(model.fc.in_features, 512),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(0.4),
                            torch.nn.Linear(512, 4)
                            )

        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model = model.to(device).eval()

        # Get target layer
        target_layer = None
        parts = layer_str.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        target_layer = obj

        cam = GradCAM(model=model, target_layers=[target_layer])

        heatmaps = []
        for img, lbl in zip(images, labels):
            img_f32 = img.astype(np.float32) / 255.0
            tensor  = transform(img).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=tensor)[0]
            overlay = show_cam_on_image(img_f32, grayscale_cam, use_rgb=True)
            heatmaps.append(overlay)
            all_heatmaps[name.capitalize()] = grayscale_cam

        _plot_gradcam_grid(images, heatmaps, labels, name, "PT")

    return all_heatmaps


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def _plot_gradcam_grid(images, overlays, labels, model_name, tag):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 3.5, 7))
    apply_dark_theme(fig, axes.flatten())

    for i in range(n):
        cls_name = CLASS_NAMES[labels[i]] if labels[i] < 4 else str(labels[i])
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"Original\n{cls_name}", color="white", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlays[i])
        axes[1, i].set_title("Grad-CAM", color="#58D9F9", fontsize=8, fontweight="bold")
        axes[1, i].axis("off")

    fig.suptitle(f"Grad-CAM — {model_name.upper()} [{tag}]",
                 fontsize=13, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"gradcam_{model_name}_{tag.lower()}.png"))


def plot_cross_model_gradcam(heatmaps_tf: dict, heatmaps_pt: dict, image: np.ndarray, label: str):
    """Side-by-side Grad-CAM from TF and PyTorch for the same image."""
    all_heatmaps = {}
    for k, v in heatmaps_tf.items():
        all_heatmaps[f"{k}-TF"] = v
    for k, v in heatmaps_pt.items():
        all_heatmaps[f"{k}-PT"] = v

    plot_xai_comparison(image, all_heatmaps, true_label=label,
                        save_path=str(OUTPUT_DIR / "gradcam_cross_model.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM XAI")
    parser.add_argument("--model",     default="resnet50",
                        choices=MODEL_NAMES + ["all"])
    parser.add_argument("--framework", default="tensorflow",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--data_dir",  default="data/test")
    args = parser.parse_args()

    print(f"\nModule 5 — Grad-CAM | Model: {args.model} | Framework: {args.framework}")

    heatmaps_tf = {}
    heatmaps_pt = {}

    if args.framework in ("tensorflow", "both"):
        heatmaps_tf = gradcam_tensorflow(args.model, args.data_dir)

    if args.framework in ("pytorch", "both"):
        heatmaps_pt = gradcam_pytorch(args.model, args.data_dir)

    if args.framework == "both" and heatmaps_tf and heatmaps_pt:
        images, labels = load_sample_images(args.data_dir, n_per_class=1)
        if len(images) > 0:
            plot_cross_model_gradcam(heatmaps_tf, heatmaps_pt,
                                     images[0], CLASS_NAMES[labels[0]])

    print(f"\nGrad-CAM visualizations saved to: {OUTPUT_DIR}")
