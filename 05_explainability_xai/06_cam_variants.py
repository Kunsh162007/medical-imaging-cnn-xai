"""
05_explainability_xai/06_cam_variants.py
=============================
CAM Variants: Score-CAM, Eigen-CAM, HiRes-CAM applied to all architectures.
Uses pytorch-grad-cam library for PyTorch; custom implementations for TF.

Run:
    python 05_explainability_xai/06_cam_variants.py --model resnet50 --framework pytorch
    python 05_explainability_xai/06_cam_variants.py --model all      --framework both
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "..")
from pathlib import Path
from utils.visualization import apply_dark_theme, _save_or_show, CLASS_NAMES

OUTPUT_DIR = Path("results/05_explainability_xai/cam_variants")
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
# PyTorch — Score-CAM, Eigen-CAM, HiRes-CAM
# ─────────────────────────────────────────────
def cam_variants_pytorch(model_name="resnet50", data_dir="data/test"):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from pytorch_grad_cam import ScoreCAM, EigenCAM, HiResCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from utils.model_utils import get_device

    device  = get_device()
    names   = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model_configs = {
        "alexnet":   ("results/04_advanced_architectures/alexnet_pt_best.pt",   "features.10"),
        "vggnet":    ("results/04_advanced_architectures/vggnet_pt_best.pt",    "features.28"),
        "googlenet": ("results/04_advanced_architectures/googlenet_pt_best.pt", "Mixed_7c"),
        "resnet50":  ("results/04_advanced_architectures/resnet50_pt_best.pt",  "layer4"),
    }

    cam_classes = {
        "Score-CAM":     ScoreCAM,
        "Eigen-CAM":     EigenCAM,
        "HiRes-CAM":     HiResCAM,
        "GradCAM++":     GradCAMPlusPlus,
    }

    for name in names:
        if name not in model_configs:
            continue
        ckpt_path, layer_str = model_configs[name]
        if not Path(ckpt_path).exists():
            print(f"  [PT-CAM] Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"  [PT-CAM] CAM variants for {name}...")

        if "alexnet" in name:
            m = models.alexnet(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
        elif "vgg" in name:
            m = models.vgg16(weights=None)
            m.classifier[-1] = nn.Linear(4096, 4)
        elif "google" in name:
            m = models.inception_v3(weights=None, aux_logits=False)
            m.fc = nn.Sequential(
                    torch.nn.Linear(m.fc.in_features, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(512, 4)
                    )
        else:
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

        # Get target layer
        parts = layer_str.split(".")
        obj = m
        for p in parts:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        target_layer = [obj]

        for img, lbl in zip(images[:2], labels[:2]):  # 2 examples per model
            img_f32 = img
            tensor  = transform(img).unsqueeze(0).to(device)
            true_name = CLASS_NAMES[lbl] if lbl < 4 else str(lbl)

            cams_dict = {}
            for cam_name, cam_cls in cam_classes.items():
                try:
                    cam_obj = cam_cls(model=m, target_layers=target_layer)
                    grayscale_cam = cam_obj(input_tensor=tensor)[0]
                    overlay = show_cam_on_image(img_f32, grayscale_cam, use_rgb=True)
                    cams_dict[cam_name] = overlay
                except Exception as e:
                    print(f"    [{cam_name}] error: {e}")

            if cams_dict:
                _plot_cam_variants(img_f32, cams_dict, true_name, name, "PT")

    print("  [PT-CAM] Done.")


# ─────────────────────────────────────────────
# TensorFlow — Score-CAM (custom implementation)
# ─────────────────────────────────────────────
def cam_variants_tensorflow(model_name="resnet50", data_dir="data/test"):
    import tensorflow as tf

    model_paths = {
        "alexnet":   "results/04_advanced_architectures/alexnet_tf.h5",
        "vggnet":    "results/04_advanced_architectures/vggnet_tf.h5",
        "googlenet": "results/04_advanced_architectures/googlenet_tf.h5",
        "resnet50":  "results/04_advanced_architectures/resnet50_tf_final.h5",
    }
    target_layers = {
        "alexnet":   None,
        "vggnet":    "block5_conv3",
        "googlenet": "mixed10",
        "resnet50":  "conv5_block3_out",
    }
    names = MODEL_NAMES if model_name == "all" else [model_name]
    images, labels = load_samples(data_dir)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)

    for name in names:
        path = model_paths.get(name, "")
        if not Path(path).exists():
            print(f"  [TF-CAM] Model not found: {path}. Skipping.")
            continue

        print(f"  [TF-CAM] Score-CAM (TF) for {name}...")
        model = tf.keras.models.load_model(path)

        layer_name = target_layers.get(name)
        if layer_name is None:
            conv_layers = [l.name for l in model.layers
                           if isinstance(l, tf.keras.layers.Conv2D)]
            layer_name  = conv_layers[-1]

        feat_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )

        for img, lbl in zip(images[:2], labels[:2]):
            img_norm = ((img - mean) / std)[np.newaxis].astype(np.float32)
            feature_maps = feat_model(img_norm)[0].numpy()  # H' x W' x C

            # Score-CAM: weight each channel by its masked prediction score
            n_channels = feature_maps.shape[-1]
            weights    = np.zeros(n_channels)
            for c in range(n_channels):
                fm = feature_maps[:, :, c]
                fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                fm_resized = tf.image.resize(fm_norm[:, :, np.newaxis], (128, 128)).numpy()[:, :, 0]
                masked_img = img_norm[0] * fm_resized[:, :, np.newaxis]
                pred = model.predict(masked_img[np.newaxis], verbose=0)[0]
                weights[c] = pred[lbl]

            cam = np.sum(feature_maps * weights, axis=-1)
            cam = np.maximum(cam, 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            import cv2
            cam_resized  = cv2.resize(cam, (128, 128))
            cam_color    = cv2.applyColorMap((cam_resized * 255).astype(np.uint8),
                                              cv2.COLORMAP_JET)
            cam_color    = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
            overlay      = cv2.addWeighted((img * 255).astype(np.uint8),
                                           0.6, cam_color, 0.4, 0)

            true_name = CLASS_NAMES[lbl] if lbl < 4 else str(lbl)
            _plot_cam_variants(img, {"Score-CAM (TF)": overlay}, true_name, name, "TF")

    print("  [TF-CAM] Done.")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def _plot_cam_variants(original, cams_dict, true_name, model_name, tag):
    n = len(cams_dict)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 3.5, 4))
    apply_dark_theme(fig, axes)

    axes[0].imshow(original)
    axes[0].set_title(f"Original\n{true_name}", color="white", fontsize=8)
    axes[0].axis("off")

    cam_colors = ["#58D9F9", "#FB923C", "#4ADE80", "#C084FC"]
    for idx, (cam_name, cam_img) in enumerate(cams_dict.items()):
        axes[idx + 1].imshow(cam_img)
        axes[idx + 1].set_title(cam_name, color=cam_colors[idx % 4],
                                 fontsize=9, fontweight="bold")
        axes[idx + 1].axis("off")

    fig.suptitle(f"CAM Variants — {model_name.upper()} [{tag}]",
                 fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, str(OUTPUT_DIR / f"cam_{model_name}_{tag.lower()}_{true_name[:8]}.png"))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="resnet50", choices=MODEL_NAMES + ["all"])
    parser.add_argument("--framework", default="pytorch",
                        choices=["tensorflow", "pytorch", "both"])
    parser.add_argument("--data_dir",  default="data/test")
    args = parser.parse_args()

    print(f"\nModule 5 — CAM Variants | Model: {args.model} | Framework: {args.framework}")

    if args.framework in ("pytorch", "both"):
        cam_variants_pytorch(args.model, args.data_dir)

    if args.framework in ("tensorflow", "both"):
        cam_variants_tensorflow(args.model, args.data_dir)

    print(f"\nCAM outputs saved to: {OUTPUT_DIR}")
