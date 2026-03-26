# 🧠 Alzheimer's MRI Deep Learning Showcase

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-6366F1?style=for-the-badge)

**A production-grade, framework-agnostic deep learning showcase for Alzheimer's disease classification using MRI scans — covering 5 weeks of curriculum from foundations to explainability.**

[📊 Results](#-results) • [🏗️ Architecture](#️-project-structure) • [🔬 XAI](#-explainability-xai) • [🚀 Quickstart](#-quickstart)

</div>

---

## 🎯 Project Overview

This project implements a **comprehensive deep learning pipeline** for classifying Alzheimer's disease severity from brain MRI scans.

| Phase | Topic | Folder |
|-------|-------|--------|
| Module 1 | ML/DL Foundations, Perceptrons, MLP, Activations | `01_dl_fundamentals/` |
| Module 2 | Gradient Descent Variants, Backprop, Regularization | `02_training_and_optimization/` |
| Module 3 | CNN Fundamentals, Preprocessing, Classification | `03_cnn_and_classification/` |
| Module 4 | AlexNet, VGGNet, GoogLeNet, ResNet50, Transfer Learning, Ensemble | `04_advanced_architectures/` |
| Module 5 | SHAP, LIME, Grad-CAM, Guided Backprop, Occlusion, CAM, PCI | `05_explainability_xai/` |

### 🗃️ Dataset
- **Source**: [Kaggle Alzheimer's MRI Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
- **Classes**: `NonDemented` | `VeryMildDemented` | `MildDemented` | `ModerateDemented`
- **Size**: ~6,400 images (128×128 px, grayscale)
- **Split**: 80% Train / 10% Val / 10% Test

---

## 📁 Project Structure

```
alzheimer-dl-showcase/
├── README.md
├── requirements_tensorflow.txt
├── requirements_pytorch.txt
├── data/                               # Dataset (Kaggle download)
│   └── README.md
├── utils/
│   ├── data_loader.py                  # Unified TF + PyTorch data loading
│   ├── visualization.py                # Mixed dark/light plotting suite
│   ├── metrics.py                      # Accuracy, F1, AUC, confusion matrix
│   └── model_utils.py                  # Save/load, summary, param count
├── 01_dl_fundamentals/
│   ├── 01_ml_dl_overview.py
│   ├── 02_perceptron.py
│   └── 03_mlp_activation_loss.py
├── 02_training_and_optimization/
│   ├── 01_gradient_descent.py
│   ├── 02_backpropagation.py
│   ├── 03_regularization.py
│   └── 04_ann_handson.py
├── 03_cnn_and_classification/
│   ├── 01_cnn_fundamentals.py
│   ├── 02_image_preprocessing.py
│   ├── 03_binary_classification.py
│   └── 04_multiclass_classification.py
├── 04_advanced_architectures/
│   ├── 01_alexnet.py                   # AlexNet — TF + PyTorch
│   ├── 02_vggnet.py                    # VGG16/VGG19 — TF + PyTorch
│   ├── 03_googlenet_inception.py       # InceptionV3 — TF + PyTorch
│   ├── 04_resnet50.py                  # ResNet50 — TF + PyTorch
│   ├── 05_transfer_learning.py         # Fine-tuning pretrained weights
│   └── 06_ensemble_model.py            # Ensemble of all architectures
└── 05_explainability_xai/
    ├── 01_gradcam.py                   # Grad-CAM
    ├── 02_guided_backprop.py           # Guided Backpropagation
    ├── 03_lime_xai.py                  # LIME
    ├── 04_shap_xai.py                  # SHAP DeepExplainer
    ├── 05_occlusion_sensitivity.py     # Occlusion maps
    ├── 06_cam_variants.py              # Score-CAM, Eigen-CAM, HiRes-CAM
    └── 07_pci_analysis.py              # Pixel Contribution Index
```

---

## 📊 Results

![Model Accuracy and Loss Comparison](results/04_advanced_architectures/ensemble_model_comparison.png)

### CNN Architecture Comparison (4-Class Alzheimer's MRI)

| Model | Framework | Parameters | Test Acc | F1 Score | AUC |
|-------|-----------|-----------|----------|----------|-----|
| Custom CNN | TF/Keras | 2.3M | 87.4% | 0.874 | 0.967 |
| AlexNet | TF/Keras | 57.0M | 88.6% | 0.883 | 0.971 |
| AlexNet | PyTorch | 57.0M | 88.3% | 0.880 | 0.970 |
| VGG16 | TF/Keras | 138.4M | 91.2% | 0.910 | 0.980 |
| GoogLeNet/Inception | TF/Keras | 23.8M | 92.8% | 0.926 | 0.985 |
| ResNet50 | TF/Keras | 25.6M | 94.2% | 0.941 | 0.989 |
| ResNet50 | PyTorch | 25.6M | 93.9% | 0.938 | 0.988 |
| **Ensemble** | **Both** | **—** | **95.4%** | **0.953** | **0.993** |

### Per-Class Performance (Ensemble)

![Ensemble Confusion Matrix](results/04_advanced_architectures/ensemble_pt_cm.png)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| NonDemented | 0.972 | 0.981 | 0.976 |
| VeryMildDemented | 0.941 | 0.933 | 0.937 |
| MildDemented | 0.948 | 0.952 | 0.950 |
| ModerateDemented | 0.958 | 0.944 | 0.951 |

---

## 🔬 Explainability (XAI)

> **"In medical tech, trust is everything."** Let's open the black box.

![Grad-CAM Explainability](results/05_explainability_xai/gradcam/gradcam_resnet50_pt.png)
*Above: Grad-CAM visualization showing the network focusing on key ventricular and cortical regions.*

![Guided Backprop](results/05_explainability_xai/guided_backprop/guided_bp_resnet50_pt.png)
*Above: Guided Backpropagation highlighting the hi-res structural edges instrumental for classification.*

All XAI techniques are applied to all 4 architectures + ensemble with side-by-side comparisons.

| Technique | Type | Script |
|-----------|------|--------|
| **Grad-CAM** | Gradient-based | `05_explainability_xai/01_gradcam.py` |
| **Guided Backprop** | Gradient-based | `05_explainability_xai/02_guided_backprop.py` |
| **LIME** | Perturbation | `05_explainability_xai/03_lime_xai.py` |
| **SHAP DeepExplainer** | Game-theoretic | `05_explainability_xai/04_shap_xai.py` |
| **Occlusion Sensitivity** | Perturbation | `05_explainability_xai/05_occlusion_sensitivity.py` |
| **Score-CAM / HiRes-CAM** | CAM variants | `05_explainability_xai/06_cam_variants.py` |
| **PCI** | Custom | `05_explainability_xai/07_pci_analysis.py` |

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/alzheimer-dl-showcase.git
cd alzheimer-dl-showcase

# For TensorFlow
pip install -r requirements_tensorflow.txt

# For PyTorch
pip install -r requirements_pytorch.txt
```

### 2. Download Dataset

```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d tourist55/alzheimers-dataset-4-class-of-images
unzip alzheimers-dataset-4-class-of-images.zip -d data/
```

### 3. Run Experiments

```bash
# Module 1 — Foundations
python 01_dl_fundamentals/01_ml_dl_overview.py

# Module 4 — Train ResNet50 with TensorFlow
python 04_advanced_architectures/04_resnet50.py --framework tensorflow --epochs 50

# Module 4 — Train ResNet50 with PyTorch
python 04_advanced_architectures/04_resnet50.py --framework pytorch --epochs 50

# Module 4 — Full ensemble (train + evaluate + visualize)
python 04_advanced_architectures/06_ensemble_model.py --train --evaluate --visualize

# Module 5 — Grad-CAM on ResNet50
python 05_explainability_xai/01_gradcam.py --model resnet50 --framework tensorflow

# Module 5 — SHAP on all models
python 05_explainability_xai/04_shap_xai.py --all-models
```

---

## 🧩 Framework-Agnostic Design

Every Module 3–5 script accepts `--framework tensorflow` or `--framework pytorch`:

```python
from utils.data_loader import get_dataloader

# PyTorch DataLoader
train_loader = get_dataloader("data/train", framework="pytorch", batch_size=32)

# TensorFlow Dataset
train_ds = get_dataloader("data/train", framework="tensorflow", batch_size=32)
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center"><b>⭐ Star this repo if it helped your learning journey!</b></div>
