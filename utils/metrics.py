"""
utils/metrics.py
================
Evaluation utilities shared across all experiments.

Usage:
    from utils.metrics import evaluate_model, print_report, compute_auc
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score
)

CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray = None) -> dict:
    """
    Compute full evaluation metrics.

    Args:
        y_true  : Ground truth labels (int array).
        y_pred  : Predicted labels (int array).
        y_proba : Predicted probabilities (N x num_classes), optional for AUC.

    Returns:
        dict with accuracy, f1_macro, f1_weighted, kappa, auc, confusion_matrix.
    """
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            pass

    results = {
        "accuracy":     round(acc, 4),
        "f1_macro":     round(f1_macro, 4),
        "f1_weighted":  round(f1_weighted, 4),
        "kappa":        round(kappa, 4),
        "confusion_matrix": cm,
    }
    if auc is not None:
        results["auc"] = round(auc, 4)

    return results


def print_report(y_true: np.ndarray, y_pred: np.ndarray,
                 model_name: str = "Model"):
    """Print a formatted classification report."""
    print(f"\n{'='*60}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))


def compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute macro one-vs-rest AUC."""
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return per-class precision, recall, F1."""
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES,
        zero_division=0, output_dict=True
    )
    return {cls: report[cls] for cls in CLASS_NAMES}


def compare_models(results_dict: dict) -> None:
    """Print a comparison table for multiple models.

    results_dict = {
        "AlexNet-TF": {"accuracy": 0.886, "f1_macro": 0.883, "auc": 0.971},
        ...
    }
    """
    header = f"{'Model':<22} {'Accuracy':>9} {'F1 Macro':>10} {'F1 Weighted':>13} {'AUC':>8}"
    print(f"\n{'='*70}")
    print("  Model Comparison Summary")
    print(f"{'='*70}")
    print(header)
    print("-" * 70)
    for name, m in results_dict.items():
        auc_str = f"{m['auc']:.4f}" if "auc" in m else "  N/A  "
        print(f"{name:<22} {m['accuracy']:>9.4f} {m['f1_macro']:>10.4f} "
              f"{m.get('f1_weighted', 0):>13.4f} {auc_str:>8}")
    print("=" * 70)
