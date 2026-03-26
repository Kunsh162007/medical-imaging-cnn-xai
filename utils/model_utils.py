"""
utils/model_utils.py
====================
Framework-agnostic model saving, loading, summarization, and parameter counting.
"""

import os
import json
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# TensorFlow Helpers
# ─────────────────────────────────────────────
def save_tf_model(model, path: str, save_format: str = "h5"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"[TF] Model saved → {path}")


def load_tf_model(path: str):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    print(f"[TF] Model loaded ← {path}")
    return model


def count_tf_params(model) -> dict:
    total    = model.count_params()
    trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    frozen   = total - trainable
    return {"total": total, "trainable": int(trainable), "frozen": int(frozen)}


def tf_model_summary(model, save_path: str = None):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    summary_str = "\n".join(lines)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(summary_str)
        print(f"[TF] Summary saved → {save_path}")
    else:
        print(summary_str)
    return summary_str


# ─────────────────────────────────────────────
# PyTorch Helpers
# ─────────────────────────────────────────────
def save_torch_model(model, path: str, extra: dict = None):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[PyTorch] Model saved → {path}")


def load_torch_model(model, path: str):
    import torch
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    print(f"[PyTorch] Weights loaded ← {path}")
    return model


def count_torch_params(model) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def torch_model_summary(model, input_size=(1, 3, 128, 128), save_path: str = None):
    """Print torchinfo summary (falls back to str(model) if torchinfo not installed)."""
    try:
        from torchinfo import summary
        s = summary(model, input_size=input_size, verbose=0)
        summary_str = str(s)
    except ImportError:
        summary_str = str(model)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(summary_str)
        print(f"[PyTorch] Summary saved → {save_path}")
    else:
        print(summary_str)
    return summary_str


# ─────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────
def save_results_json(results: dict, path: str):
    """Save experiment results dict as JSON."""
    def _convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} not JSON serializable")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"[results] Saved → {path}")


def load_results_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_device():
    """Return best available device string for PyTorch."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
