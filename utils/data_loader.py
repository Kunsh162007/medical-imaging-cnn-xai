"""
utils/data_loader.py
====================
Unified dataset loading for both TensorFlow and PyTorch.

Usage:
    from utils.data_loader import get_dataloader, get_class_names, get_dataset_stats

    # PyTorch
    train_loader = get_dataloader("data/train", framework="pytorch", batch_size=32)

    # TensorFlow
    train_ds = get_dataloader("data/train", framework="tensorflow", batch_size=32)
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union

# ─────────────────────────────────────────────
# Shared Constants
# ─────────────────────────────────────────────
IMG_SIZE = (128, 128)
NUM_CLASSES = 4
CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
CLASS_WEIGHTS = {0: 1.0, 1: 1.2, 2: 1.8, 3: 2.5}  # Adjust for class imbalance
SEED = 42


# ─────────────────────────────────────────────
# PyTorch Data Loader
# ─────────────────────────────────────────────
def _get_pytorch_loader(
    data_dir: str,
    batch_size: int = 32,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """Returns a PyTorch DataLoader with optional augmentation."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    if augment:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        
    )
    return loader


# ─────────────────────────────────────────────
# TensorFlow Data Loader
# ─────────────────────────────────────────────
def _get_tensorflow_dataset(
    data_dir: str,
    batch_size: int = 32,
    augment: bool = True,
    shuffle: bool = True,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
):
    """Returns a tf.data.Dataset with optional augmentation."""
    import tensorflow as tf

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        image = tf.image.resize(image, IMG_SIZE)
        return image, label

    def augment_fn(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.keras.layers.RandomRotation(factor=0.1)(image)
        image = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(image)
        return image, label

    kwargs = dict(
        directory=data_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        seed=SEED,
        shuffle=shuffle,
    )
    if validation_split is not None and subset is not None:
        kwargs["validation_split"] = validation_split
        kwargs["subset"] = subset

    ds = tf.keras.preprocessing.image_dataset_from_directory(**kwargs)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ─────────────────────────────────────────────
# Unified Entry Point
# ─────────────────────────────────────────────
def get_dataloader(
    data_dir: str,
    framework: str = "tensorflow",
    batch_size: int = 32,
    augment: bool = True,
    shuffle: bool = True,
    **kwargs,
):
    """
    Unified data loader entry point.

    Args:
        data_dir    : Path to image directory (ImageFolder structure expected).
        framework   : 'tensorflow' or 'pytorch'.
        batch_size  : Batch size.
        augment     : Apply augmentation transforms.
        shuffle     : Shuffle dataset.
        **kwargs    : Extra args forwarded to framework-specific loader.

    Returns:
        DataLoader (PyTorch) or tf.data.Dataset (TensorFlow).
    """
    framework = framework.lower().strip()
    if framework in ("tensorflow", "tf", "keras"):
        return _get_tensorflow_dataset(data_dir, batch_size, augment, shuffle, **kwargs)
    elif framework in ("pytorch", "torch"):
        return _get_pytorch_loader(data_dir, batch_size, augment, shuffle, **kwargs)
    else:
        raise ValueError(f"Unknown framework '{framework}'. Choose 'tensorflow' or 'pytorch'.")


def get_class_names() -> list:
    return CLASS_NAMES


def get_dataset_stats(data_dir: str) -> dict:
    """Count images per class in a directory."""
    stats = {}
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = Path(data_dir) / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
            stats[cls] = count
            total += count
        else:
            stats[cls] = 0
    stats["total"] = total
    return stats


def split_train_val(data_dir: str, val_ratio: float = 0.1, framework: str = "tensorflow", batch_size: int = 32):
    """Returns (train_loader, val_loader) split from a single directory."""
    if framework in ("tensorflow", "tf", "keras"):
        train_ds = _get_tensorflow_dataset(
            data_dir, batch_size, augment=True, shuffle=True,
            validation_split=val_ratio, subset="training"
        )
        val_ds = _get_tensorflow_dataset(
            data_dir, batch_size, augment=False, shuffle=False,
            validation_split=val_ratio, subset="validation"
        )
        return train_ds, val_ds
    else:
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, random_split

        base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        full_ds = datasets.ImageFolder(root=data_dir, transform=base_transform)
        val_size = int(len(full_ds) * val_ratio)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader, val_loader
