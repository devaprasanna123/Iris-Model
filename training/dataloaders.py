"""PyTorch DataLoader factory functions for the MedicalAI project.

This module intentionally contains ONLY DataLoader creation helpers.
It does not implement models, inference, training loops, or augmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.datasets.oct_dataset import OctDataset


def _pin_memory_if_cuda_available() -> bool:
    """Return True when CUDA is available; otherwise False."""

    return bool(torch.cuda.is_available())


def create_train_loader(
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = True,
    pin_memory: bool | None = None,
) -> DataLoader[Any]:
    """Create the training DataLoader.

    Args:
        dataset_path: Root dataset directory. Expected layout:
            {dataset_path}/train/images and {dataset_path}/train/masks
        batch_size: Batch size.
        num_workers: Number of DataLoader worker processes.
        shuffle: Shuffle training data. Defaults to True.
        pin_memory: Whether to pin memory. If None, it is set automatically
            to True only when CUDA is available.

    Returns:
        Configured PyTorch DataLoader.
    """

    ds_root = Path(dataset_path)
    images_dir = ds_root / "train" / "images"
    masks_dir = ds_root / "train" / "masks"

    resolved_pin_memory = _pin_memory_if_cuda_available() if pin_memory is None else pin_memory

    dataset = OctDataset(images_dir=images_dir, masks_dir=masks_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
    )


def create_val_loader(
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = False,
    pin_memory: bool | None = None,
) -> DataLoader[Any]:
    """Create the validation DataLoader.

    Args:
        dataset_path: Root dataset directory. Expected layout:
            {dataset_path}/val/images and {dataset_path}/val/masks
        batch_size: Batch size.
        num_workers: Number of DataLoader worker processes.
        shuffle: Shuffle validation data. Defaults to False.
        pin_memory: Whether to pin memory. If None, it is set automatically
            to True only when CUDA is available.

    Returns:
        Configured PyTorch DataLoader.
    """

    ds_root = Path(dataset_path)
    images_dir = ds_root / "val" / "images"
    masks_dir = ds_root / "val" / "masks"

    resolved_pin_memory = _pin_memory_if_cuda_available() if pin_memory is None else pin_memory

    dataset = OctDataset(images_dir=images_dir, masks_dir=masks_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
    )


def create_test_loader(
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = False,
    pin_memory: bool | None = None,
) -> DataLoader[Any]:
    """Create the test DataLoader.

    Args:
        dataset_path: Root dataset directory. Expected layout:
            {dataset_path}/test/images and {dataset_path}/test/masks
        batch_size: Batch size.
        num_workers: Number of DataLoader worker processes.
        shuffle: Shuffle test data. Defaults to False.
        pin_memory: Whether to pin memory. If None, it is set automatically
            to True only when CUDA is available.

    Returns:
        Configured PyTorch DataLoader.
    """

    ds_root = Path(dataset_path)
    images_dir = ds_root / "test" / "images"
    masks_dir = ds_root / "test" / "masks"

    resolved_pin_memory = _pin_memory_if_cuda_available() if pin_memory is None else pin_memory

    dataset = OctDataset(images_dir=images_dir, masks_dir=masks_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
    )

