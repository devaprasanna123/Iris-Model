"""PyTorch DataLoader factory functions for the MedicalAI project.

This module intentionally contains ONLY DataLoader creation helpers.
It does not implement models, inference, training loops, or augmentation.

Dataset pipeline v2 integration:
- Passes mode (train/val/test) and deterministic preprocessing/augmentation
  config from TrainingConfig into OctDataset.

Backward compatibility:
- If TrainingConfig is not provided, defaults are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.datasets.oct_dataset import OctDataset


def _pin_memory_if_cuda_available() -> bool:
    """Return True when CUDA is available; otherwise False."""

    return bool(torch.cuda.is_available())


def _build_oct_dataset(
    *,
    images_dir: Path,
    masks_dir: Path,
    mode: str,
    cfg: TrainingConfig,
) -> OctDataset:
    preprocess_cfg = {
        "resize_hw": (int(cfg.preprocess.image_height), int(cfg.preprocess.image_width)),
        "normalization_mode": str(cfg.preprocess.normalization_mode),
        "clahe_enabled": bool(cfg.preprocess.clahe_enabled),
        "clahe_clip_limit": float(cfg.preprocess.clahe_clip_limit),
        "clahe_tile_grid_size": tuple(cfg.preprocess.clahe_tile_grid_size),
        "contrast_enabled": bool(cfg.preprocess.contrast_enabled),
        "contrast_limit": float(cfg.preprocess.contrast_limit),
        "brightness_enabled": bool(cfg.preprocess.brightness_enabled),
        "brightness_limit": float(cfg.preprocess.brightness_limit),
        "gamma_enabled": bool(cfg.preprocess.gamma_enabled),
        "gamma_limit": float(cfg.preprocess.gamma_limit),
        "noise_robustness_enabled": bool(cfg.preprocess.noise_robustness_enabled),
        "noise_sigma": float(cfg.preprocess.noise_sigma),
    }

    augmentation_cfg = None
    if mode == "train":
        augmentation_cfg = {
            "hflip_enabled": bool(cfg.augmentation.hflip_enabled),
            "rotation_degrees": float(cfg.augmentation.rotation_degrees),
            "shift_pixels": float(cfg.augmentation.shift_pixels),
            "scale_range": tuple(cfg.augmentation.scale_range),
            "brightness_enabled": bool(cfg.augmentation.brightness_enabled),
            "brightness_limit": float(cfg.augmentation.brightness_limit),
            "contrast_enabled": bool(cfg.augmentation.contrast_enabled),
            "contrast_limit": float(cfg.augmentation.contrast_limit),
            "gaussian_noise_enabled": bool(cfg.augmentation.gaussian_noise_enabled),
            "gaussian_noise_sigma": float(cfg.augmentation.gaussian_noise_sigma),
            "blur_enabled": bool(cfg.augmentation.blur_enabled),
            "blur_kernel_choices": tuple(cfg.augmentation.blur_kernel_choices),
            "gamma_enabled": bool(cfg.augmentation.gamma_enabled),
            "gamma_limit": float(cfg.augmentation.gamma_limit),
            "seed": cfg.training.seed,
        }

    return OctDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        strict_pairing=True,
        mode=mode,
        image_height=int(cfg.preprocess.image_height),
        image_width=int(cfg.preprocess.image_width),
        preprocess_overrides=preprocess_cfg,
        augmentation_overrides=augmentation_cfg,
    )


def create_train_loader(
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = True,
    pin_memory: bool | None = None,
    cfg: Optional[TrainingConfig] = None,
) -> DataLoader[Any]:
    """Create the training DataLoader."""

    ds_root = Path(dataset_path)
    images_dir = ds_root / "train" / "images"
    masks_dir = ds_root / "train" / "masks"

    resolved_pin_memory = _pin_memory_if_cuda_available() if pin_memory is None else pin_memory

    use_cfg = cfg if cfg is not None else TrainingConfig()
    dataset = _build_oct_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mode="train",
        cfg=use_cfg,
    )

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
    cfg: Optional[TrainingConfig] = None,
) -> DataLoader[Any]:
    """Create the validation DataLoader."""

    ds_root = Path(dataset_path)
    images_dir = ds_root / "val" / "images"
    masks_dir = ds_root / "val" / "masks"

    resolved_pin_memory = _pin_memory_if_cuda_available() if pin_memory is None else pin_memory

    use_cfg = cfg if cfg is not None else TrainingConfig()
    dataset = _build_oct_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mode="val",
        cfg=use_cfg,
    )

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
    cfg: Optional[TrainingConfig] = None,
) -> DataLoader[Any]:
    """Create the test DataLoader."""

    ds_root = Path(dataset_path)
    images_dir = ds_root / "test" / "images"
    masks_dir = ds_root / "test" / "masks"

    resolved_pin_memory = _pin_memory_if_cuda_available() if pin_memory is None else pin_memory

    use_cfg = cfg if cfg is not None else TrainingConfig()
    dataset = _build_oct_dataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mode="test",
        cfg=use_cfg,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
    )

