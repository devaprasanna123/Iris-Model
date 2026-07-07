"""MedicalAI training entrypoint.

Responsibilities:
- Load TrainingConfig
- Set random seed
- Auto detect CUDA and print GPU name
- Print configuration
- Create model, dataloaders, optimizer, scheduler
- Create Trainer and start training
- Save logs

Constraints honored:
- Only creates this file; does not modify other existing modules.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from training.config import TrainingConfig
from training.dataloaders import create_test_loader, create_train_loader, create_val_loader

from training.losses import CombinedLoss
from training.metrics import MetricsSpec
from training.models.unet import UNet
from training.trainer import Trainer, _set_trainer_num_epochs
from training.utils.checkpoint import CheckpointManager
from training.utils.logger import Logger

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility (best-effort)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _create_optimizer(cfg: TrainingConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    lr = float(cfg.training.learning_rate)
    wd = float(cfg.training.weight_decay)
    name = str(cfg.training.optimizer).lower()

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def _create_scheduler(cfg: TrainingConfig, optimizer: torch.optim.Optimizer) -> Optional[object]:
    name = str(cfg.training.scheduler).lower()

    if name == "none":
        return None

    if name == "step":
        # Conservative defaults.
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs))

    raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler}")


def main() -> None:
    cfg = TrainingConfig()

    # Auto-detect CUDA and print GPU name (config already auto-detects, but we also print here).
    device = torch.device(cfg.device.device)
    cuda_available = bool(torch.cuda.is_available())

    print("===== MedicalAI Training Configuration =====")
    cfg.print_config()

    set_random_seed(int(cfg.training.seed))

    print("===== Device Info =====")
    if cuda_available and device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown"
        print(f"CUDA Available: {cuda_available} | Device: {device} | GPU: {gpu_name}")
    else:
        print(f"CUDA Available: {cuda_available} | Device: {device}")

    # Model (not required for DataLoader-only verification)
    # model = UNet(in_channels=cfg.image.channels, num_classes=cfg.classes.number_of_classes)
    # model.to(device)


    # Dataloaders (verification-only; do not start training)
    dataset_root = Path(cfg.dataset.dataset_root)

    train_images_path = dataset_root / cfg.dataset.train_folder / "images"
    train_masks_path = dataset_root / cfg.dataset.train_folder / "masks"
    val_images_path = dataset_root / cfg.dataset.val_folder / "images"
    val_masks_path = dataset_root / cfg.dataset.val_folder / "masks"
    test_images_path = dataset_root / cfg.dataset.test_folder / "images"
    test_masks_path = dataset_root / cfg.dataset.test_folder / "masks"

    print("===== Dataset Root / Paths =====")
    print(f"Dataset Root: {dataset_root}")
    print(f"Train Images Path: {train_images_path}")
    print(f"Train Masks Path: {train_masks_path}")
    print(f"Validation Images Path: {val_images_path}")
    print(f"Validation Masks Path: {val_masks_path}")
    print(f"Test Images Path: {test_images_path}")
    print(f"Test Masks Path: {test_masks_path}")

    required_paths = {
        "Train Images Path": train_images_path,
        "Train Masks Path": train_masks_path,
        "Validation Images Path": val_images_path,
        "Validation Masks Path": val_masks_path,
        "Test Images Path": test_images_path,
        "Test Masks Path": test_masks_path,
    }

    missing = [name for name, p in required_paths.items() if not Path(p).exists()]
    for name in missing:
        print(f"MISSING: {name} -> {required_paths[name]}")

    if missing:
        raise FileNotFoundError(
            "One or more required dataset directories are missing:\n" + "\n".join(
                f"- {name}: {required_paths[name]}" for name in missing
            )
        )

    train_loader = create_train_loader(
        dataset_path=dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=True,
    )
    val_loader = create_val_loader(
        dataset_path=dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=False,
    )
    test_loader = create_test_loader(
        dataset_path=dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=False,
    )

    # Only verify DataLoader initialization.
    print("===== DataLoader initialization: OK =====")

    # ============================
    # Training setup (verification run)
    # ============================
    device = torch.device(cfg.device.device)

    # Print required training configuration block BEFORE training starts
    epochs_for_verification = 1
    print("\n============================")
    print("Training Configuration")
    print("============================")
    print(f"Device: {device}")
    print(f"Dataset Root: {dataset_root}")
    print(f"Batch Size: {int(cfg.training.batch_size)}")
    print(f"Epochs: {epochs_for_verification}")
    print(f"Learning Rate: {float(cfg.training.learning_rate)}")
    print(f"Optimizer: AdamW (lr={float(cfg.training.learning_rate)}, weight_decay={float(cfg.training.weight_decay)})")
    print(f"Scheduler: {str(cfg.training.scheduler).lower()}")

    # Create UNet model
    model = UNet(in_channels=cfg.image.channels, num_classes=cfg.classes.number_of_classes)
    model.to(device)

    # Create CombinedLoss
    loss_fn = CombinedLoss(dice_weight=1.0)

    # Create AdamW optimizer using config values
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    # Create learning-rate scheduler (if enabled in config)
    scheduler = _create_scheduler(cfg, optimizer)

    # Ensure lr scheduler (if any) is consistent with 1-epoch verification
    # (trainer.fit() will step scheduler each epoch)

    # Logger
    logger = Logger(
        name="MedicalAI.train",
        log_dir=cfg.logs.log_dir,
        level=None if False else 20,  # INFO
        enable_console=True,
        enable_file=True,
        use_color=True,
    )
    logger.info("Starting 1-epoch verification training...")

    # Checkpoints (save best_model.pt and last_model.pt)
    checkpoint_manager = CheckpointManager(
        cfg.checkpoint.checkpoint_dir,
        best_model_name="best_model.pt",
        last_model_name="last_model.pt",
        device=device,
    )

    # TensorBoard
    tensorboard_dir = cfg.logs.tensorboard_dir
    if SummaryWriter is not None:
        os.makedirs(tensorboard_dir, exist_ok=True)

    # Trainer
    metrics_spec = MetricsSpec(num_classes=cfg.classes.number_of_classes)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics_spec=metrics_spec,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        tensorboard_dir=tensorboard_dir,
        mixed_precision=bool(cfg.training.mixed_precision),
        early_stopping=bool(cfg.training.early_stopping),
        early_stopping_patience=int(cfg.training.patience),
        resume=True,
        resume_which="last",
    )

    # Force exactly 1 epoch for verification
    trainer = _set_trainer_num_epochs(trainer, epochs=epochs_for_verification)

    result = trainer.fit()

    logger.info(
        "First epoch done. best_dice=%s best_epoch=%s last_epoch=%s",
        result.best_dice,
        result.best_epoch,
        result.last_epoch,
    )
    print("✅ First training epoch completed successfully.")

    logger.close()


if __name__ == "__main__":
    main()

