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
from training.dataloaders import create_train_loader, create_val_loader
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

    # Model
    model = UNet(in_channels=cfg.image.channels, num_classes=cfg.classes.number_of_classes)
    model.to(device)

    # Dataloaders
    dataset_root = cfg.dataset.dataset_root
    train_loader = create_train_loader(
        dataset_path=dataset_root / "train",
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=True,
    )
    val_loader = create_val_loader(
        dataset_path=dataset_root / "val",
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=False,
    )

    # Loss
    loss_fn = CombinedLoss(dice_weight=1.0)

    # Optimizer / scheduler
    optimizer = _create_optimizer(cfg, model)
    scheduler = _create_scheduler(cfg, optimizer)

    # Logger
    logger = Logger(
        name="MedicalAI.train",
        log_dir=cfg.logs.log_dir,
        level=None if False else 20,  # INFO
        enable_console=True,
        enable_file=True,
        use_color=True,
    )

    logger.info("Starting training...")

    # Checkpoints
    checkpoint_manager = CheckpointManager(
        cfg.checkpoint.checkpoint_dir,
        best_model_name=cfg.checkpoint.best_model_name.replace(".pt", ".pth")
        if cfg.checkpoint.best_model_name.endswith(".pt")
        else cfg.checkpoint.best_model_name,
        last_model_name=cfg.checkpoint.last_model_name.replace(".pt", ".pth")
        if cfg.checkpoint.last_model_name.endswith(".pt")
        else cfg.checkpoint.last_model_name,
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

    trainer = _set_trainer_num_epochs(trainer, epochs=int(cfg.training.epochs))

    result = trainer.fit()

    logger.info("Training complete. best_dice=%s best_epoch=%s last_epoch=%s", result.best_dice, result.best_epoch, result.last_epoch)
    print(f"Training complete. best_dice={result.best_dice:.6f} best_epoch={result.best_epoch} last_epoch={result.last_epoch}")

    logger.close()


if __name__ == "__main__":
    main()

