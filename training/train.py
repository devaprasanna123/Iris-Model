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

from training.loss_factory import create_loss
from training.optimizer_factory import create_optimizer
from training.scheduler_factory import create_scheduler

from training.metrics import MetricsSpec
from training.models.model_factory import create_model
from training.trainer import Trainer
from training.utils.checkpoint import CheckpointManager
from training.utils.logger import Logger
from training.monitor import TrainingMonitor

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
    """Build LR scheduler from config.

    v2 required schedulers:
      - none
      - CosineAnnealingLR
      - ReduceLROnPlateau
      - OneCycleLR
      - CosineAnnealingWarmRestarts

    Backward compatible with legacy cfg.training.scheduler values:
      - "none", "step", "cosine"
    """

    name = str(cfg.training.scheduler).strip().lower()

    if name == "none":
        return None

    # Legacy aliases
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(getattr(cfg.training, "epochs", cfg.training.epochs)),
            eta_min=float(getattr(cfg.training, "cosine_annealing_eta_min", 0.0)),
        )

    if name == "step":
        # Legacy StepLR preserved as a conservative option.
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # v2 names / aliases
    if name in {"cosineannealinglr", "cosine_annealinglr", "cosineannealinglr"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(getattr(cfg.training, "cosine_annealing_t_max", cfg.training.epochs) or cfg.training.epochs),
            eta_min=float(getattr(cfg.training, "cosine_annealing_eta_min", 0.0)),
        )

    if name in {"reducelronplateau", "reducelronplateaulr", "reduce_on_plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Dice is maximized
            factor=float(getattr(cfg.training, "plateau_factor", 0.1)),
            patience=int(getattr(cfg.training, "plateau_patience", 5)),
            min_lr=float(getattr(cfg.training, "plateau_min_lr", 0.0)),
        )

    if name in {"onecyclelr", "one_cycle_lr"}:
        max_lr = getattr(cfg.training, "onecycle_max_lr", None)
        if max_lr is None:
            max_lr = float(cfg.training.learning_rate)

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(max_lr),
            epochs=int(cfg.training.epochs),
            steps_per_epoch=1,  # will be overridden by Trainer via per-batch stepping if needed
            pct_start=float(getattr(cfg.training, "onecycle_pct_start", 0.3)),
            div_factor=float(getattr(cfg.training, "onecycle_div_factor", 25.0)),
            final_div_factor=float(getattr(cfg.training, "onecycle_final_div_factor", 1e4)),
        )

    if name in {"cosineannealingwarmrestars", "cosineannealingwarmrestaris", "cosineannealingwarmrestarestars", "cosineannealingwarmrestar", "cosineannealingwarmrestarst"}:
        # normalize for common typo; actual expected config key is CosineAnnealingWarmRestarts
        name = "cosineannealingwarmrestar"  # fallthrough handling below

    if name in {"cosineannealingwarmrestatrs", "cosineannealingwarmrestar", "cosineannealingwarmrestarns", "cosineannealingwarmrestar", "cosineannealingwarmrest"} or name == "cosineannealingwarmrestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(getattr(cfg.training, "warm_restarts_t_0", 10)),
            T_mult=int(getattr(cfg.training, "warm_restarts_t_mult", 1)),
            eta_min=float(getattr(cfg.training, "warm_restarts_eta_min", 0.0)),
        )

    raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler} (resolved name='{name}')")



def main() -> None:
    cfg = TrainingConfig()

    validation_errors = cfg.validate()
    if validation_errors:
        print("CONFIG VALIDATION FAILED:")
        for err in validation_errors:
            print(f"- {err}")
        raise ValueError("Training configuration validation failed. Fix config values and retry.")

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

    # Dataloaders
    dataset_root = Path(cfg.dataset.dataset_root)

    # Verify required dataset paths exist before constructing loaders.
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

    print("===== Dataset validation: OK =====")

    train_loader = create_train_loader(
        dataset_path=dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=True,
        cfg=cfg,
    )
    val_loader = create_val_loader(
        dataset_path=dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=False,
        cfg=cfg,
    )
    test_loader = create_test_loader(
        dataset_path=dataset_root,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.workers),
        shuffle=False,
        cfg=cfg,
    )
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

    print("===== DataLoader initialization: OK =====")
    cfg.logs.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.output.prediction_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Print required training configuration block BEFORE training starts
    print("\n==================================")
    print("Training Configuration")
    print("==================================")
    print(f"Device: {device}")
    print(f"Dataset Root: {dataset_root}")
    print(f"Batch Size: {int(cfg.training.batch_size)}")
    print(f"Epochs: {int(cfg.training.epochs)}")
    print(f"Learning Rate: {float(cfg.training.learning_rate)}")
    print(f"Optimizer: {str(cfg.training.optimizer).capitalize()}")
    print(f"Scheduler: {str(cfg.training.scheduler)}")
    print(f"Checkpoint Path: {cfg.checkpoint.checkpoint_dir}")
    print(f"Log Path: {cfg.logs.log_dir}")
    print(f"TensorBoard Path: {cfg.logs.tensorboard_dir}")
    print(f"Report Path: {cfg.logs.report_dir}")
    print(f"Prediction Path: {cfg.output.prediction_dir}")

    logger = Logger(
        name="MedicalAI.train",
        log_dir=cfg.logs.log_dir,
        level=None if False else 20,  # INFO
        enable_console=True,
        enable_file=True,
        use_color=True,
    )
    logger.info("Starting full training...")

    # Create monitor
    monitor = TrainingMonitor(out_dir=cfg.logs.log_dir, writer=None, logger=logger)

    # Create segmentation model
    model = create_model(cfg)
    model.to(device)


    # Create loss from config (backward compatible default: CE + Dice(dice_weight=1.0))
    loss_fn = create_loss(cfg)


    # Create optimizer using config values (factory)
    optimizer = create_optimizer(cfg, model)

    # Create learning-rate scheduler (if enabled in config).
    # Pass steps_per_epoch when available for OneCycleLR correctness.
    scheduler = create_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader) if train_loader is not None else None)

    # If OneCycleLR was chosen, recreate with correct steps_per_epoch
    try:
        from torch.optim.lr_scheduler import OneCycleLR
        if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
            max_lr = getattr(cfg.training, "onecycle_max_lr", None)
            if max_lr is None:
                max_lr = float(cfg.training.learning_rate)

            scheduler = OneCycleLR(
                optimizer,
                max_lr=float(max_lr),
                epochs=int(cfg.training.epochs),
                steps_per_epoch=max(1, len(train_loader)),
                pct_start=float(getattr(cfg.training, "onecycle_pct_start", 0.3)),
                div_factor=float(getattr(cfg.training, "onecycle_div_factor", 25.0)),
                final_div_factor=float(getattr(cfg.training, "onecycle_final_div_factor", 1e4)),
            )
    except Exception:
        pass

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
        num_epochs=int(cfg.training.epochs),
        tensorboard_dir=tensorboard_dir,
        mixed_precision=bool(cfg.training.mixed_precision),
        early_stopping=bool(cfg.training.early_stopping),
        early_stopping_patience=int(cfg.training.patience),
        resume=True,
        resume_which="last",
        training_config=cfg._as_jsonable(),
        monitor=monitor,
    )

    result = trainer.fit()

    logger.info(
        "Training completed. best_dice=%s best_epoch=%s last_epoch=%s",
        result.best_dice,
        result.best_epoch,
        result.last_epoch,
    )
    print("✅ Training completed successfully.")

    logger.close()


if __name__ == "__main__":
    main()

