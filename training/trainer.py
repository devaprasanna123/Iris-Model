"""MedicalAI Trainer.

This file contains the missing training pipeline implementation for the
MedicalAI project.

Constraints honored:
- No modifications to existing dataset/dataloader/loss/metrics/config/logger/checkpoint.
- Uses the existing Trainer dependencies from:
  - training/config.py
  - training/losses.py
  - training/metrics.py
  - training/dataloaders.py
  - training/models/unet.py
  - training/utils/logger.py
  - training/utils/checkpoint.py

The Trainer supports:
- Training + validation loops
- Mixed Precision (torch.cuda.amp)
- Gradient Scaler
- Optimizer + LR scheduler stepping
- Early stopping
- Metric calculation (Dice/IoU/Pixel Accuracy + precision/recall/f1)
- tqdm progress bars
- Automatic checkpoint saving (best_model.pth + last_model.pth)
- TensorBoard logging
- Resume training from last/best checkpoint

"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from training.metrics import MetricsSpec, dice_score, iou_score, pixel_accuracy, precision_score, recall_score, f1_score
from training.losses import CombinedLoss
from training.utils.checkpoint import CheckpointManager
from training.utils.logger import Logger


@dataclass
class TrainResult:
    """Structured result for a fit() call."""

    best_dice: float
    best_epoch: int
    last_epoch: int


class EarlyStopping:
    """Simple early stopping utility.

    Monitors validation Dice (mean) and stops when there is no improvement
    for `patience` epochs.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, score: float) -> bool:
        """Return True when training should stop."""

        if self.best_score is None:
            self.best_score = score
            self.num_bad_epochs = 0
            return False

        if score > (self.best_score + self.min_delta):
            self.best_score = score
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


class Trainer:
    """Training harness for MedicalAI.

    Args:
        model: U-Net model returning logits of shape (B, C, H, W).
        train_loader: DataLoader yielding (image, mask).
        val_loader: DataLoader yielding (image, mask).
        device: torch.device or device string.
        optimizer: Optimizer instance.
        scheduler: Optional LR scheduler.
        loss_fn: Loss function (expects (logits, target)).
        metrics_spec: MetricsSpec controlling number of classes.
        logger: Project Logger.
        checkpoint_manager: CheckpointManager for best/last saving + resume.
        tensorboard_dir: Optional directory for TensorBoard.
        mixed_precision: Whether to use AMP.
        early_stopping: Whether to enable early stopping.
        early_stopping_patience: Patience for early stopping.
        resume: Whether to resume from last checkpoint if available.
        resume_which: 'last' or 'best'.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: nn.Module,
        metrics_spec: MetricsSpec,
        logger: Logger,
        checkpoint_manager: CheckpointManager,
        num_epochs: int,
        tensorboard_dir: str | Path | None = None,
        mixed_precision: bool = True,
        early_stopping: bool = True,
        early_stopping_patience: int = 7,
        resume: bool = True,
        resume_which: str = "last",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.metrics_spec = metrics_spec
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.num_epochs = int(num_epochs)

        self.mixed_precision = bool(mixed_precision)
        self.scaler = GradScaler(enabled=self.mixed_precision)

        self.early_stopping_enabled = bool(early_stopping)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        self.writer = None
        if tensorboard_dir is not None and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        # State for resume
        self.start_epoch = 1
        self.best_dice = -float("inf")
        self.best_epoch = 0

        if resume:
            self._try_resume(resume_which=resume_which)

        # Deterministic-ish
        random.seed(0)
        torch.manual_seed(0)

    def _try_resume(self, resume_which: str) -> None:
        """Resume model/optimizer/scheduler state if checkpoint exists."""

        try:
            if not self.checkpoint_manager.exists(which=resume_which):
                self.logger.info("No resume checkpoint found (%s). Starting fresh.", resume_which)
                return

            metadata, _extra = self.checkpoint_manager.load(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                which=resume_which,
                strict=True,
            )

            # metadata.epoch is the epoch that produced the checkpoint.
            self.start_epoch = int(metadata.epoch) + 1
            self.best_dice = float(metadata.best_dice)
            self.best_epoch = int(metadata.epoch)

            self.logger.info(
                "Resumed from %s checkpoint: epoch=%s best_dice=%s",
                resume_which,
                metadata.epoch,
                metadata.best_dice,
            )
        except Exception as e:  # pragma: no cover
            self.logger.warning("Resume failed (%s). Starting fresh. Error: %s", resume_which, e)

    def _current_lr(self) -> float:
        """Return current learning rate from first param group."""

        for pg in self.optimizer.param_groups:
            return float(pg.get("lr", 0.0))
        return 0.0

    def _to_device_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs, masks = batch
        return imgs.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)

    def _compute_metrics_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """Compute metrics required by the pipeline from logits+target."""

        dice = dice_score(logits, target, spec=self.metrics_spec, input_is_logits=True)
        iou = iou_score(logits, target, spec=self.metrics_spec, input_is_logits=True)
        acc = pixel_accuracy(logits, target, spec=self.metrics_spec, input_is_logits=True)
        prec = precision_score(logits, target, spec=self.metrics_spec, input_is_logits=True)
        rec = recall_score(logits, target, spec=self.metrics_spec, input_is_logits=True)
        f1 = f1_score(logits, target, spec=self.metrics_spec, input_is_logits=True)

        return {
            "dice": dice,
            "iou": iou,
            "pixel_accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    def _epoch_loop(self, *, train: bool, epoch: int) -> Dict[str, Any]:
        """Run one epoch for training or validation."""

        if train:
            self.model.train()
            loader: Iterable[Any] = self.train_loader
            desc = f"Epoch {epoch} [train]"
        else:
            self.model.eval()
            loader = self.val_loader
            desc = f"Epoch {epoch} [val]"

        total_loss = 0.0
        num_batches = 0

        # Accumulate metrics across batches by summation then divide.
        dice_sum = 0.0
        iou_sum = 0.0
        acc_sum = 0.0
        prec_mean_sum = 0.0
        rec_mean_sum = 0.0
        f1_mean_sum = 0.0

        iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            imgs, masks = self._to_device_batch(batch)

            num_batches += 1
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.mixed_precision and self.device.type == "cuda")):
                logits = self.model(imgs)
                loss = self.loss_fn(logits, masks)

            loss_value = float(loss.detach().item())
            total_loss += loss_value

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            with torch.no_grad():
                metrics = self._compute_metrics_from_logits(logits, masks)
                dice_sum += float(metrics["dice"]["mean"])
                iou_sum += float(metrics["iou"]["mean"])
                acc_sum += float(metrics["pixel_accuracy"])
                prec_mean_sum += float(metrics["precision"]["mean"])
                rec_mean_sum += float(metrics["recall"]["mean"])
                f1_mean_sum += float(metrics["f1"]["mean"])

            iterator.set_postfix({"loss": f"{loss_value:.4f}"})

        # averages
        avg_loss = total_loss / max(1, num_batches)
        avg_dice = dice_sum / max(1, num_batches)
        avg_iou = iou_sum / max(1, num_batches)
        avg_acc = acc_sum / max(1, num_batches)
        avg_prec = prec_mean_sum / max(1, num_batches)
        avg_rec = rec_mean_sum / max(1, num_batches)
        avg_f1 = f1_mean_sum / max(1, num_batches)

        return {
            "loss": avg_loss,
            "dice_mean": avg_dice,
            "iou_mean": avg_iou,
            "pixel_accuracy": avg_acc,
            "precision_mean": avg_prec,
            "recall_mean": avg_rec,
            "f1_mean": avg_f1,
        }

    def _maybe_step_scheduler(self) -> None:
        """Step scheduler based on scheduler type/name."""

        if self.scheduler is None:
            return

        # Most schedulers should be stepped once per epoch.
        try:
            self.scheduler.step()
        except TypeError:
            # Some schedulers (e.g. ReduceLROnPlateau) require metric.
            # We won't use it here because config supports only step/cosine/none,
            # but keep this guard for robustness.
            pass

    def fit(self) -> TrainResult:
        """Run training until completion or early stopping."""

        self.model.to(self.device)

        # If AMP disabled on CPU, disable scaler.
        if self.device.type != "cuda":
            self.scaler = GradScaler(enabled=False)

        num_epochs = int(getattr(self, "num_epochs", 0))

        # We infer epochs from train_loader length is not reliable.
        # num_epochs should be set externally via trainer.fit(epochs=...).
        # For simplicity, we read from optimizer/scheduler config not possible.
        # So: set by caller using attribute.
        if num_epochs <= 0:
            # Determine epochs by best guess: keep loop controlled by attribute.
            raise RuntimeError("Trainer.fit() requires self.num_epochs to be set before calling.")

        start_epoch = int(self.start_epoch)
        stop_best_dice = float(self.best_dice)

        for epoch in range(start_epoch, num_epochs + 1):
            train_stats = self._epoch_loop(train=True, epoch=epoch)
            val_stats = self._epoch_loop(train=False, epoch=epoch)

            lr = self._current_lr()

            train_loss = float(train_stats["loss"])
            val_loss = float(val_stats["loss"])
            dice_mean = float(val_stats["dice_mean"])
            iou_mean = float(val_stats["iou_mean"])
            pix_acc = float(val_stats["pixel_accuracy"])

            # Required every epoch print
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f} | "
                f"Dice: {dice_mean:.6f} | IoU: {iou_mean:.6f} | "
                f"Pixel Accuracy: {pix_acc:.6f} | "
                f"Learning Rate: {lr:.8f}"
            )

            self.logger.info(
                "Epoch %s: train_loss=%.6f val_loss=%.6f dice=%.6f iou=%.6f pixel_acc=%.6f lr=%.8f",
                epoch,
                train_loss,
                val_loss,
                dice_mean,
                iou_mean,
                pix_acc,
                lr,
            )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Dice/val_mean", dice_mean, epoch)
                self.writer.add_scalar("IoU/val_mean", iou_mean, epoch)
                self.writer.add_scalar("PixelAccuracy/val", pix_acc, epoch)
                self.writer.add_scalar("LR", lr, epoch)

                # Precision/Recall/F1
                self.writer.add_scalar("Precision/val_mean", float(val_stats["precision_mean"]), epoch)
                self.writer.add_scalar("Recall/val_mean", float(val_stats["recall_mean"]), epoch)
                self.writer.add_scalar("F1/val_mean", float(val_stats["f1_mean"]), epoch)

            # Save last every epoch
            self.checkpoint_manager.save_last(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                dice=dice_mean,
                best_dice=self.best_dice,
                training_config={},
                extra={
                    "lr": lr,
                    "iou_mean": iou_mean,
                    "pixel_accuracy": pix_acc,
                    "precision_mean": float(val_stats["precision_mean"]),
                    "recall_mean": float(val_stats["recall_mean"]),
                    "f1_mean": float(val_stats["f1_mean"]),
                },
            )

            improved = dice_mean > self.best_dice
            if improved:
                self.best_dice = dice_mean
                self.best_epoch = epoch
                stop_best_dice = float(self.best_dice)

                self.checkpoint_manager.save_best(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    dice=dice_mean,
                    best_dice=self.best_dice,
                    training_config={},
                    extra={
                        "lr": lr,
                        "iou_mean": iou_mean,
                        "pixel_accuracy": pix_acc,
                        "precision_mean": float(val_stats["precision_mean"]),
                        "recall_mean": float(val_stats["recall_mean"]),
                        "f1_mean": float(val_stats["f1_mean"]),
                    },
                )

            # Scheduler step
            self._maybe_step_scheduler()

            # Early stopping
            if self.early_stopping_enabled:
                should_stop = self.early_stopping.step(dice_mean)
                if should_stop:
                    self.logger.info("Early stopping triggered at epoch %s.", epoch)
                    break

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        last_epoch = epoch
        return TrainResult(best_dice=self.best_dice, best_epoch=self.best_epoch, last_epoch=last_epoch)


def _set_trainer_num_epochs(trainer: Trainer, epochs: int) -> Trainer:
    """Internal helper to attach epochs onto the Trainer."""

    trainer.num_epochs = int(epochs)  # type: ignore[attr-defined]
    return trainer


if __name__ == "__main__":
    # Lightweight self-test without dataset.
    # We validate that the Trainer can run one synthetic train/val epoch
    # end-to-end (forward, AMP, metrics, checkpoint save).

    from torch.utils.data import TensorDataset

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic data: logits-friendly masks
    B, H, W = 4, 32, 32
    x = torch.rand(B, 3, H, W)
    y = torch.randint(0, 3, (B, H, W), dtype=torch.long)

    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=2)

    model = TinyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_fn = CombinedLoss(dice_weight=1.0)
    metrics_spec = MetricsSpec(num_classes=3)

    logger = Logger(name="MedicalAI.trainer_self_test", log_dir=Path("MedicalAI") / "training" / "logs" / "_self_test")
    ckpt_dir = Path("MedicalAI") / "training" / "checkpoints" / "_self_test"
    ckpt = CheckpointManager(ckpt_dir, best_model_name="best_model.pth", last_model_name="last_model.pth", device=device)

    trainer = Trainer(
        model=model,
        train_loader=dl,
        val_loader=dl,
        device=device,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        metrics_spec=metrics_spec,
        logger=logger,
        checkpoint_manager=ckpt,
        tensorboard_dir=Path("MedicalAI") / "training" / "tensorboard" / "_self_test",
        mixed_precision=True,
        early_stopping=False,
        resume=False,
    )

    trainer = _set_trainer_num_epochs(trainer, epochs=1)
    result = trainer.fit()

    best_exists = ckpt.exists("best")
    last_exists = ckpt.exists("last")

    print("Self-test result:", result)
    print("best_model.pth exists:", best_exists)
    print("last_model.pth exists:", last_exists)

    logger.close()

