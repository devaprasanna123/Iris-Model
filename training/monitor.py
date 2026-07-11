"""Training monitoring utilities.

Provides `TrainingMonitor` to track per-epoch metrics, write CSV/JSON
history files, log to TensorBoard, and print professional console summaries.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore

import torch


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    lr: float
    grad_norm: float
    gpu_mem_mb: float
    epoch_time_s: float
    total_time_s: float


class TrainingMonitor:
    def __init__(self, *, out_dir: str | Path, writer: Optional[SummaryWriter] = None, logger: Optional[Any] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "training_history.csv"
        self.json_path = self.out_dir / "training_history.json"
        self.writer = writer
        self.logger = logger
        self.history: List[Dict[str, Any]] = []
        self.train_start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.total_epochs: Optional[int] = None
        self.device_name: Optional[str] = None
        # training context (optional, set by Trainer)
        self._context: Dict[str, Any] = {}

    def set_writer(self, writer: Optional[SummaryWriter]) -> None:
        self.writer = writer

    def on_train_start(self, total_epochs: int) -> None:
        self.train_start_time = time.perf_counter()
        self.total_epochs = int(total_epochs)
        if self.logger:
            self.logger.info("TrainingMonitor: tracking %s epochs", total_epochs)

    def on_epoch_start(self, epoch: int) -> None:
        self.epoch_start_time = time.perf_counter()

    def _gpu_memory_mb(self, device: torch.device | str) -> float:
        try:
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                # Use allocated memory which reflects current usage
                return float(torch.cuda.memory_allocated(int(torch.cuda.current_device())) / (1024 ** 2))
        except Exception:
            pass
        return 0.0

    def on_epoch_end(self, *, epoch: int, total_epochs: int, train_stats: Dict[str, Any], val_stats: Dict[str, Any], lr: float, grad_norm: float, device: torch.device | str) -> None:
        end_time = time.perf_counter()
        epoch_time = 0.0
        if self.epoch_start_time is not None:
            epoch_time = end_time - self.epoch_start_time

        total_time = 0.0
        if self.train_start_time is not None:
            total_time = end_time - self.train_start_time

        gpu_mem = self._gpu_memory_mb(device)
        try:
            self.device_name = str(device)
        except Exception:
            pass

        record = {
            "epoch": int(epoch),
            "train_loss": float(train_stats.get("loss", 0.0)),
            "val_loss": float(val_stats.get("loss", 0.0)),
            "dice": float(val_stats.get("dice_mean", 0.0)),
            "iou": float(val_stats.get("iou_mean", 0.0)),
            "precision": float(val_stats.get("precision_mean", 0.0)),
            "recall": float(val_stats.get("recall_mean", 0.0)),
            "f1": float(val_stats.get("f1_mean", 0.0)),
            "lr": float(lr),
            "grad_norm": float(grad_norm),
            "gpu_mem_mb": float(gpu_mem),
            "epoch_time_s": float(epoch_time),
            "total_time_s": float(total_time),
        }

        self.history.append(record)

        # Write CSV and JSON (overwrite each epoch to keep files current)
        self._write_csv()
        self._write_json()

        # TensorBoard logging
        if self.writer is not None:
            step = int(epoch)
            self.writer.add_scalar("Loss/train", record["train_loss"], step)
            self.writer.add_scalar("Loss/val", record["val_loss"], step)
            self.writer.add_scalar("Dice/val_mean", record["dice"], step)
            self.writer.add_scalar("IoU/val_mean", record["iou"], step)
            self.writer.add_scalar("Precision/val_mean", record["precision"], step)
            self.writer.add_scalar("Recall/val_mean", record["recall"], step)
            self.writer.add_scalar("F1/val_mean", record["f1"], step)
            self.writer.add_scalar("LR", record["lr"], step)
            self.writer.add_scalar("GradNorm/avg", record["grad_norm"], step)
            self.writer.add_scalar("GPU/memory_mb", record["gpu_mem_mb"], step)
            self.writer.add_scalar("Time/epoch_s", record["epoch_time_s"], step)

        # Console professional print
        self._print_epoch_summary(epoch, total_epochs, record)

    def _print_epoch_summary(self, epoch: int, total_epochs: int, record: Dict[str, Any]) -> None:
        lines = []
        header = f"Epoch {epoch}/{total_epochs}"
        lines.append(header)
        lines.append("")
        lines.append(f"Train Loss:      {record['train_loss']:.6f}")
        lines.append(f"Validation Loss: {record['val_loss']:.6f}")
        lines.append(f"Dice:            {record['dice']:.6f}")
        lines.append(f"IoU:             {record['iou']:.6f}")
        lines.append(f"Precision:       {record['precision']:.6f}")
        lines.append(f"Recall:          {record['recall']:.6f}")
        lines.append(f"F1:              {record['f1']:.6f}")
        lines.append(f"Learning Rate:   {record['lr']:.8f}")
        lines.append(f"Grad Norm:       {record['grad_norm']:.6f}")
        lines.append(f"GPU Memory (MB): {record['gpu_mem_mb']:.2f}")
        lines.append(f"Epoch Time (s):  {record['epoch_time_s']:.2f}")

        # ETA estimate: average epoch time * remaining
        avg_epoch = sum(r["epoch_time_s"] for r in self.history) / max(1, len(self.history))
        remaining = max(0, total_epochs - epoch)
        eta = avg_epoch * remaining
        lines.append(f"ETA (s):         {eta:.1f}")

        out = "\n".join(lines)
        print(out)
        if self.logger:
            self.logger.info("Epoch summary written to monitor")

    def _write_csv(self) -> None:
        if not self.history:
            return
        keys = [
            "epoch",
            "train_loss",
            "val_loss",
            "dice",
            "iou",
            "precision",
            "recall",
            "f1",
            "lr",
            "grad_norm",
            "gpu_mem_mb",
            "epoch_time_s",
            "total_time_s",
        ]
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                for r in self.history:
                    writer.writerow({k: r.get(k, "") for k in keys})
        except Exception:
            if self.logger:
                self.logger.warning("Failed to write training CSV at %s", self.csv_path)

    def _write_json(self) -> None:
        try:
            with open(self.json_path, "w", encoding="utf-8") as fh:
                json.dump(self.history, fh, indent=2)
        except Exception:
            if self.logger:
                self.logger.warning("Failed to write training JSON at %s", self.json_path)

    def on_train_end(self) -> None:
        # Flush writer
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
        # Generate summary report and plots if possible
        try:
            self.generate_report()
        except Exception:
            if self.logger:
                self.logger.warning("TrainingMonitor: failed to generate report")

    def set_training_context(self, *, model: Any = None, optimizer: Any = None, scheduler: Any = None, loss_fn: Any = None, train_loader: Any = None, val_loader: Any = None, training_config: Any = None) -> None:
        """Optional: Trainer may call this to provide additional context for reporting."""
        try:
            self._context.update({
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "loss_fn": loss_fn,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "training_config": training_config,
            })
        except Exception:
            pass

    def generate_report(self) -> None:
        """Produce training_summary.json, training_summary.md and plots/ images.

        Uses self.history and optional context set by `set_training_context`.
        """
        if not self.history:
            return

        out = Path(self.out_dir)
        plots_dir = out / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Basic summary values
        epochs = len(self.history)
        total_time = float(self.history[-1].get("total_time_s", 0.0))
        avg_epoch = sum(r.get("epoch_time_s", 0.0) for r in self.history) / max(1, epochs)

        # Best epoch by dice
        best_rec = max(self.history, key=lambda r: float(r.get("dice", 0.0)))
        best_epoch = int(best_rec.get("epoch", 0))
        best_dice = float(best_rec.get("dice", 0.0))
        best_iou = float(best_rec.get("iou", 0.0))

        # Final metrics (last epoch)
        final = self.history[-1]

        # Context introspection
        model = self._context.get("model")
        optimizer = self._context.get("optimizer")
        scheduler = self._context.get("scheduler")
        loss_fn = self._context.get("loss_fn")
        train_loader = self._context.get("train_loader")
        training_config = self._context.get("training_config")

        def _name(obj: Any) -> str:
            try:
                return obj.__class__.__name__
            except Exception:
                return str(obj)

        model_name = _name(model) if model is not None else "Unknown"
        # Try to infer encoder/backbone
        encoder = None
        try:
            if model is not None:
                for attr in ("encoder", "backbone", "encoder_name", "backbone_name"):
                    if hasattr(model, attr):
                        encoder = getattr(model, attr)
                        break
                if encoder is not None:
                    encoder = _name(encoder)
        except Exception:
            encoder = None

        optimizer_name = _name(optimizer) if optimizer is not None else "Unknown"
        scheduler_name = _name(scheduler) if scheduler is not None else "None"
        loss_name = _name(loss_fn) if loss_fn is not None else "Unknown"

        batch_size = None
        dataset_size = None
        try:
            if train_loader is not None:
                batch_size = int(getattr(train_loader, "batch_size", -1))
                try:
                    dataset_size = int(len(getattr(train_loader, "dataset", train_loader)))
                except Exception:
                    dataset_size = None
        except Exception:
            pass

        # LR: initial and final
        try:
            lrs = [float(r.get("lr", 0.0)) for r in self.history]
            initial_lr = lrs[0] if lrs else 0.0
            final_lr = lrs[-1] if lrs else 0.0
        except Exception:
            initial_lr = 0.0
            final_lr = 0.0

        summary = {
            "model": model_name,
            "encoder": encoder or "Unknown",
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "loss": loss_name,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_dice": best_dice,
            "best_iou": best_iou,
            "training_time_s": total_time,
            "avg_epoch_time_s": avg_epoch,
            "gpu": self.device_name or "Unknown",
            "batch_size": batch_size,
            "initial_lr": initial_lr,
            "final_lr": final_lr,
            "dataset_size": dataset_size,
            "final_metrics": {
                "train_loss": float(final.get("train_loss", 0.0)),
                "val_loss": float(final.get("val_loss", 0.0)),
                "dice": float(final.get("dice", 0.0)),
                "iou": float(final.get("iou", 0.0)),
                "precision": float(final.get("precision", 0.0)),
                "recall": float(final.get("recall", 0.0)),
                "f1": float(final.get("f1", 0.0)),
            },
        }

        # Merge config-specified readable names if available
        try:
            if training_config is not None:
                tc = training_config
                # common config patterns
                if isinstance(tc, dict):
                    summary.setdefault("config", {}).update(tc)
                else:
                    # attempt dataclass-like extraction
                    summary.setdefault("config", {}).update({k: getattr(tc, k) for k in dir(tc) if not k.startswith("_")})
        except Exception:
            pass

        # Write JSON summary
        try:
            with open(out / "training_summary.json", "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
        except Exception:
            if self.logger:
                self.logger.warning("Failed to write training_summary.json")

        # Write Markdown summary
        try:
            md_lines = []
            md_lines.append(f"# Training Summary")
            md_lines.append("")
            md_lines.append(f"- **Model**: {summary['model']}")
            md_lines.append(f"- **Encoder**: {summary['encoder']}")
            md_lines.append(f"- **Optimizer**: {summary['optimizer']}")
            md_lines.append(f"- **Scheduler**: {summary['scheduler']}")
            md_lines.append(f"- **Loss**: {summary['loss']}")
            md_lines.append(f"- **Epochs**: {summary['epochs']}")
            md_lines.append(f"- **Best Epoch**: {summary['best_epoch']}")
            md_lines.append(f"- **Best Dice**: {summary['best_dice']:.6f}")
            md_lines.append(f"- **Best IoU**: {summary['best_iou']:.6f}")
            md_lines.append(f"- **Training Time (s)**: {summary['training_time_s']:.2f}")
            md_lines.append(f"- **Average Epoch Time (s)**: {summary['avg_epoch_time_s']:.2f}")
            md_lines.append(f"- **GPU**: {summary['gpu']}")
            md_lines.append(f"- **Batch Size**: {summary['batch_size']}")
            md_lines.append(f"- **Initial LR**: {summary['initial_lr']}")
            md_lines.append(f"- **Final LR**: {summary['final_lr']}")
            md_lines.append(f"- **Dataset Size**: {summary['dataset_size']}")
            md_lines.append("")
            md_lines.append("## Final Metrics")
            md_lines.append("")
            for k, v in summary["final_metrics"].items():
                md_lines.append(f"- **{k}**: {v}")

            md_content = "\n".join(md_lines)
            with open(out / "training_summary.md", "w", encoding="utf-8") as fh:
                fh.write(md_content)
        except Exception:
            if self.logger:
                self.logger.warning("Failed to write training_summary.md")

        # Generate plots using matplotlib if available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs_x = [int(r.get("epoch", i + 1)) for i, r in enumerate(self.history)]

            # Loss curve
            try:
                plt.figure()
                train_losses = [r.get("train_loss", 0.0) for r in self.history]
                val_losses = [r.get("val_loss", 0.0) for r in self.history]
                plt.plot(epochs_x, train_losses, label="train_loss")
                plt.plot(epochs_x, val_losses, label="val_loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / "loss_curve.png")
                plt.close()
            except Exception:
                pass

            def _plot_metric(key: str, fname: str, ylabel: str = None):
                try:
                    vals = [r.get(key, 0.0) for r in self.history]
                    plt.figure()
                    plt.plot(epochs_x, vals, marker="o")
                    plt.xlabel("Epoch")
                    plt.ylabel(ylabel or key)
                    plt.tight_layout()
                    plt.savefig(plots_dir / fname)
                    plt.close()
                except Exception:
                    pass

            _plot_metric("dice", "dice_curve.png", "Dice")
            _plot_metric("iou", "iou_curve.png", "IoU")
            _plot_metric("lr", "learning_rate_curve.png", "Learning Rate")
            _plot_metric("precision", "precision_curve.png", "Precision")
            _plot_metric("recall", "recall_curve.png", "Recall")
            _plot_metric("f1", "f1_curve.png", "F1")
        except Exception:
            if self.logger:
                self.logger.info("matplotlib not available, skipping plots")
