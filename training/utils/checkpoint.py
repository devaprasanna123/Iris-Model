"""MedicalAI training checkpoint utilities.

This module provides :class:`~CheckpointManager`, a reusable checkpoint manager
intended to be used by a future ``trainer.py``.

It supports:
- Saving model / optimizer / scheduler state dicts
- Tracking training/validation loss, dice scores, best dice
- Saving ``best_model.pth`` and ``last_model.pth``
- Resuming training via ``load``
- Basic filesystem helpers (exists/delete/latest_checkpoint)

Only standard library modules + torch are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata stored alongside state_dicts inside a checkpoint."""

    epoch: int
    train_loss: float
    val_loss: float
    dice: float
    best_dice: float
    training_config: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "dice": float(self.dice),
            "best_dice": float(self.best_dice),
            "training_config": self.training_config,
            "timestamp": self.timestamp,
        }


class CheckpointManager:
    """Reusable checkpoint manager for training resume + best/last saving.

    Args:
        checkpoint_dir: Directory where checkpoints are stored.
        best_model_name: Filename for best checkpoint.
        last_model_name: Filename for last checkpoint.
        device: If provided, will map-loaded tensors to this device.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        best_model_name: str = "best_model.pth",
        last_model_name: str = "last_model.pth",
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.checkpoint_dir / best_model_name
        self.last_model_path = self.checkpoint_dir / last_model_name

        self._map_location: Optional[str]
        if device is None:
            self._map_location = None
        else:
            self._map_location = str(device)

    def exists(self, which: str = "last") -> bool:
        """Check whether a checkpoint exists.

        Args:
            which: "best" or "last".

        Returns:
            True if checkpoint file exists.
        """

        path = self._path_for(which)
        return path.exists() and path.is_file()

    def delete(self, which: str = "last") -> None:
        """Delete a checkpoint file if it exists."""

        path = self._path_for(which)
        if path.exists():
            path.unlink()

    def latest_checkpoint(self) -> Optional[Path]:
        """Return the latest checkpoint path based on mtime.

        If neither checkpoint exists, returns None.
        """

        paths = [
            self.best_model_path if self.best_model_path.exists() else None,
            self.last_model_path if self.last_model_path.exists() else None,
        ]
        candidates = [p for p in paths if p is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _build_payload(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metadata: CheckpointMetadata,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metadata": metadata.to_dict(),
            "extra": extra or {},
        }
        return payload

    def save_best(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any] = None,
        epoch: int,
        train_loss: float,
        val_loss: float,
        dice: float,
        best_dice: float,
        training_config: Dict[str, Any],
        timestamp: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint to ``best_model.pth``."""

        ts = timestamp or datetime.now().isoformat()
        metadata = CheckpointMetadata(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            dice=dice,
            best_dice=best_dice,
            training_config=training_config,
            timestamp=ts,
        )

        payload = self._build_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            extra=extra,
        )

        torch.save(payload, self.best_model_path)
        return self.best_model_path

    def save_last(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any] = None,
        epoch: int,
        train_loss: float,
        val_loss: float,
        dice: float,
        best_dice: float,
        training_config: Dict[str, Any],
        timestamp: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint to ``last_model.pth``."""

        ts = timestamp or datetime.now().isoformat()
        metadata = CheckpointMetadata(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            dice=dice,
            best_dice=best_dice,
            training_config=training_config,
            timestamp=ts,
        )

        payload = self._build_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            extra=extra,
        )

        torch.save(payload, self.last_model_path)
        return self.last_model_path

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        which: str = "last",
        strict: bool = True,
    ) -> Tuple[CheckpointMetadata, Dict[str, Any]]:
        """Load checkpoint and restore model/optimizer/scheduler.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to restore state dict into (if present).
            scheduler: Scheduler to restore state dict into (if present).
            which: "best" or "last".
            strict: Passed to ``model.load_state_dict``.

        Returns:
            (metadata, extra) where metadata is :class:`CheckpointMetadata`.
        """

        path = self._path_for(which)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = self._map_location
        checkpoint = torch.load(path, map_location=map_location)

        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        opt_sd = checkpoint.get("optimizer_state_dict")
        if optimizer is not None and opt_sd is not None:
            optimizer.load_state_dict(opt_sd)

        sch_sd = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and sch_sd is not None:
            scheduler.load_state_dict(sch_sd)

        md_raw = checkpoint.get("metadata", {})
        metadata = CheckpointMetadata(
            epoch=int(md_raw.get("epoch", 0)),
            train_loss=float(md_raw.get("train_loss", 0.0)),
            val_loss=float(md_raw.get("val_loss", 0.0)),
            dice=float(md_raw.get("dice", 0.0)),
            best_dice=float(md_raw.get("best_dice", 0.0)),
            training_config=dict(md_raw.get("training_config", {})),
            timestamp=str(md_raw.get("timestamp", "")),
        )

        extra = dict(checkpoint.get("extra", {}))
        return metadata, extra

    def _path_for(self, which: str) -> Path:
        which_norm = which.strip().lower()
        if which_norm in {"best", "best_model"}:
            return self.best_model_path
        if which_norm in {"last", "last_model"}:
            return self.last_model_path
        raise ValueError("which must be one of: 'best', 'last'")


def _jsonable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort conversion to JSON-serializable dict."""

    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return obj

    return {k: convert(v) for k, v in (config or {}).items()}


if __name__ == "__main__":
    # Self-test
    import torch.nn as nn

    ckpt_dir = Path("MedicalAI") / "training" / "checkpoints" / "_self_test"
    manager = CheckpointManager(ckpt_dir)

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    torch.manual_seed(0)

    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Fake a scheduler step to ensure it has state.
    scheduler.step()

    training_cfg = {"epochs": 2, "lr": 1e-3}

    saved_path = manager.save_last(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        train_loss=0.123,
        val_loss=0.234,
        dice=0.5,
        best_dice=0.5,
        training_config=_jsonable_config(training_cfg),
    )

    # Load into a new model/optimizer/scheduler
    model2 = DummyModel()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.1)

    metadata, extra = manager.load(
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        which="last",
        strict=True,
    )

    success = saved_path.exists() and metadata.epoch == 1

    print(f"Checkpoint saved to: {saved_path}")
    print("Self-test status:", "OK" if success else "FAILED")

