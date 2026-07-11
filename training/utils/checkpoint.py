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
import subprocess
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
    git_commit: Optional[str] = None
    training_duration_s: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "train_loss": float(self.train_loss),
            "val_loss": float(self.val_loss),
            "dice": float(self.dice),
            "best_dice": float(self.best_dice),
            "training_config": self.training_config,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "training_duration_s": self.training_duration_s,
            "metrics": self.metrics,
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
        top_k: int = 5,
        save_every: Optional[int] = 10,
        keep_periodic: int = 5,
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
        # Top-K best checkpoints to keep (by dice)
        self.top_k = int(top_k)
        # Save periodic epoch snapshots every N epochs (None to disable)
        self.save_every = int(save_every) if save_every is not None else None
        # How many periodic snapshots to keep
        self.keep_periodic = int(keep_periodic)
        # Index file keeping track of best checkpoints
        self._best_index_path = self.checkpoint_dir / "best_index.json"

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
        scaler: Optional[Any] = None,
        metadata: CheckpointMetadata,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "checkpoint_version": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
            "metadata": metadata.to_dict(),
            "extra": extra or {},
        }

        return payload

    def _get_git_commit(self) -> Optional[str]:
        try:
            # Attempt to get short commit hash
            p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
            return p.stdout.strip()
        except Exception:
            return None

    def _read_best_index(self) -> Dict[str, Any]:
        if not self._best_index_path.exists():
            return {}
        try:
            with open(self._best_index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _write_best_index(self, index: Dict[str, Any]) -> None:
        try:
            with open(self._best_index_path, "w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=2)
        except Exception:
            pass

    def _prune_bests(self) -> None:
        index = self._read_best_index()
        if not index:
            return
        # index maps filename -> {"epoch": int, "dice": float, "path": str}
        items = list(index.items())
        # sort by dice desc
        items.sort(key=lambda it: float(it[1].get("dice", 0.0)), reverse=True)
        to_keep = items[: self.top_k]
        keep_names = {name for name, _ in to_keep}
        # delete others
        for name, meta in items[self.top_k :]:
            p = Path(meta.get("path", ""))
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            index.pop(name, None)
        # rewrite index
        self._write_best_index(index)

    def _prune_periodic(self) -> None:
        # keep only newest `keep_periodic` epoch_*.pt files
        pattern = "epoch_*.pt"
        files = list(self.checkpoint_dir.glob(pattern))
        if not files:
            return
        # sort by epoch number parsed from filename
        def _epoch_from_path(p: Path) -> int:
            try:
                name = p.stem  # e.g., 'epoch_10'
                parts = name.split("_")
                return int(parts[1])
            except Exception:
                return 0

        files.sort(key=_epoch_from_path, reverse=True)
        for p in files[self.keep_periodic :]:
            try:
                p.unlink()
            except Exception:
                pass

    def save_best(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
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
        git_commit = self._get_git_commit()
        metadata = CheckpointMetadata(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            dice=dice,
            best_dice=best_dice,
            training_config=training_config,
            timestamp=ts,
            git_commit=git_commit,
            training_duration_s=extra.get("training_duration_s") if isinstance(extra, dict) and "training_duration_s" in extra else None,
            metrics=extra.get("metrics") if isinstance(extra, dict) and "metrics" in extra else None,
        )

        payload = self._build_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metadata=metadata,
            extra=extra,
        )

        # Save a named best file (best_epoch_{epoch}.pt) and also update legacy best_model file
        best_epoch_path = self.checkpoint_dir / f"best_epoch_{epoch}.pt"
        try:
            torch.save(payload, best_epoch_path)
        except Exception:
            # fallback to .pth
            best_epoch_path = self.checkpoint_dir / f"best_epoch_{epoch}.pth"
            torch.save(payload, best_epoch_path)

        # update best model (legacy path)
        try:
            torch.save(payload, self.best_model_path)
        except Exception:
            pass

        # update best index and prune
        try:
            index = self._read_best_index()
            entry_name = best_epoch_path.name
            index[entry_name] = {"epoch": int(epoch), "dice": float(dice), "path": str(best_epoch_path), "timestamp": ts}
            self._write_best_index(index)
            self._prune_bests()
        except Exception:
            pass

        return best_epoch_path

    def save_last(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
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
        git_commit = self._get_git_commit()
        metadata = CheckpointMetadata(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            dice=dice,
            best_dice=best_dice,
            training_config=training_config,
            timestamp=ts,
            git_commit=git_commit,
            training_duration_s=extra.get("training_duration_s") if isinstance(extra, dict) and "training_duration_s" in extra else None,
            metrics=extra.get("metrics") if isinstance(extra, dict) and "metrics" in extra else None,
        )

        payload = self._build_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metadata=metadata,
            extra=extra,
        )

        # Save last (legacy path)
        try:
            torch.save(payload, self.last_model_path)
        except Exception:
            pass

        # Periodic snapshot if configured
        periodic_path = None
        try:
            if self.save_every is not None and self.save_every > 0 and (epoch % self.save_every == 0):
                periodic_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
                try:
                    torch.save(payload, periodic_path)
                except Exception:
                    periodic_path = self.checkpoint_dir / f"epoch_{epoch}.pth"
                    torch.save(payload, periodic_path)
                # prune older periodic snapshots
                try:
                    self._prune_periodic()
                except Exception:
                    pass
        except Exception:
            periodic_path = None

        return self.last_model_path

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
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

        sca_sd = checkpoint.get("scaler_state_dict")
        if scaler is not None and sca_sd is not None and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(sca_sd)
            except Exception:
                # Non-fatal: if scaler state can't be loaded, continue without failing resume
                pass

        md_raw = checkpoint.get("metadata", {})
        metadata = CheckpointMetadata(
            epoch=int(md_raw.get("epoch", 0)),
            train_loss=float(md_raw.get("train_loss", 0.0)),
            val_loss=float(md_raw.get("val_loss", 0.0)),
            dice=float(md_raw.get("dice", 0.0)),
            best_dice=float(md_raw.get("best_dice", 0.0)),
            training_config=dict(md_raw.get("training_config", {})),
            timestamp=str(md_raw.get("timestamp", "")),
            git_commit=md_raw.get("git_commit"),
            training_duration_s=md_raw.get("training_duration_s"),
            metrics=md_raw.get("metrics"),
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

