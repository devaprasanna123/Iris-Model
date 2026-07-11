"""Scheduler factory for MedicalAI.

Creates LR scheduler instances from TrainingConfig. Supported:
- None
- CosineAnnealingLR
- CosineAnnealingWarmRestarts
- ReduceLROnPlateau
- OneCycleLR

The factory reads parameters from `cfg.training` to avoid hardcoding.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _normalize(name: Any) -> str:
    return str(name).strip().lower()


def create_scheduler(cfg: object, optimizer: torch.optim.Optimizer, steps_per_epoch: Optional[int] = None) -> Optional[object]:
    name = _normalize(getattr(getattr(cfg, "training", cfg), "scheduler", "none"))

    if name in {"none", "", "null"}:
        return None

    # CosineAnnealingLR
    if name in {"cosine", "cosineannealinglr", "cosine_annealinglr", "cosineannealing"}:
        t_max = int(getattr(getattr(cfg, "training", cfg), "cosine_annealing_t_max", None) or getattr(getattr(cfg, "training", cfg), "epochs", None) or 1)
        eta_min = float(getattr(getattr(cfg, "training", cfg), "cosine_annealing_eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    # CosineAnnealingWarmRestarts
    if name in {"cosineannealingwarmrestarts", "cosineannealingwarmrestarts", "cosineannealingwarmrestart", "cosineannealingwarmrestarts"}:
        t_0 = int(getattr(getattr(cfg, "training", cfg), "warm_restarts_t_0", 10))
        t_mult = int(getattr(getattr(cfg, "training", cfg), "warm_restarts_t_mult", 1))
        eta_min = float(getattr(getattr(cfg, "training", cfg), "warm_restarts_eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)

    # ReduceLROnPlateau
    if name in {"reducelronplateau", "reducelronplateaulr", "reduce_on_plateau", "reducelronplateaulf"}:
        factor = float(getattr(getattr(cfg, "training", cfg), "plateau_factor", 0.1))
        patience = int(getattr(getattr(cfg, "training", cfg), "plateau_patience", 5))
        min_lr = float(getattr(getattr(cfg, "training", cfg), "plateau_min_lr", 0.0))
        mode = "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

    # OneCycleLR
    if name in {"onecyclelr", "one_cycle_lr", "onecycle", "one_cycle"}:
        max_lr = getattr(getattr(cfg, "training", cfg), "onecycle_max_lr", None)
        if max_lr is None:
            max_lr = float(getattr(getattr(cfg, "training", cfg), "learning_rate", 1e-3))
        else:
            max_lr = float(max_lr)

        epochs = int(getattr(getattr(cfg, "training", cfg), "epochs", 1))
        steps = int(steps_per_epoch or 1)
        pct_start = float(getattr(getattr(cfg, "training", cfg), "onecycle_pct_start", 0.3))
        div_factor = float(getattr(getattr(cfg, "training", cfg), "onecycle_div_factor", 25.0))
        final_div_factor = float(getattr(getattr(cfg, "training", cfg), "onecycle_final_div_factor", 1e4))

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(max_lr),
            epochs=epochs,
            steps_per_epoch=max(1, steps),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    raise ValueError(f"Unsupported scheduler: {name}")
