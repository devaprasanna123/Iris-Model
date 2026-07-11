"""Optimizer factory for MedicalAI.

Creates optimizer instances from TrainingConfig without hardcoding values.
Supported optimizers: Adam, AdamW, SGD.
"""

from __future__ import annotations

import torch
from typing import Any


def _normalize(name: Any) -> str:
    return str(name).strip().lower()


def create_optimizer(cfg: object, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from `cfg` and model parameters.

    Expects `cfg.training` to contain fields:
      - optimizer
      - learning_rate
      - weight_decay
      - sgd_momentum (for SGD)

    Returns a torch optimizer instance.
    """

    name = _normalize(getattr(getattr(cfg, "training", cfg), "optimizer", "adamw"))
    lr = float(getattr(getattr(cfg, "training", cfg), "learning_rate", 1e-3))
    wd = float(getattr(getattr(cfg, "training", cfg), "weight_decay", 0.0))

    if name in {"adam", "torch.optim.adam"}:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if name in {"adamw", "adam_w", "torch.optim.adamw"}:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if name in {"sgd", "stochasticgradientdescent"}:
        momentum = float(getattr(getattr(cfg, "training", cfg), "sgd_momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

    raise ValueError(f"Unsupported optimizer: {name}")
