"""MedicalAI model factory (v2 migration).

This module is responsible for:
- Reading model configuration
- Instantiating the requested segmentation architecture

It must return a model that outputs **logits** with shape (B, C, H, W)
where C == cfg.classes.number_of_classes.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from training.config import TrainingConfig
from training.models.unetplusplus import UNetPlusPlusModel
from training.models.unet import UNet

logger = logging.getLogger(__name__)


def create_model(cfg: TrainingConfig) -> torch.nn.Module:
    """Create the segmentation model based on ``cfg.model``.

    Args:
        cfg: Training configuration.

    Returns:
        A segmentation model that outputs logits (no activation).
    """

    if not hasattr(cfg, "model") or cfg.model is None:  # type: ignore[attr-defined]
        logger.info("cfg.model missing -> falling back to v1 UNet")
        return UNet(in_channels=int(cfg.image.channels), num_classes=int(cfg.classes.number_of_classes))

    architecture = str(getattr(cfg.model, "architecture", "unetplusplus")).lower()

    if architecture in {"unet++", "unetplusplus", "unet_plus_plus", "unetpp"}:
        return UNetPlusPlusModel(cfg=cfg)

    if architecture in {"unet", "unet-v1", "v1-unet"}:
        return UNet(in_channels=int(cfg.model.in_channels), num_classes=int(cfg.model.classes))

    raise ValueError(f"Unsupported model architecture: {architecture}")


def create_model_from_cfg_dict(cfg: dict[str, Any], base_cfg: TrainingConfig | None = None) -> torch.nn.Module:
    """Helper that can instantiate models from a dict-like config.

    Kept for future extensibility.
    """

    if base_cfg is None:
        base_cfg = TrainingConfig()

    # Best-effort override of base_cfg.model fields.
    model_cfg = getattr(base_cfg, "model")
    for k, v in cfg.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)

    return create_model(base_cfg)

