"""Loss factory for MedicalAI.

Provides a single entrypoint `create_loss(cfg)` which returns a torch.nn.Module
loss built from `TrainingConfig`-like objects. Reuses implementations from
`training.losses` and keeps backward-compatible defaults.

Supported loss names (case-insensitive):
 - dice
 - cross_entropy, ce
 - dice_cross_entropy, combined, dice+ce
 - weighted_cross_entropy, weighted_ce, wce
 - focal, focal_loss
 - tversky, tversky_loss
 - lovasz (optional) -> if unavailable raises ValueError
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from training import losses as _losses


def create_loss(cfg: object) -> nn.Module:
    """Create a loss module from a TrainingConfig-like object.

    This is a thin, explicit factory that maps config fields to constructors
    in `training.losses`.
    """

    loss_name = str(getattr(getattr(cfg, "training", cfg), "loss_name", "dice_cross_entropy")).lower()
    dice_weight = float(getattr(getattr(cfg, "training", cfg), "dice_weight", 1.0))

    # Dice
    if loss_name in {"dice", "diceloss"}:
        smooth = float(getattr(getattr(cfg, "training", cfg), "dice_smooth", 1.0))
        eps = float(getattr(getattr(cfg, "training", cfg), "dice_eps", 1e-7))
        return _losses.DiceLoss(smooth=smooth, eps=eps)

    # Cross entropy
    if loss_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss()

    # Combined (weighted CE + dice)
    if loss_name in {"dice_cross_entropy", "dice+ce", "combined", "dice_ce", "weighted_dice_cross_entropy", "weighted_dice_ce"}:
        dice_smooth = float(getattr(getattr(cfg, "training", cfg), "dice_smooth", 1.0))
        dice_eps = float(getattr(getattr(cfg, "training", cfg), "dice_eps", 1e-7))
        weights = getattr(getattr(cfg, "training", cfg), "weighted_ce_class_weights", None)
        return _losses.CombinedLoss(
            dice_weight=dice_weight,
            dice_smooth=dice_smooth,
            dice_eps=dice_eps,
            class_weights=weights,
        )

    # Weighted CE
    if loss_name in {"weighted_cross_entropy", "weighted_ce", "wce"}:
        weights = getattr(getattr(cfg, "training", cfg), "weighted_ce_class_weights", None)
        if weights is None:
            return nn.CrossEntropyLoss()
        class_weights = torch.tensor(list(weights), dtype=torch.float32)
        return _losses.WeightedCrossEntropyLoss(class_weights=class_weights)

    # Dice + Weighted Focal
    if loss_name in {"dice_weighted_focal", "dice_focal"}:
        dice_smooth = float(getattr(getattr(cfg, "training", cfg), "dice_smooth", 1.0))
        dice_eps = float(getattr(getattr(cfg, "training", cfg), "dice_eps", 1e-7))
        gamma = float(getattr(getattr(cfg, "training", cfg), "focal_gamma", 2.0))
        alpha = getattr(getattr(cfg, "training", cfg), "focal_alpha", None)
        alpha_t = None
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha_t = torch.tensor(list(alpha), dtype=torch.float32)
            else:
                alpha_t = torch.tensor(float(alpha), dtype=torch.float32)
        return _losses.DiceFocalLoss(
            dice_smooth=dice_smooth,
            dice_eps=dice_eps,
            focal_gamma=gamma,
            focal_alpha=alpha_t,
        )

    # Focal
    if loss_name in {"focal", "focal_loss"}:
        gamma = float(getattr(getattr(cfg, "training", cfg), "focal_gamma", 2.0))
        alpha = getattr(getattr(cfg, "training", cfg), "focal_alpha", None)
        alpha_t = None
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha_t = torch.tensor(list(alpha), dtype=torch.float32)
            else:
                alpha_t = torch.tensor(float(alpha), dtype=torch.float32)
        return _losses.FocalLoss(gamma=gamma, alpha=alpha_t)

    # Tversky
    if loss_name in {"tversky", "tversky_loss"}:
        alpha = float(getattr(getattr(cfg, "training", cfg), "tversky_alpha", 0.5))
        beta = float(getattr(getattr(cfg, "training", cfg), "tversky_beta", 0.5))
        smooth = float(getattr(getattr(cfg, "training", cfg), "tversky_smooth", 1.0))
        return _losses.TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    # Lovasz (optional) - not implemented here; keep placeholder for future
    if loss_name in {"lovasz", "lovasz_hinge", "lovasz_softmax"}:
        raise ValueError("Lovasz Loss not implemented in this workspace. Consider adding an implementation if needed.")

    raise ValueError(f"Unsupported loss_name: {loss_name}")
