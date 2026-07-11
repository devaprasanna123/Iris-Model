"""MedicalAI training losses.

This module provides production-quality loss functions for semantic segmentation.

Model outputs:
    - logits: (B, C, H, W) where C=3 (Background=0, Cornea=1, Iris=2)

Ground truth masks:
    - target: (B, H, W) with integer class IDs in {0,1,2}

Only PyTorch is used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multiclass Dice loss computed from logits.

    Notes:
        - Applies softmax internally.
        - Uses one-hot targets.
        - Numerically stable via smoothing and epsilon.
        - Ignores division-by-zero by construction (smooth/eps terms).
        - Supports batch dimension.

    Args:
        smooth: Smoothing constant added to numerator and denominator.
            Typically 1.0 or 0.0 are used; 1.0 is common for stability.
        eps: Small epsilon to avoid numerical issues.
        class_dim: Channel dimension for logits (default: 1 for (B,C,H,W)).
        reduction: 'mean' (default) or 'none'.
    """

    def __init__(
        self,
        *,
        smooth: float = 1.0,
        eps: float = 1e-7,
        class_dim: int = 1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.class_dim = int(class_dim)
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute dice loss.

        Args:
            pred_logits: Model logits of shape (B, C, H, W).
            target: Ground truth integer masks of shape (B, H, W).

        Returns:
            Scalar loss if reduction='mean', else a tensor of shape (B,).
        """

        if pred_logits.ndim != 4:
            raise ValueError(
                f"pred_logits must have shape (B,C,H,W); got {tuple(pred_logits.shape)}"
            )
        if target.ndim != 3:
            raise ValueError(f"target must have shape (B,H,W); got {tuple(target.shape)}")

        # Ensure spatial dimensions match.
        if pred_logits.shape[0] != target.shape[0] or pred_logits.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                "Shape mismatch: pred_logits is (B,C,H,W) but target is (B,H,W) with different B/H/W"
            )

        c = pred_logits.shape[self.class_dim]
        b = pred_logits.shape[0]

        # Softmax over class/channel dimension.
        probs = F.softmax(pred_logits, dim=self.class_dim)

        # Convert target (B,H,W) -> one_hot (B,C,H,W)
        # scatter_ is used to avoid external dependencies.
        one_hot = torch.zeros(
            (b, c, *target.shape[-2:]),
            device=pred_logits.device,
            dtype=probs.dtype,
        )
        one_hot.scatter_(
            dim=1,
            index=target.unsqueeze(1).long(),
            value=1.0,
        )

        # Flatten spatial dims for dice computation.
        # intersection and sums computed per-class and per-batch.
        probs_flat = probs.reshape(b, c, -1)
        one_hot_flat = one_hot.reshape(b, c, -1)

        intersection = (probs_flat * one_hot_flat).sum(dim=-1)  # (B,C)
        probs_sum = probs_flat.sum(dim=-1)  # (B,C)
        target_sum = one_hot_flat.sum(dim=-1)  # (B,C)

        # Dice score per class:
        # dice_c = (2*|X∩Y| + smooth) / (|X| + |Y| + smooth)
        dice = (2.0 * intersection + self.smooth) / (
            probs_sum + target_sum + self.smooth + self.eps
        )  # (B,C)

        # Convert to loss: 1 - dice.
        loss_per_class = 1.0 - dice  # (B,C)

        # Mean over classes -> (B,)
        loss_per_batch = loss_per_class.mean(dim=1)

        if self.reduction == "none":
            return loss_per_batch
        return loss_per_batch.mean()


def cross_entropy_loss() -> torch.nn.CrossEntropyLoss:
    """Helper to create a standard CrossEntropyLoss.

    Returns:
        torch.nn.CrossEntropyLoss() with default settings.
    """

    return torch.nn.CrossEntropyLoss()


class CombinedLoss(nn.Module):
    """CrossEntropyLoss + dice_weight * DiceLoss."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        dice_eps: float = 1e-7,
        ignore_index: Optional[int] = None,
    ) -> None:
        """Initialize combined loss.

        Args:
            dice_weight: Weight applied to Dice loss.
            dice_smooth: Smoothing constant for DiceLoss.
            dice_eps: Epsilon for numerical stability in DiceLoss.
            ignore_index: Optional ignore_index for CrossEntropyLoss.
                (If None, CrossEntropyLoss will not ignore any label.)
        """

        super().__init__()
        self.dice_weight = float(dice_weight)

        # CrossEntropyLoss operates on logits and integer targets.
        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.dice = DiceLoss(smooth=dice_smooth, eps=dice_eps)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred_logits: (B, C, H, W)
            target: (B, H, W)

        Returns:
            Scalar combined loss.
        """

        ce_loss = self.ce(pred_logits, target)
        dice_loss = self.dice(pred_logits, target)
        return ce_loss + (self.dice_weight * dice_loss)


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy for multiclass segmentation.

    Expects:
      - logits: (B,C,H,W)
      - target: (B,H,W)

    Args:
      - class_weights: list/tuple/torch.Tensor of shape (C,)
      - ignore_index: optional
      - label_smoothing: optional float
    """

    def __init__(
        self,
        *,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index if ignore_index is not None else -100,
            label_smoothing=float(label_smoothing),
        )
        self.ignore_index = ignore_index

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is None:
            # We created CE with ignore_index=-100 to avoid errors; override behavior by using logits/target normally.
            return nn.CrossEntropyLoss(
                weight=self.ce.weight,
                ignore_index=-100,
                label_smoothing=self.ce.label_smoothing,
            )(pred_logits, target)
        return self.ce(pred_logits, target)


class FocalLoss(nn.Module):
    """Multiclass Focal Loss.

    Implementation based on CE with modulating factor.

    Args:
      - gamma: focusing parameter (>=0). gamma=0 reduces to CE.
      - alpha: optional class weights (vector of shape C or scalar in [0,1] for binary).
      - ignore_index: optional
      - reduction: 'mean' or 'sum' or 'none'
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean','sum','none'")
        self.gamma = float(gamma)
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred_logits.ndim != 4:
            raise ValueError(f"pred_logits must be (B,C,H,W), got {tuple(pred_logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"target must be (B,H,W), got {tuple(target.shape)}")

        # Compute log-probabilities
        log_probs = F.log_softmax(pred_logits, dim=1)  # (B,C,H,W)
        probs = torch.exp(log_probs)

        # Gather per-pixel probability of the target class
        target_unsq = target.unsqueeze(1)  # (B,1,H,W)
        pt = probs.gather(dim=1, index=target_unsq).squeeze(1)  # (B,H,W)
        log_pt = log_probs.gather(dim=1, index=target_unsq).squeeze(1)  # (B,H,W)

        if self.ignore_index is not None:
            valid = target != int(self.ignore_index)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        focal_weight = (1.0 - pt).clamp(min=0.0) ** self.gamma

        if self.alpha is not None:
            # alpha as class-wise weights
            if isinstance(self.alpha, torch.Tensor) and self.alpha.ndim == 1:
                alpha_t = self.alpha.to(pred_logits.device).gather(dim=0, index=target.view(-1)).view_as(target)
            else:
                # scalar alpha not supported here; treat as constant weight
                alpha_t = torch.as_tensor(self.alpha, device=pred_logits.device, dtype=pred_logits.dtype)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        loss = loss * valid.to(loss.dtype)

        if self.reduction == "mean":
            denom = valid.sum().clamp(min=1).to(loss.dtype)
            return loss.sum() / denom
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """Multiclass Tversky loss computed from logits.

    Uses one-hot targets and probabilities (softmax).

    For each class:
      TP, FP, FN computed over (B,H,W)
      Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
      loss = 1 - Tversky

    Args:
      - alpha: weight for FP
      - beta: weight for FN
      - smooth: smoothing
      - reduction: 'mean' or 'none'
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        eps: float = 1e-7,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.eps = float(eps)
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred_logits.ndim != 4:
            raise ValueError(f"pred_logits must be (B,C,H,W), got {tuple(pred_logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"target must be (B,H,W), got {tuple(target.shape)}")

        b, c, h, w = pred_logits.shape
        if target.shape[0] != b or target.shape[-2:] != (h, w):
            raise ValueError("Shape mismatch between logits and target")

        probs = F.softmax(pred_logits, dim=1)  # (B,C,H,W)
        one_hot = torch.zeros((b, c, h, w), device=pred_logits.device, dtype=probs.dtype)
        one_hot.scatter_(dim=1, index=target.unsqueeze(1).long(), value=1.0)

        probs_flat = probs.reshape(b, c, -1)
        one_hot_flat = one_hot.reshape(b, c, -1)

        tp = (probs_flat * one_hot_flat).sum(dim=-1)  # (B,C)
        fp = (probs_flat * (1.0 - one_hot_flat)).sum(dim=-1)  # (B,C)
        fn = ((1.0 - probs_flat) * one_hot_flat).sum(dim=-1)  # (B,C)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps)
        loss_per_class = 1.0 - tversky  # (B,C)
        loss_per_batch = loss_per_class.mean(dim=1)  # (B,)

        if self.reduction == "none":
            return loss_per_batch
        return loss_per_batch.mean()


def build_loss_from_config(cfg: "object") -> nn.Module:
    """Deprecated wrapper kept for backward-compatibility.

    Delegates to `training.loss_factory.create_loss(cfg)` which contains the
    canonical, configurable mapping for all supported losses.
    """

    try:
        from training.loss_factory import create_loss

        return create_loss(cfg)
    except Exception:
        # Fall back to a conservative default: CombinedLoss(dice_weight=1.0)
        return CombinedLoss(dice_weight=float(getattr(getattr(cfg, "training", cfg), "dice_weight", 1.0)))


if __name__ == "__main__":
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randint(low=0, high=3, size=(2, 32, 32))

    dice_loss_fn = DiceLoss()
    combined_loss_fn = CombinedLoss(dice_weight=1.0)

    dice_loss = dice_loss_fn(pred, target)
    combined_loss = combined_loss_fn(pred, target)

    print("Dice Loss:", dice_loss.item())
    print("Combined Loss:", combined_loss.item())


