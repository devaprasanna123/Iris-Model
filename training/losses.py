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


if __name__ == "__main__":
    pred = torch.randn(2, 3, 512, 512)
    target = torch.randint(low=0, high=3, size=(2, 512, 512))

    dice_loss_fn = DiceLoss()
    combined_loss_fn = CombinedLoss(dice_weight=1.0)

    dice_loss = dice_loss_fn(pred, target)
    combined_loss = combined_loss_fn(pred, target)

    print("Dice Loss:", dice_loss.item())
    print("Combined Loss:", combined_loss.item())

