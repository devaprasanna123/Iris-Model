"""MedicalAI semantic segmentation metrics (multiclass: background/cornea/iris).

Model outputs:
    - pred_logits: (B, C, H, W) where C=3
Ground truth:
    - target: (B, H, W) with integer class IDs in {0,1,2}

This module is intentionally framework-light:
    - Uses only: torch, torch.nn.functional
    - No sklearn/monai/torchmetrics

It provides reusable metrics for evaluation and can be imported by
trainer.py / evaluate.py later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union, overload

import torch
import torch.nn.functional as F


ClassName = Literal["background", "cornea", "iris"]


@dataclass(frozen=True)
class MetricsSpec:
    """Configuration for metrics."""

    num_classes: int = 3
    class_names: Tuple[ClassName, ...] = ("background", "cornea", "iris")
    eps: float = 1e-7


_DEFAULT_SPEC = MetricsSpec()


def _to_pred_labels(pred_logits: torch.Tensor) -> torch.Tensor:
    """Convert (B,C,H,W) logits to predicted labels (B,H,W)."""

    if pred_logits.ndim != 4:
        raise ValueError(
            f"pred_logits must have shape (B,C,H,W); got {tuple(pred_logits.shape)}"
        )
    # Argmax over class/channel dimension.
    return pred_logits.argmax(dim=1)


def _ensure_target_batch(target: torch.Tensor) -> torch.Tensor:
    """Ensure target has batch dimension.

    Accepts:
        - (H,W) -> (1,H,W)
        - (B,H,W) -> unchanged
    """

    if target.ndim == 2:
        return target.unsqueeze(0)
    if target.ndim == 3:
        return target
    raise ValueError(
        f"target must have shape (H,W) or (B,H,W); got {tuple(target.shape)}"
    )


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float) -> torch.Tensor:
    return num / (den + eps)


def _confusion_counts(
    pred_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute TP/FP/FN per class over entire batch.

    Args:
        pred_labels: (B,H,W) integer labels.
        target: (B,H,W) integer labels.
        num_classes: number of classes.

    Returns:
        (tp, fp, fn) each of shape (C,).

    Notes:
        - Computes per-class counts using vectorized boolean masks.
        - No ignore_index behavior is assumed (all labels count).
    """

    if pred_labels.ndim != 3 or target.ndim != 3:
        raise ValueError(
            "pred_labels and target must be (B,H,W) tensors "
            f"but got pred_labels={tuple(pred_labels.shape)} target={tuple(target.shape)}"
        )
    if pred_labels.shape != target.shape:
        raise ValueError(
            "pred_labels and target must have the same shape; "
            f"got pred_labels={tuple(pred_labels.shape)} target={tuple(target.shape)}"
        )

    # Flatten spatial + batch for simpler counting.
    pred_flat = pred_labels.reshape(-1)
    tgt_flat = target.reshape(-1)

    tp_list = []
    fp_list = []
    fn_list = []
    for cls in range(num_classes):
        pred_is = pred_flat == cls
        tgt_is = tgt_flat == cls
        tp = (pred_is & tgt_is).sum().to(dtype=torch.float32)
        fp = (pred_is & ~tgt_is).sum().to(dtype=torch.float32)
        fn = (~pred_is & tgt_is).sum().to(dtype=torch.float32)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    tp = torch.stack(tp_list, dim=0)  # (C,)
    fp = torch.stack(fp_list, dim=0)  # (C,)
    fn = torch.stack(fn_list, dim=0)  # (C,)
    return tp, fp, fn


def _overall_pixel_accuracy(pred_labels: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute pixel accuracy over entire batch."""

    if pred_labels.shape != target.shape:
        raise ValueError(
            f"pred_labels and target must match shapes; got {tuple(pred_labels.shape)} vs {tuple(target.shape)}"
        )

    correct = (pred_labels == target).sum().to(dtype=torch.float32)
    total = torch.tensor(target.numel(), device=target.device, dtype=torch.float32)
    return correct / (total + 1e-7)


def confusion_matrix(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> torch.Tensor:
    """Compute the confusion matrix for multiclass segmentation.

    Returns a matrix of shape (C, C) where rows are ground truth classes and
    columns are predicted classes.
    """

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    if pred_labels.shape != target_b.shape:
        raise ValueError(
            f"Shape mismatch after batching: pred_labels={tuple(pred_labels.shape)} target={tuple(target_b.shape)}"
        )

    num_classes = spec.num_classes
    pred_flat = pred_labels.reshape(-1)
    target_flat = target_b.reshape(-1)

    one_hot_pred = F.one_hot(pred_flat, num_classes=num_classes).to(dtype=torch.float64)
    one_hot_target = F.one_hot(target_flat, num_classes=num_classes).to(dtype=torch.float64)

    return one_hot_target.T @ one_hot_pred


def _extract_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Extract a binary boundary map from a class mask tensor."""

    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W); got {tuple(mask.shape)}")

    # Use morphological erosion on the binary mask to extract boundary pixels.
    mask_float = mask.to(dtype=torch.float32).unsqueeze(1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask_float, kernel_size=3, stride=1, padding=1)
    boundary = (mask_float - eroded) > 0.5
    return boundary.squeeze(1).to(dtype=torch.uint8)


def boundary_iou(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> Dict[str, float]:
    """Compute boundary IoU per class + mean."""

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    if pred_labels.shape != target_b.shape:
        raise ValueError(
            f"Shape mismatch after batching: pred_labels={tuple(pred_labels.shape)} target={tuple(target_b.shape)}"
        )

    output: Dict[str, float] = {}
    scores = []
    for i, class_name in enumerate(spec.class_names):
        pred_mask = (pred_labels == i).to(dtype=torch.uint8)
        target_mask = (target_b == i).to(dtype=torch.uint8)

        pred_boundary = _extract_boundary(pred_mask)
        target_boundary = _extract_boundary(target_mask)

        intersection = (pred_boundary & target_boundary).sum().to(dtype=torch.float32)
        union = (pred_boundary | target_boundary).sum().to(dtype=torch.float32)
        if union.item() == 0:
            score = 1.0
        else:
            score = float(intersection / (union + spec.eps))

        output[class_name] = float(score)
        scores.append(score)

    output["mean"] = float(sum(scores) / len(scores)) if scores else 0.0
    return output


def dice_score(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> Dict[str, float]:
    """Multiclass Dice score per class + mean.

    Args:
        pred_logits_or_labels: Either
            - logits (B,C,H,W)
            - labels (B,H,W) or (H,W)
        target: (B,H,W) integer labels.
        spec: MetricsSpec
        input_is_logits: If provided, forces interpretation.
            If None, inferred from tensor rank.

    Returns:
        dict with keys: background, cornea, iris, mean
    """

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    if pred_labels.shape != target_b.shape:
        raise ValueError(
            f"Shape mismatch after batching: pred_labels={tuple(pred_labels.shape)} target={tuple(target_b.shape)}"
        )

    tp, fp, fn = _confusion_counts(pred_labels, target_b, num_classes=spec.num_classes)

    # Dice per class: 2TP / (2TP + FP + FN)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + spec.eps)

    out: Dict[str, float] = {}
    for i, name in enumerate(spec.class_names):
        out[name] = float(dice[i].item())
    out["mean"] = float(dice.mean().item())
    return out


def iou_score(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> Dict[str, float]:
    """Multiclass IoU score per class + mean."""

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    if pred_labels.shape != target_b.shape:
        raise ValueError(
            f"Shape mismatch after batching: pred_labels={tuple(pred_labels.shape)} target={tuple(target_b.shape)}"
        )

    tp, fp, fn = _confusion_counts(pred_labels, target_b, num_classes=spec.num_classes)

    # IoU per class: TP / (TP + FP + FN)
    iou = _safe_div(tp, tp + fp + fn, spec.eps)

    out: Dict[str, float] = {}
    for i, name in enumerate(spec.class_names):
        out[name] = float(iou[i].item())
    out["mean"] = float(iou.mean().item())
    return out


def pixel_accuracy(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> float:
    """Pixel accuracy (overall): correct / total pixels."""

    _ = spec

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    if pred_labels.shape != target_b.shape:
        raise ValueError(
            f"Shape mismatch after batching: pred_labels={tuple(pred_labels.shape)} target={tuple(target_b.shape)}"
        )

    return float(_overall_pixel_accuracy(pred_labels, target_b).item())


def precision_score(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> Dict[str, float]:
    """Multiclass precision per class + mean."""

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    tp, fp, _fn = _confusion_counts(pred_labels, target_b, num_classes=spec.num_classes)

    precision = _safe_div(tp, tp + fp, spec.eps)

    out: Dict[str, float] = {}
    for i, name in enumerate(spec.class_names):
        out[name] = float(precision[i].item())
    out["mean"] = float(precision.mean().item())
    return out


def recall_score(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> Dict[str, float]:
    """Multiclass recall per class + mean."""

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    tp, _fp, fn = _confusion_counts(pred_labels, target_b, num_classes=spec.num_classes)

    recall = _safe_div(tp, tp + fn, spec.eps)

    out: Dict[str, float] = {}
    for i, name in enumerate(spec.class_names):
        out[name] = float(recall[i].item())
    out["mean"] = float(recall.mean().item())
    return out


def f1_score(
    pred_logits_or_labels: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
    input_is_logits: Optional[bool] = None,
) -> Dict[str, float]:
    """Multiclass F1 score per class + mean.

    Uses TP/FP/FN definition for consistency:
        F1 = 2TP / (2TP + FP + FN)
    """

    if input_is_logits is None:
        input_is_logits = pred_logits_or_labels.ndim == 4

    if input_is_logits:
        pred_labels = _to_pred_labels(pred_logits_or_labels)
    else:
        pred_labels = pred_logits_or_labels

    target_b = _ensure_target_batch(target)
    if pred_labels.ndim == 2:
        pred_labels = pred_labels.unsqueeze(0)

    tp, fp, fn = _confusion_counts(pred_labels, target_b, num_classes=spec.num_classes)

    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + spec.eps)

    out: Dict[str, float] = {}
    for i, name in enumerate(spec.class_names):
        out[name] = float(f1[i].item())
    out["mean"] = float(f1.mean().item())
    return out


def flatten_per_class_metrics(
    metrics: Dict[str, object],
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
) -> Dict[str, float]:
    """Flatten per-class metrics into a simple key/value mapping.

    Example output keys:
        background_dice, cornea_dice, iris_dice, background_iou, ...
    """

    flattened: Dict[str, float] = {}
    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_value, dict):
            continue
        if metric_name not in {"dice", "iou", "precision", "recall", "f1"}:
            continue
        for class_name in spec.class_names:
            if class_name in metric_value:
                flattened[f"{class_name}_{metric_name}"] = float(metric_value[class_name])
        if "mean" in metric_value:
            flattened[f"{metric_name}_mean"] = float(metric_value["mean"])
    return flattened


def evaluate_all_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: MetricsSpec = _DEFAULT_SPEC,
) -> Dict[str, object]:
    """Compute all required metrics and return a structured dict."""

    return {
        "dice": dice_score(pred_logits, target, spec=spec, input_is_logits=True),
        "iou": iou_score(pred_logits, target, spec=spec, input_is_logits=True),
        "pixel_accuracy": pixel_accuracy(
            pred_logits, target, spec=spec, input_is_logits=True
        ),
        "precision": precision_score(pred_logits, target, spec=spec, input_is_logits=True),
        "recall": recall_score(pred_logits, target, spec=spec, input_is_logits=True),
        "f1": f1_score(pred_logits, target, spec=spec, input_is_logits=True),
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    # Self-test using random tensors.
    B, C, H, W = 2, 3, 64, 64
    pred_logits = torch.randn(B, C, H, W)
    target = torch.randint(low=0, high=C, size=(B, H, W), dtype=torch.long)

    pred_labels = _to_pred_labels(pred_logits)

    print("Pred labels shape:", tuple(pred_labels.shape))
    print("Target shape:", tuple(target.shape))

    dice = dice_score(pred_logits, target, input_is_logits=True)
    iou = iou_score(pred_logits, target, input_is_logits=True)
    prec = precision_score(pred_logits, target, input_is_logits=True)
    rec = recall_score(pred_logits, target, input_is_logits=True)
    f1 = f1_score(pred_logits, target, input_is_logits=True)
    acc = pixel_accuracy(pred_logits, target, input_is_logits=True)

    print("\nDice:", dice)
    print("\nIoU:", iou)
    print("\nPrecision:", prec)
    print("\nRecall:", rec)
    print("\nF1:", f1)
    print("\nPixel Accuracy:", acc)

