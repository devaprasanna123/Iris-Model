"""MedicalAI evaluation utilities.

This module provides end-to-end evaluation for semantic segmentation.
It computes per-class metrics, confusion matrix, gallery images, and report
artifacts automatically.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.metrics import (
    MetricsSpec,
    boundary_iou,
    confusion_matrix,
    dice_score,
    f1_score,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
)

LABEL_COLORS_BGR = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 128, 128),
    (128, 0, 0),
    (0, 128, 0),
]


@dataclass
class EvaluationArtifactPaths:
    root: Path
    json: Path
    markdown: Path
    confusion_matrix: Path
    dice: Path
    iou: Path
    precision: Path
    recall: Path
    gallery: Path


def _safe_class_names(spec: MetricsSpec) -> List[str]:
    return [str(name) for name in spec.class_names]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _get_image_info(loader: Any, index: int) -> Tuple[Optional[Path], Optional[Path]]:
    dataset = getattr(loader, "dataset", None)
    for candidate in (dataset, getattr(dataset, "dataset", None)):
        if candidate is None:
            continue
        if hasattr(candidate, "_pairs"):
            pairs = getattr(candidate, "_pairs")
            if 0 <= index < len(pairs):
                return pairs[index]
    return None, None


def _merge_confusion_matrices(matrices: Sequence[torch.Tensor]) -> torch.Tensor:
    if not matrices:
        raise ValueError("No confusion matrices to merge")
    total = matrices[0].clone().to(torch.float64)
    for m in matrices[1:]:
        total += m.to(torch.float64)
    return total


def _per_class_metrics_from_confusion(cm: torch.Tensor, spec: MetricsSpec) -> Dict[str, Dict[str, float]]:
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    eps = spec.eps

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    output: Dict[str, Dict[str, float]] = {
        "dice": {},
        "iou": {},
        "precision": {},
        "recall": {},
        "f1": {},
    }
    class_names = _safe_class_names(spec)
    for i, name in enumerate(class_names):
        output["dice"][name] = float(dice[i].item())
        output["iou"][name] = float(iou[i].item())
        output["precision"][name] = float(precision[i].item())
        output["recall"][name] = float(recall[i].item())
        output["f1"][name] = float(f1[i].item())

    output["dice"]["mean"] = float(dice.mean().item())
    output["iou"]["mean"] = float(iou.mean().item())
    output["precision"]["mean"] = float(precision.mean().item())
    output["recall"]["mean"] = float(recall.mean().item())
    output["f1"]["mean"] = float(f1.mean().item())
    return output


def _colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color = LABEL_COLORS_BGR[cls % len(LABEL_COLORS_BGR)]
        out[mask == cls] = color
    return out


def _make_prediction_gallery_image(
    original_bgr: np.ndarray,
    target_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_names: Sequence[str],
    title: str,
) -> np.ndarray:
    resized = cv2.resize(original_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    gt_color = _colorize_mask(target_mask, num_classes=len(class_names))
    pred_color = _colorize_mask(pred_mask, num_classes=len(class_names))
    gt_overlay = cv2.addWeighted(resized, 0.6, gt_color, 0.4, 0)
    pred_overlay = cv2.addWeighted(resized, 0.6, pred_color, 0.4, 0)
    combined = np.concatenate([resized, gt_overlay, pred_overlay], axis=1)
    cv2.putText(
        combined,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return combined


def _plot_bar(metric: Dict[str, float], title: str, output_path: Path) -> None:
    labels = [k for k in metric.keys() if k != "mean"]
    values = [metric[k] for k in labels]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color="#4C72B0")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.savefig(output_path)
    plt.close()


def _plot_confusion_matrix(cm: torch.Tensor, class_names: Sequence[str], output_path: Path) -> None:
    matrix = cm.cpu().numpy().astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Ground Truth",
        xlabel="Prediction",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _create_artifact_paths(root: Path) -> EvaluationArtifactPaths:
    eval_root = root
    eval_root.mkdir(parents=True, exist_ok=True)
    gallery_dir = eval_root / "prediction_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    return EvaluationArtifactPaths(
        root=eval_root,
        json=eval_root / "evaluation.json",
        markdown=eval_root / "evaluation.md",
        confusion_matrix=eval_root / "confusion_matrix.png",
        dice=eval_root / "dice_per_class.png",
        iou=eval_root / "iou_per_class.png",
        precision=eval_root / "precision_per_class.png",
        recall=eval_root / "recall_per_class.png",
        gallery=gallery_dir,
    )


def _read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image for gallery: {path}")
    return image


def evaluate_model(
    *,
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    spec: MetricsSpec,
    output_root: Path,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    model.eval()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = _create_artifact_paths(output_root)

    class_names = _safe_class_names(spec)
    sample_summaries: List[Dict[str, Any]] = []
    confusion_matrices: List[torch.Tensor] = []
    total_correct = 0
    total_pixels = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            imgs, masks = batch
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1)

            confusion_matrices.append(
                confusion_matrix(preds, masks, spec=spec, input_is_logits=False)
            )
            total_correct += int((preds == masks).sum().item())
            total_pixels += int(masks.numel())
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            for sample_offset in range(preds.shape[0]):
                sample_idx = batch_index * test_loader.batch_size + sample_offset
                pred_sample = preds[sample_offset : sample_offset + 1]
                mask_sample = masks[sample_offset : sample_offset + 1]

                dice = float(dice_score(pred_sample, mask_sample, spec=spec, input_is_logits=False)["mean"])
                iou = float(iou_score(pred_sample, mask_sample, spec=spec, input_is_logits=False)["mean"])
                prec = float(precision_score(pred_sample, mask_sample, spec=spec, input_is_logits=False)["mean"])
                rec = float(recall_score(pred_sample, mask_sample, spec=spec, input_is_logits=False)["mean"])
                f1 = float(f1_score(pred_sample, mask_sample, spec=spec, input_is_logits=False)["mean"])
                acc = float(pixel_accuracy(pred_sample, mask_sample, spec=spec, input_is_logits=False))

                image_path, mask_path = _get_image_info(test_loader, sample_idx)
                sample_summaries.append(
                    {
                        "index": int(sample_idx),
                        "image_path": str(image_path) if image_path is not None else None,
                        "mask_path": str(mask_path) if mask_path is not None else None,
                        "dice": dice,
                        "iou": iou,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "pixel_accuracy": acc,
                    }
                )

    global_cm = _merge_confusion_matrices(confusion_matrices)
    global_cm = global_cm.to(torch.float64)
    per_class = _per_class_metrics_from_confusion(global_cm, spec)
    overall_pixel_accuracy = float(total_correct / total_pixels if total_pixels > 0 else 0.0)

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    boundary_scores: Optional[Dict[str, float]] = None
    try:
        boundary_scores = boundary_iou(
            all_preds_tensor,
            all_targets_tensor,
            spec=spec,
            input_is_logits=False,
        )
    except Exception:
        boundary_scores = None

    sample_summaries.sort(key=lambda x: x["dice"])
    worst_samples = sample_summaries[: top_k]
    best_samples = list(reversed(sample_summaries[-top_k:]))

    class_names_list = list(class_names)
    cm_stats = {
        "total_pixels": int(total_pixels),
        "overall_pixel_accuracy": overall_pixel_accuracy,
        "confusion_matrix": global_cm.tolist(),
        "class_names": class_names_list,
    }

    failure_pairs: List[Dict[str, Any]] = []
    for true_idx in range(global_cm.shape[0]):
        for pred_idx in range(global_cm.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(global_cm[true_idx, pred_idx].item())
            if count <= 0:
                continue
            failure_pairs.append(
                {
                    "true_class": class_names_list[true_idx],
                    "predicted_class": class_names_list[pred_idx],
                    "count": count,
                }
            )
    failure_pairs.sort(key=lambda x: x["count"], reverse=True)

    failure_analysis = {
        "most_confused_classes": failure_pairs[:10],
        "worst_classes_by_iou": sorted(
            [
                {
                    "class_name": class_names_list[i],
                    "iou": per_class["iou"][class_names_list[i]],
                    "dice": per_class["dice"][class_names_list[i]],
                }
                for i in range(len(class_names_list))
            ],
            key=lambda x: x["iou"],
        )[:3],
    }

    report = {
        "dataset_size": len(test_loader.dataset),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "checkpoint": checkpoint_metadata or {},
        "overall": {
            "pixel_accuracy": overall_pixel_accuracy,
            "dice_mean": per_class["dice"]["mean"],
            "iou_mean": per_class["iou"]["mean"],
            "precision_mean": per_class["precision"]["mean"],
            "recall_mean": per_class["recall"]["mean"],
            "f1_mean": per_class["f1"]["mean"],
        },
        "per_class": per_class,
        "confusion_matrix": cm_stats,
        "boundary_iou": boundary_scores,
        "failure_analysis": failure_analysis,
        "best_predictions": best_samples,
        "worst_predictions": worst_samples,
    }

    if boundary_scores is not None:
        report["boundary_iou"] = boundary_scores

    # Save plots and gallery
    _plot_confusion_matrix(global_cm, class_names_list, artifact_paths.confusion_matrix)
    _plot_bar(per_class["dice"], "Dice per class", artifact_paths.dice)
    _plot_bar(per_class["iou"], "IoU per class", artifact_paths.iou)
    _plot_bar(per_class["precision"], "Precision per class", artifact_paths.precision)
    _plot_bar(per_class["recall"], "Recall per class", artifact_paths.recall)

    # Save sample gallery images
    for category, items in (("worst", worst_samples), ("best", best_samples)):
        for rank, item in enumerate(items, start=1):
            if item.get("image_path") is None:
                continue
            image_path = Path(item["image_path"])
            try:
                original = _read_image_bgr(image_path)
            except Exception:
                continue
            mask_path = Path(item["mask_path"]) if item.get("mask_path") else None
            if mask_path is None or not mask_path.exists():
                continue
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                continue
            gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            pred_mask = all_preds_tensor[item["index"]].numpy().astype(np.uint8)
            pred_mask = cv2.resize(pred_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            title = (
                f"{category.title()} {rank}: {image_path.name} | "
                f"Dice={item['dice']:.4f} IoU={item['iou']:.4f}"
            )
            gallery_image = _make_prediction_gallery_image(
                original, gt_mask, pred_mask, class_names_list, title=title
            )
            out_path = artifact_paths.gallery / f"{category}_{rank:02d}_{image_path.stem}.png"
            cv2.imwrite(str(out_path), gallery_image)

    # Save JSON and markdown reports
    with open(artifact_paths.json, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2)

    _save_markdown_report(report, artifact_paths.markdown, class_names_list)

    return report


def _save_markdown_report(report: Dict[str, Any], path: Path, class_names: Sequence[str]) -> None:
    lines: List[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"**Dataset Size:** {report.get('dataset_size')}")
    checkpoint = report.get("checkpoint", {})
    if checkpoint:
        lines.append(f"**Checkpoint Epoch:** {checkpoint.get('epoch', 'N/A')}")
        lines.append(f"**Checkpoint Best Dice:** {checkpoint.get('best_dice', 'N/A')}")
        lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    overall = report["overall"]
    for key in ["pixel_accuracy", "dice_mean", "iou_mean", "precision_mean", "recall_mean", "f1_mean"]:
        lines.append(f"- **{key.replace('_', ' ').title()}**: {overall.get(key, 0.0):.6f}")
    lines.append("")
    if report.get("boundary_iou") is not None:
        lines.append("## Boundary IoU")
        lines.append("")
        for class_name in class_names:
            lines.append(f"- **{class_name}**: {report['boundary_iou'].get(class_name, 0.0):.6f}")
        lines.append("")
    lines.append("## Per-Class Metrics")
    lines.append("")
    lines.append("| Class | Dice | IoU | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    per_class = report["per_class"]
    for name in class_names:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                name,
                per_class["dice"][name],
                per_class["iou"][name],
                per_class["precision"][name],
                per_class["recall"][name],
                per_class["f1"][name],
            )
        )
    lines.append("")
    lines.append("## Failure Analysis")
    lines.append("")
    fa = report["failure_analysis"]
    if fa["most_confused_classes"]:
        lines.append("### Most Confused Class Pairs")
        for pair in fa["most_confused_classes"]:
            lines.append(
                f"- {pair['true_class']} -> {pair['predicted_class']}: {pair['count']} pixels"
            )
        lines.append("")
    if fa["worst_classes_by_iou"]:
        lines.append("### Worst Classes by IoU")
        for item in fa["worst_classes_by_iou"]:
            lines.append(
                f"- {item['class_name']}: IoU={item['iou']:.4f}, Dice={item['dice']:.4f}"
            )
        lines.append("")
    lines.append("## Prediction Gallery")
    lines.append("")
    lines.append("Saved images to `prediction_gallery/` including the 20 best and 20 worst predictions.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly.")
