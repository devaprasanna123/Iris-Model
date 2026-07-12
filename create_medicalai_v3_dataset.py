#!/usr/bin/env python3
"""Generate a MedicalAI V3 iris-boundary dataset without touching the original dataset.

This script reads the existing MedicalAI dataset split folders, extracts the iris class
from each segmentation mask, generates:
- a binary iris mask
- an iris-only image with everything outside the iris set to black
- a padded crop around the iris
- ordered posterior-iris boundary points
- a smoothed boundary representation
- an overlay image with the posterior boundary over the original image
- per-sample metadata and a combined dataset report

The original dataset remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent / "dataset"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "MedicalAI_V3"
PADDING_PIXELS = 40


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create MedicalAI V3 iris boundary dataset")
    parser.add_argument("--source_root", type=str, default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--padding", type=int, default=PADDING_PIXELS)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_image_mask_pairs(images_dir: Path, masks_dir: Path, split_name: str) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    supported_exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in supported_exts:
            continue
        stem = img_path.stem
        mask_path = masks_dir / f"{stem}.png"
        if mask_path.exists() and mask_path.is_file():
            pairs.append((img_path, mask_path))
    return pairs


def find_annotation_path(annotations_dir: Path, stem: str) -> Path | None:
    for candidate in sorted(annotations_dir.glob("*.json")):
        if candidate.stem == stem:
            return candidate
    return None


def make_binary_iris_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    iris_mask = (mask_u8 == 2).astype(np.uint8)
    if iris_mask.sum() == 0:
        # Some source masks only encode iris as class 1 or do not expose a distinct iris label.
        # Fall back to the largest connected non-background region.
        foreground = (mask_u8 > 0).astype(np.uint8)
        if foreground.sum() == 0:
            return np.zeros_like(mask_u8, dtype=np.uint8)

        # Use connected components and select the largest region.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        if num_labels <= 1:
            return foreground

        largest_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        iris_mask = (labels == largest_component).astype(np.uint8)
    return iris_mask


def smooth_contour(points: np.ndarray, num_points: int = 128) -> np.ndarray:
    if len(points) < 3:
        return points.astype(np.int32)

    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 1:
        return pts.astype(np.int32)

    # Remove duplicates while preserving order.
    unique_pts = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - unique_pts[-1]) > 1e-6:
            unique_pts.append(p)
    pts = np.array(unique_pts, dtype=np.float32)

    if len(pts) < 3:
        return pts.astype(np.int32)

    dist = np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(dist)])
    target_dist = np.linspace(0.0, cum_dist[-1], num_points, endpoint=False)

    x_smooth = np.interp(target_dist, cum_dist[:-1], pts[:, 0], period=cum_dist[-1])
    y_smooth = np.interp(target_dist, cum_dist[:-1], pts[:, 1], period=cum_dist[-1])
    return np.column_stack([x_smooth, y_smooth]).astype(np.int32)


def order_contour(points: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    if len(points) == 0:
        return points
    # Rotate contour so the first point is the left-most point (proxy for scleral-spur-side start).
    h, w = image_shape
    centroid = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    point_order = np.argsort(np.linalg.norm(points - centroid, axis=1))
    ordered = points[point_order]
    left_idx = int(np.argmin(ordered[:, 0]))
    ordered = np.vstack([ordered[left_idx:], ordered[:left_idx]])
    return ordered.astype(np.int32)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_pdf(report_path: Path, stats: Dict[str, Any], sample_examples: List[Dict[str, Any]], quality_summary: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("MedicalAI V3 Dataset Report", fontsize=14)

    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.02,
        0.98,
        "Dataset Overview\n\n"
        f"Total images: {stats['total_images']}\n"
        f"Images with iris: {stats['images_with_iris']}\n"
        f"Average iris area: {stats['average_iris_area']:.2f}px\n"
        f"Average bounding box area: {stats['average_bbox_area']:.2f}px\n"
        f"Average boundary points: {stats['average_boundary_points']:.2f}",
        va="top",
        ha="left",
        fontsize=10,
    )

    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.02,
        0.98,
        "Folder Structure\n\n"
        "MedicalAI_V3/\n"
        "  train/images\n"
        "  train/iris_masks\n"
        "  train/iris_only\n"
        "  train/cropped_iris\n"
        "  train/posterior_boundary\n"
        "  train/posterior_boundary_smooth\n"
        "  train/overlay\n"
        "  train/metadata\n"
        "  val/...\n"
        "  test/...\n",
        va="top",
        ha="left",
        fontsize=10,
    )

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.02,
        0.98,
        "Quality Control Summary\n\n"
        f"Accepted: {quality_summary['accepted']}\n"
        f"Rejected: {quality_summary['rejected']}\n"
        f"Split counts: {quality_summary['split_counts']}",
        va="top",
        ha="left",
        fontsize=10,
    )

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "Example Samples\n\n" + "\n".join(
            [f"- {item['stem']} ({item['split']})" for item in sample_examples[:6]]
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def process_split(
    split_name: str,
    image_dir: Path,
    mask_dir: Path,
    output_root: Path,
    annotations_dir: Path,
    padding: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    split_output_root = output_root / split_name
    split_images_dir = ensure_dir(split_output_root / "images")
    split_iris_masks_dir = ensure_dir(split_output_root / "iris_masks")
    split_iris_only_dir = ensure_dir(split_output_root / "iris_only")
    split_cropped_dir = ensure_dir(split_output_root / "cropped_iris")
    split_boundary_dir = ensure_dir(split_output_root / "posterior_boundary")
    split_boundary_smooth_dir = ensure_dir(split_output_root / "posterior_boundary_smooth")
    split_overlay_dir = ensure_dir(split_output_root / "overlay")
    split_metadata_dir = ensure_dir(split_output_root / "metadata")
    split_rejected_dir = ensure_dir(output_root / "rejected" / split_name)

    pairs = find_image_mask_pairs(image_dir, mask_dir, split_name)
    valid_samples: List[Dict[str, Any]] = []
    rejected_samples: List[Dict[str, Any]] = []
    stats_values: List[Dict[str, Any]] = []

    for img_path, mask_path in pairs:
        stem = img_path.stem
        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not read source image")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError("Could not read source mask")
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.astype(np.uint8)

            iris_mask = make_binary_iris_mask(mask)
            if iris_mask.sum() == 0:
                raise ValueError("No iris pixels found")

            # Save binary iris mask.
            iris_mask_path = split_iris_masks_dir / f"{stem}.png"
            cv2.imwrite(str(iris_mask_path), iris_mask * 255)

            # Save iris-only image.
            iris_only = image.copy()
            iris_only[iris_mask == 0] = 0
            iris_only_path = split_iris_only_dir / f"{stem}.png"
            cv2.imwrite(str(iris_only_path), iris_only)

            # Save copy of original image in the split images folder.
            image_out_path = split_images_dir / f"{stem}.png"
            cv2.imwrite(str(image_out_path), image)

            # Bounding box with padding.
            ys, xs = np.nonzero(iris_mask)
            if len(xs) == 0 or len(ys) == 0:
                raise ValueError("Bounding box could not be computed")
            x0 = max(0, int(xs.min()) - padding)
            y0 = max(0, int(ys.min()) - padding)
            x1 = min(image.shape[1], int(xs.max()) + padding + 1)
            y1 = min(image.shape[0], int(ys.max()) + padding + 1)
            cropped = image[y0:y1, x0:x1]
            cropped_path = split_cropped_dir / f"{stem}.png"
            cv2.imwrite(str(cropped_path), cropped)

            # Posterior boundary extraction.
            contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                raise ValueError("No contour found")
            contour = max(contours, key=cv2.contourArea)
            contour_points = contour[:, 0, :]
            ordered_points = order_contour(contour_points, iris_mask.shape)
            smooth_points = smooth_contour(ordered_points)

            boundary_path = split_boundary_dir / f"{stem}.json"
            smooth_boundary_path = split_boundary_smooth_dir / f"{stem}.json"
            save_json(boundary_path, [[int(x), int(y)] for x, y in ordered_points.tolist()])
            save_json(smooth_boundary_path, [[int(x), int(y)] for x, y in smooth_points.tolist()])

            # Overlay.
            overlay = image.copy()
            cv2.polylines(overlay, [smooth_points.reshape(-1, 1, 2)], isClosed=False, color=(255, 0, 0), thickness=2)
            overlay_path = split_overlay_dir / f"{stem}.png"
            cv2.imwrite(str(overlay_path), overlay)

            # Metadata.
            annotation_path = find_annotation_path(annotations_dir, stem)
            metadata = {
                "image_name": stem,
                "original_image_path": str(img_path),
                "original_mask_path": str(mask_path),
                "binary_iris_mask_path": str(iris_mask_path),
                "iris_only_path": str(iris_only_path),
                "cropped_iris_path": str(cropped_path),
                "posterior_boundary_path": str(boundary_path),
                "posterior_boundary_smooth_path": str(smooth_boundary_path),
                "overlay_path": str(overlay_path),
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "bounding_box": {
                    "x": int(x0),
                    "y": int(y0),
                    "width": int(x1 - x0),
                    "height": int(y1 - y0),
                },
                "pixel_count": int(iris_mask.sum()),
                "iris_area_pixels": int(iris_mask.sum()),
                "split": split_name,
                "padding": padding,
                "source_annotation_path": str(annotation_path) if annotation_path else None,
            }
            metadata_path = split_metadata_dir / f"{stem}.json"
            save_json(metadata_path, metadata)

            valid_samples.append({
                "stem": stem,
                "split": split_name,
                "mask_area": int(iris_mask.sum()),
                "bbox_area": int((x1 - x0) * (y1 - y0)),
                "boundary_points": len(smooth_points),
            })
            stats_values.append({
                "stem": stem,
                "iris_area": int(iris_mask.sum()),
                "bbox_area": int((x1 - x0) * (y1 - y0)),
                "boundary_points": len(smooth_points),
            })
        except Exception as exc:  # noqa: BLE001
            rejected_samples.append({
                "stem": stem,
                "split": split_name,
                "reason": str(exc),
            })
            rejected_item_dir = split_rejected_dir / stem
            rejected_item_dir.mkdir(parents=True, exist_ok=True)
            (rejected_item_dir / "reason.txt").write_text(str(exc), encoding="utf-8")
            # Copy the source image and mask into the rejection folder for inspection.
            shutil.copy2(str(img_path), rejected_item_dir / img_path.name)
            mask_copy_name = str(mask_path.name)
            shutil.copy2(str(mask_path), rejected_item_dir / mask_copy_name)

    stats = {
        "split": split_name,
        "total_images": len(pairs),
        "images_with_iris": len(valid_samples),
        "average_iris_area": float(np.mean([s["mask_area"] for s in valid_samples])) if valid_samples else 0.0,
        "minimum_iris_area": int(min([s["mask_area"] for s in valid_samples])) if valid_samples else 0,
        "maximum_iris_area": int(max([s["mask_area"] for s in valid_samples])) if valid_samples else 0,
        "average_bbox_area": float(np.mean([s["bbox_area"] for s in valid_samples])) if valid_samples else 0.0,
        "average_boundary_points": float(np.mean([s["boundary_points"] for s in valid_samples])) if valid_samples else 0.0,
    }
    return valid_samples, rejected_samples, stats


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")

    annotations_dir = source_root / "annotations"
    if not annotations_dir.exists():
        annotations_dir.mkdir(parents=True, exist_ok=True)

    all_samples: List[Dict[str, Any]] = []
    all_rejected: List[Dict[str, Any]] = []
    split_stats: List[Dict[str, Any]] = []

    for split_name in args.splits:
        source_split_root = source_root / split_name
        image_dir = source_split_root / "images"
        mask_dir = source_split_root / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            continue

        valid_samples, rejected_samples, stats = process_split(
            split_name=split_name,
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_root=output_root,
            annotations_dir=annotations_dir,
            padding=args.padding,
        )
        all_samples.extend(valid_samples)
        all_rejected.extend(rejected_samples)
        split_stats.append(stats)

    dataset_statistics = {
        "total_images": len(all_samples),
        "images_with_iris": len(all_samples),
        "average_iris_area": float(np.mean([s["mask_area"] for s in all_samples])) if all_samples else 0.0,
        "minimum_iris_area": int(min([s["mask_area"] for s in all_samples])) if all_samples else 0,
        "maximum_iris_area": int(max([s["mask_area"] for s in all_samples])) if all_samples else 0,
        "average_bbox_area": float(np.mean([s["bbox_area"] for s in all_samples])) if all_samples else 0.0,
        "average_boundary_points": float(np.mean([s["boundary_points"] for s in all_samples])) if all_samples else 0.0,
        "per_split": split_stats,
    }
    save_json(output_root / "dataset_statistics.json", dataset_statistics)

    quality_report = {
        "accepted": len(all_samples),
        "rejected": len(all_rejected),
        "split_counts": {stat["split"]: stat["total_images"] for stat in split_stats},
        "rejected_samples": all_rejected,
    }
    save_json(output_root / "quality_report.json", quality_report)
    save_json(output_root / "rejected_report.json", {"rejected_samples": all_rejected})

    sample_examples = [
        {"stem": sample["stem"], "split": sample["split"]}
        for sample in all_samples[:6]
    ]
    generate_pdf(
        output_root / "dataset_report.pdf",
        dataset_statistics,
        sample_examples,
        quality_report,
    )

    print(f"Created MedicalAI V3 dataset at {output_root}")
    print(json.dumps(dataset_statistics, indent=2))


if __name__ == "__main__":
    main()
