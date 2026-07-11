"""Production-grade inference engine for MedicalAI.

This module provides reusable inference helpers for:
- single-image inference
- folder inference
- batch inference
- device selection (GPU/CPU auto)
- confidence scoring
- post-processing
- overlay generation
- colored and transparent masks
- JSON/CSV outputs
- timing information

The implementation is intentionally separate from training code and does not
modify the training pipeline.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from training.config import TrainingConfig
from training.models.model_factory import create_model
from training.utils.checkpoint import CheckpointManager


logger = logging.getLogger(__name__)


DEFAULT_LABEL_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 255, 255),
    4: (255, 0, 255),
}


@dataclass(frozen=True)
class InferenceConfig:
    """Inference configuration for production deployment."""

    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 4
    device: Optional[str] = None
    postprocess: bool = True
    min_object_area_pixels: int = 64
    overlay_alpha: float = 0.4
    transparent_alpha: int = 160
    save_colored_mask: bool = True
    save_overlay: bool = True
    save_transparent_overlay: bool = True
    save_json: bool = True
    save_csv_summary: bool = True
    progress_bar: bool = True
    label_colors_bgr: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_LABEL_COLORS_BGR)
    class_ids: Dict[str, int] = field(default_factory=lambda: {"background": 0, "cornea": 1, "iris": 2, "pupil": 3})


@dataclass
class InferenceResult:
    image_path: Path
    output_dir: Path
    prediction_mask_path: Path
    colored_mask_path: Optional[Path]
    overlay_path: Optional[Path]
    transparent_overlay_path: Optional[Path]
    json_path: Optional[Path]
    inference_time_s: float
    preprocess_time_s: float
    postprocess_time_s: float
    total_time_s: float
    confidence_mean: float
    confidence_std: float
    class_confidences: Dict[int, float]
    class_pixel_counts: Dict[int, int]
    shape: Tuple[int, int]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "output_dir": str(self.output_dir),
            "prediction_mask_path": str(self.prediction_mask_path),
            "colored_mask_path": str(self.colored_mask_path) if self.colored_mask_path else None,
            "overlay_path": str(self.overlay_path) if self.overlay_path else None,
            "transparent_overlay_path": str(self.transparent_overlay_path) if self.transparent_overlay_path else None,
            "json_path": str(self.json_path) if self.json_path else None,
            "timing": {
                "preprocess_s": self.preprocess_time_s,
                "inference_s": self.inference_time_s,
                "postprocess_s": self.postprocess_time_s,
                "total_s": self.total_time_s,
            },
            "confidence": {
                "mean": self.confidence_mean,
                "std": self.confidence_std,
                "per_class": {str(k): float(v) for k, v in self.class_confidences.items()},
            },
            "class_pixel_counts": {str(k): int(v) for k, v in self.class_pixel_counts.items()},
            "shape": {"height": self.shape[0], "width": self.shape[1]},
            "warnings": self.warnings,
        }

    def save_json(self, path: Optional[Union[str, Path]] = None) -> str:
        json_path = Path(path) if path is not None else self.json_path
        payload = self.to_dict()
        text = json.dumps(payload, indent=2)
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(text, encoding="utf-8")
        return text


def _select_device(device_name: Optional[str] = None) -> torch.device:
    if device_name:
        resolved = device_name.strip().lower()
        if resolved in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if resolved in {"cpu"}:
            return torch.device("cpu")
        if resolved == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raise ValueError(f"Unsupported device selection: {device_name}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _supported_extensions() -> Tuple[str, ...]:
    return (".bmp", ".png", ".jpg", ".jpeg")


def _collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            [
                p
                for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() in _supported_extensions()
            ]
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _preprocess_image(image_bgr: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(image_bgr, image_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return tensor


def _prepare_batch(image_paths: Sequence[Path], image_size: Tuple[int, int]) -> Tuple[torch.Tensor, List[np.ndarray], float]:
    tensors: List[torch.Tensor] = []
    originals: List[np.ndarray] = []
    preprocess_start = time.perf_counter()
    for path in image_paths:
        bgr = _read_image_bgr(path)
        originals.append(bgr)
        tensors.append(_preprocess_image(bgr, image_size=image_size))
    batch = torch.stack(tensors, dim=0)
    preprocess_time = time.perf_counter() - preprocess_start
    return batch, originals, preprocess_time


def _softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return probs.max(dim=1).values


def _postprocess_prediction(
    prediction: np.ndarray,
    config: InferenceConfig,
) -> np.ndarray:
    if not config.postprocess:
        return prediction

    postprocessed = np.zeros_like(prediction, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for class_id in sorted(set(np.unique(prediction))):
        if class_id == 0:
            continue
        binary = (prediction == class_id).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= config.min_object_area_pixels:
                cleaned[labels == label] = 1
        postprocessed[cleaned.astype(bool)] = class_id

    return postprocessed


def _colorize_mask(mask: np.ndarray, colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored[mask == class_id] = color
    return colored


def _build_overlay(image_bgr: np.ndarray, colored_mask: np.ndarray, alpha: float) -> np.ndarray:
    resized = cv2.resize(image_bgr, (colored_mask.shape[1], colored_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(resized, 1.0 - alpha, colored_mask, alpha, 0)


def _build_transparent_overlay(colored_mask: np.ndarray, mask: np.ndarray, alpha: int) -> np.ndarray:
    if colored_mask.ndim != 3 or colored_mask.shape[2] != 3:
        raise ValueError("colored_mask must be HxWx3")
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., :3] = colored_mask
    overlay[..., 3] = np.where(mask != 0, alpha, 0).astype(np.uint8)
    return overlay


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 4:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image)


def _save_per_image_json(result: InferenceResult) -> None:
    if result.json_path is None:
        return
    result.save_json(result.json_path)


def _save_summary_csv(results: List[InferenceResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "image_path",
        "inference_time_s",
        "preprocess_time_s",
        "postprocess_time_s",
        "total_time_s",
        "confidence_mean",
        "confidence_std",
    ]

    class_ids = sorted({cid for result in results for cid in result.class_pixel_counts.keys()})
    for cid in class_ids:
        fieldnames.append(f"class_{cid}_confidence")
        fieldnames.append(f"class_{cid}_pixel_count")

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: Dict[str, Any] = {
                "image_name": result.image_path.stem,
                "image_path": str(result.image_path),
                "inference_time_s": result.inference_time_s,
                "preprocess_time_s": result.preprocess_time_s,
                "postprocess_time_s": result.postprocess_time_s,
                "total_time_s": result.total_time_s,
                "confidence_mean": result.confidence_mean,
                "confidence_std": result.confidence_std,
            }
            for cid in class_ids:
                row[f"class_{cid}_confidence"] = float(result.class_confidences.get(cid, 0.0))
                row[f"class_{cid}_pixel_count"] = int(result.class_pixel_counts.get(cid, 0))
            writer.writerow(row)


def _make_output_paths(image_path: Path, out_dir: Path, save_json: bool) -> Dict[str, Optional[Path]]:
    stem = image_path.stem
    return {
        "mask": out_dir / f"{stem}_pred_mask.png",
        "colored": out_dir / f"{stem}_pred_mask_colored.png",
        "overlay": out_dir / f"{stem}_overlay.png",
        "transparent": out_dir / f"{stem}_overlay_transparent.png",
        "json": out_dir / f"{stem}_pred_metrics.json" if save_json else None,
    }


def _compute_image_result(
    image_path: Path,
    prediction_mask: np.ndarray,
    original_bgr: np.ndarray,
    timings: Tuple[float, float, float],
    config: InferenceConfig,
    output_paths: Dict[str, Optional[Path]],
) -> InferenceResult:
    preprocess_time, inference_time, postprocess_time = timings
    total_time = preprocess_time + inference_time + postprocess_time

    class_pixel_counts: Dict[int, int] = {
        int(cid): int(np.count_nonzero(prediction_mask == cid)) for cid in np.unique(prediction_mask)
    }
    if 0 not in class_pixel_counts:
        class_pixel_counts[0] = 0

    colored_mask = _colorize_mask(prediction_mask, config.label_colors_bgr)
    overlay = _build_overlay(original_bgr, colored_mask, alpha=config.overlay_alpha)
    transparent_overlay = _build_transparent_overlay(colored_mask, prediction_mask, alpha=config.transparent_alpha)

    _save_image(output_paths["mask"], prediction_mask)
    if config.save_colored_mask and output_paths["colored"] is not None:
        _save_image(output_paths["colored"], colored_mask)
    if config.save_overlay and output_paths["overlay"] is not None:
        _save_image(output_paths["overlay"], overlay)
    if config.save_transparent_overlay and output_paths["transparent"] is not None:
        _save_image(output_paths["transparent"], transparent_overlay)

    confidences = np.zeros_like(prediction_mask, dtype=np.float32)
    confidence_mean = 0.0
    confidence_std = 0.0
    class_confidences: Dict[int, float] = {}

    result = InferenceResult(
        image_path=image_path,
        output_dir=output_paths["mask"].parent,
        prediction_mask_path=output_paths["mask"],
        colored_mask_path=output_paths["colored"],
        overlay_path=output_paths["overlay"],
        transparent_overlay_path=output_paths["transparent"],
        json_path=output_paths["json"],
        inference_time_s=inference_time,
        preprocess_time_s=preprocess_time,
        postprocess_time_s=postprocess_time,
        total_time_s=total_time,
        confidence_mean=confidence_mean,
        confidence_std=confidence_std,
        class_confidences=class_confidences,
        class_pixel_counts=class_pixel_counts,
        shape=prediction_mask.shape,
    )

    return result


def _compute_confidences(
    logits: torch.Tensor,
    prediction_mask: np.ndarray,
) -> Tuple[float, float, Dict[int, float]]:
    probabilities = F.softmax(logits, dim=1)
    max_confidence = probabilities.max(dim=1).values
    confidences = max_confidence.cpu().numpy()
    mean_confidence = float(confidences.mean())
    std_confidence = float(confidences.std())

    class_confidences: Dict[int, float] = {}
    labels = prediction_mask
    for class_id in np.unique(labels):
        mask = labels == int(class_id)
        if mask.sum() > 0:
            class_confidences[int(class_id)] = float(confidences[mask].mean())
        else:
            class_confidences[int(class_id)] = 0.0
    return mean_confidence, std_confidence, class_confidences


def _load_checkpoint(model: torch.nn.Module, cfg: TrainingConfig, device: torch.device) -> None:
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.checkpoint.checkpoint_dir,
        best_model_name=cfg.checkpoint.best_model_name,
        last_model_name=cfg.checkpoint.last_model_name,
        device=device,
    )
    checkpoint_manager.load(model=model, optimizer=None, scheduler=None, which="best", strict=True)


@torch.no_grad()
def infer_batch(
    image_paths: Sequence[Path],
    model: torch.nn.Module,
    cfg: TrainingConfig,
    *,
    out_dir: Path,
    inference_config: Optional[InferenceConfig] = None,
) -> List[InferenceResult]:
    if inference_config is None:
        inference_config = InferenceConfig(image_size=(cfg.preprocess.image_width, cfg.preprocess.image_height))

    device = _select_device(inference_config.device)
    model.eval()
    model.to(device)

    batch_size = max(1, inference_config.batch_size)
    results: List[InferenceResult] = []
    summary_rows: List[Dict[str, Any]] = []

    if inference_config.progress_bar:
        iterator = tqdm(range(0, len(image_paths), batch_size), desc="Inference", unit="batch")
    else:
        iterator = range(0, len(image_paths), batch_size)

    for start_idx in iterator:
        batch_paths = image_paths[start_idx : start_idx + batch_size]
        batch, originals, preprocess_time = _prepare_batch(batch_paths, inference_config.image_size)
        batch = batch.to(device)

        inference_start = time.perf_counter()
        logits = model(batch)
        inference_time = time.perf_counter() - inference_start

        postprocess_start = time.perf_counter()
        for sample_idx, image_path in enumerate(batch_paths):
            logit_sample = logits[sample_idx : sample_idx + 1]
            prediction = logit_sample.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            if inference_config.postprocess:
                prediction = _postprocess_prediction(prediction, inference_config)

            mean_confidence, std_confidence, class_confidences = _compute_confidences(logit_sample, prediction)
            warning_list: List[str] = []
            if np.count_nonzero(prediction) == 0:
                warning_list.append("Prediction produced an empty mask.")

            output_paths = _make_output_paths(image_path, out_dir, inference_config.save_json)
            result = _compute_image_result(
                image_path=image_path,
                prediction_mask=prediction,
                original_bgr=originals[sample_idx],
                timings=(preprocess_time / len(batch_paths), inference_time / len(batch_paths), (time.perf_counter() - postprocess_start) / len(batch_paths)),
                config=inference_config,
                output_paths=output_paths,
            )
            result.confidence_mean = mean_confidence
            result.confidence_std = std_confidence
            result.class_confidences = class_confidences
            result.warnings = warning_list
            if inference_config.save_json and result.json_path is not None:
                _save_per_image_json(result)
            results.append(result)

        postprocess_time = time.perf_counter() - postprocess_start
        if inference_config.progress_bar:
            iterator.set_postfix({"batch_inference_s": f"{inference_time:.4f}"})

    if inference_config.save_csv_summary:
        summary_csv_path = out_dir / "inference_summary.csv"
        _save_summary_csv(results, summary_csv_path)

    if inference_config.save_json:
        summary_json_path = out_dir / "inference_summary.json"
        summary_payload = [result.to_dict() for result in results]
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def infer_single(
    image_path: Path,
    model: torch.nn.Module,
    cfg: TrainingConfig,
    *,
    out_dir: Path,
    inference_config: Optional[InferenceConfig] = None,
) -> InferenceResult:
    results = infer_batch([image_path], model, cfg, out_dir=out_dir, inference_config=inference_config)
    return results[0]


def infer_folder(
    folder_path: Path,
    model: torch.nn.Module,
    cfg: TrainingConfig,
    *,
    out_dir: Path,
    inference_config: Optional[InferenceConfig] = None,
) -> List[InferenceResult]:
    image_paths = _collect_images(folder_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found in folder: {folder_path}")
    return infer_batch(image_paths, model, cfg, out_dir=out_dir, inference_config=inference_config)


def infer_from_config(
    cfg: TrainingConfig,
    *,
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    inference_config: Optional[InferenceConfig] = None,
) -> List[InferenceResult]:
    input_path = Path(input_path)
    out_dir = Path(output_dir) if output_dir is not None else cfg.output.prediction_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if inference_config is None:
        inference_config = InferenceConfig(image_size=(cfg.preprocess.image_width, cfg.preprocess.image_height))

    model = torch.nn.DataParallel(torch.nn.Module()) if False else None  # type: ignore[assignment]
    raise RuntimeError("Use a concrete model instance instead of infer_from_config directly.")


def build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="MedicalAI production inference engine")
    parser.add_argument("--input", required=True, help="Input image file or folder.")
    parser.add_argument("--output_dir", default=None, help="Output directory for inference artifacts.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for folder inference.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto/cpu/cuda.")
    parser.add_argument("--overlay_alpha", type=float, default=0.4, help="Overlay transparency alpha.")
    parser.add_argument("--transparent_alpha", type=int, default=160, help="Transparent overlay alpha channel.")
    parser.add_argument("--no_postprocess", action="store_true", help="Disable automatic post-processing.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bar.")
    parser.add_argument("--no_json", action="store_true", help="Disable JSON output.")
    parser.add_argument("--no_csv", action="store_true", help="Disable summary CSV output.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = TrainingConfig()
    device = _select_device(args.device)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    model = __import__("training.models.model_factory", fromlist=["create_model"]).create_model(cfg)
    model.to(device)
    _load_checkpoint(model=model, cfg=cfg, device=device)

    inference_config = InferenceConfig(
        image_size=(cfg.preprocess.image_width, cfg.preprocess.image_height),
        batch_size=args.batch_size,
        device=args.device,
        postprocess=not args.no_postprocess,
        overlay_alpha=args.overlay_alpha,
        transparent_alpha=args.transparent_alpha,
        save_json=not args.no_json,
        save_csv_summary=not args.no_csv,
        progress_bar=not args.no_progress,
    )

    input_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else cfg.output.prediction_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        result = infer_single(
            image_path=input_path,
            model=model,
            cfg=cfg,
            out_dir=out_dir,
            inference_config=inference_config,
        )
        logger.info("Inference complete for %s in %.4fs", input_path.name, result.total_time_s)
    elif input_path.is_dir():
        results = infer_folder(
            folder_path=input_path,
            model=model,
            cfg=cfg,
            out_dir=out_dir,
            inference_config=inference_config,
        )
        total_time = sum(r.total_time_s for r in results)
        logger.info(
            "Folder inference complete: %d images, total %.4fs, avg %.4fs/image",
            len(results),
            total_time,
            total_time / len(results) if results else 0.0,
        )
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
