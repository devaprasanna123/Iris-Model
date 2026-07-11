"""MedicalAI prediction (production-ready).

This script rewrites the entire inference pipeline to be compatible with the
rest of the project (U-Net + OctDataset preprocessing).

Supported:
- Single image: python -m training.predict --input image.bmp
- Folder: python -m training.predict --input dataset/test/images

Artifacts saved per image:
- <stem>_pred_mask.png              (grayscale class-id mask)
- <stem>_pred_mask_colored.png    (colored semantic mask)
- <stem>_overlay.png                (overlay of colored mask on resized original)

Color mapping (BGR for OpenCV):
- Background: Black
- Cornea: Blue
- Iris: Green
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from training.config import TrainingConfig
from training.models.model_factory import create_model
from training.utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


LABEL_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # Background - Black
    1: (255, 0, 0),  # Cornea - Blue in BGR
    2: (0, 255, 0),  # Iris - Green in BGR
}


def _configure_logging() -> None:
    # Keep console logs simple for CLI usage.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _supported_extensions() -> Tuple[str, ...]:
    return (".bmp", ".png", ".jpg", ".jpeg")


def _iter_images(input_dir: Path) -> List[Path]:
    exts = {e.lower() for e in _supported_extensions()}
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return files


def _read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _preprocess_like_training(bgr_img: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
    """Match training preprocessing exactly.

    Training contract (from repo spec):
    - Resize to 512x512 with INTER_LINEAR
    - Convert BGR -> RGB
    - float32 and divide by 255
    - Tensor shape (3, 512, 512)
    """

    resized = cv2.resize(bgr_img, image_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    chw = np.ascontiguousarray(chw)
    return torch.from_numpy(chw).float()


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H,W), got {mask.shape}")

    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in LABEL_COLORS_BGR.items():
        out[mask == cls_id] = color
    return out


def _overlay_resized(original_bgr_512: np.ndarray, colored_mask_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(original_bgr_512, 1.0 - alpha, colored_mask_bgr, alpha, 0)


@torch.no_grad()
def _predict_mask(
    *,
    model: torch.nn.Module,
    device: torch.device,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    model.eval()
    batch = input_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
    logits = model(batch)  # (1,C,H,W)
    pred = logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()  # (H,W)
    return pred


def _save_outputs(
    *,
    image_path: Path,
    pred_mask: np.ndarray,
    colored_mask_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    pred_mask_path = out_dir / f"{stem}_pred_mask.png"
    colored_path = out_dir / f"{stem}_pred_mask_colored.png"
    overlay_path = out_dir / f"{stem}_overlay.png"

    # Grayscale mask of class ids.
    cv2.imwrite(str(pred_mask_path), pred_mask)
    cv2.imwrite(str(colored_path), colored_mask_bgr)
    cv2.imwrite(str(overlay_path), overlay_bgr)


def predict_one(
    *,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
    image_size: Tuple[int, int] = (512, 512),
) -> float:
    """Predict a single image.

    Returns:
        Inference time in seconds.
    """

    bgr = _read_image_bgr(image_path)

    # Preprocess input for model.
    input_tensor = _preprocess_like_training(bgr, image_size=image_size)  # (3,512,512)

    # Inference.
    t0 = time.perf_counter()
    pred_mask = _predict_mask(model=model, device=device, input_tensor=input_tensor)
    t1 = time.perf_counter()

    # Prepare overlay at EXACTLY 512x512 to match predicted mask size.
    bgr_512 = cv2.resize(bgr, image_size, interpolation=cv2.INTER_LINEAR)
    colored_mask = _colorize_mask(pred_mask)
    overlay = _overlay_resized(bgr_512, colored_mask, alpha=0.5)

    _save_outputs(
        image_path=image_path,
        pred_mask=pred_mask,
        colored_mask_bgr=colored_mask,
        overlay_bgr=overlay,
        out_dir=out_dir,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return t1 - t0


def predict_folder(
    *,
    folder_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
) -> None:
    images = _iter_images(folder_path)
    if not images:
        raise FileNotFoundError(
            f"No supported images found in folder: {folder_path}. Supported: {', '.join(_supported_extensions())}"
        )

    total = len(images)
    for idx, img_path in enumerate(images, start=1):
        logger.info("Predicting %d / %d : %s", idx, total, img_path.name)
        try:
            dt = predict_one(image_path=img_path, model=model, device=device, out_dir=out_dir)
            logger.info("  Inference time: %.4fs", dt)
        except Exception:
            # Continue with remaining images but fail loudly at the end.
            logger.exception("Failed on image: %s", img_path)
            raise

    print("Prediction completed.")
    print(f"Total Images: {total}")
    print(f"Output Folder: {out_dir.resolve().as_posix()}")


def _load_best_checkpoint(*, model: torch.nn.Module, cfg: TrainingConfig, device: torch.device) -> None:
    ckpt_dir = cfg.checkpoint.checkpoint_dir
    best_name = cfg.checkpoint.best_model_name

    # Ensure extension is .pt (project requirement).
    if not str(best_name).lower().endswith(".pt"):
        raise ValueError(
            f"cfg.checkpoint.best_model_name must be a .pt file. Got: {best_name}"
        )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        best_model_name=best_name,
        last_model_name=cfg.checkpoint.last_model_name,
        device=device,
    )

    # Load best weights into model.
    checkpoint_manager.load(model=model, optimizer=None, scheduler=None, which="best", strict=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedicalAI U-Net prediction")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image file OR a folder containing images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to TrainingConfig().output.prediction_dir",
    )
    return parser


def main() -> None:
    _configure_logging()

    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = TrainingConfig()
    device = torch.device(cfg.device.device)

    # Output folder.
    out_dir = Path(args.output_dir) if args.output_dir else cfg.output.prediction_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(cfg)
    model.to(device)


    _load_best_checkpoint(model=model, cfg=cfg, device=device)

    input_path = Path(args.input)
    if input_path.is_file():
        dt = predict_one(image_path=input_path, model=model, device=device, out_dir=out_dir)
        print("Prediction completed.")
        print("Total Images: 1")
        print(f"Inference Time: {dt:.4f}s")
        print(f"Output Folder: {out_dir.resolve().as_posix()}")
        return

    if input_path.is_dir():
        # Print header-style stats.
        images = _iter_images(input_path)
        total = len(images)
        if total == 0:
            raise FileNotFoundError(
                f"No supported images found in folder: {input_path}. Supported: {', '.join(_supported_extensions())}"
            )

        # Progress printing is handled in predict_folder.
        logger.info("Starting folder prediction. Total Images: %d", total)
        predict_folder(folder_path=input_path, model=model, device=device, out_dir=out_dir)
        return

    raise FileNotFoundError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()

