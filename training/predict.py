"""MedicalAI prediction utilities.

Responsibilities:
- Load best_model.pth
- Predict one image OR predict a folder
- Save predicted masks into outputs/predictions/
- Support bmp, png, jpg, jpeg
- Overlay prediction (Original image + colored predicted mask)
- Color mapping:
  Background = Black
  Cornea = Blue
  Iris = Green

No visualization using matplotlib; only saves output images.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

import cv2

from training.config import TrainingConfig
from training.metrics import MetricsSpec
from training.models.unet import UNet
from training.utils.checkpoint import CheckpointManager


LABEL_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # Background - Black
    1: (255, 0, 0),  # Cornea - Blue in BGR
    2: (0, 255, 0),  # Iris - Green in BGR
}


def _read_image_cv2(path: Path) -> np.ndarray:
    """Read image via OpenCV in BGR and keep dtype."""

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def _to_tensor_rgb01(
    bgr_img: np.ndarray,
    image_size: tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """
    Convert OpenCV BGR image to RGB tensor (1,3,512,512)
    using exactly the same preprocessing as training.
    """

    # Resize exactly like OctDataset
    bgr_img = cv2.resize(
        bgr_img,
        image_size,
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    rgb = rgb.astype(np.float32) / 255.0

    rgb = np.transpose(rgb, (2, 0, 1))

    rgb = np.ascontiguousarray(rgb)

    tensor = torch.from_numpy(rgb).unsqueeze(0).float()

    return tensor


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert (H,W) int mask to color mask in BGR (H,W,3)."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be (H,W), got shape={mask.shape}")

    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in LABEL_COLORS_BGR.items():
        out[mask == cls_id] = color
    return out


def _overlay(original_bgr: np.ndarray, colored_mask_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha blend overlay image."""

    return cv2.addWeighted(original_bgr, 1.0 - alpha, colored_mask_bgr, alpha, 0)


def predict_one(
    *,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Run prediction for a single image and save results."""

    bgr = _read_image_cv2(image_path)
    inp = _to_tensor_rgb01(bgr).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(inp)
        pred_mask = logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
        print("Prediction Shape :", pred_mask.shape)
        print("Classes Present :", np.unique(pred_mask))

 # Resize original image to match prediction
bgr = cv2.resize(
    bgr,
    (512, 512),
    interpolation=cv2.INTER_LINEAR,
)

colored = _colorize_mask(pred_mask)

overlay = _overlay(
    bgr,
    colored,
    alpha=0.5,
)

out_dir.mkdir(parents=True, exist_ok=True)

stem = image_path.stem

pred_mask_path = out_dir / f"{stem}_pred_mask.png"
overlay_path = out_dir / f"{stem}_overlay.png"
colored_path = out_dir / f"{stem}_pred_mask_colored.png"

cv2.imwrite(str(pred_mask_path), pred_mask)
cv2.imwrite(str(colored_path), colored)
cv2.imwrite(str(overlay_path), overlay) 
    stem = image_path.stem

    pred_mask_path = out_dir / f"{stem}_pred_mask.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    colored_path = out_dir / f"{stem}_pred_mask_colored.png"

    cv2.imwrite(str(pred_mask_path), pred_mask)
    cv2.imwrite(str(colored_path), colored)
    cv2.imwrite(str(overlay_path), overlay)


def predict_folder(
    *,
    folder_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Run prediction for all supported images in folder."""

    exts = {".bmp", ".png", ".jpg", ".jpeg"}
    for p in sorted(folder_path.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            predict_one(image_path=p, model=model, device=device, out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="MedicalAI prediction (best_model.pth).")
    parser.add_argument("--input", type=str, required=True, help="Path to an image or a folder of images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("MedicalAI") / "training" / "predictions"),
        help="Output directory for prediction artifacts",
    )
    args = parser.parse_args()

    cfg = TrainingConfig()

    device = torch.device(cfg.device.device)

    # Logger not required by spec; keep minimal.

    model = UNet(in_channels=cfg.image.channels, num_classes=cfg.classes.number_of_classes)
    model.to(device)

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.checkpoint.checkpoint_dir,
        best_model_name=cfg.checkpoint.best_model_name,
        last_model_name=cfg.checkpoint.last_model_name,
        device=device,
    )


    checkpoint_manager.load(model=model, optimizer=None, scheduler=None, which="best", strict=True)

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if input_path.is_file():
        predict_one(image_path=input_path, model=model, device=device, out_dir=out_dir)
    elif input_path.is_dir():
        predict_folder(folder_path=input_path, model=model, device=device, out_dir=out_dir)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Predictions saved to: {out_dir.resolve().as_posix()}")


if __name__ == "__main__":
    main()

