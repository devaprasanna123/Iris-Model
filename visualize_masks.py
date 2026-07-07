import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from dataset_exclusion import load_excluded_samples, is_excluded_sample





LABEL_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),   # Cornea (BGR)
    2: (0, 255, 0),   # Iris (BGR)
}


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("visualize_masks")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


def load_image_any(path: Path):
    # returns BGR or grayscale as loaded by OpenCV
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_bgr.copy()
    # Create a colored overlay for classes 1 and 2
    colored = np.zeros_like(image_bgr, dtype=np.uint8)
    for class_id, color in LABEL_COLORS.items():
        if class_id == 0:
            continue
        colored[mask == class_id] = color

    cv2.addWeighted(colored, alpha, overlay, 1.0, 0, overlay)
    return overlay


def find_pairs(images_dir: Path, masks_dir: Path, excluded: set[str]) -> List[Tuple[Path, Path]]:
    masks = {p.stem: p for p in masks_dir.glob("*.png") if p.is_file() and not is_excluded_sample(p.stem, excluded)}
    pairs: List[Tuple[Path, Path]] = []

    exts = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    for m_stem, m_path in masks.items():
        img_path = None
        for ext in exts:
            cand = images_dir / f"{m_stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is not None:
            pairs.append((img_path, m_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Visualize AS-OCT segmentation masks overlays.")
    parser.add_argument("--dataset_root", type=str, default=str(Path("MedicalAI") / "dataset"))
    parser.add_argument("--num_samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    vis_dir = dataset_root / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(dataset_root / "logs" / "visualize_masks.log")

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Expected images_dir and masks_dir. Run convert_labelme_to_masks.py first.")

    excluded = load_excluded_samples(Path(__file__).resolve().parent)
    pairs = find_pairs(images_dir, masks_dir, excluded)

    if not pairs:
        raise FileNotFoundError(f"No image-mask pairs found in {images_dir} and {masks_dir}")

    random.seed(args.seed)
    k = min(args.num_samples, len(pairs))
    samples = random.sample(pairs, k=k)

    for img_path, mask_path in tqdm(samples, desc="Creating overlays"):
        stem = img_path.stem
        img = load_image_any(img_path)
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            logger.warning(f"Failed to load mask: {mask_path}")
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        img_bgr = ensure_bgr(img)

        overlay = overlay_mask(img_bgr, mask, alpha=args.alpha)

        # Also create a clean color mask visualization
        color_mask = np.zeros_like(img_bgr)
        for class_id, color in LABEL_COLORS.items():
            if class_id == 0:
                continue
            color_mask[mask == class_id] = color

        # Outline edges for easier QC
        edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
        edges_bgr = ensure_bgr(edges)
        outline = overlay.copy()
        outline[edges > 0] = (0, 0, 255)

        cv2.imwrite(str(vis_dir / f"original_{stem}.png"), img_bgr)
        cv2.imwrite(str(vis_dir / f"mask_{stem}.png"), color_mask)
        cv2.imwrite(str(vis_dir / f"overlay_{stem}.png"), outline)

    logger.info(f"Visualization complete. Saved to: {vis_dir}")


if __name__ == "__main__":
    main()

