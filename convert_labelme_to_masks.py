import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from dataset_exclusion import load_excluded_samples, is_excluded_sample


LABEL_TO_ID = {
    "Cornea": 1,
    "Iris": 2,
}



def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("convert_labelme_to_masks")
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


def safe_read_json(json_path: Path) -> Tuple[bool, dict]:
    try:
        return True, json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, {"_error": str(e)}


def polygon_to_int32(points: List[List[float]]) -> np.ndarray:
    arr = np.array(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int32)
    return arr.astype(np.int32)


def extract_polygons(labelme: dict) -> List[Tuple[int, np.ndarray]]:
    """Returns list of (class_id, polygon_points Nx2 int32)."""
    shapes = labelme.get("shapes", [])
    out: List[Tuple[int, np.ndarray]] = []

    for sh in shapes:
        label = sh.get("label")
        if label not in LABEL_TO_ID:
            # ignore Lesion and any unknown labels
            continue
        pts = sh.get("points", [])
        poly = polygon_to_int32(pts)
        if poly.shape[0] < 3:
            # need at least 3 points for polygon fill
            continue
        out.append((LABEL_TO_ID[label], poly))

    return out


def labelme_image_size(labelme: dict) -> Tuple[int, int]:
    h = int(labelme.get("imageHeight", 0))
    w = int(labelme.get("imageWidth", 0))
    return h, w


def main():
    parser = argparse.ArgumentParser(description="Convert merged LabelMe JSON polygons to segmentation masks.")
    parser.add_argument("--dataset_root", type=str, default=str(Path("MedicalAI") / "dataset"))
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images"
    anns_dir = dataset_root / "annotations"
    masks_dir = Path(args.masks_dir) if args.masks_dir else dataset_root / "masks"

    logger = setup_logger(dataset_root / "logs" / "convert_labelme_to_masks.log")

    masks_dir.mkdir(parents=True, exist_ok=True)

    excluded = load_excluded_samples(Path(__file__).resolve().parent)
    json_paths = sorted([p for p in anns_dir.glob("*.json") if p.is_file() and not is_excluded_sample(p.stem, excluded)])

    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {anns_dir}")

    skipped = 0
    failed = 0
    written = 0

    for jp in tqdm(json_paths, desc="Converting LabelMe -> masks"):
        stem = jp.stem

        # Find corresponding image by stem (any supported extension)
        img_path = None
        for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            logger.warning(f"Missing image for JSON: {jp.name}. Skipping.")
            failed += 1
            continue

        out_mask_path = masks_dir / f"{stem}.png"
        if args.skip_existing and out_mask_path.exists():
            skipped += 1
            continue

        ok, labelme = safe_read_json(jp)
        if not ok:
            logger.warning(f"Corrupted JSON (skipping): {jp} -> {labelme.get('_error')}")
            failed += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Corrupted/unreadable image (skipping): {img_path}")
            failed += 1
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygons = extract_polygons(labelme)
        for class_id, poly in polygons:
            # poly: Nx2 int32
            cv2.fillPoly(mask, [poly], int(class_id))

        cv2.imwrite(str(out_mask_path), mask)
        written += 1

    summary = {
        "json_files": len(json_paths),
        "written_masks": written,
        "skipped_existing": skipped,
        "failed_or_missing": failed,
    }
    (dataset_root / "statistics").mkdir(parents=True, exist_ok=True)
    (dataset_root / "statistics" / "conversion_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info(f"Conversion done: {summary}")


if __name__ == "__main__":
    main()

