import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
from tqdm import tqdm

from dataset_exclusion import load_excluded_samples, is_excluded_sample


LABELS = {
    "Cornea": 1,
    "Iris": 2,
}



def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dataset_statistics")
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


def polygon_area(contour: np.ndarray) -> float:
    # contour: Nx2 int32
    if contour.shape[0] < 3:
        return 0.0
    return float(cv2.contourArea(contour))


def polygon_points_count(contour: np.ndarray) -> int:
    return int(contour.shape[0])


def safe_read_json(json_path: Path) -> Optional[dict]:
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for merged AS-OCT dataset.")
    parser.add_argument("--dataset_root", type=str, default=str(Path("MedicalAI") / "dataset"))
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images"
    anns_dir = dataset_root / "annotations"

    masks_dir = dataset_root / "masks"  # optional

    logger = setup_logger(dataset_root / "logs" / "dataset_statistics.log")
    logger.info("Computing dataset statistics...")

    excluded = load_excluded_samples(Path(__file__).resolve().parent)

    json_paths = sorted([
        p for p in anns_dir.glob("*.json") if p.is_file() and not is_excluded_sample(p.stem, excluded)
    ])


    image_paths: List[Path] = []
    if images_dir.exists():
        for p in images_dir.iterdir():
            if (
                p.is_file()
                and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                and not is_excluded_sample(p.stem, excluded)
            ):
                image_paths.append(p)


    mask_paths: List[Path] = []
    if masks_dir.exists():
        mask_paths = [p for p in masks_dir.glob("*.png") if p.is_file() and not is_excluded_sample(p.stem, excluded)]


    total_images = len(image_paths)
    total_masks = len(mask_paths)
    total_annotations = len(json_paths)

    corrupted_json_files = 0
    corrupted_mask_files = 0
    missing_masks = 0

    cornea_polygons = 0
    iris_polygons = 0
    polygon_points_list: List[int] = []
    iris_areas: List[float] = []
    cornea_areas: List[float] = []

    image_resolutions: List[Tuple[int, int]] = []
    image_sizes_hist: Dict[str, int] = defaultdict(int)

    def avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    # Image resolutions + corrupted images
    for ip in tqdm(image_paths, desc="Reading image resolutions"):
        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        image_resolutions.append((w, h))
        image_sizes_hist[f"{w}x{h}"] += 1

    # Mask presence + corruption + pixel class check
    img_by_stem = {p.stem: p for p in image_paths}
    mask_by_stem = {p.stem: p for p in mask_paths}

    for stem, ipath in tqdm(img_by_stem.items(), desc="Verifying masks (existence + read)"):
        if stem not in mask_by_stem:
            missing_masks += 1
            continue
        mpath = mask_by_stem[stem]
        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if m is None:
            corrupted_mask_files += 1
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        ih, iw = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED).shape[:2] if cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED) is not None else (None, None)
        if ih is None:
            continue
        if m.shape[0] != ih or m.shape[1] != iw:
            # still count as corrupted/invalid
            corrupted_mask_files += 1
            continue
        # ensure pixel classes only contain 0,1,2
        uniq = np.unique(m)
        if not np.all(np.isin(uniq, np.array([0, 1, 2], dtype=np.uint8))):
            corrupted_mask_files += 1

    # Parse JSON polygons
    for jp in tqdm(json_paths, desc="Parsing LabelMe JSON polygons"):
        labelme = safe_read_json(jp)
        if labelme is None:
            corrupted_json_files += 1
            continue

        for sh in labelme.get("shapes", []):
            lbl = sh.get("label")
            if lbl not in LABELS:
                continue
            pts = sh.get("points", [])
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            contour = np.array(pts, dtype=np.int32)
            if contour.ndim != 2 or contour.shape[1] != 2 or contour.shape[0] < 3:
                continue

            pts_count = polygon_points_count(contour)
            polygon_points_list.append(pts_count)
            area = polygon_area(contour)

            if lbl == "Cornea":
                cornea_polygons += 1
                cornea_areas.append(area)
            elif lbl == "Iris":
                iris_polygons += 1
                iris_areas.append(area)

    avg_polygon_points = float(np.mean(polygon_points_list)) if polygon_points_list else 0.0

    avg_w = float(np.mean([w for w, _ in image_resolutions])) if image_resolutions else 0.0
    avg_h = float(np.mean([h for _, h in image_resolutions])) if image_resolutions else 0.0

    stats = {
        "total_images": total_images,
        "total_masks": total_masks,
        "total_annotations": total_annotations,
        "corrupted_json_files": corrupted_json_files,
        "corrupted_mask_files": corrupted_mask_files,
        "missing_masks": missing_masks,
        "cornea_count": cornea_polygons,
        "iris_count": iris_polygons,
        "average_polygon_points": avg_polygon_points,
        "average_image_resolution": {"width": avg_w, "height": avg_h},
        "average_iris_area_pixels": avg(iris_areas),
        "average_cornea_area_pixels": avg(cornea_areas),
    }

    # Print to console
    print("Dataset statistics")
    print("-------------------")
    print(f"Total images: {total_images}")
    print(f"Total masks: {total_masks}")
    print(f"Missing masks: {missing_masks}")
    print(f"Corrupted JSON files: {corrupted_json_files}")
    print(f"Corrupted mask files: {corrupted_mask_files}")
    print(f"Cornea count (polygons): {cornea_polygons}")
    print(f"Iris count (polygons): {iris_polygons}")
    print(f"Average polygon points: {avg_polygon_points:.3f}")
    print(f"Average Cornea area (pixels): {avg(cornea_areas):.3f}")
    print(f"Average Iris area (pixels): {avg(iris_areas):.3f}")

    if image_resolutions:
        ws = [w for w, _ in image_resolutions]
        hs = [h for _, h in image_resolutions]
        print(f"Average image resolution (w,h): ({sum(ws)/len(ws):.1f}, {sum(hs)/len(hs):.1f})")

    stats_dir = dataset_root / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_path = stats_dir / "dataset_statistics.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


