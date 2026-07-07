import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

from dataset_exclusion import load_excluded_samples, is_excluded_sample





@dataclass
class FileIssue:
    path: str
    reason: str


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("verify_dataset")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if script is re-run in same process
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


def safe_read_json(json_path: Path) -> Tuple[bool, Optional[dict], Optional[str]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return True, data, None
    except Exception as e:
        return False, None, str(e)


def detect_image_corruption(image_path: Path) -> Tuple[bool, Optional[Tuple[int, int]]]:
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, None
        h, w = img.shape[:2]
        return True, (w, h)
    except Exception:
        return False, None


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []

    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    paths: List[Path] = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


def list_jsons(ann_dir: Path) -> List[Path]:
    if not ann_dir.exists():
        return []
    return [p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]


def stem_to_json(jsons: List[Path]) -> Dict[str, Path]:
    # LabelMe JSON filename typically matches image stem
    return {p.stem: p for p in jsons}


def stem_to_image(images: List[Path]) -> Dict[str, Path]:
    return {p.stem: p for p in images}


def build_dataset_index(images_dir: Path, ann_dir: Path, logger: logging.Logger, excluded: set[str]) -> dict:
    images = [p for p in list_images(images_dir) if not is_excluded_sample(p.stem, excluded)]
    jsons = [p for p in list_jsons(ann_dir) if not is_excluded_sample(p.stem, excluded)]

    img_by_stem = stem_to_image(images)
    json_by_stem = stem_to_json(jsons)

    duplicated_image_filenames = sorted(
        [name for name, cnt in _filename_counts(images).items() if cnt > 1]
    )

    duplicated_json_filenames = sorted(
        [name for name, cnt in _filename_counts(jsons).items() if cnt > 1]
    )

    missing_json = []
    missing_images = []

    # For each image stem, ensure corresponding JSON exists
    for stem, img_path in img_by_stem.items():
        if stem not in json_by_stem:
            missing_json.append(str(img_path))

    # For each JSON stem, ensure corresponding image exists
    for stem, json_path in json_by_stem.items():
        if stem not in img_by_stem:
            missing_images.append(str(json_path))

    corrupted_jsons: List[FileIssue] = []
    corrupted_images: List[FileIssue] = []
    image_sizes: Dict[str, Tuple[int, int]] = {}

    all_json_paths = list(json_by_stem.values())
    # Validate JSONs
    for jp in tqdm(all_json_paths, desc=f"Validating JSONs in {images_dir.name}"):
        ok, _, err = safe_read_json(jp)
        if not ok:
            corrupted_jsons.append(FileIssue(path=str(jp), reason=err or "unknown error"))

    # Validate images
    all_img_paths = list(img_by_stem.values())
    for ip in tqdm(all_img_paths, desc=f"Validating images in {images_dir.name}"):
        ok, size = detect_image_corruption(ip)
        if not ok:
            corrupted_images.append(FileIssue(path=str(ip), reason="cv2.imread failed"))
        else:
            if size:
                w, h = size
                image_sizes[str(ip)] = (int(w), int(h))

    # Summary stats for polygon counts (computed lazily in dataset_statistics)
    return {
        "image_count": len(images),
        "json_count": len(jsons),
        "duplicated_image_filenames": duplicated_image_filenames,
        "duplicated_json_filenames": duplicated_json_filenames,
        "corrupted_images": [issue.__dict__ for issue in corrupted_images],
        "corrupted_jsons": [issue.__dict__ for issue in corrupted_jsons],
        "missing_json": missing_json,
        "missing_images": missing_images,
        "image_sizes": {k: [v[0], v[1]] for k, v in image_sizes.items()},
    }



def _filename_counts(paths: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in paths:
        counts[p.name] = counts.get(p.name, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Verify AS-OCT dataset health (images + LabelMe JSON).")

    parser.add_argument("--dataset_root", type=str,
                        default=r"D:\\OCT Images\\AIDK_Dataset\\AIDK_Dataset")

    parser.add_argument("--out_dir", type=str,
                        default=str(Path("MedicalAI") / "dataset" / "logs"))

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    full_images = dataset_root / "Full-frame_Dataset" / "Original_AS-OCT_Images"
    full_anns = dataset_root / "Full-frame_Dataset" / "Experts_Annotations"
    partial_images = dataset_root / "Partial-frame_Dataset" / "Original_AS-OCT_Images"
    partial_anns = dataset_root / "Partial-frame_Dataset" / "Experts_Annotations"

    out_dir = Path(args.out_dir)
    logger = setup_logger(out_dir / "verify_dataset.log")

    logger.info("Starting dataset verification...")

    excluded = load_excluded_samples(Path(__file__).resolve().parent)

    full_index = build_dataset_index(full_images, full_anns, logger, excluded)
    partial_index = build_dataset_index(partial_images, partial_anns, logger, excluded)

    excluded_count = len(excluded)


    # Cross-dataset duplicates (by filename, not stem)
    full_img_files = list_images(full_images)
    partial_img_files = list_images(partial_images)

    full_names = {p.name for p in full_img_files}
    partial_names = {p.name for p in partial_img_files}
    dup_names = sorted(full_names.intersection(partial_names))

    total_images = full_index["image_count"] + partial_index["image_count"]
    total_json = full_index["json_count"] + partial_index["json_count"]

    report = {
        "dataset_root": str(dataset_root),
        "excluded_samples": sorted(list(excluded)),
        "excluded_count": excluded_count,
        "full_frame": full_index,
        "partial_frame": partial_index,
        "duplicate_filenames_across_full_and_partial": dup_names,
        "totals": {
            "total_images": total_images,
            "total_json": total_json,
            "corrupted_images": len(full_index["corrupted_images"]) + len(partial_index["corrupted_images"]),
            "corrupted_jsons": len(full_index["corrupted_jsons"]) + len(partial_index["corrupted_jsons"]),
            "missing_images": len(full_index["missing_images"]) + len(partial_index["missing_images"]),
            "missing_json": len(full_index["missing_json"]) + len(partial_index["missing_json"]),
            "duplicate_filenames_count": len(dup_names),
        },
    }


    dataset_dir = Path("MedicalAI") / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = dataset_dir / "verification_report.json"
    txt_report_path = dataset_dir / "verification_report.txt"

    with json_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    totals = report["totals"]
    with txt_report_path.open("w", encoding="utf-8") as f:
        f.write("AS-OCT Dataset Verification Report\n")
        f.write("=================================\n\n")
        f.write(f"Dataset root: {report['dataset_root']}\n\n")
        f.write("Totals\n------\n")
        for k, v in totals.items():
            f.write(f"{k}: {v}\n")
        f.write("\nDuplicate filenames across datasets:\n")
        if dup_names:
            for name in dup_names:
                f.write(f"- {name}\n")
        else:
            f.write("None\n")

    logger.info("Verification completed.")
    logger.info(f"JSON report: {json_report_path}")
    logger.info(f"TXT report: {txt_report_path}")


if __name__ == "__main__":
    main()

