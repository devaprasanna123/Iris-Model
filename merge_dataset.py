import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2

from dataset_exclusion import load_excluded_samples, is_excluded_sample


@dataclass
class CopyItem:
    src: Path
    dst: Path



LABELME_IMAGEPATH_KEY_CANDIDATES = ["imagePath"]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("merge_dataset")
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


def list_images(images_dir: Path) -> Dict[str, Path]:
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if not images_dir.exists():
        return {}
    out: Dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.name] = p
    return out


def list_jsons(ann_dir: Path) -> Dict[str, Path]:
    if not ann_dir.exists():
        return {}
    out: Dict[str, Path] = {}
    for p in ann_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            out[p.stem + ".json"] = p
    return out


def safe_copy(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True


def update_labelme_imagepath(json_path: Path, new_image_filename: str, logger: logging.Logger) -> None:
    # We only update imagePath; handle common variations robustly.
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read JSON for imagePath update: {json_path} ({e})")
        return

    changed = False
    for key in LABELME_IMAGEPATH_KEY_CANDIDATES:
        if key in data:
            data[key] = new_image_filename
            changed = True
            break

    if not changed:
        # Some labelme exports might use different casing; do a best-effort fallback.
        for k in list(data.keys()):
            if k.lower() == "imagepath":
                data[k] = new_image_filename
                changed = True
                break

    if changed:
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_merged_structure(root: Path) -> None:
    (root / "dataset" / "images").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "annotations").mkdir(parents=True, exist_ok=True)


def resolve_duplicates_across(full_images: Dict[str, Path], partial_images: Dict[str, Path]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Returns mapping old_filename->new_filename for full and partial.

    If a filename exists in both datasets, we rename them to:
      full_<name>, partial_<name>

    For non-conflicting names, we keep original filenames.
    """
    full_new: Dict[str, str] = {}
    partial_new: Dict[str, str] = {}

    full_names = set(full_images.keys())
    partial_names = set(partial_images.keys())
    dup = full_names.intersection(partial_names)

    for name in full_names:
        full_new[name] = f"full_{name}" if name in dup else name
    for name in partial_names:
        partial_new[name] = f"partial_{name}" if name in dup else name

    return full_new, partial_new


def main():
    parser = argparse.ArgumentParser(description="Merge Full-frame and Partial-frame datasets safely.")
    parser.add_argument("--dataset_root", type=str,
                        default=r"D:\\OCT Images\\AIDK_Dataset\\AIDK_Dataset")
    parser.add_argument("--out_root", type=str, default=str(Path("MedicalAI") / "dataset"))

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    full_images_dir = dataset_root / "Full-frame_Dataset" / "Original_AS-OCT_Images"
    full_anns_dir = dataset_root / "Full-frame_Dataset" / "Experts_Annotations"
    partial_images_dir = dataset_root / "Partial-frame_Dataset" / "Original_AS-OCT_Images"
    partial_anns_dir = dataset_root / "Partial-frame_Dataset" / "Experts_Annotations"

    out_root = Path("MedicalAI") / "dataset"
    merged_images_dir = out_root / "images"
    merged_anns_dir = out_root / "annotations"
    out_root.mkdir(parents=True, exist_ok=True)
    merged_images_dir.mkdir(parents=True, exist_ok=True)
    merged_anns_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "logs" / "merge_dataset.log")
    logger.info("Starting merge...")

    excluded = load_excluded_samples(Path(__file__).resolve().parent)

    full_images = {k: v for k, v in list_images(full_images_dir).items() if not is_excluded_sample(Path(k).stem, excluded)}
    partial_images = {k: v for k, v in list_images(partial_images_dir).items() if not is_excluded_sample(Path(k).stem, excluded)}

    # Resolve duplicates by filename between datasets
    full_name_map, partial_name_map = resolve_duplicates_across(full_images, partial_images)


    # JSON filename resolution: assume json stem matches image stem; but use JSON stem mapping explicitly.
    full_jsons = {k: v for k, v in list_jsons(full_anns_dir).items() if not is_excluded_sample(Path(k).stem, excluded)}
    partial_jsons = {k: v for k, v in list_jsons(partial_anns_dir).items() if not is_excluded_sample(Path(k).stem, excluded)}


    # Build stem->json path
    full_json_by_stem = {p.stem: p for p in full_jsons.values()}
    partial_json_by_stem = {p.stem: p for p in partial_jsons.values()}

    copied = {
        "full": {"images": 0, "annotations": 0, "renamed_images": []},
        "partial": {"images": 0, "annotations": 0, "renamed_images": []},
    }

    # Copy images + JSON with matching renames; update JSON imagePath.
    def process_one(dataset_kind: str, images_dict: Dict[str, Path], json_by_stem: Dict[str, Path], name_map: Dict[str, str]):
        nonlocal copied
        for orig_image_name, src_image_path in images_dict.items():
            new_image_name = name_map[orig_image_name]
            dst_image_path = merged_images_dir / new_image_name

            img_stem = src_image_path.stem
            json_path = json_by_stem.get(img_stem)
            if json_path is None:
                logger.warning(f"[{dataset_kind}] Missing JSON for image stem {img_stem}. Skipping image.")
                continue

            dst_json_name = f"{Path(new_image_name).stem}.json"
            dst_json_path = merged_anns_dir / dst_json_name

            did_image = safe_copy(src_image_path, dst_image_path)
            if did_image:
                copied[dataset_kind]["images"] += 1

            if new_image_name != orig_image_name:
                copied[dataset_kind]["renamed_images"].append({"from": orig_image_name, "to": new_image_name})

            # Copy JSON only if destination doesn't exist
            if not dst_json_path.exists():
                shutil.copy2(json_path, dst_json_path)
                copied[dataset_kind]["annotations"] += 1
                # Update labelme imagePath inside json to refer to renamed image
                update_labelme_imagepath(dst_json_path, new_image_name, logger)
            else:
                logger.info(f"[{dataset_kind}] Destination JSON already exists, skipping: {dst_json_path}")

    process_one("full", full_images, full_json_by_stem, full_name_map)
    process_one("partial", partial_images, partial_json_by_stem, partial_name_map)

    # Write merge manifest
    manifest = {
        "source": {
            "full_images": str(full_images_dir),
            "full_annotations": str(full_anns_dir),
            "partial_images": str(partial_images_dir),
            "partial_annotations": str(partial_anns_dir),
        },
        "output": {
            "images_dir": str(merged_images_dir),
            "annotations_dir": str(merged_anns_dir),
        },
        "summary": copied,
        "duplicate_resolution": {
            "full_prefix": "full_",
            "partial_prefix": "partial_",
        }
    }

    manifest_path = out_root / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Merge completed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

