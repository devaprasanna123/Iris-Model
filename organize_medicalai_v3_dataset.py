#!/usr/bin/env python3
"""Copy the generated MedicalAI V3 dataset into a completely independent folder.

Creates (by COPY, never MOVE):

MedicalAI_V3_Dataset/
  train/images/
  train/iris_masks/
  train/posterior_boundary/
  train/metadata/
  val/... (same)
  test/... (same)

Verification includes:
- Missing files (mask/boundary/metadata per image)
- Duplicate filenames
- Broken JSON (posterior_boundary + metadata)
- Corrupted images

This script does NOT touch the V2 dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
JSON_EXT = ".json"


@dataclass
class VerifyIssues:
    missing_masks: List[str]
    missing_boundaries: List[str]
    missing_metadata: List[str]
    duplicate_images: List[str]
    duplicate_masks: List[str]
    duplicate_boundaries: List[str]
    duplicate_metadata: List[str]
    broken_boundary_json: List[str]
    broken_metadata_json: List[str]
    corrupted_images: List[str]

    def any_issues(self) -> bool:
        return any(
            len(getattr(self, k)) > 0
            for k in (
                "missing_masks",
                "missing_boundaries",
                "missing_metadata",
                "duplicate_images",
                "duplicate_masks",
                "duplicate_boundaries",
                "duplicate_metadata",
                "broken_boundary_json",
                "broken_metadata_json",
                "corrupted_images",
            )
        )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def collect_files(dir_path: Path, exts: Optional[set[str]] = None) -> List[Path]:
    if not dir_path.exists():
        return []
    out: List[Path] = []
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if exts is None or p.suffix.lower() in exts:
            out.append(p)
    return out


def stem_counts(files: Iterable[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in files:
        counts[f.stem] = counts.get(f.stem, 0) + 1
    return counts


def filename_counts(files: Iterable[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in files:
        counts[f.name] = counts.get(f.name, 0) + 1
    return counts


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))


def try_read_image(p: Path) -> bool:
    try:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return img is not None
    except Exception:
        return False


def safe_read_json(p: Path) -> bool:
    try:
        with p.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def verify_split(dest_split_root: Path, split_name: str) -> VerifyIssues:
    images_dir = dest_split_root / "images"
    masks_dir = dest_split_root / "iris_masks"
    boundaries_dir = dest_split_root / "posterior_boundary"
    metadata_dir = dest_split_root / "metadata"

    image_files = [p for p in collect_files(images_dir, IMAGE_EXTS) if p.is_file()]
    mask_files = [p for p in collect_files(masks_dir, IMAGE_EXTS) if p.is_file()]
    boundary_files = [p for p in collect_files(boundaries_dir, {JSON_EXT}) if p.is_file()]
    metadata_files = [p for p in collect_files(metadata_dir, {JSON_EXT}) if p.is_file()]

    # Duplicates by stem (shouldn't normally happen, but detect anyway)
    dup_images = sorted([s for s, c in stem_counts(image_files).items() if c > 1])
    dup_masks = sorted([s for s, c in stem_counts(mask_files).items() if c > 1])
    dup_boundaries = sorted([s for s, c in stem_counts(boundary_files).items() if c > 1])
    dup_metadata = sorted([s for s, c in stem_counts(metadata_files).items() if c > 1])

    img_stems = {p.stem for p in image_files}
    mask_stems = {p.stem for p in mask_files}
    boundary_stems = {p.stem for p in boundary_files}
    metadata_stems = {p.stem for p in metadata_files}

    missing_masks = sorted([s for s in img_stems if s not in mask_stems])
    missing_boundaries = sorted([s for s in img_stems if s not in boundary_stems])
    missing_metadata = sorted([s for s in img_stems if s not in metadata_stems])

    broken_boundary_json: List[str] = []
    for p in boundary_files:
        if not safe_read_json(p):
            broken_boundary_json.append(str(p))

    broken_metadata_json: List[str] = []
    for p in metadata_files:
        if not safe_read_json(p):
            broken_metadata_json.append(str(p))

    corrupted_images: List[str] = []
    for p in image_files:
        if not try_read_image(p):
            corrupted_images.append(str(p))

    issues = VerifyIssues(
        missing_masks=[str(dest_split_root / "iris_masks" / f"{s}.png") for s in missing_masks],
        missing_boundaries=[
            str(dest_split_root / "posterior_boundary" / f"{s}.json") for s in missing_boundaries
        ],
        missing_metadata=[str(dest_split_root / "metadata" / f"{s}.json") for s in missing_metadata],
        duplicate_images=dup_images,
        duplicate_masks=dup_masks,
        duplicate_boundaries=dup_boundaries,
        duplicate_metadata=dup_metadata,
        broken_boundary_json=sorted(broken_boundary_json),
        broken_metadata_json=sorted(broken_metadata_json),
        corrupted_images=sorted(corrupted_images),
    )

    print(f"\n[Verify] split={split_name}")
    print(f"  images: {len(image_files)}")
    print(f"  masks: {len(mask_files)}")
    print(f"  boundaries: {len(boundary_files)}")
    print(f"  metadata: {len(metadata_files)}")

    if issues.any_issues():
        print("  Issues found:")
        if issues.missing_masks:
            print(f"    Missing masks: {len(issues.missing_masks)}")
        if issues.missing_boundaries:
            print(f"    Missing boundaries: {len(issues.missing_boundaries)}")
        if issues.missing_metadata:
            print(f"    Missing metadata: {len(issues.missing_metadata)}")
        if issues.duplicate_images:
            print(f"    Duplicate image stems: {len(issues.duplicate_images)}")
        if issues.duplicate_masks:
            print(f"    Duplicate mask stems: {len(issues.duplicate_masks)}")
        if issues.duplicate_boundaries:
            print(f"    Duplicate boundary stems: {len(issues.duplicate_boundaries)}")
        if issues.duplicate_metadata:
            print(f"    Duplicate metadata stems: {len(issues.duplicate_metadata)}")
        if issues.broken_boundary_json:
            print(f"    Broken boundary JSON: {len(issues.broken_boundary_json)}")
        if issues.broken_metadata_json:
            print(f"    Broken metadata JSON: {len(issues.broken_metadata_json)}")
        if issues.corrupted_images:
            print(f"    Corrupted images: {len(issues.corrupted_images)}")
    else:
        print("  OK: no missing/corrupt/broken files detected for this split")

    return issues


def copy_split(source_root: Path, dest_root: Path, split_name: str) -> Tuple[int, int, int, int]:
    src_split = source_root / split_name
    if not src_split.exists():
        raise FileNotFoundError(f"Source split not found: {src_split}")

    # Source folders created by generator script
    src_images = src_split / "images"
    src_masks = src_split / "iris_masks"
    src_boundaries = src_split / "posterior_boundary"
    src_metadata = src_split / "metadata"

    dst_split = dest_root / split_name
    dst_images = dst_split / "images"
    dst_masks = dst_split / "iris_masks"
    dst_boundaries = dst_split / "posterior_boundary"
    dst_metadata = dst_split / "metadata"

    # Destination structure
    ensure_dir(dst_images)
    ensure_dir(dst_masks)
    ensure_dir(dst_boundaries)
    ensure_dir(dst_metadata)

    # Copy by stem->filename naming scheme; keep identical filenames.
    images = collect_files(src_images, IMAGE_EXTS)
    masks = collect_files(src_masks, IMAGE_EXTS)
    boundaries = collect_files(src_boundaries, {JSON_EXT})
    metadata = collect_files(src_metadata, {JSON_EXT})

    # Copy all files present in the source folders
    for p in images:
        safe_copy(p, dst_images / p.name)
    for p in masks:
        safe_copy(p, dst_masks / p.name)
    for p in boundaries:
        safe_copy(p, dst_boundaries / p.name)
    for p in metadata:
        safe_copy(p, dst_metadata / p.name)

    return len(images), len(masks), len(boundaries), len(metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize generated MedicalAI V3 into independent folder")
    parser.add_argument(
        "--source_root",
        type=str,
        default=str(Path("MedicalAI") / "MedicalAI_V3"),
        help="Root folder containing generated V3 splits (default: MedicalAI/MedicalAI_V3)",
    )
    parser.add_argument(
        "--dest_root",
        type=str,
        default=str(Path("MedicalAI") / "MedicalAI_V3_Dataset"),
        help="Destination root folder to create independent dataset (default: MedicalAI/MedicalAI_V3_Dataset)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root).resolve()
    dest_root = Path(args.dest_root).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")

    # Do not delete anything: only create missing folders and copy.
    ensure_dir(dest_root)

    total_train = total_val = total_test = 0
    total_images = 0
    total_masks = 0
    total_boundaries = 0
    total_metadata = 0

    all_issues: Dict[str, VerifyIssues] = {}

    for split_name in args.splits:
        n_images, n_masks, n_boundaries, n_metadata = copy_split(source_root, dest_root, split_name)

        if split_name == "train":
            total_train = n_images
        elif split_name == "val":
            total_val = n_images
        elif split_name == "test":
            total_test = n_images

        total_images += n_images
        total_masks += n_masks
        total_boundaries += n_boundaries
        total_metadata += n_metadata

        dest_split_root = dest_root / split_name
        issues = verify_split(dest_split_root, split_name)
        all_issues[split_name] = issues

    # Print required summary
    print("\n==================== SUMMARY ====================")
    print(f"Total train samples: {total_train}")
    print(f"Total validation samples: {total_val}")
    print(f"Total test samples: {total_test}")

    print(f"Total images: {total_images}")
    print(f"Total masks: {total_masks}")
    print(f"Total boundary files: {total_boundaries}")
    print(f"Total metadata files: {total_metadata}")

    # If any issues exist, print a compact warning but still finish.
    any_bad = any(issues.any_issues() for issues in all_issues.values())
    if any_bad:
        print("\nWARNING: Verification found issues. See split verification output above.")
    else:
        print("\nVerification passed with no issues.")

    print("MedicalAI V3 Dataset successfully organized and ready for training.")


if __name__ == "__main__":
    main()

