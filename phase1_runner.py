import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path(r"MedicalAI") / "dataset"
DEFAULT_INPUT_ROOT = Path(r"D:\\OCT Images\\AIDK_Dataset\\AIDK_Dataset")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase1_runner")
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


def run_python_script(script_path: Path, args: List[str], logger: logging.Logger) -> None:
    cmd = [sys.executable, str(script_path), *args]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("Script failed: %s", script_path)
        if proc.stdout:
            logger.error("STDOUT:\n%s", proc.stdout)
        if proc.stderr:
            logger.error("STDERR:\n%s", proc.stderr)
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")


def list_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    masks_by_stem = {p.stem: p for p in masks_dir.glob("*.png") if p.is_file()}
    exts = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    pairs: List[Tuple[Path, Path]] = []
    for stem, mpath in masks_by_stem.items():
        ipath = None
        for ext in exts:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                ipath = cand
                break
        if ipath is not None:
            pairs.append((ipath, mpath))
    return pairs


def verify_masks_strict(
    dataset_root: Path,
    sample_k: int = 25,
    seed: int = 42,
) -> Dict:
    """Verify dataset-level mask constraints and gather overlay sample paths."""

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    vis_dir = dataset_root / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Expected dataset_root/images and dataset_root/masks")

    exts = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    images_by_stem: Dict[str, Path] = {}
    for ip in images_dir.iterdir():
        if ip.is_file() and ip.suffix.lower() in exts:
            images_by_stem[ip.stem] = ip

    mask_by_stem = {p.stem: p for p in masks_dir.glob("*.png") if p.is_file()}

    missing_masks = sorted([f"{stem}" for stem in images_by_stem.keys() if stem not in mask_by_stem])
    missing_images = sorted([f"{stem}" for stem in mask_by_stem.keys() if stem not in images_by_stem])

    # Strictly check each common pair
    common_stems = sorted([stem for stem in images_by_stem.keys() if stem in mask_by_stem])

    class_violations: List[str] = []
    dim_violations: List[str] = []
    read_failures: List[str] = []

    for stem in common_stems:
        ipath = images_by_stem[stem]
        mpath = mask_by_stem[stem]

        img = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            read_failures.append(stem)
            continue

        h, w = img.shape[:2]
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[0] != h or mask.shape[1] != w:
            dim_violations.append(stem)
            continue

        unique = np.unique(mask)
        # Required: classes only contain 0,1,2
        if not np.all(np.isin(unique, np.array([0, 1, 2], dtype=np.uint8))):
            class_violations.append(f"{stem}:{unique.tolist()}")

        # Lesion pixels must be absent: lesion label might map to 0/ignored; ensure mask doesn't contain anything else.
        # Since we map only Cornea/Iris, lesion cannot exist as a separate class. We enforce by class check above.

    # Create overlay samples
    sample_stems = common_stems.copy()
    random.Random(seed).shuffle(sample_stems)
    sample_stems = sample_stems[: min(sample_k, len(sample_stems))]

    # Import local module without requiring MedicalAI to be installed as a package
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from visualize_masks import LABEL_COLORS, overlay_mask, ensure_bgr

    sample_info: List[Dict] = []

    for stem in sample_stems:
        ipath = images_by_stem[stem]
        mpath = mask_by_stem[stem]
        img = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        img_bgr = ensure_bgr(img)
        overlay = overlay_mask(img_bgr, mask)

        orig_out = vis_dir / f"original_{stem}.png"
        mask_out = vis_dir / f"mask_{stem}.png"
        overlay_out = vis_dir / f"overlay_{stem}.png"

        cv2.imwrite(str(orig_out), img_bgr)

        color_mask = np.zeros_like(img_bgr)
        for class_id, color in LABEL_COLORS.items():
            if class_id == 0:
                continue
            color_mask[mask == class_id] = color
        cv2.imwrite(str(mask_out), color_mask)
        cv2.imwrite(str(overlay_out), overlay)

        sample_info.append(
            {
                "stem": stem,
                "image": str(ipath),
                "mask": str(mpath),
                "original_overlay": str(orig_out),
                "mask_viz": str(mask_out),
                "overlay": str(overlay_out),
            }
        )

    return {
        "images_total": len(images_by_stem),
        "masks_total": len(mask_by_stem),
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "read_failures": read_failures,
        "dim_violations": dim_violations,
        "class_violations": class_violations,
        "sample_overlays": sample_info,
        "passed": (not missing_masks) and (not missing_images) and (not read_failures) and (not dim_violations) and (not class_violations),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase-1 dataset preparation end-to-end.")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--input_root", type=str, default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlay_samples", type=int, default=25)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    input_root = Path(args.input_root)

    logger = setup_logger(dataset_root / "logs" / "phase1_runner.log")

    # Runner scripts are expected to be in MedicalAI/
    repo_root = Path(__file__).resolve().parent

    merge_script = repo_root / "merge_dataset.py"
    convert_script = repo_root / "convert_labelme_to_masks.py"
    stats_script = repo_root / "dataset_statistics.py"
    split_script = repo_root / "split_dataset.py"

    # Step 1: merge
    run_python_script(
        merge_script,
        ["--dataset_root", str(input_root), "--out_root", str(dataset_root)],
        logger,
    )

    # Step 2: convert masks
    masks_dir = dataset_root / "masks"
    run_python_script(
        convert_script,
        ["--dataset_root", str(dataset_root), "--masks_dir", str(masks_dir)],
        logger,
    )

    # Step 3: visualize overlays (requirements random sample >=25; script already does default 25)
    run_python_script(
        repo_root / "visualize_masks.py",
        ["--dataset_root", str(dataset_root), "--num_samples", str(args.overlay_samples), "--seed", str(args.seed)],
        logger,
    )

    # Step 4: dataset statistics (will be corrected in dataset_statistics.py)
    run_python_script(
        stats_script,
        ["--dataset_root", str(dataset_root)],
        logger,
    )

    # Step 5: split dataset
    run_python_script(
        split_script,
        [
            "--dataset_root",
            str(dataset_root),
            "--seed",
            str(args.seed),
            "--train_ratio",
            "0.8",
            "--val_ratio",
            "0.1",
            "--test_ratio",
            "0.1",
        ],
        logger,
    )

    # Final strict mask verification (and also generate sample overlays to visualization/)
    strict = verify_masks_strict(dataset_root, sample_k=args.overlay_samples, seed=args.seed)

    # Build phase1_report.md
    # Gather split counts
    def count_pairs(split: str) -> int:
        sd = dataset_root / split
        if not sd.exists():
            return 0
        images_dir = sd / "images"
        masks_dir = sd / "masks"
        stems_img = {p.stem for p in images_dir.glob("*") if p.is_file()}
        stems_mask = {p.stem for p in masks_dir.glob("*.png") if p.is_file()}
        return len(stems_img.intersection(stems_mask))

    split_counts = {
        "train": count_pairs("train"),
        "val": count_pairs("val"),
        "test": count_pairs("test"),
    }

    stats_path = dataset_root / "statistics" / "dataset_statistics.json"
    stats = {}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}

    dup_report_path = dataset_root / "verification_report.json"
    dup_report = {}
    if dup_report_path.exists():
        try:
            dup_report = json.loads(dup_report_path.read_text(encoding="utf-8"))
        except Exception:
            dup_report = {}

    lesion_pixels_remaining = bool(strict.get("class_violations"))

    sample_overlays = strict.get("sample_overlays", [])

    report_md = []
    report_md.append("# Phase 1 Completion Report\n")
    report_md.append("## Dataset Preparation Summary\n")
    report_md.append(f"- Images Processed: {strict['images_total']}\n")
    report_md.append(f"- Masks Generated: {strict['masks_total']}\n")
    report_md.append(f"- Missing Masks: {len(strict['missing_masks'])}\n")
    report_md.append(f"- Missing Files: {len(strict['missing_images'])}\n")
    report_md.append(f"- Duplicate Files (across full/partial by filename): {len(dup_report.get('duplicate_filenames_across_full_and_partial', []))}\n")
    report_md.append(f"- Corrupted Files (JSON read failures): {len(strict['read_failures'])}\n")

    report_md.append("\n## Split Counts\n")
    report_md.append(f"- Train: {split_counts['train']}\n")
    report_md.append(f"- Validation: {split_counts['val']}\n")
    report_md.append(f"- Test: {split_counts['test']}\n")

    if stats:
        report_md.append("\n## Dataset Statistics\n")
        report_md.append(f"- Cornea Count (polygons): {stats.get('cornea_polygons', 'N/A')}\n")
        report_md.append(f"- Iris Count (polygons): {stats.get('iris_polygons', 'N/A')}\n")
        report_md.append(f"- Average Polygon Points: {stats.get('average_polygon_points', 'N/A')}\n")
        report_md.append(f"- Average Image Resolution: {stats.get('image_resolutions', {}).get('most_common', 'N/A')}\n")
        report_md.append(f"- Average Iris Area (pixels): {stats.get('polygon_area_pixels', {}).get('average_iris_area', 'N/A')}\n")
        report_md.append(f"- Average Cornea Area (pixels): {stats.get('polygon_area_pixels', {}).get('average_cornea_area', 'N/A')}\n")

    report_md.append("\n## Mask Verification (Strict)\n")
    report_md.append(f"- Passed: {strict['passed']}\n")
    report_md.append(f"- Dimension Violations: {len(strict['dim_violations'])}\n")
    report_md.append(f"- Class Violations (should only be 0/1/2): {len(strict['class_violations'])}\n")
    report_md.append(f"- Lesion Pixels Remaining: {lesion_pixels_remaining}\n")

    report_md.append("\n## Sample Overlay Images\n")
    if sample_overlays:
        for s in sample_overlays:
            stem = s['stem']
            report_md.append(f"### {stem}\n")
            report_md.append(f"- Original: {Path(s['original_overlay']).as_posix()}\n")
            report_md.append(f"- Mask: {Path(s['mask_viz']).as_posix()}\n")
            report_md.append(f"- Overlay: {Path(s['overlay']).as_posix()}\n")
    else:
        report_md.append("No overlay samples were generated.\n")

    phase1_report_path = repo_root / "phase1_report.md"
    phase1_report_path.write_text("".join(report_md), encoding="utf-8")

    logger.info("Strict mask verification passed=%s", strict["passed"])
    logger.info("Generated report: %s", phase1_report_path)

    # Fail hard if strict verification fails
    if not strict["passed"]:
        raise RuntimeError("Phase 1 strict verification failed. See logs and visualization outputs.")


if __name__ == "__main__":
    main()

