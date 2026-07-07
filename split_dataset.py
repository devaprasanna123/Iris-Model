import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from dataset_exclusion import load_excluded_samples, is_excluded_sample





def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("split_dataset")
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


def find_image_mask_pairs(images_dir: Path, masks_dir: Path, excluded: set[str]) -> List[Tuple[Path, Path]]:
    images = {p.stem: p for p in images_dir.iterdir() if p.is_file() and not is_excluded_sample(p.stem, excluded)}
    pairs: List[Tuple[Path, Path]] = []
    for stem, img_path in images.items():
        if is_excluded_sample(stem, excluded):
            continue
        mask_path = masks_dir / f"{stem}.png"
        if mask_path.exists() and mask_path.is_file() and not is_excluded_sample(mask_path.stem, excluded):
            pairs.append((img_path, mask_path))
    return pairs



def copy_pair(pair: Tuple[Path, Path], out_images: Path, out_masks: Path) -> None:
    img_path, mask_path = pair
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(img_path), str(out_images / img_path.name))
    shutil.copy2(str(mask_path), str(out_masks / mask_path.name))


def main():
    parser = argparse.ArgumentParser(description="Split merged dataset into train/val/test.")
    parser.add_argument("--dataset_root", type=str, default=str(Path("MedicalAI") / "dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Expected dataset_root/images and dataset_root/masks. Run convert_labelme_to_masks.py first.")

    logger = setup_logger(dataset_root / "logs" / "split_dataset.log")
    excluded = load_excluded_samples(Path(__file__).resolve().parent)
    logger.info("Finding image-mask pairs...")
    pairs = find_image_mask_pairs(images_dir, masks_dir, excluded)

    if not pairs:
        raise FileNotFoundError("No image-mask pairs found. Check masks_dir naming.")

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    # ensure remainder goes to test
    n_test = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    out_root = dataset_root
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        out_images = out_root / split_name / "images"
        out_masks = out_root / split_name / "masks"
        for pair in split_pairs:
            copy_pair(pair, out_images, out_masks)
        logger.info(f"{split_name}: {len(split_pairs)} pairs")

    # counts
    stats = {
        "seed": args.seed,
        "total_pairs": n,
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    }
    stats_path = out_root / "logs" / "split_report.json"

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(__import__("json").dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

