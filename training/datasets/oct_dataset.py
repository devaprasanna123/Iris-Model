from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_TO_ID: dict[str, int] = {
    "Background": 0,
    "Cornea": 1,
    "Iris": 2,
}


@dataclass(frozen=True)
class OctDatasetConfig:
    """Configuration for the OCT segmentation dataset.

    Notes:
        - No resizing
        - No augmentation
        - No normalization besides dividing the image by 255
        - Masks are kept as grayscale int IDs
    """

    images_dir: Path
    masks_dir: Path
    image_extensions: Tuple[str, ...] = (".bmp",)


class OctDataset(Dataset):
    """AS-OCT dataset for cornea + iris semantic segmentation.

    Expected on-disk layout (per split):
        {split}/images/*.bmp
        {split}/masks/*.png

    Mask values:
        - 0: Background
        - 1: Cornea
        - 2: Iris

    Returns:
        (image, mask)

        - image: torch.FloatTensor of shape (3, H, W), values in [0, 1]
        - mask:  torch.LongTensor of shape (H, W), values in {0, 1, 2}
    """

    def __init__(
        self,
        images_dir: str | os.PathLike[str],
        masks_dir: str | os.PathLike[str],
        *,
        strict_pairing: bool = True,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        """Create the dataset.

        Args:
            images_dir: Directory containing BMP images.
            masks_dir: Directory containing PNG masks.
            strict_pairing: If True, raises if any image/mask pair is missing.
            transform: Optional callable applied to (image, mask) tensors.
                      This is intentionally not used for augmentation; it can be
                      used only for deterministic preprocessing.
        """

        cfg = OctDatasetConfig(images_dir=Path(images_dir), masks_dir=Path(masks_dir))
        self._cfg = cfg
        self._strict_pairing = strict_pairing
        self._transform = transform

        self._pairs: List[Tuple[Path, Path]] = self._build_pairs()

    def _build_pairs(self) -> List[Tuple[Path, Path]]:
        if not self._cfg.images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {self._cfg.images_dir}")
        if not self._cfg.masks_dir.exists():
            raise FileNotFoundError(f"masks_dir not found: {self._cfg.masks_dir}")

        image_paths: List[Path] = []
        for p in sorted(self._cfg.images_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() in {ext.lower() for ext in self._cfg.image_extensions}:
                image_paths.append(p)

        if not image_paths:
            raise FileNotFoundError(f"No images found in: {self._cfg.images_dir} (expected {self._cfg.image_extensions})")

        pairs: List[Tuple[Path, Path]] = []
        missing: List[str] = []

        for img_path in image_paths:
            stem = img_path.stem
            # Masks are always PNG per project spec
            mask_path = self._cfg.masks_dir / f"{stem}.png"
            if not mask_path.exists() or not mask_path.is_file():
                missing.append(str(mask_path))
                if self._strict_pairing:
                    # Fail early with meaningful context
                    raise FileNotFoundError(
                        f"Missing mask for image '{img_path.name}'. Expected: '{mask_path}'."
                    )
                continue
            pairs.append((img_path, mask_path))

        if not pairs:
            raise FileNotFoundError(
                f"No valid (image, mask) pairs found. Missing masks: {len(missing)}. images_dir={self._cfg.images_dir}, masks_dir={self._cfg.masks_dir}"
            )

        # Verify matching filenames (stem equality)
        for img_path, mask_path in pairs:
            if img_path.stem != mask_path.stem:
                raise ValueError(
                    f"Filename mismatch: image stem '{img_path.stem}' vs mask stem '{mask_path.stem}'."
                )

        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    @staticmethod
    def _load_bmp_rgb(image_path: Path) -> np.ndarray:
        """Load BMP image via OpenCV and convert to RGB."""
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image with cv2.imread: {image_path}")

        # Convert to RGB.
        # - If grayscale: convert GRAY -> BGR
        # - Then BGR -> RGB
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            # Some BMPs may contain alpha; drop alpha
            img_bgr = img_bgr[:, :, :3]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    @staticmethod
    def _load_mask_grayscale(mask_path: Path) -> np.ndarray:
        """Load PNG mask via OpenCV as grayscale (single channel)."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask with cv2.imread: {mask_path}")

        # Ensure single channel grayscale.
        if mask.ndim == 3:
            # If mask is stored as BGR/RGB, convert to grayscale.
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        return mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self._pairs[index]

        # Image: BMP -> RGB -> float32 -> /255
        img_rgb = self._load_bmp_rgb(img_path)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        # (H,W,3) -> (3,H,W)
        img_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).contiguous()

        # Mask: PNG -> grayscale -> int64
        mask = self._load_mask_grayscale(mask_path)
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).contiguous()

        if self._transform is not None:
            img_tensor, mask_tensor = self._transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

