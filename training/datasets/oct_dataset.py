from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_TO_ID: dict[str, int] = {
    "Background": 0,
    "Cornea": 1,
    "Iris": 2,
}


# Default training image size required by the task.
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_IMAGE_WIDTH = 512


@dataclass(frozen=True)
class OctDatasetConfig:
    """Configuration for the OCT segmentation dataset.

    Backward compatible notes:
    - If no mode/config is provided, preprocessing defaults match the v1 behavior:
      Resize -> /255 normalization -> tensor (3,H,W)
      Mask resize with NEAREST -> int64 tensor (H,W)
    - No augmentation is applied unless mode='train'.
    """

    images_dir: Path
    masks_dir: Optional[Path]
    image_extensions: Tuple[str, ...] = (".bmp",)
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH


class _MedicalPreprocessing:
    """Deterministic medical preprocessing for images.

    Implemented with OpenCV/numpy to avoid hard dependency on Albumentations.
    Albumentations can be added later as an optional backend.
    """

    def __init__(
        self,
        *,
        resize_hw: Tuple[int, int],
        normalization_mode: str,
        clahe_enabled: bool,
        clahe_clip_limit: float,
        clahe_tile_grid_size: Tuple[int, int],
        contrast_enabled: bool,
        contrast_limit: float,
        brightness_enabled: bool,
        brightness_limit: float,
        gamma_enabled: bool,
        gamma_limit: float,
        noise_robustness_enabled: bool,
        noise_sigma: float,
    ) -> None:
        self._resize_hw = resize_hw
        self._normalization_mode = normalization_mode

        self._clahe_enabled = clahe_enabled
        self._clahe_clip_limit = float(clahe_clip_limit)
        self._clahe_tile_grid_size = tuple(clahe_tile_grid_size)

        self._contrast_enabled = contrast_enabled
        self._contrast_limit = float(contrast_limit)

        self._brightness_enabled = brightness_enabled
        self._brightness_limit = float(brightness_limit)

        self._gamma_enabled = gamma_enabled
        self._gamma_limit = float(gamma_limit)

        self._noise_robustness_enabled = noise_robustness_enabled
        self._noise_sigma = float(noise_sigma)

    def _apply_clahe_rgb(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        # Apply CLAHE on luminance approximation (Y channel) for stability.
        img_bgr = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip_limit,
            tileGridSize=self._clahe_tile_grid_size,
        )
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

    def _contrast_brightness(self, img_rgb_u8: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        # img in uint8.
        out = img_rgb_u8.astype(np.float32) * alpha + beta
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return out

    def _gamma_correction(self, img_rgb_u8: np.ndarray, gamma: float) -> np.ndarray:
        # gamma > 0
        if gamma <= 0:
            return img_rgb_u8
        table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
        table = np.clip(table, 0, 255).astype(np.uint8)
        return cv2.LUT(img_rgb_u8, table)

    def _noise_robustness(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        if self._noise_sigma <= 0:
            return img_rgb_u8
        noise = np.random.normal(0.0, self._noise_sigma, img_rgb_u8.shape).astype(np.float32)
        out = img_rgb_u8.astype(np.float32) + noise
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        return out

    def __call__(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        if img_rgb_u8.ndim != 3 or img_rgb_u8.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got {img_rgb_u8.shape}")

        resized = cv2.resize(
            img_rgb_u8,
            (self._resize_hw[1], self._resize_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        out = resized

        if self._clahe_enabled:
            out = self._apply_clahe_rgb(out)

        # Contrast/Brightness/Gamma/Noise: applied in uint8 space.
        if self._contrast_enabled and self._contrast_limit > 0:
            # alpha in [1-limit, 1+limit]
            alpha = 1.0 + random.uniform(-self._contrast_limit, self._contrast_limit)
        else:
            alpha = 1.0

        if self._brightness_enabled and self._brightness_limit > 0:
            # beta in [-limit*255, +limit*255]
            beta = random.uniform(-self._brightness_limit, self._brightness_limit) * 255.0
        else:
            beta = 0.0

        if (self._contrast_enabled and self._contrast_limit > 0) or (
            self._brightness_enabled and self._brightness_limit > 0
        ):
            out = self._contrast_brightness(out, alpha=alpha, beta=beta)

        if self._gamma_enabled and self._gamma_limit > 0:
            # gamma in [1-limit, 1+limit]
            gamma = 1.0 + random.uniform(-self._gamma_limit, self._gamma_limit)
            out = self._gamma_correction(out, gamma=gamma)

        if self._noise_robustness_enabled:
            out = self._noise_robustness(out)

        # Normalization
        if self._normalization_mode == "divide_255":
            out_f = out.astype(np.float32) / 255.0
        elif self._normalization_mode == "none":
            out_f = out.astype(np.float32)
        else:
            raise ValueError(f"Unsupported normalization_mode: {self._normalization_mode}")

        # (H,W,3) -> (3,H,W)
        chw = np.transpose(out_f, (2, 0, 1))
        chw = np.ascontiguousarray(chw)
        return chw


def _resize_mask(mask_u8: np.ndarray, resize_hw: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        mask_u8,
        (resize_hw[1], resize_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def _load_bmp_rgb(image_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image with cv2.imread: {image_path}")

    # Convert to RGB.
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _load_mask_grayscale(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask with cv2.imread: {mask_path}")

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask


def _safe_int_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    mask_i = mask.astype(np.int64, copy=False)
    mask_i = np.clip(mask_i, 0, num_classes - 1)
    return mask_i


class _TrainingAugmentation:
    """Training-only augmentation.

    Implemented without Albumentations (since it's not installed).
    """

    def __init__(
        self,
        *,
        num_classes: int,
        hflip_enabled: bool,
        rotation_degrees: float,
        shift_pixels: float,
        scale_range: Tuple[float, float],
        brightness_enabled: bool,
        brightness_limit: float,
        contrast_enabled: bool,
        contrast_limit: float,
        gaussian_noise_enabled: bool,
        gaussian_noise_sigma: float,
        blur_enabled: bool,
        blur_kernel_choices: Tuple[int, ...],
        gamma_enabled: bool,
        gamma_limit: float,
        seed: Optional[int] = None,
    ) -> None:
        self._num_classes = int(num_classes)
        self._hflip_enabled = bool(hflip_enabled)

        self._rotation_degrees = float(rotation_degrees)
        self._shift_pixels = float(shift_pixels)
        self._scale_min = float(scale_range[0])
        self._scale_max = float(scale_range[1])

        self._brightness_enabled = bool(brightness_enabled)
        self._brightness_limit = float(brightness_limit)
        self._contrast_enabled = bool(contrast_enabled)
        self._contrast_limit = float(contrast_limit)

        self._gaussian_noise_enabled = bool(gaussian_noise_enabled)
        self._gaussian_noise_sigma = float(gaussian_noise_sigma)

        self._blur_enabled = bool(blur_enabled)
        self._blur_kernel_choices = tuple(int(x) for x in blur_kernel_choices)

        self._gamma_enabled = bool(gamma_enabled)
        self._gamma_limit = float(gamma_limit)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _maybe_horizontal_flip(self, img_u8: np.ndarray, mask_i64: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._hflip_enabled:
            return img_u8, mask_i64
        if random.random() < 0.5:
            img_u8 = np.ascontiguousarray(np.flip(img_u8, axis=1))
            mask_i64 = np.ascontiguousarray(np.flip(mask_i64, axis=1))
        return img_u8, mask_i64

    def _affine_rotate_shift_scale(
        self,
        img_u8: np.ndarray,
        mask_i64: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img_u8.shape[:2]

        angle = random.uniform(-self._rotation_degrees, self._rotation_degrees) if self._rotation_degrees > 0 else 0.0
        tx = random.uniform(-self._shift_pixels, self._shift_pixels) if self._shift_pixels > 0 else 0.0
        scale = (
            random.uniform(self._scale_min, self._scale_max) if (self._scale_max > self._scale_min) else 1.0
        )

        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += tx

        # Image
        img_warp = cv2.warpAffine(
            img_u8,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        # Mask: nearest neighbor and constant padding as background=0
        mask_warp = cv2.warpAffine(
            mask_i64.astype(np.int32),
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_warp = _safe_int_mask(mask_warp, num_classes=self._num_classes)

        return img_warp, mask_warp

    def _photometric(self, img_u8: np.ndarray) -> np.ndarray:
        out = img_u8

        # Contrast/Brightness in uint8 space
        if self._contrast_enabled and self._contrast_limit > 0:
            alpha = 1.0 + random.uniform(-self._contrast_limit, self._contrast_limit)
        else:
            alpha = 1.0

        if self._brightness_enabled and self._brightness_limit > 0:
            beta = random.uniform(-self._brightness_limit, self._brightness_limit) * 255.0
        else:
            beta = 0.0

        if alpha != 1.0 or beta != 0.0:
            out = out.astype(np.float32) * alpha + beta
            out = np.clip(out, 0.0, 255.0).astype(np.uint8)

        # Gamma
        if self._gamma_enabled and self._gamma_limit > 0:
            gamma = 1.0 + random.uniform(-self._gamma_limit, self._gamma_limit)
            if gamma > 0:
                table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
                table = np.clip(table, 0, 255).astype(np.uint8)
                out = cv2.LUT(out, table)

        # Gaussian noise
        if self._gaussian_noise_enabled and self._gaussian_noise_sigma > 0:
            noise = np.random.normal(0.0, self._gaussian_noise_sigma, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        # Blur
        if self._blur_enabled and self._blur_kernel_choices:
            if random.random() < 0.3:
                k = random.choice(self._blur_kernel_choices)
                if k % 2 == 0:
                    k += 1
                if k > 1:
                    out = cv2.GaussianBlur(out, (k, k), 0)

        return out

    def __call__(self, img_u8: np.ndarray, mask_i64: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_u8, mask_i64 = self._maybe_horizontal_flip(img_u8, mask_i64)
        img_u8, mask_i64 = self._affine_rotate_shift_scale(img_u8, mask_i64)
        img_u8 = self._photometric(img_u8)
        mask_i64 = _safe_int_mask(mask_i64, num_classes=self._num_classes)
        return img_u8, mask_i64


class OctDataset(Dataset):
    """AS-OCT dataset for cornea + iris semantic segmentation.

    Modes:
      - train: deterministic preprocessing + training-only augmentation
      - val: deterministic preprocessing, no augmentation
      - test: deterministic preprocessing, no augmentation
      - predict: deterministic preprocessing + dummy zero mask
      - external: deterministic preprocessing + dummy zero mask

    Output contract for all modes:
      (image_tensor, mask_tensor)

    - image_tensor: FloatTensor (3,H,W), normalized
    - mask_tensor: LongTensor (H,W), int class ids. For predict/external: zeros.
    """

    def __init__(
        self,
        images_dir: str | os.PathLike[str],
        masks_dir: str | os.PathLike[str] | None = None,
        *,
        strict_pairing: bool = True,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        mode: str = "train",
        num_classes: int = 3,
        # Optional config object (TrainingConfig.preprocess/augmentation are supported).
        preprocess_overrides: Optional[Dict[str, Any]] = None,
        augmentation_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._mode = str(mode).lower()
        self._num_classes = int(num_classes)

        masks_path = Path(masks_dir) if masks_dir is not None else None

        cfg = OctDatasetConfig(
            images_dir=Path(images_dir),
            masks_dir=masks_path,
            image_height=int(image_height),
            image_width=int(image_width),
        )
        self._cfg = cfg
        self._strict_pairing = bool(strict_pairing)
        self._transform = transform

        self._pairs: List[Tuple[Path, Optional[Path]]] = self._build_pairs()

        # Defaults for v2 preprocessing/augmentation.
        # These are expected to be overridden by training/config.py via dataloaders.
        preprocess_defaults: Dict[str, Any] = {
            "resize_hw": (self._cfg.image_height, self._cfg.image_width),
            "normalization_mode": "divide_255",
            "clahe_enabled": False,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": (8, 8),
            "contrast_enabled": True,
            "contrast_limit": 0.15,
            "brightness_enabled": True,
            "brightness_limit": 0.15,
            "gamma_enabled": True,
            "gamma_limit": 0.2,
            "noise_robustness_enabled": False,
            "noise_sigma": 5.0,
        }
        if preprocess_overrides:
            preprocess_defaults.update(preprocess_overrides)

        self._preprocess = _MedicalPreprocessing(
            resize_hw=tuple(preprocess_defaults["resize_hw"]),
            normalization_mode=str(preprocess_defaults["normalization_mode"]),
            clahe_enabled=bool(preprocess_defaults["clahe_enabled"]),
            clahe_clip_limit=float(preprocess_defaults["clahe_clip_limit"]),
            clahe_tile_grid_size=tuple(preprocess_defaults["clahe_tile_grid_size"]),
            contrast_enabled=bool(preprocess_defaults["contrast_enabled"]),
            contrast_limit=float(preprocess_defaults["contrast_limit"]),
            brightness_enabled=bool(preprocess_defaults["brightness_enabled"]),
            brightness_limit=float(preprocess_defaults["brightness_limit"]),
            gamma_enabled=bool(preprocess_defaults["gamma_enabled"]),
            gamma_limit=float(preprocess_defaults["gamma_limit"]),
            noise_robustness_enabled=bool(preprocess_defaults["noise_robustness_enabled"]),
            noise_sigma=float(preprocess_defaults["noise_sigma"]),
        )

        aug_defaults: Dict[str, Any] = {
            "hflip_enabled": False,
            "rotation_degrees": 10.0,
            "shift_pixels": 10.0,
            "scale_range": (0.9, 1.1),
            "brightness_enabled": True,
            "brightness_limit": 0.10,
            "contrast_enabled": True,
            "contrast_limit": 0.10,
            "gaussian_noise_enabled": True,
            "gaussian_noise_sigma": 3.0,
            "blur_enabled": True,
            "blur_kernel_choices": (3, 5),
            "gamma_enabled": True,
            "gamma_limit": 0.15,
            "seed": None,
        }
        if augmentation_overrides:
            aug_defaults.update(augmentation_overrides)

        self._augmentation = _TrainingAugmentation(
            num_classes=self._num_classes,
            hflip_enabled=bool(aug_defaults["hflip_enabled"]),
            rotation_degrees=float(aug_defaults["rotation_degrees"]),
            shift_pixels=float(aug_defaults["shift_pixels"]),
            scale_range=tuple(aug_defaults["scale_range"]),
            brightness_enabled=bool(aug_defaults["brightness_enabled"]),
            brightness_limit=float(aug_defaults["brightness_limit"]),
            contrast_enabled=bool(aug_defaults["contrast_enabled"]),
            contrast_limit=float(aug_defaults["contrast_limit"]),
            gaussian_noise_enabled=bool(aug_defaults["gaussian_noise_enabled"]),
            gaussian_noise_sigma=float(aug_defaults["gaussian_noise_sigma"]),
            blur_enabled=bool(aug_defaults["blur_enabled"]),
            blur_kernel_choices=tuple(aug_defaults["blur_kernel_choices"]),
            gamma_enabled=bool(aug_defaults["gamma_enabled"]),
            gamma_limit=float(aug_defaults["gamma_limit"]),
            seed=aug_defaults.get("seed", None),
        )

        self._resize_hw = (self._cfg.image_height, self._cfg.image_width)

        # Validate mode
        if self._mode not in {"train", "val", "test", "predict", "external"}:
            raise ValueError(
                f"Unsupported mode='{self._mode}'. Expected one of: train, val, test, predict, external."
            )

        if self._mode in {"train", "val", "test"}:
            if self._cfg.masks_dir is None:
                raise ValueError("masks_dir must be provided for train/val/test modes")

    def _build_pairs(self) -> List[Tuple[Path, Optional[Path]]]:
        if not self._cfg.images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {self._cfg.images_dir}")

        image_paths: List[Path] = []
        for p in sorted(self._cfg.images_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() in {ext.lower() for ext in self._cfg.image_extensions}:
                image_paths.append(p)

        if not image_paths:
            raise FileNotFoundError(
                f"No images found in: {self._cfg.images_dir} (expected {self._cfg.image_extensions})"
            )

        if self._mode in {"predict", "external"}:
            return [(img_path, None) for img_path in image_paths]

        if self._cfg.masks_dir is None:
            raise ValueError("masks_dir is required for non-predict modes")

        pairs: List[Tuple[Path, Optional[Path]]] = []
        missing: List[str] = []

        for img_path in image_paths:
            stem = img_path.stem
            mask_path = self._cfg.masks_dir / f"{stem}.png"
            if not mask_path.exists() or not mask_path.is_file():
                missing.append(str(mask_path))
                if self._strict_pairing:
                    raise FileNotFoundError(
                        f"Missing mask for image '{img_path.name}'. Expected: '{mask_path}'."
                    )
                continue
            pairs.append((img_path, mask_path))

        if not pairs:
            raise FileNotFoundError(
                "No valid (image, mask) pairs found. "
                f"Missing masks: {len(missing)}. images_dir={self._cfg.images_dir}, masks_dir={self._cfg.masks_dir}"
            )

        # Verify matching filenames (stem equality)
        for img_path, mask_path in pairs:
            if mask_path is None:
                continue
            if img_path.stem != mask_path.stem:
                raise ValueError(
                    f"Filename mismatch: image stem '{img_path.stem}' vs mask stem '{mask_path.stem}'."
                )

        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    @staticmethod
    def _to_tensor_image_from_chw_float(chw_float: np.ndarray) -> torch.Tensor:
        # chw_float expected float32 (3,H,W)
        if chw_float.dtype != np.float32:
            chw_float = chw_float.astype(np.float32)
        return torch.from_numpy(chw_float).contiguous()

    @staticmethod
    def _to_tensor_mask(mask_i64: np.ndarray) -> torch.Tensor:
        if mask_i64.dtype != np.int64:
            mask_i64 = mask_i64.astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(mask_i64)).contiguous()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self._pairs[index]

        img_rgb = _load_bmp_rgb(img_path)  # uint8 RGB, variable size

        # Deterministic preprocessing (resize + optional medical ops + normalization)
        # returns chw_float in np.float32
        img_chw = self._preprocess(img_rgb)

        # For masks: resize separately with NEAREST to keep class ids.
        if mask_path is not None:
            mask = _load_mask_grayscale(mask_path)
            mask = _resize_mask(mask, self._resize_hw)
            mask_i64 = _safe_int_mask(mask, num_classes=self._num_classes)
        else:
            # dummy mask for predict/external
            mask_i64 = np.zeros(self._resize_hw, dtype=np.int64)

        # Training augmentation: apply in uint8 mask+image space.
        if self._mode == "train" and mask_path is not None:
            # Recreate uint8 RGB for augmentation.
            # We apply augmentation BEFORE normalization; therefore we need the resized uint8 image.
            img_rgb_u8 = cv2.resize(
                img_rgb,
                (self._resize_hw[1], self._resize_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            img_rgb_u8, mask_i64 = self._augmentation(img_rgb_u8, mask_i64)
            # Apply normalization exactly like preprocessing would (without resizing again)
            # We'll reuse preprocessing by temporarily running medical preprocessing on augmented u8.
            # Set clahe/contrast/etc inside _preprocess; it will also resize, so avoid double resize by letting it run.
            img_chw = self._preprocess(img_rgb_u8)

        image_tensor = self._to_tensor_image_from_chw_float(img_chw)
        mask_tensor = self._to_tensor_mask(mask_i64)

        if self._transform is not None:
            image_tensor, mask_tensor = self._transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor


def _self_test_first_five_samples() -> None:
    """Quick runtime validation.

    Contract:
      - image: (3,512,512) float32
      - mask: (512,512) int64
    """

    dataset_root = Path("MedicalAI") / "dataset"
    train_images_dir = dataset_root / "train" / "images"
    train_masks_dir = dataset_root / "train" / "masks"

    if not train_images_dir.exists() or not train_masks_dir.exists():
        print(
            "[OctDataset self-test] Skipped: expected dataset paths not found. "
            f"images={train_images_dir} masks={train_masks_dir}"
        )
        return

    ds = OctDataset(train_images_dir, train_masks_dir, mode="train")
    n = min(5, len(ds))
    for i in range(n):
        img, mask = ds[i]
        print(
            f"[sample {i}] Image: {tuple(img.shape)} dtype={img.dtype} ; "
            f"Mask: {tuple(mask.shape)} dtype={mask.dtype}"
        )
        assert tuple(img.shape) == (3, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
        assert img.dtype == torch.float32
        assert tuple(mask.shape) == (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
        assert mask.dtype == torch.int64


if __name__ == "__main__":
    _self_test_first_five_samples()

