"""Training configuration (single source of truth).

This module centralizes every training configuration for the MedicalAI project.
It is intentionally standalone and does not create any trainer/evaluation.

It provides:
- TrainingConfig dataclass
- JSON save/load helpers
- A small self-test when run as a script

Phase note:
- Dataset pipeline parameters for preprocessing and augmentation live here
  so the dataset loader does not hardcode values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, Literal, Tuple

import torch


OptimizerName = Literal["adam", "sgd", "adamw"]
SchedulerName = Literal["none", "step", "cosine"]


def _detect_device() -> Dict[str, Any]:
    """Auto-detect device and CUDA availability."""

    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        device = torch.device("cuda")
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown"
        return {
            "device_type": "cuda",
            "cuda_available": True,
            "gpu_name": str(gpu_name),
            "device": str(device),
        }

    return {
        "device_type": "cpu",
        "cuda_available": False,
        "gpu_name": "N/A",
        "device": str(torch.device("cpu")),
    }


def _default_workers() -> int:
    """Choose DataLoader workers automatically."""

    try:
        cpu_count = len(torch.multiprocessing.get_all_start_methods())
    except Exception:
        cpu_count = 0

    try:
        import os

        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = cpu_count or 0

    if cpu_count <= 0:
        return 2

    return max(0, min(8, cpu_count - 1))


def _default_dataset_root() -> Path:
    """Pick the dataset root path dynamically."""

    candidates = [
        Path("/content/dataset"),
        Path("/content/drive/MyDrive/MedicalAI_Dataset/dataset"),
        Path("MedicalAI") / "dataset",
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[-1]


@dataclass(frozen=True)
class DatasetConfig:
    dataset_root: Path = field(default_factory=_default_dataset_root)
    train_folder: str = "train"
    val_folder: str = "val"
    test_folder: str = "test"


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_dir = Path("/content/drive/MyDrive/MedicalAI/checkpoints")
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_model.pt"


@dataclass(frozen=True)
class LogsConfig:
    log_dir: Path = Path("MedicalAI") / "training" / "logs"
    tensorboard_dir: Path = Path("MedicalAI") / "training" / "tensorboard"


@dataclass(frozen=True)
class OutputConfig:
    prediction_dir: Path = Path("MedicalAI") / "training" / "predictions"


@dataclass
class DeviceConfig:
    device_type: Literal["cuda", "cpu"] = "cpu"
    cuda_available: bool = False
    gpu_name: str = "N/A"
    device: str = "cpu"

    @staticmethod
    def from_detection() -> "DeviceConfig":
        info = _detect_device()
        return DeviceConfig(
            device_type=info["device_type"],
            cuda_available=info["cuda_available"],
            gpu_name=info["gpu_name"],
            device=info["device"],
        )


@dataclass(frozen=True)
class ImageConfig:
    channels: int = 3
    normalization: bool = True


@dataclass(frozen=True)
class DatasetPreprocessConfig:
    """Dataset deterministic preprocessing parameters."""

    # Automatic resize target
    image_height: int = 512
    image_width: int = 512

    # Normalization
    normalization_mode: Literal["divide_255", "none"] = "divide_255"

    # CLAHE
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Contrast/Brightness
    contrast_enabled: bool = True
    contrast_limit: float = 0.15  # relative intensity range

    brightness_enabled: bool = True
    brightness_limit: float = 0.15  # relative (fraction of 255)

    # Gamma correction
    gamma_enabled: bool = True
    gamma_limit: float = 0.2

    # Noise robustness (applied where appropriate)
    noise_robustness_enabled: bool = False
    noise_sigma: float = 5.0


@dataclass(frozen=True)
class DatasetAugmentationConfig:
    """Training-only augmentation parameters."""

    hflip_enabled: bool = False  # gated conservatively; flip may be anatomically invalid

    rotation_degrees: float = 10.0
    shift_pixels: float = 10.0
    scale_range: Tuple[float, float] = (0.9, 1.1)

    brightness_enabled: bool = True
    brightness_limit: float = 0.10

    contrast_enabled: bool = True
    contrast_limit: float = 0.10

    gaussian_noise_enabled: bool = True
    gaussian_noise_sigma: float = 3.0

    blur_enabled: bool = True
    blur_kernel_choices: Tuple[int, ...] = (3, 5)

    gamma_enabled: bool = True
    gamma_limit: float = 0.15

    seed: int | None = None


@dataclass
class TrainingHyperparams:
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 30

    optimizer: OptimizerName = "adamw"
    weight_decay: float = 1e-4

    scheduler: SchedulerName = "cosine"

    early_stopping: bool = True
    patience: int = 7

    mixed_precision: bool = True
    seed: int = 42

    workers: int = 2


@dataclass(frozen=True)
class ClassesConfig:
    number_of_classes: int = 3
    class_names: Tuple[str, ...] = ("Background", "Cornea", "Iris")


@dataclass(frozen=True)
class ModelConfig:
    """Segmentation model configuration.

    Important contract:
    - The model must output logits (activation=None) so trainer/loss/metrics work.
    """

    architecture: str = "unetplusplus"

    # segmentation_models_pytorch encoder config
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"

    # task output contract
    classes: int = 3
    in_channels: int = 3

    # Must be None to return logits.
    activation: str | None = None

    # Optional architecture knobs (kept configurable)
    decoder_attention: bool = False
    auxiliary_head: bool = False


@dataclass
class TrainingConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logs: LogsConfig = field(default_factory=LogsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig.from_detection)

    classes: ClassesConfig = field(default_factory=ClassesConfig)
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    image: ImageConfig = field(default_factory=ImageConfig)

    # Segmentation model config
    model: ModelConfig = field(default_factory=ModelConfig)

    # v2 dataset pipeline parameters (moved from hardcoded values)
    preprocess: DatasetPreprocessConfig = field(default_factory=DatasetPreprocessConfig)
    augmentation: DatasetAugmentationConfig = field(default_factory=DatasetAugmentationConfig)


    def _as_jsonable(self) -> Dict[str, Any]:
        def convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, torch.device):
                return str(obj)
            return obj

        raw = asdict(self)

        def walk(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, list):
                return [walk(v) for v in x]
            if isinstance(x, tuple):
                return [walk(v) for v in x]
            if isinstance(x, Path):
                return str(x)
            return convert(x)

        return walk(raw)

    def print_config(self) -> None:
        data = self._as_jsonable()
        print(json.dumps(data, indent=2, sort_keys=True))

    def save_json(self, path: str | Path = "config.json") -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self._as_jsonable(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainingConfig":
        in_path = Path(path)
        data = json.loads(in_path.read_text(encoding="utf-8"))

        dataset = DatasetConfig(
            dataset_root=Path(data["dataset"]["dataset_root"]),
            train_folder=data["dataset"]["train_folder"],
            val_folder=data["dataset"]["val_folder"],
            test_folder=data["dataset"]["test_folder"],
        )

        checkpoint = CheckpointConfig(
            checkpoint_dir=Path(data["checkpoint"]["checkpoint_dir"]),
            best_model_name=data["checkpoint"]["best_model_name"],
            last_model_name=data["checkpoint"]["last_model_name"],
        )

        logs = LogsConfig(
            tensorboard_dir=Path(data["logs"]["tensorboard_dir"]),
            log_dir=Path(data["logs"]["log_dir"]),
        )

        output = OutputConfig(prediction_dir=Path(data["output"]["prediction_dir"]))

        device_raw = data["device"]
        device = DeviceConfig(
            device_type=device_raw["device_type"],
            cuda_available=bool(device_raw["cuda_available"]),
            gpu_name=device_raw["gpu_name"],
            device=device_raw["device"],
        )

        classes_raw = data["classes"]
        classes = ClassesConfig(
            number_of_classes=int(classes_raw["number_of_classes"]),
            class_names=tuple(classes_raw["class_names"]),
        )

        training_raw = data["training"]
        training = TrainingHyperparams(
            batch_size=int(training_raw["batch_size"]),
            learning_rate=float(training_raw["learning_rate"]),
            epochs=int(training_raw["epochs"]),
            optimizer=training_raw["optimizer"],
            weight_decay=float(training_raw["weight_decay"]),
            scheduler=training_raw["scheduler"],
            early_stopping=bool(training_raw["early_stopping"]),
            patience=int(training_raw["patience"]),
            mixed_precision=bool(training_raw["mixed_precision"]),
            seed=int(training_raw["seed"]),
            workers=int(training_raw["workers"]),
        )

        image_raw = data["image"]
        image = ImageConfig(
            channels=int(image_raw["channels"]),
            normalization=bool(image_raw["normalization"]),
        )

        # Model config (optional for backward compatibility)
        model_raw = data.get("model", {})
        model = ModelConfig(
            architecture=model_raw.get("architecture", ModelConfig.architecture),
            encoder_name=model_raw.get("encoder_name", ModelConfig.encoder_name),
            encoder_weights=model_raw.get("encoder_weights", ModelConfig.encoder_weights),
            classes=int(model_raw.get("classes", classes.number_of_classes)),
            in_channels=int(model_raw.get("in_channels", image.channels)),
            activation=model_raw.get("activation", ModelConfig.activation),
            decoder_attention=bool(model_raw.get("decoder_attention", ModelConfig.decoder_attention)),
            auxiliary_head=bool(model_raw.get("auxiliary_head", ModelConfig.auxiliary_head)),
        )


        preprocess_raw = data.get("preprocess", {})
        preprocess = DatasetPreprocessConfig(
            image_height=int(preprocess_raw.get("image_height", DatasetPreprocessConfig.image_height)),
            image_width=int(preprocess_raw.get("image_width", DatasetPreprocessConfig.image_width)),
            normalization_mode=preprocess_raw.get("normalization_mode", DatasetPreprocessConfig.normalization_mode),
            clahe_enabled=bool(preprocess_raw.get("clahe_enabled", DatasetPreprocessConfig.clahe_enabled)),
            clahe_clip_limit=float(preprocess_raw.get("clahe_clip_limit", DatasetPreprocessConfig.clahe_clip_limit)),
            clahe_tile_grid_size=tuple(
                preprocess_raw.get("clahe_tile_grid_size", list(DatasetPreprocessConfig.clahe_tile_grid_size))
            ),
            contrast_enabled=bool(preprocess_raw.get("contrast_enabled", DatasetPreprocessConfig.contrast_enabled)),
            contrast_limit=float(preprocess_raw.get("contrast_limit", DatasetPreprocessConfig.contrast_limit)),
            brightness_enabled=bool(preprocess_raw.get("brightness_enabled", DatasetPreprocessConfig.brightness_enabled)),
            brightness_limit=float(preprocess_raw.get("brightness_limit", DatasetPreprocessConfig.brightness_limit)),
            gamma_enabled=bool(preprocess_raw.get("gamma_enabled", DatasetPreprocessConfig.gamma_enabled)),
            gamma_limit=float(preprocess_raw.get("gamma_limit", DatasetPreprocessConfig.gamma_limit)),
            noise_robustness_enabled=bool(
                preprocess_raw.get("noise_robustness_enabled", DatasetPreprocessConfig.noise_robustness_enabled)
            ),
            noise_sigma=float(preprocess_raw.get("noise_sigma", DatasetPreprocessConfig.noise_sigma)),
        )

        augmentation_raw = data.get("augmentation", {})
        augmentation = DatasetAugmentationConfig(
            hflip_enabled=bool(augmentation_raw.get("hflip_enabled", DatasetAugmentationConfig.hflip_enabled)),
            rotation_degrees=float(augmentation_raw.get("rotation_degrees", DatasetAugmentationConfig.rotation_degrees)),
            shift_pixels=float(augmentation_raw.get("shift_pixels", DatasetAugmentationConfig.shift_pixels)),
            scale_range=tuple(augmentation_raw.get("scale_range", list(DatasetAugmentationConfig.scale_range))),
            brightness_enabled=bool(
                augmentation_raw.get("brightness_enabled", DatasetAugmentationConfig.brightness_enabled)
            ),
            brightness_limit=float(
                augmentation_raw.get("brightness_limit", DatasetAugmentationConfig.brightness_limit)
            ),
            contrast_enabled=bool(
                augmentation_raw.get("contrast_enabled", DatasetAugmentationConfig.contrast_enabled)
            ),
            contrast_limit=float(
                augmentation_raw.get("contrast_limit", DatasetAugmentationConfig.contrast_limit)
            ),
            gaussian_noise_enabled=bool(
                augmentation_raw.get("gaussian_noise_enabled", DatasetAugmentationConfig.gaussian_noise_enabled)
            ),
            gaussian_noise_sigma=float(
                augmentation_raw.get("gaussian_noise_sigma", DatasetAugmentationConfig.gaussian_noise_sigma)
            ),
            blur_enabled=bool(augmentation_raw.get("blur_enabled", DatasetAugmentationConfig.blur_enabled)),
            blur_kernel_choices=tuple(
                augmentation_raw.get("blur_kernel_choices", list(DatasetAugmentationConfig.blur_kernel_choices))
            ),
            gamma_enabled=bool(augmentation_raw.get("gamma_enabled", DatasetAugmentationConfig.gamma_enabled)),
            gamma_limit=float(augmentation_raw.get("gamma_limit", DatasetAugmentationConfig.gamma_limit)),
            seed=augmentation_raw.get("seed", DatasetAugmentationConfig.seed),
        )

        return cls(
            dataset=dataset,
            checkpoint=checkpoint,
            logs=logs,
            output=output,
            device=device,
            classes=classes,
            training=training,
            image=image,
            model=model,
            preprocess=preprocess,
            augmentation=augmentation,
        )



def _verify_roundtrip(original: TrainingConfig, loaded: TrainingConfig) -> None:
    assert original.dataset.dataset_root == loaded.dataset.dataset_root

    assert original.dataset.train_folder == loaded.dataset.train_folder
    assert original.dataset.val_folder == loaded.dataset.val_folder
    assert original.dataset.test_folder == loaded.dataset.test_folder

    assert original.checkpoint.checkpoint_dir == loaded.checkpoint.checkpoint_dir
    assert original.checkpoint.best_model_name == loaded.checkpoint.best_model_name
    assert original.checkpoint.last_model_name == loaded.checkpoint.last_model_name

    assert original.logs.tensorboard_dir == loaded.logs.tensorboard_dir
    assert original.logs.log_dir == loaded.logs.log_dir

    assert original.output.prediction_dir == loaded.output.prediction_dir

    assert original.classes.number_of_classes == loaded.classes.number_of_classes
    assert list(original.classes.class_names) == list(loaded.classes.class_names)

    assert original.training.batch_size == loaded.training.batch_size
    assert original.training.learning_rate == loaded.training.learning_rate
    assert original.training.epochs == loaded.training.epochs
    assert original.training.optimizer == loaded.training.optimizer
    assert original.training.weight_decay == loaded.training.weight_decay
    assert original.training.scheduler == loaded.training.scheduler
    assert original.training.early_stopping == loaded.training.early_stopping
    assert original.training.patience == loaded.training.patience
    assert original.training.mixed_precision == loaded.training.mixed_precision
    assert original.training.seed == loaded.training.seed
    assert original.training.workers == loaded.training.workers

    assert original.image.channels == loaded.image.channels
    assert original.image.normalization == loaded.image.normalization

    assert original.preprocess.image_height == loaded.preprocess.image_height
    assert original.preprocess.image_width == loaded.preprocess.image_width

    assert original.augmentation.hflip_enabled == loaded.augmentation.hflip_enabled

    assert original.device.device_type == loaded.device.device_type
    assert original.device.cuda_available == loaded.device.cuda_available
    assert original.device.gpu_name == loaded.device.gpu_name
    assert original.device.device == loaded.device.device


if __name__ == "__main__":
    cfg = TrainingConfig()
    print("\n===== TrainingConfig (runtime instance) =====")
    cfg.print_config()

    out = cfg.save_json("config.json")
    print(f"\nSaved config to: {out.resolve().as_posix()}")

    cfg_loaded = TrainingConfig.load_json(out)
    _verify_roundtrip(cfg, cfg_loaded)

    print("\nSelf-test: JSON roundtrip verified successfully.")

