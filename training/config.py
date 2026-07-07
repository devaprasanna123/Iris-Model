"""Training configuration (single source of truth).

This module centralizes every training configuration for the MedicalAI project.
It is intentionally standalone and does not create any trainer/evaluation.

It provides:
- TrainingConfig dataclass
- JSON save/load helpers
- A small self-test when run as a script
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch


OptimizerName = Literal["adam", "sgd", "adamw"]
SchedulerName = Literal["none", "step", "cosine"]


def _detect_device() -> Dict[str, Any]:
    """Auto-detect device and CUDA availability.

    Returns:
        Dict with keys:
            - device_type: "cuda" or "cpu"
            - cuda_available: bool
            - gpu_name: str ("N/A" if CPU)
            - device: torch.device string
    """

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
    """Choose DataLoader workers automatically.

    Requirement note:
    - Windows/Linux supported.
    - We'll use CPU count minus 1, clamped to [0, 8].

    Returns:
        int: num_workers value.
    """

    try:
        cpu_count = len(torch.multiprocessing.get_all_start_methods())  # not CPU count; fallback below
    except Exception:
        cpu_count = 0

    # Better CPU count method in pure Python
    try:
        import os

        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = cpu_count or 0

    # If unknown, fall back to 2.
    if cpu_count <= 0:
        return 2

    return max(0, min(8, cpu_count - 1))


def _default_dataset_root() -> Path:
    """Pick the dataset root path dynamically.

    Order:
    1) /content/dataset (Colab)
    2) /content/drive/MyDrive/MedicalAI_Dataset/dataset (Colab)
    3) MedicalAI/dataset (repo fallback)
    """

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
    # Automatically detected
    device_type: Literal["cuda", "cpu"] = "cpu"
    cuda_available: bool = False
    gpu_name: str = "N/A"
    device: str = "cpu"  # torch.device string representation

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


@dataclass
class TrainingHyperparams:
    """Training hyperparameters."""

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


@dataclass
class TrainingConfig:
    """Centralized training configuration.

    This is designed to be a reusable single source of truth for all future
    training-related modules.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logs: LogsConfig = field(default_factory=LogsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig.from_detection)

    classes: ClassesConfig = field(default_factory=ClassesConfig)
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    image: ImageConfig = field(default_factory=ImageConfig)

    def _as_jsonable(self) -> Dict[str, Any]:
        """Convert config into JSON-serializable dict.

        Ensures Path and tuples are converted into strings/lists.
        """

        def convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, torch.device):
                return str(obj)
            return obj

        raw = asdict(self)
        # asdict already converts dataclasses, but may still contain Paths/tuples.
        # Walk the structure to convert special types.
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
        """Pretty-print the full configuration to stdout."""

        data = self._as_jsonable()
        print(json.dumps(data, indent=2, sort_keys=True))

    def save_json(self, path: str | Path = "config.json") -> Path:
        """Save the configuration as JSON.

        Args:
            path: Output file path.

        Returns:
            Resolved output path.
        """

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self._as_jsonable(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load_json(cls, path: str | Path) -> "TrainingConfig":
        """Load TrainingConfig from a JSON file.

        Args:
            path: JSON config path.

        Returns:
            TrainingConfig instance.
        """

        in_path = Path(path)
        data = json.loads(in_path.read_text(encoding="utf-8"))

        # Reconstruct dataclasses.
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

        output = OutputConfig(
            prediction_dir=Path(data["output"]["prediction_dir"]),
        )

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

        return cls(
            dataset=dataset,
            checkpoint=checkpoint,
            logs=logs,
            output=output,
            device=device,
            classes=classes,
            training=training,
            image=image,
        )


def _verify_roundtrip(original: TrainingConfig, loaded: TrainingConfig) -> None:
    """Verify that key values match after JSON roundtrip."""

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

    # Device fields: verify basic keys exist + equality.
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

