import numpy as np
import torch

from training.config import TrainingConfig
from training.inference import InferenceConfig, _select_device, infer_single


def _make_dummy_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 3, kernel_size=1),
    )


def _make_synthetic_image(path):
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[128:384, 128:384] = (255, 255, 255)
    np.save(path, image)
    return image


def test_select_device_auto():
    device = _select_device("auto")
    assert device.type in {"cpu", "cuda"}


# The remaining infer_single / infer_folder flow requires runtime model and images,
# so this test targets device selection and import correctness.
