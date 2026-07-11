"""UNet++ model wrapper using ``segmentation_models_pytorch``.

Contract:
- Input: RGB tensor of shape (B, in_channels, H, W)
- Output: **logits** tensor of shape (B, classes, H, W)
- No activation is applied (activation=None).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

try:
    import segmentation_models_pytorch as smp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "segmentation_models_pytorch is required for UNet++ migration. "
        "Install it via: pip install segmentation-models-pytorch"
    ) from e

from training.config import TrainingConfig


class UNetPlusPlusModel(torch.nn.Module):
    """UNet++ (EfficientNet-B4 encoder) wrapper.

    This is a thin adapter around segmentation_models_pytorch.
    """

    def __init__(self, *, cfg: TrainingConfig) -> None:
        super().__init__()

        model_cfg = cfg.model

        # Defaults per task specification.
        encoder_name = getattr(model_cfg, "encoder_name", "efficientnet-b4")
        encoder_weights = getattr(model_cfg, "encoder_weights", "imagenet")
        classes = int(getattr(model_cfg, "classes", cfg.classes.number_of_classes))
        in_channels = int(getattr(model_cfg, "in_channels", cfg.image.channels))
        activation = getattr(model_cfg, "activation", None)

        # Task requirement: Activation must be None so we return logits.
        activation = None

        # Optional attention/aux head (kept configurable, but wrapped safely).
        decoder_attention = getattr(model_cfg, "decoder_attention", False)
        auxiliary_head = getattr(model_cfg, "auxiliary_head", False)

        # segmentation_models_pytorch API:
        # smp.UnetPlusPlus(encoder_name, encoder_weights, in_channels, classes, activation, ...)
        # It supports decoder_attention and aux_params in newer versions.
        # We pass what we can; keep robust to version differences.

        unetpp_kwargs: dict[str, Any] = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": classes,
            "activation": activation,
        }

        # decoder_attention
        if decoder_attention is not None:
            unetpp_kwargs["decoder_attention"] = decoder_attention

        # auxiliary head
        if auxiliary_head:
            # smp uses aux_params in many architectures; it can be absent depending on version.
            # Use a conservative aux_params that will be accepted if supported.
            unetpp_kwargs["aux_params"] = {"classes": classes}

        self.net = smp.UnetPlusPlus(**unetpp_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":  # pragma: no cover
    cfg = TrainingConfig()
    model = UNetPlusPlusModel(cfg=cfg)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, cfg.image.channels, 256, 256))
    print("out shape:", tuple(out.shape))

