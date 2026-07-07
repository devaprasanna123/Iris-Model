"""MedicalAI U-Net model.

This module implements a standard U-Net for semantic segmentation.

Input:
    - RGB image tensor of shape (B, 3, H, W)

Output:
    - Logits tensor of shape (B, 3, H, W)

Notes:
    - No activation (Softmax/Sigmoid) is applied; this is intentional so that
      torch.nn.CrossEntropyLoss can be used directly on logits.
    - The network is fully convolutional and supports arbitrary spatial sizes
      (H, W) as long as they are compatible with 4 levels of downsampling.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2 block.

    Uses 3x3 kernels with padding=1 to preserve spatial resolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Encoder block: MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Decoder block with skip connections.

    Performs:
        - ConvTranspose2d upsampling by factor 2
        - Concatenation with corresponding encoder feature map (skip)
        - DoubleConv refinement

    Spatial alignment:
        - If shapes differ by 1-2 pixels due to odd input sizes,
          the upsampled tensor is center-cropped to match the skip tensor.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize Up block.

        Args:
            in_channels: Number of channels coming from the previous layer.
                         (Before concatenation.)
            out_channels: Number of output channels after the DoubleConv.
        """

        super().__init__()
        # After upconv: channels become out_channels
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        # After concat: channels become out_channels (from skip) + out_channels (from up)
        self.conv = DoubleConv(in_channels=out_channels * 2, out_channels=out_channels)

    @staticmethod
    def _center_crop_to_match(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Center-crop src spatially to match target (H, W) if needed."""

        if src.shape[-2:] == target.shape[-2:]:
            return src

        src_h, src_w = src.shape[-2], src.shape[-1]
        tgt_h, tgt_w = target.shape[-2], target.shape[-1]

        dh = src_h - tgt_h
        dw = src_w - tgt_w

        crop_top = dh // 2
        crop_left = dw // 2

        return src[..., crop_top : crop_top + tgt_h, crop_left : crop_left + tgt_w]

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._center_crop_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net with channels: 64-128-256-512-1024 bottleneck."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__()

        self.in_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.in_conv(x)  # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16)

        # Decoder
        y = self.up1(x5, x4)  # 1024 -> 512
        y = self.up2(y, x3)   # 512 -> 256
        y = self.up3(y, x2)   # 256 -> 128
        y = self.up4(y, x1)   # 128 -> 64

        return self.out_conv(y)  # (B, 3, H, W)


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=3)
    model.eval()

    inp = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(inp)

    print("Input Shape:", inp.shape)
    print("Output Shape:", out.shape)
    print("Expected output:", "torch.Size([2, 3, 512, 512])")

