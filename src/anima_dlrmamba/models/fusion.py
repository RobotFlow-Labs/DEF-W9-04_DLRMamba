from __future__ import annotations

import torch
from torch import nn


class PixelFusion(nn.Module):
    """Pixel-level RGB/IR fusion: I^f = F_fusion(I^v, I^i)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, ir], dim=1)
        return self.proj(x)
