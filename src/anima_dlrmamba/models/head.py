from __future__ import annotations

import torch
from torch import nn


class _SingleScaleHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.box = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls(x), self.box(x)


class DecoupledDetectionHead(nn.Module):
    def __init__(self, channels: list[int], num_classes: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([_SingleScaleHead(c, num_classes) for c in channels])

    def forward(self, pyramids: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        cls_out, box_out = [], []
        for feat, head in zip(pyramids, self.heads):
            c, b = head(feat)
            cls_out.append(c)
            box_out.append(b)
        return {"cls": cls_out, "box": box_out}
