from __future__ import annotations

from torch import nn
import torch

from .ss2d import LowRankSS2D


class DLRMambaBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        embed_dim: int = 64,
        num_blocks: int = 3,
        state_dim: int = 64,
        rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [LowRankSS2D(embed_dim, state_dim=state_dim, rank_ratio=rank_ratio) for _ in range(num_blocks)]
        )

        self.down_p4 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim * 2),
            nn.SiLU(inplace=True),
        )
        self.down_p5 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim * 4),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        x = self.stem(x)
        states = []
        for blk in self.blocks:
            x, state_seq = blk(x)
            states.append(state_seq)

        p3 = x
        p4 = self.down_p4(p3)
        p5 = self.down_p5(p4)
        return [p3, p4, p5], states
