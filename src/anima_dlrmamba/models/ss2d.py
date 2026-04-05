from __future__ import annotations

import torch
from torch import nn


class LowRankSS2D(nn.Module):
    """Low-rank SS2D approximation.

    Paper Eq. (5): h_t^s = (U V^T) h_{t-1}^s + B x_t
    """

    def __init__(self, dim: int, state_dim: int, rank_ratio: float = 0.5) -> None:
        super().__init__()
        rank = max(1, int(state_dim * rank_ratio))
        self.state_dim = state_dim
        self.rank = rank

        self.in_proj = nn.Linear(dim, state_dim)
        self.U = nn.Parameter(torch.randn(state_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(state_dim, rank) * 0.02)
        self.B = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def transition(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # (UV^T)h + Bx
        low = h @ self.V
        low = low @ self.U.T
        return low + self.B(x)

    def _scan_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L, C]
        B, L, C = seq.shape
        x = self.in_proj(seq)
        h = torch.zeros(B, self.state_dim, device=seq.device, dtype=seq.dtype)
        outs = []
        for t in range(L):
            h = self.transition(h, x[:, t, :])
            outs.append(h)
        y = torch.stack(outs, dim=1)
        y = self.out_proj(y)
        return y

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        seq_lr = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        seq_rl = torch.flip(seq_lr, dims=[1])

        y_lr = self._scan_sequence(seq_lr)
        y_rl = torch.flip(self._scan_sequence(seq_rl), dims=[1])

        y = 0.5 * (y_lr + y_rl)
        y = self.norm(y + seq_lr)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return y, y_lr
