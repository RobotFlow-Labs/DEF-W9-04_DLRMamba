import torch

from anima_dlrmamba.models.ss2d import LowRankSS2D


def test_low_rank_ss2d_shapes() -> None:
    layer = LowRankSS2D(dim=16, state_dim=16, rank_ratio=0.5)
    x = torch.randn(2, 16, 32, 32)
    y, state_seq = layer(x)
    assert y.shape == x.shape
    assert state_seq.shape[0] == 2
    assert state_seq.shape[1] == 32 * 32
