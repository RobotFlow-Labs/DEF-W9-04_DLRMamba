import torch

from anima_dlrmamba.losses import StructureAwareDistillationLoss


def test_distillation_components_positive() -> None:
    criterion = StructureAwareDistillationLoss()

    u_s = torch.randn(32, 8)
    v_s = torch.randn(32, 8)
    a_t = torch.randn(32, 32)

    svd = criterion.svd_alignment_loss(u_s, v_s, a_t)
    assert svd.item() >= 0.0

    s1 = torch.randn(2, 64, 32)
    s2 = torch.randn(2, 64, 32)
    state = criterion.state_alignment_loss(s1, s2)
    assert state.item() >= 0.0

    f1 = torch.randn(2, 16, 32, 32)
    f2 = torch.randn(2, 16, 32, 32)
    feat = criterion.feature_reconstruction_loss(f1, f2)
    assert feat.item() >= 0.0


def test_total_objective() -> None:
    criterion = StructureAwareDistillationLoss()
    total = criterion(
        task_loss=torch.tensor(1.0),
        svd_loss=torch.tensor(0.5),
        state_loss=torch.tensor(0.25),
        feat_loss=torch.tensor(0.75),
    )
    assert total.item() > 0.0
