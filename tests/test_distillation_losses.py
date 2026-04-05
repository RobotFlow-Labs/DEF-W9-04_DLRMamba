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


def test_detection_loss() -> None:
    criterion = StructureAwareDistillationLoss()
    cls_logits = [torch.randn(2, 8, 16, 16, requires_grad=True), torch.randn(2, 8, 8, 8, requires_grad=True)]
    box_deltas = [torch.randn(2, 4, 16, 16, requires_grad=True), torch.randn(2, 4, 8, 8, requires_grad=True)]
    targets = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "labels": torch.tensor([3])},
        {"boxes": torch.tensor([[0.3, 0.7, 0.2, 0.2]]), "labels": torch.tensor([1])},
    ]
    loss = criterion.detection_loss(cls_logits, box_deltas, targets)
    assert loss.item() >= 0.0
    assert loss.requires_grad


def test_state_alignment_different_shapes() -> None:
    criterion = StructureAwareDistillationLoss()
    s1 = torch.randn(2, 64, 32)
    s2 = torch.randn(2, 64, 48)
    state = criterion.state_alignment_loss(s1, s2)
    assert state.item() >= 0.0


def test_feature_reconstruction_different_shapes() -> None:
    criterion = StructureAwareDistillationLoss()
    f1 = torch.randn(2, 16, 32, 32)
    f2 = torch.randn(2, 32, 64, 64)
    feat = criterion.feature_reconstruction_loss(f1, f2)
    assert feat.item() >= 0.0
