import torch

from anima_dlrmamba.models.model import DLRMambaDetector


def test_detector_forward_and_decode() -> None:
    model = DLRMambaDetector(
        num_classes=8,
        in_channels=3,
        fusion_channels=16,
        embed_dim=32,
        num_blocks=2,
        state_dim=32,
        rank_ratio=0.5,
    )
    sample = torch.randn(1, 2, 3, 128, 128)

    out = model(sample)
    assert len(out.pyramids) == 3
    assert len(out.cls_logits) == 3
    assert len(out.box_deltas) == 3

    preds = model.decode(out, conf_threshold=0.2, topk=20)
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], list)
