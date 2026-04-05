from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from .config import load_config
from .data import RGBIRPairDataset, RandomRGBIRDataset, collate_detection
from .losses import StructureAwareDistillationLoss
from .models.model import DLRMambaDetector


def build_dataloader(cfg, train: bool = True) -> DataLoader:
    root = cfg.data.train_root if train else cfg.data.val_root
    if root and Path(root).exists():
        ds = RGBIRPairDataset(root=root, image_size=cfg.data.image_size)
    else:
        ds = RandomRGBIRDataset(image_size=cfg.data.image_size, num_classes=cfg.model.num_classes, length=32)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=train, num_workers=cfg.data.num_workers, collate_fn=collate_detection)


def train_loop(config_path: str, max_steps: int | None = None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = DLRMambaDetector(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        fusion_channels=cfg.model.fusion_channels,
        embed_dim=cfg.model.embed_dim,
        num_blocks=cfg.model.num_blocks,
        state_dim=cfg.model.state_dim,
        rank_ratio=cfg.model.rank_ratio,
    ).to(device)

    teacher = DLRMambaDetector(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        fusion_channels=cfg.model.fusion_channels,
        embed_dim=cfg.model.embed_dim,
        num_blocks=max(cfg.model.num_blocks + 1, 4),
        state_dim=cfg.model.state_dim,
        rank_ratio=1.0,
    ).to(device)
    teacher.eval()

    optimizer = SGD(student.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    criterion = StructureAwareDistillationLoss(
        lambda_task=cfg.train.lambda_task,
        lambda_svd=cfg.train.lambda_svd,
        lambda_state=cfg.train.lambda_state,
        lambda_feat=cfg.train.lambda_feat,
    )

    loader = build_dataloader(cfg, train=True)

    step = 0
    for epoch in range(cfg.train.epochs):
        for sample, _targets in loader:
            sample = sample.to(device)

            out_s = student(sample)
            with torch.no_grad():
                out_t = teacher(sample)

            task_loss = criterion.detection_surrogate_loss(out_s.cls_logits, out_s.box_deltas)

            # Use first SS2D block matrices for matrix-level alignment.
            ss2d_s = student.backbone.blocks[0]
            ss2d_t = teacher.backbone.blocks[0]
            a_t = ss2d_t.U @ ss2d_t.V.T
            svd_loss = criterion.svd_alignment_loss(ss2d_s.U, ss2d_s.V, a_t.detach())

            state_loss = criterion.state_alignment_loss(out_s.states[0], out_t.states[0])
            feat_loss = criterion.feature_reconstruction_loss(out_s.pyramids[0], out_t.pyramids[0])

            total = criterion(task_loss, svd_loss, state_loss, feat_loss)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"step={step} total={total.item():.4f} task={task_loss.item():.4f}")

            if max_steps is not None and step >= max_steps:
                print("Reached max_steps; stopping early.")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train_loop(config_path=args.config, max_steps=args.max_steps)
