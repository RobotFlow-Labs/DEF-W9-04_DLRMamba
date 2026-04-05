from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from .config import AppConfig, load_config
from .data import RGBIRPairDataset, RandomRGBIRDataset, collate_detection
from .losses import StructureAwareDistillationLoss
from .models.model import DLRMambaDetector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


class CheckpointManager:
    def __init__(self, save_dir: str, keep_top_k: int = 2, metric: str = "val_loss", mode: str = "min"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, epoch: int) -> Path:
        path = self.save_dir / f"checkpoint_epoch{epoch:04d}_val{metric_value:.4f}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        best_val, best_path = self.history[0]
        best_dst = self.save_dir / "best.pth"
        shutil.copy2(best_path, best_dst)
        return path


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta)
            if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_dataloader(cfg: AppConfig, split: str = "train") -> DataLoader:
    if split == "train":
        root = cfg.data.train_root
    elif split == "val":
        root = cfg.data.val_root
    else:
        root = cfg.data.test_root

    if root and Path(root).exists():
        ds = RGBIRPairDataset(root=root, image_size=cfg.data.image_size)
    else:
        length = 64 if split == "train" else 16
        ds = RandomRGBIRDataset(
            image_size=cfg.data.image_size, num_classes=cfg.model.num_classes, length=length
        )
        if root:
            print(f"[WARN] {split} root '{root}' not found — using synthetic data")

    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_detection,
        drop_last=(split == "train"),
    )


def train_loop(config_path: str, max_steps: int | None = None, resume: str = "") -> None:
    cfg = load_config(config_path)
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float16

    # --- Models ---
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
    for p in teacher.parameters():
        p.requires_grad_(False)

    param_count = sum(p.numel() for p in student.parameters())
    print(f"[MODEL] Student params: {param_count / 1e6:.2f}M")
    print(f"[CONFIG] {config_path}")
    print(f"[DEVICE] {device}, AMP={use_amp} ({cfg.train.precision})")

    # --- Optimizer ---
    optimizer = SGD(
        student.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = StructureAwareDistillationLoss(
        lambda_task=cfg.train.lambda_task,
        lambda_svd=cfg.train.lambda_svd,
        lambda_state=cfg.train.lambda_state,
        lambda_feat=cfg.train.lambda_feat,
    )

    # --- Data ---
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    total_steps = len(train_loader) * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, cfg.train.min_lr)

    # --- Checkpoint manager ---
    ckpt_mgr = CheckpointManager(
        save_dir=cfg.checkpoint.output_dir,
        keep_top_k=cfg.checkpoint.keep_top_k,
        metric=cfg.checkpoint.metric,
        mode=cfg.checkpoint.mode,
    )

    early_stop = EarlyStopping(
        patience=cfg.early_stopping.patience,
        min_delta=cfg.early_stopping.min_delta,
        mode=cfg.checkpoint.mode,
    ) if cfg.early_stopping.enabled else None

    # --- AMP scaler ---
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    resume_path = resume or cfg.train.resume
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[RESUME] from epoch {start_epoch}, step {global_step}")

    # --- Log dir ---
    log_dir = Path(cfg.checkpoint.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "training_history.jsonl"

    print(f"[BATCH] batch_size={cfg.train.batch_size}")
    print(f"[DATA] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    print(f"[TRAIN] {cfg.train.epochs} epochs, lr={cfg.train.lr}, optimizer=SGD")
    print(f"[CKPT] save to {cfg.checkpoint.output_dir}, keep best {cfg.checkpoint.keep_top_k}")

    # --- Training ---
    best_val_loss = float("inf")
    for epoch in range(start_epoch, cfg.train.epochs):
        student.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for sample, _targets in train_loader:
            sample = sample.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out_s = student(sample)
                with torch.no_grad():
                    out_t = teacher(sample)

                task_loss = criterion.detection_loss(out_s.cls_logits, out_s.box_deltas, _targets)

                ss2d_s = student.backbone.blocks[0]
                ss2d_t = teacher.backbone.blocks[0]
                a_t = ss2d_t.U @ ss2d_t.V.T
                svd_loss = criterion.svd_alignment_loss(ss2d_s.U, ss2d_s.V, a_t.detach())
                state_loss = criterion.state_alignment_loss(out_s.states[0], out_t.states[0])
                feat_loss = criterion.feature_reconstruction_loss(out_s.pyramids[0], out_t.pyramids[0])

                total = criterion(task_loss, svd_loss, state_loss, feat_loss)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()

            if cfg.train.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), cfg.train.gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += total.item()
            epoch_steps += 1

            if global_step % cfg.train.log_interval == 0:
                lr = scheduler.get_lr()
                print(
                    f"[Epoch {epoch+1}/{cfg.train.epochs}] step={global_step} "
                    f"loss={total.item():.4f} task={task_loss.item():.4f} "
                    f"svd={svd_loss.item():.4f} state={state_loss.item():.4f} "
                    f"feat={feat_loss.item():.4f} lr={lr:.6f}"
                )

            if torch.isnan(total):
                print("[FATAL] Loss is NaN — stopping training")
                print("[FIX] Reduce lr by 10x, check data, check gradient clipping")
                return

            if max_steps is not None and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}; stopping early.")
                _save_checkpoint(
                    ckpt_mgr, student, optimizer, scheduler, scaler,
                    epoch, global_step, epoch_loss / max(epoch_steps, 1), cfg
                )
                return

        train_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        samples_per_sec = len(train_loader.dataset) / elapsed

        # --- Validation ---
        val_loss = float("inf")
        if (epoch + 1) % cfg.train.val_interval == 0:
            val_loss = validate(student, teacher, val_loader, criterion, device, use_amp, amp_dtype)

        print(
            f"[Epoch {epoch+1}/{cfg.train.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={scheduler.get_lr():.6f} "
            f"time={elapsed:.1f}s ({samples_per_sec:.1f} samp/s)"
        )

        # Log
        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(scheduler.get_lr(), 8),
            "global_step": global_step,
            "time_s": round(elapsed, 1),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Checkpoint
        if val_loss < best_val_loss or (epoch + 1) % cfg.checkpoint.save_every_n_epochs == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            _save_checkpoint(
                ckpt_mgr, student, optimizer, scheduler, scaler,
                epoch, global_step, val_loss, cfg
            )

        # Early stopping
        if early_stop and early_stop.step(val_loss):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs. Stopping.")
            break

    print(f"[DONE] Training complete. Best val_loss={best_val_loss:.4f}")


def _save_checkpoint(
    ckpt_mgr: CheckpointManager,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    metric_value: float,
    cfg: AppConfig,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metric_value": metric_value,
        "config": {
            "model": {k: getattr(cfg.model, k) for k in cfg.model.__dataclass_fields__},
            "train": {k: getattr(cfg.train, k) for k in cfg.train.__dataclass_fields__},
        },
    }
    path = ckpt_mgr.save(state, metric_value, epoch)
    print(f"[CKPT] Saved {path.name} (val={metric_value:.4f})")


@torch.no_grad()
def validate(
    student: nn.Module,
    teacher: nn.Module,
    val_loader: DataLoader,
    criterion: StructureAwareDistillationLoss,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    student.eval()
    total_loss = 0.0
    count = 0
    for sample, targets in val_loader:
        sample = sample.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out_s = student(sample)
            out_t = teacher(sample)

            task_loss = criterion.detection_loss(out_s.cls_logits, out_s.box_deltas, targets)
            ss2d_s = student.backbone.blocks[0]
            ss2d_t = teacher.backbone.blocks[0]
            a_t = ss2d_t.U @ ss2d_t.V.T
            svd_loss = criterion.svd_alignment_loss(ss2d_s.U, ss2d_s.V, a_t)
            state_loss = criterion.state_alignment_loss(out_s.states[0], out_t.states[0])
            feat_loss = criterion.feature_reconstruction_loss(out_s.pyramids[0], out_t.pyramids[0])

            loss = criterion(task_loss, svd_loss, state_loss, feat_loss)

        total_loss += loss.item()
        count += 1

    student.train()
    return total_loss / max(count, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    train_loop(config_path=args.config, max_steps=args.max_steps, resume=args.resume)
