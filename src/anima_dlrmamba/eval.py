"""mAP50 evaluation harness for DLRMamba."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import RGBIRPairDataset, collate_detection
from .models.model import DLRMambaDetector


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU between two boxes in (cx, cy, w, h) format."""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_x * inter_y

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / max(union, 1e-8)


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        precs_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(precs_at_recall) if precs_at_recall else 0.0
    return ap / 11.0


def evaluate_map50(
    model: DLRMambaDetector,
    dataloader: DataLoader,
    num_classes: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute mAP@50 across all classes."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds: dict[int, list] = {c: [] for c in range(num_classes)}
    all_gts: dict[int, int] = {c: 0 for c in range(num_classes)}

    img_id = 0
    with torch.no_grad():
        for sample, targets in dataloader:
            sample = sample.to(device, non_blocking=True)
            out = model(sample)
            batch_preds = model.decode(out, conf_threshold=conf_threshold)

            for b in range(len(batch_preds)):
                gt_boxes = targets[b]["boxes"].numpy()
                gt_labels = targets[b]["labels"].numpy()

                for i in range(len(gt_labels)):
                    cls_id = int(gt_labels[i])
                    if 0 <= cls_id < num_classes:
                        all_gts[cls_id] += 1

                matched_gt = set()
                preds_sorted = sorted(batch_preds[b], key=lambda p: p["score"], reverse=True)

                for pred in preds_sorted:
                    cls_id = pred["class_id"]
                    if cls_id < 0 or cls_id >= num_classes:
                        continue
                    pred_box = np.array([pred["bx"], pred["by"], pred["bw"], pred["bh"]])
                    score = pred["score"]

                    best_iou = 0.0
                    best_gt_idx = -1
                    for gi in range(len(gt_labels)):
                        if int(gt_labels[gi]) != cls_id:
                            continue
                        if gi in matched_gt:
                            continue
                        iou = compute_iou(pred_box, gt_boxes[gi])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gi

                    tp = 1 if best_iou >= iou_threshold and best_gt_idx >= 0 else 0
                    if tp:
                        matched_gt.add(best_gt_idx)

                    all_preds[cls_id].append((score, tp, img_id))

                img_id += 1

    # Compute per-class AP
    aps = {}
    for cls_id in range(num_classes):
        preds_cls = sorted(all_preds[cls_id], key=lambda x: x[0], reverse=True)
        n_gt = all_gts[cls_id]
        if n_gt == 0:
            continue

        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        for score, tp, _ in preds_cls:
            if tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / n_gt)

        aps[cls_id] = compute_ap(precisions, recalls)

    map50 = float(np.mean(list(aps.values()))) if aps else 0.0
    return {"mAP50": round(map50, 4), "per_class_AP": {k: round(v, 4) for k, v in aps.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="DLRMamba evaluation")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DLRMambaDetector(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        fusion_channels=cfg.model.fusion_channels,
        embed_dim=cfg.model.embed_dim,
        num_blocks=cfg.model.num_blocks,
        state_dim=cfg.model.state_dim,
        rank_ratio=cfg.model.rank_ratio,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)

    root = cfg.data.val_root if args.split == "val" else cfg.data.test_root
    ds = RGBIRPairDataset(root=root, image_size=cfg.data.image_size)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_detection)

    results = evaluate_map50(model, loader, cfg.model.num_classes, device=device)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
