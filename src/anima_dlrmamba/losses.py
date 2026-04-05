from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class StructureAwareDistillationLoss(nn.Module):
    def __init__(
        self,
        lambda_task: float = 1.0,
        lambda_svd: float = 0.5,
        lambda_state: float = 0.1,
        lambda_feat: float = 1.5,
    ) -> None:
        super().__init__()
        self.lambda_task = lambda_task
        self.lambda_svd = lambda_svd
        self.lambda_state = lambda_state
        self.lambda_feat = lambda_feat

    @staticmethod
    def svd_alignment_loss(u_s: torch.Tensor, v_s: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        u_t, s_t, v_t = torch.linalg.svd(a_t, full_matrices=False)
        r = u_s.shape[1]
        s_root = torch.sqrt(s_t[:r].clamp(min=1e-8))
        u_ref = u_t[:, :r] * s_root.unsqueeze(0)
        v_ref = v_t[:r, :].T * s_root.unsqueeze(0)
        return F.mse_loss(u_s, u_ref) + F.mse_loss(v_s, v_ref)

    @staticmethod
    def state_alignment_loss(
        state_s: torch.Tensor, state_t: torch.Tensor, proj: nn.Module | None = None
    ) -> torch.Tensor:
        if proj is not None:
            state_t = proj(state_t)
        if state_s.shape != state_t.shape:
            # Reshape target to match student: [B, L, C] → use linear interp
            B, L_s, C_s = state_s.shape
            # Pool along sequence dim (dim 1)
            state_t = state_t.permute(0, 2, 1)  # [B, C_t, L_t]
            state_t = F.adaptive_avg_pool1d(state_t, L_s)  # [B, C_t, L_s]
            state_t = state_t.permute(0, 2, 1)  # [B, L_s, C_t]
            # Pool along feature dim (dim 2)
            if state_t.shape[2] != C_s:
                state_t = F.adaptive_avg_pool1d(state_t, C_s)  # [B, L_s, C_s]
        return F.mse_loss(state_s, state_t)

    @staticmethod
    def feature_reconstruction_loss(feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        if feat_s.shape != feat_t.shape:
            feat_t = F.adaptive_avg_pool2d(feat_t, feat_s.shape[-2:])
            if feat_t.shape[1] != feat_s.shape[1]:
                feat_t = feat_t[:, : feat_s.shape[1]]
        return F.mse_loss(feat_s, feat_t)

    @staticmethod
    def detection_loss(
        cls_logits: list[torch.Tensor],
        box_deltas: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """YOLO-style detection loss: focal BCE for cls + CIoU-inspired for box."""
        device = cls_logits[0].device
        B = cls_logits[0].shape[0]
        num_classes = cls_logits[0].shape[1]

        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)
        num_scales = len(cls_logits)

        for scale_idx in range(num_scales):
            cls_pred = cls_logits[scale_idx]  # [B, K, H, W]
            box_pred = box_deltas[scale_idx]  # [B, 4, H, W]
            _, _, H, W = cls_pred.shape

            # Build target maps for this scale
            cls_target = torch.zeros(B, num_classes, H, W, device=device)
            box_target = torch.zeros(B, 4, H, W, device=device)
            obj_mask = torch.zeros(B, 1, H, W, device=device)

            for b in range(B):
                t = targets[b]
                boxes = t["boxes"].to(device)  # [N, 4] (cx, cy, w, h) normalized
                labels = t["labels"].to(device)  # [N]

                if boxes.numel() == 0:
                    continue

                # Map normalized coords to grid
                cx = (boxes[:, 0] * W).clamp(0, W - 1).long()
                cy = (boxes[:, 1] * H).clamp(0, H - 1).long()

                for i in range(len(boxes)):
                    gx, gy = cx[i], cy[i]
                    cls_id = labels[i].clamp(0, num_classes - 1)
                    cls_target[b, cls_id, gy, gx] = 1.0
                    box_target[b, :, gy, gx] = boxes[i]
                    obj_mask[b, 0, gy, gx] = 1.0

            # Focal BCE for classification
            cls_prob = torch.sigmoid(cls_pred)
            alpha = 0.25
            gamma = 2.0
            bce = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none")
            pt = cls_target * cls_prob + (1 - cls_target) * (1 - cls_prob)
            focal_weight = alpha * (1 - pt) ** gamma
            cls_loss = cls_loss + (focal_weight * bce).mean()

            # Smooth L1 for box regression (only at positive locations)
            if obj_mask.sum() > 0:
                box_loss = box_loss + F.smooth_l1_loss(
                    box_pred * obj_mask.expand_as(box_pred),
                    box_target * obj_mask.expand_as(box_target),
                    reduction="sum",
                ) / obj_mask.sum().clamp(min=1)

        return (cls_loss + box_loss) / num_scales

    def forward(
        self,
        task_loss: torch.Tensor,
        svd_loss: torch.Tensor,
        state_loss: torch.Tensor,
        feat_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.lambda_task * task_loss
            + self.lambda_svd * svd_loss
            + self.lambda_state * state_loss
            + self.lambda_feat * feat_loss
        )
