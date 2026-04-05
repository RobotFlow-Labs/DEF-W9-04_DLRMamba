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
        # Teacher matrix SVD alignment.
        u_t, s_t, v_t = torch.linalg.svd(a_t, full_matrices=False)
        r = u_s.shape[1]
        s_root = torch.sqrt(s_t[:r])
        u_ref = u_t[:, :r] * s_root.unsqueeze(0)
        v_ref = v_t[:r, :].T * s_root.unsqueeze(0)
        return F.mse_loss(u_s, u_ref) + F.mse_loss(v_s, v_ref)

    @staticmethod
    def state_alignment_loss(state_s: torch.Tensor, state_t: torch.Tensor, proj: nn.Module | None = None) -> torch.Tensor:
        if proj is not None:
            state_t = proj(state_t)
        return F.mse_loss(state_s, state_t)

    @staticmethod
    def feature_reconstruction_loss(feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(feat_s, feat_t)

    @staticmethod
    def detection_surrogate_loss(cls_logits: list[torch.Tensor], box_deltas: list[torch.Tensor]) -> torch.Tensor:
        # Placeholder task loss for scaffold. Replace with benchmark-faithful detector loss in PRD-04.
        cls_loss = sum(torch.mean(torch.sigmoid(x)) for x in cls_logits)
        box_loss = sum(torch.mean(torch.abs(x)) for x in box_deltas)
        return cls_loss + box_loss

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
