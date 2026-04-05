from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .backbone import DLRMambaBackbone
from .fusion import PixelFusion
from .head import DecoupledDetectionHead


@dataclass(slots=True)
class ModelOutput:
    pyramids: list[torch.Tensor]
    states: list[torch.Tensor]
    cls_logits: list[torch.Tensor]
    box_deltas: list[torch.Tensor]


class DLRMambaDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        in_channels: int = 3,
        fusion_channels: int = 32,
        embed_dim: int = 64,
        num_blocks: int = 3,
        state_dim: int = 64,
        rank_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fusion = PixelFusion(in_channels=in_channels, out_channels=fusion_channels)
        self.backbone = DLRMambaBackbone(
            in_channels=fusion_channels,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            state_dim=state_dim,
            rank_ratio=rank_ratio,
        )
        self.head = DecoupledDetectionHead([embed_dim, embed_dim * 2, embed_dim * 4], num_classes)

    def forward(self, sample: torch.Tensor) -> ModelOutput:
        # sample: [B, 2, 3, H, W]
        rgb, ir = sample[:, 0], sample[:, 1]
        fused = self.fusion(rgb, ir)
        pyramids, states = self.backbone(fused)
        pred = self.head(pyramids)
        return ModelOutput(
            pyramids=pyramids,
            states=states,
            cls_logits=pred["cls"],
            box_deltas=pred["box"],
        )

    @torch.no_grad()
    def decode(
        self,
        output: ModelOutput,
        conf_threshold: float = 0.25,
        topk: int = 300,
    ) -> list[list[dict[str, float]]]:
        # Lightweight decode for local smoke tests.
        batch_preds: list[list[dict[str, float]]] = []
        B = output.cls_logits[0].shape[0]

        for b in range(B):
            preds: list[dict[str, float]] = []
            for cls_map, box_map in zip(output.cls_logits, output.box_deltas):
                c = torch.sigmoid(cls_map[b])  # [K,H,W]
                max_scores, cls_ids = torch.max(c, dim=0)
                mask = max_scores > conf_threshold
                ys, xs = torch.where(mask)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    score = float(max_scores[y, x].item())
                    cls_id = int(cls_ids[y, x].item())
                    box = box_map[b, :, y, x].detach().cpu().tolist()
                    preds.append(
                        {
                            "score": score,
                            "class_id": cls_id,
                            "bx": float(box[0]),
                            "by": float(box[1]),
                            "bw": float(box[2]),
                            "bh": float(box[3]),
                        }
                    )

            preds = sorted(preds, key=lambda p: p["score"], reverse=True)[:topk]
            batch_preds.append(preds)

        return batch_preds
