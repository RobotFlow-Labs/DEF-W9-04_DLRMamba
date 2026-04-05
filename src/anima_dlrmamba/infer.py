from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import load_config
from .models.model import DLRMambaDetector


def _load_rgb(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    return x


def run_inference(config_path: str, rgb_path: str, ir_path: str, checkpoint: str = "") -> list[dict[str, float]]:
    cfg = load_config(config_path)
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
    model.eval()

    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)

    rgb = _load_rgb(rgb_path, cfg.data.image_size)
    ir = _load_rgb(ir_path, cfg.data.image_size)
    sample = torch.stack([rgb, ir], dim=0).unsqueeze(0).to(device)  # [1,2,3,H,W]

    with torch.no_grad():
        out = model(sample)
        preds = model.decode(out, conf_threshold=cfg.infer.conf_threshold, topk=cfg.infer.topk)
    return preds[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="DLRMamba inference")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--rgb", type=str, required=False)
    parser.add_argument("--ir", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    if not args.rgb or not args.ir:
        print("Provide --rgb and --ir image paths.")
        return

    preds = run_inference(args.config, args.rgb, args.ir, args.checkpoint)
    for p in preds[:20]:
        print(p)


if __name__ == "__main__":
    main()
