"""Export pipeline: pth → safetensors → ONNX → TRT FP16 → TRT FP32."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from .config import load_config
from .models.model import DLRMambaDetector


def load_model(config_path: str, checkpoint_path: str, device: str = "cpu") -> DLRMambaDetector:
    cfg = load_config(config_path)
    model = DLRMambaDetector(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        fusion_channels=cfg.model.fusion_channels,
        embed_dim=cfg.model.embed_dim,
        num_blocks=cfg.model.num_blocks,
        state_dim=cfg.model.state_dim,
        rank_ratio=cfg.model.rank_ratio,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def export_safetensors(model: DLRMambaDetector, output_path: Path) -> Path:
    from safetensors.torch import save_file

    dst = output_path / "model.safetensors"
    save_file(model.state_dict(), str(dst))
    print(f"[EXPORT] safetensors → {dst} ({dst.stat().st_size / 1e6:.1f}MB)")
    return dst


def export_onnx(
    model: DLRMambaDetector, output_path: Path, image_size: int = 640, opset: int = 18
) -> Path:
    dst = output_path / "model.onnx"
    dummy = torch.randn(1, 2, 3, image_size, image_size)

    # Wrap model to only return cls+box tensors for ONNX compatibility
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            cls_cat = torch.cat([c.flatten(2) for c in out.cls_logits], dim=2)
            box_cat = torch.cat([b.flatten(2) for b in out.box_deltas], dim=2)
            return cls_cat, box_cat

    wrapper = OnnxWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy,
        str(dst),
        input_names=["input"],
        output_names=["cls_logits", "box_deltas"],
        opset_version=opset,
        dynamo=False,
    )
    print(f"[EXPORT] ONNX → {dst} ({dst.stat().st_size / 1e6:.1f}MB)")
    return dst


def export_trt(onnx_path: Path, output_path: Path, precision: str = "fp16") -> Path | None:
    """Export ONNX to TensorRT using trtexec or shared toolkit."""
    suffix = f"_trt_{precision}.engine"
    dst = output_path / f"model{suffix}"

    # Try shared TRT toolkit first
    trt_script = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if trt_script.exists():
        cmd = [
            "python", str(trt_script),
            "--onnx", str(onnx_path),
            "--output", str(dst),
            "--precision", precision,
        ]
        print(f"[EXPORT] TRT {precision} via shared toolkit...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[EXPORT] TRT {precision} → {dst}")
            return dst
        print(f"[WARN] TRT toolkit failed: {result.stderr[:200]}")

    # Fallback to trtexec
    trtexec = "trtexec"
    flag = "--fp16" if precision == "fp16" else ""
    cmd = f"{trtexec} --onnx={onnx_path} --saveEngine={dst} {flag} --workspace=4096"
    print(f"[EXPORT] TRT {precision} via trtexec...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[EXPORT] TRT {precision} → {dst}")
        return dst

    print(f"[WARN] TRT {precision} export failed. Install TensorRT or use shared toolkit.")
    return None


def export_all(config_path: str, checkpoint_path: str, output_dir: str, image_size: int = 640):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(config_path, checkpoint_path)

    # 1. Save pth (copy)
    pth_dst = out / "model.pth"
    torch.save(model.state_dict(), pth_dst)
    print(f"[EXPORT] pth → {pth_dst} ({pth_dst.stat().st_size / 1e6:.1f}MB)")

    # 2. safetensors
    export_safetensors(model, out)

    # 3. ONNX
    onnx_path = export_onnx(model, out, image_size=image_size)

    # 4. TRT FP16
    export_trt(onnx_path, out, precision="fp16")

    # 5. TRT FP32
    export_trt(onnx_path, out, precision="fp32")

    print(f"[EXPORT] All formats exported to {out}")


def main():
    parser = argparse.ArgumentParser(description="DLRMamba export pipeline")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/mnt/artifacts-datai/exports/dlrmamba")
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    export_all(args.config, args.checkpoint, args.output_dir, args.image_size)


if __name__ == "__main__":
    main()
