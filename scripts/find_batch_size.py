from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anima_dlrmamba.models.model import DLRMambaDetector


def estimate_batch_size(image_size: int, target: float = 0.75) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DLRMambaDetector().to(device)
    model.eval()

    if device.type != "cuda":
        return 2

    total_mem = torch.cuda.get_device_properties(device).total_memory
    low, high = 1, 64
    best = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()
            sample = torch.randn(mid, 2, 3, image_size, image_size, device=device)
            with torch.no_grad():
                _ = model(sample)
            used = torch.cuda.max_memory_allocated(device)
            if used / total_mem <= target:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        except RuntimeError:
            high = mid - 1

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--target", type=float, default=0.75)
    args = parser.parse_args()

    bs = estimate_batch_size(args.image_size, args.target)
    print(bs)
