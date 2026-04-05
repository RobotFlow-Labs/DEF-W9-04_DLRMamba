from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass(slots=True)
class DetectionTarget:
    boxes: torch.Tensor
    labels: torch.Tensor


class RGBIRPairDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Simple paired RGB/IR dataset.

    Expected folder structure under ``root``:
    - rgb/*.jpg|png
    - ir/*.jpg|png
    - labels/*.txt  (YOLO format: cls cx cy w h normalized)
    """

    def __init__(self, root: str | Path, image_size: int = 640) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.rgb_dir = self.root / "rgb"
        self.ir_dir = self.root / "ir"
        self.labels_dir = self.root / "labels"

        if not self.rgb_dir.exists() or not self.ir_dir.exists():
            self.samples: list[Path] = []
            return

        rgb_files = sorted([p for p in self.rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        self.samples = [p for p in rgb_files if (self.ir_dir / p.name).exists()]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = F.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        return x

    def _load_target(self, stem: str) -> DetectionTarget:
        label_path = self.labels_dir / f"{stem}.txt"
        if not label_path.exists():
            return DetectionTarget(boxes=torch.zeros((0, 4), dtype=torch.float32), labels=torch.zeros((0,), dtype=torch.long))

        boxes: list[list[float]] = []
        labels: list[int] = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = parts
            labels.append(int(float(cls_id)))
            boxes.append([float(cx), float(cy), float(w), float(h)])

        if not boxes:
            return DetectionTarget(boxes=torch.zeros((0, 4), dtype=torch.float32), labels=torch.zeros((0,), dtype=torch.long))

        return DetectionTarget(boxes=torch.tensor(boxes, dtype=torch.float32), labels=torch.tensor(labels, dtype=torch.long))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rgb_path = self.samples[idx]
        ir_path = self.ir_dir / rgb_path.name

        rgb = self._load_image(rgb_path)
        ir = self._load_image(ir_path)

        target = self._load_target(rgb_path.stem)
        # Keep modalities separate for fusion module.
        sample = torch.stack([rgb, ir], dim=0)  # [2, 3, H, W]

        return sample, {"boxes": target.boxes, "labels": target.labels}


class RandomRGBIRDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Synthetic fallback used for smoke tests and debug runs."""

    def __init__(self, image_size: int = 224, num_classes: int = 8, length: int = 32) -> None:
        self.image_size = image_size
        self.num_classes = num_classes
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rgb = torch.rand(3, self.image_size, self.image_size)
        ir = torch.rand(3, self.image_size, self.image_size)
        sample = torch.stack([rgb, ir], dim=0)

        n = torch.randint(1, 4, (1,)).item()
        boxes = torch.rand(n, 4)
        labels = torch.randint(0, self.num_classes, (n,))

        return sample, {"boxes": boxes, "labels": labels}


def collate_detection(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    x = torch.stack([b[0] for b in batch], dim=0)
    y = [b[1] for b in batch]
    return x, y
