"""Convert LLVIP annotations from VOC XML to YOLO txt format."""
from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Parse VOC XML and return YOLO-format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        # LLVIP is pedestrian detection — single class (0)
        cls_id = 0

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO (cx, cy, w, h) normalized
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Clamp
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0.001, min(1, w))
        h = max(0.001, min(1, h))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def prepare_llvip(
    raw_dir: str = "/mnt/train-data/datasets/llvip/LLVIP",
    out_dir: str = "/mnt/train-data/datasets/llvip",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    raw = Path(raw_dir)
    out = Path(out_dir)

    ann_dir = raw / "Annotations"
    vis_train = raw / "visible" / "train"
    vis_test = raw / "visible" / "test"
    ir_train = raw / "infrared" / "train"
    ir_test = raw / "infrared" / "test"

    # Default image size for LLVIP
    img_w, img_h = 1280, 1024

    # Get all train image stems
    train_stems = sorted([p.stem for p in vis_train.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    test_stems = sorted([p.stem for p in vis_test.iterdir() if p.suffix.lower() in {".jpg", ".png"}])

    # Split train into train/val
    random.seed(seed)
    random.shuffle(train_stems)
    n_val = int(len(train_stems) * val_ratio)
    val_stems = set(train_stems[:n_val])
    train_stems_final = train_stems[n_val:]

    print(f"Train: {len(train_stems_final)}, Val: {len(val_stems)}, Test: {len(test_stems)}")

    for split, stems, vis_dir, ir_dir_src in [
        ("train", train_stems_final, vis_train, ir_train),
        ("val", list(val_stems), vis_train, ir_train),
        ("test", test_stems, vis_test, ir_test),
    ]:
        split_dir = out / split
        vis_out = split_dir / "visible"
        ir_out = split_dir / "infrared"
        lbl_out = split_dir / "labels"

        vis_out.mkdir(parents=True, exist_ok=True)
        ir_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for stem in stems:
            # Find image
            vis_img = None
            for ext in [".jpg", ".png"]:
                candidate = vis_dir / f"{stem}{ext}"
                if candidate.exists():
                    vis_img = candidate
                    break

            if vis_img is None:
                continue

            ir_img = ir_dir_src / vis_img.name
            if not ir_img.exists():
                continue

            # Symlink images (save disk space)
            vis_link = vis_out / vis_img.name
            ir_link = ir_out / vis_img.name

            if not vis_link.exists():
                os.symlink(str(vis_img.resolve()), str(vis_link))
            if not ir_link.exists():
                os.symlink(str(ir_img.resolve()), str(ir_link))

            # Convert annotations
            xml_path = ann_dir / f"{stem}.xml"
            if xml_path.exists():
                lines = voc_to_yolo(xml_path, img_w, img_h)
                (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
            else:
                # Empty annotation
                (lbl_out / f"{stem}.txt").write_text("")

            count += 1

        print(f"  {split}: {count} images prepared")

    print("LLVIP preparation complete.")
    print(f"  Train: {out / 'train'}")
    print(f"  Val: {out / 'val'}")
    print(f"  Test: {out / 'test'}")


if __name__ == "__main__":
    prepare_llvip()
