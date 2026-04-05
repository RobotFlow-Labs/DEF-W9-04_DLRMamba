# 04_DLRMamba

## Identity
- Module: `dlrmamba`
- Paper: **DLRMamba: Distilling Low-Rank Mamba for Edge Multispectral Fusion Object Detection**
- ArXiv: `2603.06920`
- Domain: RGB-IR multispectral object detection for edge deployment

## Objective
Implement a production-ready ANIMA module that reproduces the paper's core method:
1. Pixel-level RGB/IR fusion
2. Low-Rank SS2D (matrix-factorized state transition)
3. Structure-Aware Distillation (SVD + state + feature)
4. Decoupled detection head for multi-scale predictions

## Current Scope (Local)
- Build full PRD + tasks + essential runnable code on Mac
- Keep CUDA optimization and large-scale training paths prepared but not executed locally
- Ensure code is migration-ready for server-side CUDA refinement

## Canonical Commands
```bash
# tests
python -m pytest -q

# quick model smoke
python -m anima_dlrmamba.infer --help

# train skeleton
python scripts/train.py --config configs/debug.toml --max-steps 3

# serve
uvicorn anima_dlrmamba.serve:app --host 0.0.0.0 --port 8036
```

## Notes
- No official author repository is linked in arXiv source as of 2026-04-04.
- Reference upstreams used for design grounding:
  - VMamba paper/SS2D mechanism
  - Ultralytics YOLOv8-style decoupled detection head concept
