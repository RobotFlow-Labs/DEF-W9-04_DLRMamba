# DLRMamba (ANIMA Module)

Paper-driven ANIMA module scaffold for:

> DLRMamba: Distilling Low-Rank Mamba for Edge Multispectral Fusion Object Detection (arXiv:2603.06920)

This repository contains:
- Full PRD + task suite (`prds/`, `tasks/`)
- Essential local implementation (`src/anima_dlrmamba/`)
- Train/infer/serve entrypoints
- Test suite for core model and distillation logic

## Quick Start
```bash
python -m pytest -q
python scripts/train.py --config configs/debug.toml --max-steps 3
python -m anima_dlrmamba.infer --help
```
