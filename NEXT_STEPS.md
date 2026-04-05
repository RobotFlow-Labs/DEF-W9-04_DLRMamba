# NEXT_STEPS — DLRMamba

## Session Log
- [20:58] Initialized `/anima-autopilot` flow for 04_DLRMamba.
- [21:03] Parsed paper 2603.06920 + arXiv source; confirmed no official author code repo in v1 metadata.
- [21:12] Generated full planning assets: `ASSETS.md`, `PRD.md`, `prds/`, `tasks/`.
- [21:24] Implemented essential local codebase (model, losses, train/infer/serve, configs, tests).
- [21:30] Validation pass complete: `pytest` green, debug train smoke run passes (`--max-steps 3`), infer CLI + service health verified.

## What Is Done
- Core architecture scaffold completed for paper-aligned implementation.
- Distillation objectives implemented (SVD/state/feature).
- Decoupled detector and multi-scale outputs implemented.
- Local unit/smoke tests added.

## What Is Next
1. Provision datasets (VEDAI/FLIR/LLVIP/M3FD/DroneVehicle) and teacher checkpoint.
2. Implement full benchmark evaluators and report generator (PRD-04).
3. Add ROS2 node + launch integration (PRD-06).
4. Run CUDA optimization and long training on `datai_srv7_development`.
5. Export ONNX/TRT fp16/fp32 and finalize production package (PRD-07).

## Blockers
- Missing datasets and teacher checkpoint prevent full training/evaluation.

## Resume Pointer
Resume from **PRD-04 (Evaluation Harness)** after assets are available.
