# PRD-07: Production Hardening + Export

> Module: dlrmamba | Priority: P2  
> Depends on: PRD-04  
> Status: ⬜ Not started

## Objective
Harden runtime behavior and provide export toolchain for ONNX/TRT and reproducible release artifacts.

## Context (from paper)
The paper emphasizes edge efficiency; production deployment requires robust failure handling and optimized model artifacts.

## Acceptance Criteria
- [ ] Failure-safe input validation and graceful degraded responses
- [ ] ONNX export script implemented
- [ ] TensorRT FP16/FP32 export hooks prepared for CUDA server
- [ ] Release notes and training report templates generated

## Files
| File | Purpose |
|---|---|
| `scripts/export_onnx.py` | ONNX export |
| `scripts/export_trt.py` | TRT conversion pipeline |
| `reports/TRAINING_REPORT.md` | reproducibility artifact |
| `src/anima_dlrmamba/runtime_guard.py` | hardening utilities |
