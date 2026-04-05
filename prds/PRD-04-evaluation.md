# PRD-04: Evaluation Harness

> Module: dlrmamba | Priority: P1  
> Depends on: PRD-03  
> Status: ⬜ Not started

## Objective
Build benchmark evaluators for VEDAI, FLIR, LLVIP, M3FD, and DroneVehicle and report mAP50 and speed metrics.

## Context (from paper)
The paper claims performance across five datasets and multiple devices; reproducibility requires a unified evaluator.

Paper references:
- Section V-A/C/D
- Tables I–V

## Acceptance Criteria
- [ ] Dataset-specific eval adapters implemented
- [ ] mAP50 computation matches paper definition (IoU 0.5)
- [ ] Cross-device timing script (A100/4090/RPi5 compatible)
- [ ] Generates markdown report comparing paper vs reproduced metrics

## Files
| File | Purpose |
|---|---|
| `src/anima_dlrmamba/eval.py` | unified evaluation driver |
| `src/anima_dlrmamba/metrics.py` | mAP50 and PR utilities |
| `scripts/benchmark_latency.py` | throughput/latency evaluator |
| `reports/EVAL_REPORT.md` | reproducibility output |
