# DLRMamba Task Index

## Build Order
| Task | Title | Depends | Status |
|---|---|---|---|
| PRD-0101 | Project packaging and config schema | None | ✅ Done |
| PRD-0102 | Dataset interface for RGB-IR pairs | PRD-0101 | ✅ Done |
| PRD-0103 | Baseline tests for config/data | PRD-0101 | ✅ Done |
| PRD-0201 | Pixel-level fusion module | PRD-0102 | ✅ Done |
| PRD-0202 | Low-rank SS2D transition block | PRD-0201 | ✅ Done |
| PRD-0203 | Distillation losses (SVD/state/feature) | PRD-0202 | ✅ Done |
| PRD-0204 | Backbone + decoupled detection head | PRD-0202 | ✅ Done |
| PRD-0205 | Full model assembly and smoke test | PRD-0204 | ✅ Done |
| PRD-0301 | Inference CLI with RGB/IR inputs | PRD-0205 | ✅ Done |
| PRD-0302 | Prediction decoding and output schema | PRD-0301 | ✅ Done |
| PRD-0401 | mAP50 metric implementation | PRD-0302 | ⬜ Pending |
| PRD-0402 | Multi-dataset eval adapters | PRD-0401 | ⬜ Pending |
| PRD-0403 | Latency benchmarking scripts | PRD-0402 | ⬜ Pending |
| PRD-0501 | FastAPI health/ready endpoints | PRD-0302 | ✅ Done |
| PRD-0502 | Predict endpoint + container files | PRD-0501 | ✅ Done |
| PRD-0601 | ROS2 inference node | PRD-0502 | ⬜ Pending |
| PRD-0602 | ROS2 launch + param wiring | PRD-0601 | ⬜ Pending |
| PRD-0701 | ONNX export + runtime guards | PRD-0403 | ⬜ Pending |
| PRD-0702 | TRT fp16/fp32 export pipeline | PRD-0701 | ⬜ Pending |
