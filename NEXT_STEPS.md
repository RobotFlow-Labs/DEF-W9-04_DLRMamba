# NEXT_STEPS — DLRMamba
> Last updated: 2026-04-05
> MVP Readiness: 75%

## Done
- [x] PRD-01: Foundation & Config — dataclass config with checkpoint/scheduler/early-stop sections
- [x] PRD-02: Core Model — Low-Rank SS2D, Pixel Fusion, Decoupled Detection Head
- [x] PRD-03: Inference Pipeline — CLI + batch decode
- [x] PRD-04: Evaluation Harness — mAP50 eval, proper focal+smooth_l1 detection loss
- [x] PRD-05: API & Docker Serving — FastAPI + Dockerfile.serve + docker-compose + anima_module.yaml
- [x] PRD-06: ROS2 Integration — ROS2 node skeleton (Detection2DArray publisher)
- [x] PRD-07: Export Pipeline — pth/safetensors/ONNX/TRT FP16/FP32 export script
- [x] Production training loop — checkpointing, cosine warmup scheduler, early stopping, bf16 AMP, gradient clipping, resume support
- [x] LLVIP dataset downloaded + prepared — 10823 train, 1202 val, 3463 test (symlinks)
- [x] All 8 tests pass, ruff lint clean

## In Progress
- [ ] GPU training on LLVIP (pending GPU allocation)

## TODO
- [ ] Ask user for GPU allocation, run /gpu-batch-finder
- [ ] Launch full 300-epoch training with nohup+disown
- [ ] Monitor GPU VRAM (target 60-80% of 23GB)
- [ ] Evaluate best checkpoint on test set (mAP50)
- [ ] Export best model: pth → safetensors → ONNX → TRT FP16 → TRT FP32
- [ ] Push to HuggingFace: ilessio-aiflowlab/dlrmamba
- [ ] Generate TRAINING_REPORT.md
- [ ] Git push to develop

## Blocking
- Need GPU allocation from user before training can start
- venv is on /mnt/train-data/venvs/dlrmamba (forge-data disk is 100% full)

## Downloads Needed
- None — LLVIP dataset is ready at /mnt/train-data/datasets/llvip/

## Notes
- LLVIP is pedestrian detection (1 class) — paper config uses num_classes=1
- Paper reports 97.5 mAP50 on LLVIP — our target >=96.5
- Paper uses SGD with lr=0.01, momentum=0.937, 300 epochs, batch_size=8
- Student model: ~0.5M params (debug), ~8.5M params (paper config with embed_dim=64, num_blocks=4)
