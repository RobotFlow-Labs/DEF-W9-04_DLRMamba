# NEXT_STEPS — DLRMamba
> Last updated: 2026-04-05
> MVP Readiness: 85%

## Done
- [x] PRD-01 through PRD-07: ALL COMPLETE
- [x] Production training loop: checkpointing, cosine warmup, early stopping, bf16 AMP, resume
- [x] LLVIP dataset: 10823 train, 1202 val, 3463 test
- [x] Parallel block-scan SS2D (64-step blocks + conv1d cross-block mixing)
- [x] Fixed bf16 SVD (cast to float32)
- [x] mAP50 evaluation harness
- [x] Export pipeline: pth → safetensors → ONNX → TRT FP16/FP32
- [x] Docker serving + ROS2 node
- [x] 8/8 tests pass, ruff lint clean
- [x] 7 git commits

## In Progress
- [x] Training on GPU 6 — bs=24, 320x320, 300 epochs, LLVIP
  - PID: 26570
  - Log: /mnt/artifacts-datai/logs/dlrmamba/train_20260405_0735.log
  - Epoch 1/300: train_loss=2.03, val_loss=0.39
  - VRAM: 10.3GB/23GB (45%), 100% GPU util
  - ETA: ~50 hours (300 epochs × 10 min/epoch)
  - Checkpoint: /mnt/artifacts-datai/checkpoints/dlrmamba/best.pth

## TODO (after training completes)
- [ ] Export best model: pth → safetensors → ONNX → TRT FP16 → TRT FP32
- [ ] Push to HuggingFace: ilessio-aiflowlab/dlrmamba
- [ ] Git push to origin main
- [ ] Generate TRAINING_REPORT.md
- [ ] Evaluate on test set (mAP50)

## Standing Order
User is sleeping 6h. When training completes:
1. Run full export pipeline
2. Push to git origin main
3. Push to HuggingFace
4. Save CUDA kernels to shared_infra

## Notes
- venv: /mnt/train-data/venvs/dlrmamba (forge-data disk full)
- LLVIP = pedestrian detection, 1 class, paper target 97.5 mAP50
- Student: 2.03M params, Teacher: 2.54M params
- Block-scan: 64-step blocks with depthwise conv1d mixing
