# DLRMamba — Asset Manifest

## Paper
- Title: DLRMamba: Distilling Low-Rank Mamba for Edge Multispectral Fusion Object Detection
- ArXiv: 2603.06920
- Authors: Qianqian Zhang, Leon Tabaro, Ahmed M. Abdelmoniem, Junshe An
- Submitted: 2026-03-06

## Status: ALMOST
Local code/PRD scaffolding is ready. Dataset and checkpoint provisioning still required before full training runs.

## Reference Code Repositories
- Official author repo: **NOT PROVIDED** in arXiv v1 source (checked 2026-04-04).
- Related upstreams:
  - VMamba (SS2D design reference): https://arxiv.org/abs/2401.10166
  - Ultralytics (YOLOv8 head style baseline): https://github.com/ultralytics/ultralytics

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|---|---:|---|---|---|
| VMamba backbone init | TBD | VMamba ecosystem | /mnt/forge-data/models/vmamba/ | MISSING |
| YOLOv8n init | ~6 MB | Ultralytics | /mnt/forge-data/models/yolov8n.pt | MISSING |
| Distillation teacher checkpoint (full-rank SS2D) | TBD | internal training artifact | /mnt/artifacts-datai/checkpoints/dlrmamba/teacher_latest.pth | MISSING |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---|---:|---|---|---|---|
| VEDAI | TBD | train/val/test | original benchmark | /mnt/forge-data/datasets/vedai/ | MISSING |
| FLIR ADAS | TBD | train/val | official FLIR split | /mnt/forge-data/datasets/flir_adas/ | MISSING |
| LLVIP | TBD | train/test | official benchmark | /mnt/forge-data/datasets/llvip/ | MISSING |
| M3FD | TBD | train/test | official benchmark | /mnt/forge-data/datasets/m3fd/ | MISSING |
| DroneVehicle | TBD | train/test | official benchmark | /mnt/forge-data/datasets/dronevehicle/ | MISSING |

## Hyperparameters (from paper)
| Param | Value | Paper Section |
|---|---|---|
| optimizer | SGD | V-B |
| learning_rate | 0.01 | V-B |
| momentum | 0.937 | V-B |
| weight_decay | 0.0005 | V-B |
| batch_size | 8 | V-B |
| epochs | 300 | V-B |
| distill_lambda_1 | 1.0 | IV-C Eq. (9) |
| distill_lambda_2 | 0.5 | IV-C Eq. (9) |
| distill_lambda_3 | 0.1 | IV-C Eq. (9) |
| distill_lambda_4 | 1.5 | IV-C Eq. (9) |

## Expected Metrics (from paper)
| Benchmark | Metric | Paper Value | Our Target |
|---|---|---:|---:|
| VEDAI | mAP50 | 84.7 | >=84.0 |
| FLIR | mAP50 | 80.0 | >=79.0 |
| LLVIP | mAP50 | 97.5 | >=96.5 |
| M3FD | mAP50 | 76.6 | >=75.5 |
| DroneVehicle | mAP50 | 76.5 | >=75.0 |
| Raspberry Pi 5 | FPS | 2.3 | >=2.0 |
| RTX 4090 | FPS | 29.0 | >=27.0 |

## Hardware Baseline (paper)
- Training: single NVIDIA RTX A100 80GB
- Inference platforms: NVIDIA A100, RTX 4090, Raspberry Pi 5

## Gaps Blocking Full Lifecycle
1. Required datasets are not provisioned in this local module.
2. Teacher checkpoint for structure-aware distillation is missing.
3. CUDA-optimized SS2D kernels are not yet integrated in this local version.
