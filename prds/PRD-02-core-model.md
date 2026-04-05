# PRD-02: Core Model (Fusion + Low-Rank SS2D + Distillation)

> Module: dlrmamba | Priority: P0  
> Depends on: PRD-01  
> Status: ✅ Completed

## Objective
Implement DLRMamba core architecture blocks aligned with paper equations and loss design.

## Context (from paper)
Paper contribution centers on replacing full-rank SS2D with low-rank factorization and preserving performance through structure-aware distillation.

Paper references:
- Section IV-A: low-rank SS2D, Eq. (5)
- Section IV-B: distillation losses, Eq. (6-8)
- Section IV-C: total objective, Eq. (9)

## Acceptance Criteria
- [x] Pixel-level fusion module implemented
- [x] Low-rank SS2D block implemented with rank ratio config
- [x] Distillation loss functions implemented (SVD/state/feature)
- [x] Detector head outputs class + bbox branches
- [x] Forward-pass tests pass on CPU

## Files
| File | Purpose | Paper Ref |
|---|---|---|
| `src/anima_dlrmamba/models/fusion.py` | pixel-level modality fusion | III-B / Eq. fusion |
| `src/anima_dlrmamba/models/ss2d.py` | low-rank SS2D transition | IV-A Eq. (5) |
| `src/anima_dlrmamba/models/backbone.py` | feature extractor with SS2D blocks | III / IV |
| `src/anima_dlrmamba/models/head.py` | decoupled detection branches | IV-C |
| `src/anima_dlrmamba/models/model.py` | full detector assembly | IV |
| `src/anima_dlrmamba/losses.py` | Eq. (6-9) loss functions | IV-B/C |
| `tests/test_model_forward.py` | shape and forward tests | — |
| `tests/test_low_rank_ss2d.py` | low-rank transition tests | — |
| `tests/test_distillation_losses.py` | distillation loss tests | — |

## Notes
Current implementation is architecture-faithful skeleton; full CUDA kernel optimization is deferred to server phase.
