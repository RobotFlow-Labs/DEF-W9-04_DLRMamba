# PRD-03: Inference Pipeline

> Module: dlrmamba | Priority: P0  
> Depends on: PRD-02  
> Status: ✅ Completed

## Objective
Provide a runnable inference path that loads model checkpoints, runs RGB-IR prediction, and emits structured detections.

## Context (from paper)
The method is evaluated across edge and high-end devices, so a deterministic and lightweight inference path is required.

Paper references:
- Section V-B: cross-platform inference setup
- Table V: latency/FPS emphasis

## Acceptance Criteria
- [x] CLI inference entrypoint implemented
- [x] Supports RGB + IR image inputs
- [x] Produces detection dicts with bbox/score/class
- [x] Basic smoke tests pass

## Files
| File | Purpose |
|---|---|
| `src/anima_dlrmamba/infer.py` | command-line inference entrypoint |
| `src/anima_dlrmamba/models/model.py` | decode predictions for CLI output |
| `tests/test_model_forward.py` | inference-shape sanity checks |

## Notes
Post-processing is intentionally minimal and should be replaced with benchmark-faithful decoding in PRD-04.
