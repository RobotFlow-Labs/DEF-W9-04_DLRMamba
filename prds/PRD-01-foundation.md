# PRD-01: Foundation & Config

> Module: dlrmamba | Priority: P0  
> Depends on: None  
> Status: ✅ Completed

## Objective
Create a reproducible module foundation with config loading, package layout, dataset interfaces, and project guardrails.

## Context (from paper)
The paper requires multi-dataset RGB-IR training/evaluation with fixed training hyperparameters and edge-focused deployment constraints.

Paper references:
- Section V-A: datasets (VEDAI, FLIR, LLVIP, M3FD, DroneVehicle)
- Section V-B: optimizer and training recipe

## Acceptance Criteria
- [x] `pyproject.toml` exists and package installs locally
- [x] `configs/default.toml`, `configs/debug.toml`, `configs/paper.toml` created
- [x] Dataset interface for paired RGB-IR data implemented
- [x] Unit tests for config loading and dataset-safe behavior

## Files
| File | Purpose | Paper Ref |
|---|---|---|
| `pyproject.toml` | packaging + dependencies | V-B |
| `src/anima_dlrmamba/config.py` | typed config model | V-B |
| `src/anima_dlrmamba/data.py` | paired RGB-IR dataset interface | V-A |
| `configs/*.toml` | paper/debug/default configs | V-B |
| `tests/test_config.py` | config validation tests | — |

## Notes
This PRD intentionally avoids dataset downloading and treats data paths as externally provisioned assets from `ASSETS.md`.
