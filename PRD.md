# DLRMamba Build Plan

## Scope
Build an ANIMA-ready implementation of DLRMamba from paper 2603.06920 with deployable train/infer/serve paths.

## PRD Checklist
| PRD | Title | Priority | Depends On | Status |
|---|---|---|---|---|
| PRD-01 | Foundation & Config | P0 | None | ✅ Completed |
| PRD-02 | Core Model (Fusion + Low-Rank SS2D + Distill) | P0 | PRD-01 | ✅ Completed |
| PRD-03 | Inference Pipeline | P0 | PRD-02 | ✅ Completed |
| PRD-04 | Evaluation Harness | P1 | PRD-03 | ⬜ Pending |
| PRD-05 | API & Docker Serving | P1 | PRD-03 | ✅ Completed |
| PRD-06 | ROS2 Integration | P1 | PRD-05 | ⬜ Pending |
| PRD-07 | Production Hardening + Export | P2 | PRD-04 | ⬜ Pending |

## Current Gate Snapshot
- Gate 0 Session Recovery: PASS (fresh scaffold with minor pre-existing workspace changes)
- Gate 1 Paper Alignment: PASS (paper parsed and mapped to code)
- Gate 2 Data Preflight: BLOCKED for full train (datasets/checkpoints missing)
- Gate 3 Infra Check: PASS (core infra created)
- Gate 3.5 PRD Generation: PASS (7 PRDs + tasks generated)

## Blockers
- Dataset assets not mounted in local workspace.
- Distillation teacher checkpoint unavailable.
- CUDA optimization phase deferred to server.
