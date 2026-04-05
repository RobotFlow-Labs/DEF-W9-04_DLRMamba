# DLRMamba Build Plan

## Scope
Build an ANIMA-ready implementation of DLRMamba from paper 2603.06920 with deployable train/infer/serve paths.

## PRD Checklist
| PRD | Title | Priority | Depends On | Status |
|---|---|---|---|---|
| PRD-01 | Foundation & Config | P0 | None | ✅ Completed |
| PRD-02 | Core Model (Fusion + Low-Rank SS2D + Distill) | P0 | PRD-01 | ✅ Completed |
| PRD-03 | Inference Pipeline | P0 | PRD-02 | ✅ Completed |
| PRD-04 | Evaluation Harness | P1 | PRD-03 | ✅ Completed |
| PRD-05 | API & Docker Serving | P1 | PRD-03 | ✅ Completed |
| PRD-06 | ROS2 Integration | P1 | PRD-05 | ✅ Completed |
| PRD-07 | Production Hardening + Export | P2 | PRD-04 | ✅ Completed |

## Current Gate Snapshot
- Gate 0 Session Recovery: PASS (resumed from Mac scaffold)
- Gate 1 Paper Alignment: PASS (paper parsed, hyperparams matched)
- Gate 2 Data Preflight: PASS (LLVIP downloaded + prepared on server)
- Gate 3 Infra Check: PASS (all infra files present)
- Gate 3.5 PRD Generation: PASS (7 PRDs + tasks)
- Phase 4 Build: DONE (all 7 PRDs implemented)
- Gate 5 Pre-train Verify: PENDING (needs GPU allocation)

## Next Phase
- Gate 5: Pre-train verify (batch finder + smoke test on allocated GPU)
- Phase 6: Train (300 epochs on LLVIP, nohup+disown)
- Phase 7: Export + ship (all 5 formats → HF)
