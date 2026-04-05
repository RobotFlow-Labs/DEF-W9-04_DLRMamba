# PRD-05: API & Docker Serving

> Module: dlrmamba | Priority: P1  
> Depends on: PRD-03  
> Status: ✅ Completed

## Objective
Expose model through service endpoints and container artifacts compatible with ANIMA serving workflows.

## Context (from paper)
The method targets edge deployment; API-level serving and containerization are prerequisites for operationalization.

## Acceptance Criteria
- [x] `/health` and `/ready` endpoints implemented
- [x] `/predict` endpoint accepts RGB+IR files
- [x] Dockerfile and compose template created
- [x] Local uvicorn boot path documented

## Files
| File | Purpose |
|---|---|
| `src/anima_dlrmamba/serve.py` | FastAPI server |
| `Dockerfile.serve` | container definition |
| `docker-compose.serve.yml` | local orchestration |
| `.env.serve` | runtime env template |
