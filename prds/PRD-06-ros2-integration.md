# PRD-06: ROS2 Integration

> Module: dlrmamba | Priority: P1  
> Depends on: PRD-05  
> Status: ⬜ Not started

## Objective
Integrate module into ROS2 runtime with topic I/O and launch orchestration for ANIMA robotics pipelines.

## Context (from paper)
Primary use-cases include surveillance and remote sensing deployment contexts where message-driven integration is needed.

## Acceptance Criteria
- [ ] ROS2 node class with configurable input topics
- [ ] Converts ROS image messages into model-ready tensors
- [ ] Publishes detection outputs as ROS2 detection messages
- [ ] Launch file and parameter file included

## Files
| File | Purpose |
|---|---|
| `src/anima_dlrmamba/ros2/node.py` | inference ROS2 node |
| `launch/dlrmamba.launch.py` | launch entrypoint |
| `configs/ros2.toml` | ROS runtime config |
