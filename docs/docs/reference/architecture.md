---
sidebar_position: 1
---

# Architecture

System internals and technical constraints for developers.

## Pipeline

```text
InputProvider (BVH file / Pico4)
    -> Retargeter (GMR)
    -> ObservationBuilder (167D)
    -> Controller (dual-input TemporalCNN ONNX)
    -> Robot (MuJoCo sim or Unitree G1)
```

Offline/online inference is assembled by `teleopit/runtime/` and `teleopit/pipeline.py`. The hardware state machine runs through the process-isolated runtime in `teleopit/sim2real/mp/`. Training is provided by `train_mimic/`.

## Code Structure

```text
configs / scripts
    -> runtime
    -> interfaces + pipeline state machines
    -> adapters (inputs / retargeting / controller / robot / recording)

train_mimic/scripts
    -> train_mimic/app.py
    -> tracking and RL-only ladder task configs
    -> mjlab / rsl_rl

train_mimic/scripts/data
    -> train_mimic/data/dataset_builder.py
    -> dataset_lib / motion_fk / convert_pkl_to_npz
```

## Core Boundaries

| Module | Role |
|--------|------|
| `teleopit/interfaces.py` | Stable protocols: InputProvider, Retargeter, Controller, Robot, ObservationBuilder |
| `teleopit/runtime/` | Config parsing, path normalization, component assembly, CLI validation |
| `teleopit/pipeline.py` | Lightweight facade for offline sim |
| `teleopit/sim2real/mp/` | Process-isolated sim2real state machine, IPC, and robot-control loop |
| `teleopit/controllers/observation.py` | ObservationBuilder |
| `teleopit/controllers/rl_policy.py` | Accepts dual-input ONNX whose observation dimension matches the runtime builder |
| `train_mimic/app.py` | Shared train/play/benchmark assembly |
| `train_mimic/tasks/tracking/config/` | Tracking and RL-only ladder task registration |
| `train_mimic/data/dataset_builder.py` | Sole official dataset construction entry |

## Technical Specifications

| Spec | Value |
|------|-------|
| Training tasks | `General-Tracking-G1`, `G1-Ladder-Climb-RL` |
| Inference observation | `velcmd_history` (167D) |
| ONNX signature | Dual-input `obs` (167D) + `obs_history` |
| Tracking Actor/Critic | TemporalCNN (2048, 1024, 512, 256, 128) |
| Ladder Actor/Critic | TemporalCNN + separate history/geometry Conv1d encoders + MLP (2048, 1024, 512, 256, 128), no motion dataset |
| Ladder observations | Current actor 117D / clean critic 120D; critic-only current 14D reward/FSM state; 10-frame histories; `9 x 15` torso-frame target/support-aware rung tokens with continuous hand-support markers; 24D ordered phase command with torso-frame target vectors and active-hand grip strength |
| Ladder hand release | Internal feedback-controlled `PRE_RELEASE`: 8-step load-transfer hold, reversible 20-step per-environment weld-softening ramp, then 5-step stable soft hold before detach |
| Ladder action | 29D G1 joint-position targets |
| Ladder curriculum | `LadderOnPolicyRunner`; synchronized last-100 success window and checkpointed adaptive phase state |
| Training sampling | Default `rewind`; also supports `uniform`; playback/benchmark use `start` |
| Training `window_steps` | `[0]` |
| Data format | Minimal recursive HDF5 shards (`shard_*.h5`) |

## Constraints

- `controller.policy_path` must be explicitly provided and the file must exist
- Offline BVH runs require explicit `input.bvh_file`
- `viewers` is the sole viewer configuration entry
- Observation/ONNX dimension mismatch causes immediate startup error
- sim2real also requires a dual-input ONNX whose observation dimension matches the runtime builder

## Public Surface

**Stable run modes:** offline sim2sim, offline sim2real playback, Pico4 sim2sim, G1 sim2real

**Stable training entry points:** `train.py`, `train_ladder.py`, `play.py`, `benchmark.py`, `benchmark_ladder.py`, `save_onnx.py`

**Stable data entry points:** `build_dataset.py`, `precompute_dataset.py`
