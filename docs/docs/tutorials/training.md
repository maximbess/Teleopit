---
sidebar_position: 5
---

# Training

Train either the whole-body tracking policy or the independent RL-only ladder policy. The tracking policy can be exported as ONNX for inference.

:::info
For data preparation, see [Dataset Reference](../reference/dataset). For common training issues, see [Training Troubleshooting](../reference/training-troubleshooting).
:::

## Setup

```bash
conda create -n teleopit python=3.10
conda activate teleopit
pip install -e '.[train]'
```

Verify:
```bash
python -c "import train_mimic.tasks; print('training OK')"
```

The motion dataset is required only for `General-Tracking-G1`. Download the
distributed minimal datasets and generate the combined precomputed training
dataset:

```bash
python scripts/setup/download_assets.py --only robots data
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets --outdir data/datasets_precomputed --jobs 8
```

The RL-only ladder task needs the `robots` asset group but no dataset.

## Training

### Smoke Test

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets_precomputed
```

### Full Training

```bash
python train_mimic/scripts/train.py \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

### Multi-GPU

```bash
python train_mimic/scripts/train.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

### Multi-Node Multi-GPU

Use `torchrun` directly when training across multiple machines:

```bash
torchrun \
    --nnodes=$PET_NNODES \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$PET_MASTER_ADDR \
    --master_port=$PET_MASTER_PORT \
    train_mimic/scripts/train.py \
    --num_envs 1024 \
    --max_iterations 1000 \
    --motion_file data/datasets_precomputed
```

**Notes:**
- `--num_envs` is per-GPU in multi-GPU mode
- `--num_envs` is also per-process in multi-node mode, so total environments scale with `world_size`
- Default logger is TensorBoard. Use `--logger wandb` or `--logger swanlab` to select W&B or SwanLab; the project name defaults to `experiment_name`
- `--motion_file` accepts a precomputed training dataset root directory or a single precomputed `.h5` shard; shard discovery is recursive
- If you only have the minimal distributed shards, first run `python train_mimic/scripts/data/precompute_dataset.py <minimal_dataset> --outdir <precomputed_dataset>` and pass the precomputed output to training.
- Training loads all discovered precomputed motion windows into memory at startup.
- `--max_iterations` means additional iterations; resuming from `model_12000.pt` with `--max_iterations 18000` trains to `model_30000.pt`

## RL-Only Ladder Training

Use the dedicated ladder entry point when the objective is to learn A-frame
ladder climbing directly from reinforcement learning:

```bash
python train_mimic/scripts/train_ladder.py \
    --num_envs 64 \
    --max_iterations 100
```

For a full run:

```bash
python train_mimic/scripts/train_ladder.py \
    --num_envs 4096 \
    --max_iterations 60000
```

The ladder entry point intentionally has no `--motion_file`, sampling-mode, or
rewind options. It uses a TemporalCNN policy with separate Conv1d encoders for
state history and ladder geometry. The 24D ladder command produces the 117D
current actor group and 120D privileged-critic group; each side also receives a
10-frame history and a `9 x 7` torso-frame tensor containing the finite rung
endpoints and validity bits. Hand and foot target shaping uses signed closing progress:
approaching is positive, hovering is zero, and retreating is negative. This
prevents the policy from collecting a persistent proximity reward by holding a
foot near a rung without making contact. The simulated rubber-hand welds are
part of the environment mechanics; feet use ordinary physical contacts and the
policy action remains the 29 G1 joint targets.

The command uses an explicit five-state sequence and repeats it for every rung:

1. stabilize both feet and both attached hands;
2. move the first hand;
3. move the second hand;
4. move the first foot with both hands attached;
5. move the second foot, then return to stabilization.

The policy sees this state as a five-value one-hot inside the existing 24D
command. Hand targets require three consecutive close, slow frames; foot
targets require five consecutive frames with proximity, low speed, and a real
contact-sensor hit. The adaptive curriculum records every terminal episode as
a success only when it reaches the boundary after the currently unlocked
phase; falls, low-root termination, and ordinary episode timeout are failures.
The next phase opens only when both conditions hold:

- the rolling window contains 100 episodes and its success rate is strictly
  greater than 80% (at least 81 successes);
- the current phase has accumulated at least 120,000 environment steps.

With `num_steps_per_env=24`, the minimum budget is 5,000 PPO iterations per
transition. If every phase passes immediately when eligible, the earliest
schedule is:

| Phase available | Earliest environment step | Earliest PPO iteration |
|---|---:|---:|
| Stabilize | 0 | 0 |
| First hand | 120,000 | 5,000 |
| Second hand | 240,000 | 10,000 |
| First foot | 360,000 | 15,000 |
| Second foot/full cycle | 480,000 | 20,000 |

If success is 80% or lower, the phase remains locked regardless of elapsed
steps. Reaching a locked boundary truncates the short prefix episode so PPO can
start another attempt without waiting for the normal timeout. Reward weights
switch on the matching long-run scale:

| Environment step | Hand/foot progress | Torso ascent/stability | Hand advance | Foot advance | Stable/cycle/finish |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 / 8 | 12 / 10 | 25 | 50 | 30 / 80 / 100 |
| 240,000 | 6 / 12 | 16 / 8 | 35 | 70 | 25 / 120 / 180 |
| 480,000 | 8 / 16 | 20 / 6 | 45 | 90 | 20 / 160 / 260 |

Torso ascent is active only in a foot phase with both hands attached. A dense
torso-speed reward, stronger upright reward, stronger action-rate and global
joint-velocity penalties, joint-acceleration penalty, and an additional
support-joint velocity penalty stabilize the trunk and the limbs that must stay
fixed. Foot-contact and foot-transition weights are deliberately larger than
the hand equivalents. Transition and completion rewards remain dt-independent
one-step impulses. The ladder runner merges outcome windows across all GPUs and
saves the unlocked phase, phase-start step, and recent outcomes in checkpoints.
Resuming a fixed-schedule checkpoint fails explicitly because it has no
adaptive curriculum state. Playback and benchmarking unlock every phase
immediately and use final-stage weights.

The long-run ladder PPO preset uses 60,000 iterations, saves every 1,000
iterations, starts Gaussian exploration at standard deviation 0.7, and uses a
`5e-4` learning rate with `0.005` entropy coefficient. Ladder training does not
randomize initial episode lengths, because artificial early timeouts would
pollute the last-100 curriculum success window.

The ladder geometry and grip sites are generated on top of the canonical
`assets/robots/unitree_g1/g1_29dof.xml`. Each ladder face uses fixed collidable
side rails and separate flat-topped box rungs with stiff contacts. The robot
cannot pass through these bars and the flat rung tops support the feet. A thin
invisible blocker is offset behind each face. Its separate collision mask stops
the pelvis, torso, and head from entering the A-frame while allowing hands and
feet to reach the exposed bars, so the gaps remain visually open. The ladder
collision configuration explicitly re-enables all generated rails and rungs
after G1's default collision editor. The five-phase FSM commands only one moving
limb at a time. Both hands are attached throughout both foot phases, and the
non-moving foot remains a physical support.

Every episode starts directly on the ladder. Both feet physically contact rung
2, both hands initially attach at rung 5, and a stabilization phase requires
both foot contacts plus low torso and joint speeds before the first hand is
released. The same stabilization check repeats after every complete hand/foot
cycle. This removes the ground-to-ladder approach from the learning problem.

The ladder builder removes the embedded XML `floor` and keeps one scene-owned
ground plane. The plane uses a solid, non-reflective material instead of the
default repeated checker texture. Ladder playback also uses one controlled
light with shadows and reflections disabled to avoid floor moiré and shadow-map
artifacts. Checkpoints are saved under `logs/rsl_rl/g1_ladder_rl/` and are not
exported with `save_onnx.py`.

The multi-group TemporalCNN contract is incompatible with earlier 105D/108D and
flat 117D/120D MLP ladder checkpoints. The current-frame dimensions stay
117D/120D, but the model now requires history and ladder-geometry inputs. The
phase one-hot semantics, ordered FSM, climbing keyframe, scheduled rewards,
active bar contacts, and trunk-blocking collision geometry also require a new
training run.

Multi-GPU launch uses the same per-GPU environment convention:

```bash
python train_mimic/scripts/train_ladder.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 60000
```

:::warning Warp version for multi-GPU
The supported `mjlab==1.4.0` / MuJoCo Warp 3.8 training stack requires
`warp-lang==1.15.0`. Warp 1.16.0 fails during sensor-kernel compilation with
`Referencing undefined symbol: xmat`; the simultaneous cold start on four GPUs
makes the failure appear to be a distributed-launch problem, but it happens
before PPO or NCCL initialization. Repair an existing environment with:

```bash
python -m pip install --force-reinstall "warp-lang==1.15.0"
python -c "import warp as wp; print(wp.__version__)"
```

The training entry points now validate this version before creating a CUDA
environment and print the same repair command when it is incompatible.
:::

## Export ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output track.onnx \
    --history_length 10
```

The exported model is a dual-input ONNX (`obs` + `obs_history`). The inference side expects a 167D dual-input ONNX policy matching the current `velcmd_history` observation.

## Evaluation

### Playback

```bash
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed
```

### Benchmark

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed \
    --num_envs 1
```

### Benchmark with Video

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed \
    --num_envs 1 \
    --video \
    --video_length 600
```

### Ladder Benchmark

Use the dedicated RL-only benchmark to evaluate a ladder checkpoint without
performing another PPO update or loading a motion dataset:

```bash
python train_mimic/scripts/benchmark_ladder.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --num_envs 64 \
    --num_eval_steps 5000
```

It writes text and JSON reports under `benchmark_results/` with success,
failure, timeout, rung-progress, grip, reach-distance, reward, and
episode-length statistics. A timeout is a completed unsuccessful episode, so
the default 20-second episode limit prevents stalled policies from disappearing
from the success rate.

Video uses one environment and is saved under `benchmark_results/videos/`:

```bash
python train_mimic/scripts/benchmark_ladder.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --num_envs 1 \
    --video \
    --video_length 1000
```

### Ladder Policy Video Only

Use the dedicated recorder when only an MP4 is needed. It does not calculate
benchmark metrics or create text/JSON reports. Recording uses one environment
and stops at the first episode termination or the frame limit, before an
automatic reset can appear in the video:

```bash
python train_mimic/scripts/record_ladder_video.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --output ladder.mp4 \
    --frames 1000
```

The recorder uses a fixed world-space overview from the robot's outside face
of the ladder (`azimuth=30`, `elevation=-5`, `distance=4.5`) and aims at the
middle of the climbing face. Unlike torso tracking, this keeps the complete
robot and ladder in frame and prevents the opposite ladder face from blocking
the policy motion. Use `--camera_azimuth`, `--camera_elevation`,
`--camera_distance`, `--camera_lookat X Y Z`, and `--camera_fovy` to override
the default composition.

## Training Architecture

```text
train_mimic/scripts
    -> train_mimic/app.py
    -> tracking and RL-only ladder task configs
    -> mjlab + rsl_rl
```

Key files:
- `train_mimic/app.py` - Shared entry point for train/play/benchmark
- `train_mimic/tasks/tracking/config/env.py` - General-Tracking-G1 and generated ladder env builders
- `train_mimic/tasks/tracking/config/rl.py` - TemporalCNN tracking and ladder PPO configs
- `train_mimic/tasks/tracking/mdp/ladder.py` - Ladder command, grip state, rewards, and success termination
- `train_mimic/tasks/tracking/mdp/commands.py` - Supports `uniform`, `start`, and `rewind` sampling modes. Training defaults to `rewind`; playback/benchmark use `start`.
