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
current actor group and a 120D clean critic group; each side also receives a
10-frame history and a `9 x 15` torso-frame rung-token tensor. The critic alone
receives a separate current-only 14D privileged vector. It exposes whole-body
height, episode height record and record gap, cycle/phase ascent, torso speed,
torso-orientation and support-offset errors, joint-speed RMS, two physical foot
supports, phase-required support validity, release-ramp progress, and normalized
phase dwell. Keeping this vector out of `critic_history` avoids needlessly
expanding the temporal encoder. Every rung token
contains finite endpoints, a validity bit, and separate left/right hand and
foot target/support markers. The hand and foot target vectors in the 24D
command use the same torso frame. Its former post-reset initialization scalar
now reports continuous active-hand grip strength, and the corresponding held-rung
marker fades by the same value. Task shaping uses one phase-conditioned
potential. It combines normalized whole-body height, defined as the mean of
pelvis and torso COM height, with normalized distance to the active limb target.
Positive progress is paid only while the phase-required supports and torso pose
remain valid; negative progress is always retained. The simulated rubber-hand welds are
part of the environment mechanics; feet use ordinary physical contacts and the
policy action remains the 29 G1 joint targets.

The ladder-only MuJoCo articulation scales the effort limits of every shoulder,
elbow, and wrist actuator to 70% of the standard G1 values: the 25 Nm arm group
is limited to 17.5 Nm and the 5 Nm wrist-pitch/yaw group to 3.5 Nm. This torque
saturation is applied only by the ladder robot builder; tracking and inference
retain the standard G1 limits.

The command uses an explicit five-state sequence and repeats it for every rung:

1. stabilize both feet and both attached hands;
2. move the first hand;
3. move the second hand;
4. move the first foot with both hands attached;
5. move the second foot, then return to stabilization.

Each hand movement phase starts with an internal `PRE_RELEASE` substage; it does
not add a sixth one-hot state. Both welds stay active for 8 stable preload-transfer
steps. The selected environment's weld then follows a 20-step smoothstep ramp
from its compiled equality parameters to `solref.timeconst=0.18` and
`solimp dmin=dmax=0.05`. The ramp advances only while both feet and the other
hand support the robot and the torso speed, torso-orientation error, and
support-relative COM error remain within `0.12 m/s`, `0.25 rad`, and `0.12 m`.
A violation reverses the ramp by two steps. Joint motion remains penalized but
cannot veto release. Binary detach occurs only after five more stable frames
at minimum grip strength. Remaining in `PRE_RELEASE` for 300 policy steps
terminates the episode as a failure. It receives the same `-50` terminal reward
impulse as any other unsuccessful ending, so deliberately falling cannot avoid
a larger stall-specific cost.
MJLab expands `eq_solref` and `eq_solimp` per environment so parallel releases
remain independent.

Episode diagnostics log the hand-attachment, two-foot-support, torso-speed,
torso-orientation, and support-offset gates independently, plus their combined
validity and normalized preload, softening-ramp, and final-dwell progress. This
makes a stalled release attributable to one condition instead of only reporting
the terminal `pre_release_stalled` result.

The support centroid used by the phase-validity gate is also continuous: hand
positions are weighted by grip strength and attached state, while foot positions
are weighted by physical support. Consequently the balance target moves from
the four-support geometry toward the future three-support geometry throughout
the ramp and never follows a free reaching hand.

The policy sees this state as a five-value one-hot inside the existing 24D
command. Hand targets require three consecutive close, slow frames; foot
targets require five consecutive frames with proximity, low speed, a real
contact-sensor hit, and valid whole-body height. The first foot may lower the
body by at most 0.03 m from its phase start; the second foot must raise it by at
least 0.12 m from the cycle start. Stabilization requires a per-environment random 50--100
consecutive valid frames (1--2 seconds at 50 Hz); losing a hand attachment,
foot support, torso-speed limit, or joint-speed limit resets the hold counter.
Every successful phase transition is stored in a bounded, per-phase GPU state
bank. Once a phase is unlocked, 50% of training resets sample uniformly from
the available phase boundaries; the remaining resets retain the deterministic
climbing keyframe. A restored state includes root/joint position and velocity,
hand-anchor poses, weld state, and the hand/foot rung assignments before the
requested phase is configured. Boundary-started episodes still contribute PPO
experience but do not enter the promotion window. The adaptive curriculum records every eligible full-prefix terminal episode as
a success only when it reaches the boundary after the currently unlocked
phase; falls, low-root termination, and ordinary episode timeout are failures.
The next phase opens only when both conditions hold:

- the rolling window contains 100 episodes and its success rate is strictly
  greater than 80% (at least 81 successes);
- the current phase has accumulated its configured minimum environment steps.

With `num_steps_per_env=24`, initial stabilization uses 36,000 environment
steps (1,500 PPO iterations), while every later transition uses 120,000 steps
(5,000 iterations). If every phase passes immediately when eligible, the
earliest schedule is:

| Phase available | Earliest environment step | Earliest PPO iteration |
|---|---:|---:|
| Stabilize | 0 | 0 |
| First hand | 36,000 | 1,500 |
| Second hand | 156,000 | 6,500 |
| First foot | 276,000 | 11,500 |
| Second foot/full cycle | 396,000 | 16,500 |

If success is 80% or lower, the phase remains locked regardless of elapsed
steps. Reaching a locked boundary truncates the short prefix episode so PPO can
start another attempt without waiting for the normal timeout. The reward set is
fixed throughout training:

| Term | Weight |
|---|---:|
| Supported novel maximum whole-body height | 20 |
| Phase-aware signed foot placement | 8 |
| Phase-conditioned target progress | 8 |
| Any ordered phase completion | 25 |
| Stabilization torso-orientation squared error | -1 |
| Unsuccessful episode termination | -50 |
| Final success | 100 |
| Survival outside stabilization | 3 |
| Missing required foot support (per foot) | -2 |
| Contact-independent held-foot recovery progress | 4 |
| Stabilization threshold violation | -1 |
| Action rate | -0.1 |
| Joint limits | -10 |
| Self-contact slots above 1 N | -0.1 |
| Ankle-joint acceleration squared | -2.5e-6 |


Progress and foot placement are separate terms for each of the five phases:
`ladder_<phase>_progress` and `ladder_<phase>_foot_placement`. Each has weight 8
and returns zero outside its phase; their sum equals the original unsplit
shaping reward. All histories update before masking. Height and event bonuses
are shared; orientation applies only to stabilization. Shared regularizers use
the tracking task's self-contact sensor and ankle-joint selection; there is no
additional all-joint acceleration penalty.

The reward per control step is
`dt * (20 H + sum_p I_p (8 P_p + 8 F_p) + 4 R - 2 M - I_stabilize (O + V) + 3 (1 - I_stabilize) - 0.1 A - 10 L - 0.1 C - 2.5e-6 Q) + 25 B - 50 D + 100 S`.
Here H is the supported novel-height rate, P and F are the existing clipped
progress and foot-placement rates, O is squared torso-orientation error,
A is squared action change, L is soft joint-limit excess, C counts self-contact
slots above 1 N, and Q sums squared ankle-joint accelerations. B, D, and S are
phase-completion, unsuccessful-termination, and success event indicators.
The event terms already divide by dt internally, so their bonuses are impulses.


M counts missing required physical foot supports on every active step. R is the
signed decrease of summed required-foot distances to held rungs, divided by
`2 * 0.35 m * dt`, without a contact multiplier. Foot phases exclude the selected
moving foot from both terms. Recovery history reanchors on reset and phase or
held-rung changes; stationary feet receive zero progress and retreat is negative.
V is the mean squared normalized positive threshold excess for torso linear speed,
joint-speed RMS, maximum torso/pelvis angular speed, waist-joint speed, and
support-relative torso offset. Each uses the existing stabilization threshold as
its scale (unit scale for a zero threshold); valid gates cost zero. This dense
penalty provides a signal before all gates pass, without altering the dwell or
phase-transition conditions. Survival is zero during stabilization.

`record_ladder_video.py --ladder_phase first_hand` permits stabilization to advance
into the first hand phase and freezes transitions only after that hand phase
succeeds. Reward changes affect subsequent training; they do not change the actions
of an already trained checkpoint during playback.

The terminal failure term applies to ordinary timeout, stalled `PRE_RELEASE`,
falls, and low-root endings. Final success and the intentional truncation at a
completed locked curriculum boundary are excluded.

The novel-height term records the maximum mean pelvis/torso COM height reached
since the episode reset. It always advances that record, but pays a positive
increase only while the physical supports required by the current phase are
present. An unsupported jump therefore consumes the new record without earning
reward, and returning to that height cannot collect it later. The term returns
`delta_height / step_dt`; after reward-manager timestep scaling, its maximum
episode contribution is `20 * (max_height - initial_height)` and is independent
of the control frequency.

The foot-placement term is also a signed potential difference, not a dense
reward for remaining in place. During stabilization and hand phases, both feet
are evaluated against their currently assigned rungs. During a foot phase, the
support foot remains assigned to its held rung while the selected foot is
evaluated against the new target rung. Placement quality combines physical
rung contact with an exponential distance score using a 0.06 m standard
deviation. Establishing the desired contact pays the positive potential change,
losing it applies the equal negative change, and unchanged contact produces
zero reward. A closed detach/reattach cycle therefore has zero net reward and
cannot be farmed through contact chatter; an unsupported body launch also loses
placement potential and receives no height reward.

The phase-conditioned potential contains normalized distance to the active hand
or foot target and, during hand phases, continuous release progress
`1 - grip_strength` with coefficient `1.0`; whole-body height is handled
exclusively by the novel-height term. Reversing the grip ramp therefore returns
the same progress as a negative reward, so a closed soften/regrip cycle has zero
net reward. `STABILIZE` pays only increases of its per-phase maximum normalized
dwell progress. Resetting and rebuilding an already reached partial hold pays
zero, preventing discounted dwell cycles from being farmed.
Positive target progress normally receives 25% strength while phase support or
posture constraints are invalid and full strength when they are valid. Once the
selected hand is detached, positive hand-target progress instead receives zero
unless those constraints are valid, preventing an unstable dive toward the bar
from earning progress.
Negative target progress is never attenuated. A movement phase completes only after the same support and
support-relative COM-offset check also holds with torso speed at most 0.20 m/s
and joint-speed RMS at most 1.0 rad/s. Torso orientation remains available to
reward shaping and diagnostics but no longer blocks phase transitions.
Initial stabilization additionally requires torso and pelvis angular speed at
most 0.40 rad/s and every waist-joint speed at most 0.60 rad/s. Its soft
orientation penalty discourages twisting without turning orientation into a
hard transition condition.
Phase-completion and success rewards are dt-independent
one-step impulses. The ladder runner
merges outcome windows across all GPUs and
saves the unlocked phase, phase-start step, recent outcomes, and phase-boundary
bank in checkpoints.
Resuming a fixed-schedule checkpoint fails explicitly because it has no
adaptive curriculum state. Playback and benchmarking unlock every phase
immediately and use the same fixed reward definition.

The long-run ladder PPO preset uses 60,000 iterations, saves every 1,000
iterations, starts Gaussian exploration at standard deviation 0.7, clamps its
effective value to `[0.25, 1.0]`, and uses a `5e-4` learning rate with `0.005`
entropy coefficient. The ladder runner also projects the raw scalar/log std
parameter after every update and checkpoint load, so it cannot remain below the
clamp with zero gradient. Opening a new curriculum phase restores std to `0.7`,
clears its optimizer momentum, and resets both the adaptive learning-rate state
and optimizer groups to `5e-4`; `Policy/raw_mean_std` logs the projected raw
parameter separately from the effective policy std. Ladder training does not
randomize initial episode lengths, because artificial early timeouts would
pollute the last-100 curriculum success window.

The ladder geometry and grip sites are generated on top of the canonical
`assets/robots/unitree_g1/g1_29dof.xml`. Each ladder face uses fixed collidable
side rails and separate flat-topped box rungs with stiff contacts. The robot
collides with these bars and the flat rung tops support the feet. Gaps are
physically open: there are no invisible face blockers. The ladder-only G1
collision overlay described below covers the actual body surfaces. The ladder
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
flat 117D/120D MLP ladder checkpoints. The base current-frame dimensions stay
117D/120D, but the model now requires history, `9 x 15` target-aware
ladder-geometry inputs, and the critic-only current 14D reward/FSM state.
Standard full resume from a checkpoint without that critic group is unsupported;
reusing such a policy requires an explicit actor-only warm start with a newly
initialized critic. TemporalCNN checkpoints using the earlier `9 x 7`
endpoint-only geometry are also incompatible. The
continuous grip-strength scalar and feedback-controlled soft-release mechanic
also change the meaning and transition distribution of the 24D command. The
phase one-hot semantics, ordered FSM, climbing keyframe, phase-potential reward and whole-body ascent gates,
active bar contacts, and refined surface collision geometry therefore require a
new training run. The curriculum checkpoint version is bumped so attempting to
resume a policy trained with the former scheduled multi-term reward fails fast.

### Ladder collision assets

Install the build extra and prepare the assets before training or playback:

```bash
pip install -e '.[train,collision-build]'
python scripts/setup/download_assets.py --only robots g1_collision
```

`g1_collision` explicitly downloads the G1 collision surfaces from
[OmniRetarget/Holosoma](https://github.com/amazon-far/holosoma/tree/bccd4d7451640a2800ddc77e469d911a84f91994/src/holosoma_retargeting/holosoma_retargeting/models/g1),
pinned to commit `bccd4d7451640a2800ddc77e469d911a84f91994`. It is independent
of the ModelScope/HuggingFace `--source` setting and is not included in the
default hosted-asset download. The CPU build uses CoACD 1.0.14 to produce up to
12 convex parts per source link, with at most 64 vertices per part. This is an
approximate surface model, not exact triangle-to-triangle collision. Motor housings
and ankle hinges use solid convex envelopes as decomposition input; internal CAD
bores are excluded from the contact model. Motor housings
and ankle hinges use solid convex envelopes as decomposition input; internal CAD
bores are excluded from the contact model. The first
build can take several minutes per link; completed parts are cached. The source
revision, checksums, decomposition settings and upstream license notices are
retained under `assets/robots/unitree_g1/omniretarget_collision/`. The part limit
can override CoACD's requested concavity threshold; it is not an error bound.
These assets
are ignored by Git. Missing or corrupt parts stop ladder startup with rebuild
instructions.

The overlay replaces primitive colliders on 25 source links: pelvis contour,
torso, head, hips, thighs, shins, ankles, feet, shoulder yaw links, elbows and
wrists. Fixed accessory transforms are folded into their canonical parent.
G1 rev. 1.0 uses a different waist assembly: its own torso surface is decomposed
instead of the older donor torso. Four contiguous horizontal convex sections
represent its exterior without reproducing internal CAD cavities. The head overlay is shifted 10 mm upward
in the torso frame to match the canonical visual mesh. The source torso checksum
is recorded and verified. Canonical joint-origin differences (waist roll +9 mm,
waist pitch -19 mm, shoulder pitch +10 mm) remain unchanged.
Canonical hand capsules and attachment sites are retained because this task
uses its own rubber-hand grip mechanic. The canonical G1 XML still owns all 29
joints, masses, inertias, actuator settings and visual meshes. Tracking and
inference keep their original collision model.

Rails and rungs retain their declared stiff contact parameters and use higher
contact priority than robot feet (`condim=4`, sliding friction 1.4 on rails and
1.8 on rungs). Robot-material randomization alone therefore does not vary the
ladder contact material. More collision parts increase collision work; measure
training throughput on the intended GPU before choosing an environment count.
Train a fresh policy for this contact model; loading old weights does not
restore the previous collision physics or make cached boundary states valid.
Curriculum checkpoint version 3 rejects full training resume from earlier
reward/contact models. Playback may evaluate old weights under the new physics.

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

The same entry point plays an RL-only ladder checkpoint without a motion
dataset:

```bash
python train_mimic/scripts/play.py \
    --task G1-Ladder-Climb-RL \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --ladder_phase second_hand
```

`--ladder_phase` accepts `stabilize`, `first_hand`, `second_hand`,
`first_foot`, or `second_foot` (the default). It selects the deepest phase in
the ordered curriculum prefix, not an isolated initial state. For example,
`second_hand` plays `stabilize -> first_hand -> second_hand` and resets before
`first_foot`; this keeps the hand attachments and foot supports physically
consistent. Omit the option, or choose `second_foot`, to play the complete
climb.

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
    --frames 1000 \
    --ladder_phase second_hand
```

The recorder accepts the same `--ladder_phase` values and ordered-prefix
semantics as interactive playback. When the option is supplied explicitly,
the FSM freezes after the selected phase succeeds instead of transitioning or
terminating at the curriculum boundary. The policy and physics continue until
the frame limit or a real failure, which makes `stabilize` useful for sustained
balance inspection. Omit the option to record the complete climb.

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
