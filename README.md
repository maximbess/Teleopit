<p align="center">
  <img src="assets/teleopit_logo.jpg" width="80" alt="Teleopit">
</p>

<h1 align="center">Teleopit</h1>

<p align="center">
  Lightweight, extensible whole-body teleoperation framework for humanoid robots.
  <br/>
  Real-time motion retargeting from BVH / Pico 4 VR to Unitree G1, in MuJoCo sim or on real hardware.
</p>

<p align="center">
  <a href="https://BotRunner64.github.io/Teleopit/">Documentation</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/zh-Hans/">中文文档</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/tutorials/pico-sim2sim">Pico Sim2Sim</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/tutorials/pico-sim2real">Pico Sim2Real</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/tutorials/training">Training</a>
</p>

---

## Quick Start — Minimal Sim2Sim

**1. Install**

```bash
pip install -e .
```

**2. Download assets**

```bash
pip install modelscope
python scripts/setup/download_assets.py --only robots gmr ckpt bvh
```

The canonical Unitree G1 robot model is downloaded to
`assets/robots/unitree_g1/g1_29dof.xml`. Training, sim2sim, retargeting, and FK
validation all use this same XML.

**3. Run**

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

You should see a MuJoCo viewer with the robot tracking the BVH motion.

To show the simulated D435i RGB camera view, add the explicit `camera` viewer:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    'viewers=[sim2sim,camera]'
```

For sim2real, viewers are disabled by default. Add `viewers=retarget` to show
the retargeted reference in an optional MuJoCo window.

## RL-Only Ladder Training

The ladder policy has its own entry point and does not use motion clips,
retargeting, imitation rewards, or the tracking runner:

```bash
pip install -e '.[train]'
python train_mimic/scripts/train_ladder.py \
    --num_envs 4096 \
    --max_iterations 60000
```

For a short environment check, use `--num_envs 64 --max_iterations 100`.
Multi-GPU launch is also supported with `--gpu_ids 0 1 2 3`. The task builds
the A-frame ladder and grip constraints on top of the canonical
`assets/robots/unitree_g1/g1_29dof.xml`; no copied G1 model or motion dataset is
required. Checkpoints are written under `logs/rsl_rl/g1_ladder_rl/`.

The `train` dependency set pins `warp-lang==1.15.0`. Do not use Warp 1.16.0
with the supported `mjlab==1.4.0` / MuJoCo Warp 3.8 stack: its sensor-kernel
compilation fails with `Referencing undefined symbol: xmat`, commonly during a
four-process multi-GPU cold start. Repair an existing environment before
launching training:

```bash
python -m pip install --force-reinstall "warp-lang==1.15.0"
python -c "import warp as wp; print(wp.__version__)"
```

Both ladder faces use fixed collidable side rails and individual flat-topped
box rungs. These bars are rigid, use stiff contacts, and cannot be crossed by
the robot, while the visible space between adjacent rungs remains empty. A thin
invisible blocker is offset behind each face and uses a separate collision mask:
it stops the pelvis, torso, and head from entering the A-frame, but hands and
feet pass through it to reach the exposed bars. The flat rung tops give each
foot a stable support surface. Episodes start directly in a climbing pose: both
feet are on physical rung 2 and both hands initially hold rung 5. An explicit
five-phase FSM repeats the same order on every rung: stabilize all four
supports, move the first hand, move the second hand, step with the first foot,
then step with the second foot. Only the selected limb is released. Hand and
foot targets must remain valid for 3 and 5 consecutive control steps,
respectively, so a single noisy contact cannot advance the FSM. There is no
ground-to-ladder approach phase.

Hand and foot target shaping uses signed closing progress instead of absolute
proximity: approaching a target is positive, hovering is zero, and retreating
is negative. The adaptive prefix curriculum keeps the last 100 terminal
episodes for the current phase. It opens the next phase only when that full
window has strictly more than 80% successes and the phase has received at
least 120,000 environment steps (5,000 PPO iterations with a 24-step rollout).
Thus the complete five-phase cycle cannot unlock before iteration 20,000. A
completed prefix episode ends immediately while its next phase is locked. The
reward schedule at steps 0, 240,000, and 480,000 emphasizes low torso speed,
upright posture, smooth joints, and especially low velocity in supporting limb
joints. Curriculum state is synchronized between GPUs and saved in every
checkpoint. Torso ascent is rewarded only during the two-hand-supported foot
phases, while physical foot placement and complete-cycle impulses carry larger
weights. The 24D command carries a five-state phase one-hot. The policy keeps
the 117D actor and 120D privileged-critic current-frame groups, encodes a
10-frame history for each with TemporalCNN, and separately encodes all nine
finite rung endpoints as a `9 x 7` torso-frame geometry tensor.

The ladder scene removes the canonical XML's embedded `floor` and uses one
solid-color, non-reflective terrain plane. It also uses a single controlled
light with playback shadows and reflections disabled, avoiding checker-texture
moiré, duplicate-plane contacts, and shadow artifacts in video. The resulting
policy fuses current state, temporal history, and ladder geometry through
separate Conv1d encoders before the scaled `(2048, 1024, 512, 256, 128)` MLP.
The ordered phase-command and adaptive-curriculum semantics, climbing keyframe,
scheduled rewards, active bar contacts, trunk-blocking collision geometry, and
multi-group TemporalCNN inputs require a fresh ladder training run. Earlier
117D/120D MLP checkpoints do not match this model signature, and fixed-schedule
checkpoints also lack the adaptive curriculum state.

Evaluate a ladder checkpoint without further PPO updates:

```bash
python train_mimic/scripts/benchmark_ladder.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --num_envs 64 \
    --num_eval_steps 5000
```

Add `--num_envs 1 --video --video_length 1000` to write an MP4 under
`benchmark_results/videos/`. The ladder benchmark reports success, failure,
timeout, rung-progress, grip, reach-distance, reward, and episode-length
statistics; it does not require a motion dataset.

To record only the trained policy rollout, without benchmark metrics or report
files, use the dedicated recorder:

```bash
python train_mimic/scripts/record_ladder_video.py \
    --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
    --output ladder.mp4 \
    --frames 1000
```

The recorder runs one environment and stops before an automatic episode reset.
Its default camera is a fixed world-space overview on the robot's outside face
of the ladder, rather than a torso-tracking view through the rungs. The complete
robot and ladder remain framed throughout the climb. Override the view with
`--camera_azimuth`, `--camera_elevation`, `--camera_distance`,
`--camera_lookat X Y Z`, and `--camera_fovy` when needed.

## Pico Motion Recording

Record many Pico clips as training-ready G1 motion NPZ files:

```bash
pip install -e '.[pico4]'
python scripts/run/record_pico_motion.py
```

The recorder starts the Pico receiver and live Retarget viewer before waiting
for clip names, so preview keeps running while the terminal is idle. Enter a
semantic clip name, then use `R` to start, `S` to save, `D` to discard, `N` for
a new name, and `Q` to quit. Saved clips are written to
`data/pico_motion/clips/` using the semantic label in the filename, with no
sidecar JSON.

Merge recorded clips into the standard HDF5 shard dataset:

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec data/pico_motion/pico_recorded.yaml --force
```

## Sim2Real HDF5 Recording

Pico sim2real can also record manual HDF5 episodes from the real G1:

```bash
pip install -e '.[recording]'
# If you use RealSense video, install pyrealsense2 manually for your platform.
# On Arm machines, prefer conda-forge:
# conda install -c conda-forge pyrealsense2
python scripts/run/run_sim2real.py --config-name sim2real_record \
    controller.policy_path=track.onnx \
    recording.task="walk forward"
```

Recording uses the terminal controls `R` start, `S` save, `D` discard, and `Q`
shutdown. `STANDING`, `MOCAP`, `ARMS`, and paused mocap can be recorded. Saved
episodes are written as `.h5` files under `data/recordings/sim2real_hdf5/episodes/`.
`sim2real_record.yaml` stores camera frames as compressed MP4 sidecar files under
`data/recordings/sim2real_hdf5/videos/` and keeps `frame_index` / `timestamp`
sync metadata in the HDF5 episode. The low-dimensional HDF5 schema records
`observation.state(68)`, `observation.mode(1)`, `action(36)` as the aligned
reference qpos sent to the policy path, and `action.hand(12)` as the latest
LinkerHand left/right 6D pose commands.

## Documentation

Full docs at **[BotRunner64.github.io/Teleopit](https://BotRunner64.github.io/Teleopit/)**, covering installation profiles, all tutorials, configuration reference, and architecture.

## Changelog

### v0.4.0 (2026-06-25)

- Improved Pico realtime control with pico-bridge 0.2.1, `ARMS` mode, armed sim2real mocap entry, and retargeter-preserving pause/arms resets.
- Added optional LinkerHand L6/O6 sim2real control, including Pico gripper input and low-latency L6 `vr_hand_pose`.
- Added manual Pico sim2real HDF5 recording and an interactive Pico motion recorder for training NPZ clips.
- Refined the training data path with minimal HDF5 shards, explicit precompute, rewind sampling, and updated tracking rewards.

### v0.3.0 (2026-05-12)

- Consolidated realtime input around pico-bridge 0.2.0 and removed the old ZMQ/onboard Pico path.
- Unified sim/sim2real reference buffering, resume realignment, and velocity smoothing.
- Added UDP BVH realtime input, online sim config, multi-viewer support, and fixed camera viewing.
- Split sim2real reference/safety runtime modules and updated the G1 MuJoCo camera asset.

### v0.2.0 (2026-04-03)

- Added Pico 4 teleoperation through pico-bridge and the G1 Bridge SDK.
- Added offline playback keyboard controls, Pico sim2sim mode control, and a standalone standing controller.
- Improved realtime mocap buffering/catch-up and upgraded the released model to the 30k checkpoint.

### v0.1.1 (2026-03-28)

- Dataset shard-only refactor
- External asset management (ModelScope), repository slimming

### v0.1.0 (2026-03-25)

- Initial public release: General-Tracking-G1 training, ONNX sim2sim inference, Pico 4 VR teleoperation, Unitree G1 hardware deployment

## License

[Apache 2.0](LICENSE)
