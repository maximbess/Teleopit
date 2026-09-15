# AGENTS.md

## Project Overview

Teleopit is a lightweight, extensible, self-contained humanoid robot whole-body teleoperation framework. It integrates GMR (General Motion Retargeting) and supports train_mimic-exported ONNX RL policy inference.

Language: Python 3.10+
Package: `teleopit` (installed via `pip install -e .`)
Config: Hydra/OmegaConf YAML files in `teleopit/configs/`

## Architecture

```
InputProvider (BVH file / Pico4 VR) → Retargeter (GMR) → ObservationBuilder (167D) → Controller (dual-input TemporalCNN ONNX) → Robot (MuJoCo + PD / Unitree SDK)
```

Module-internal isolation: all modules run in-process and communicate via `InProcessBus` (zero-copy). Core interfaces are defined as `typing.Protocol` in `teleopit/interfaces.py`.

## Supported Surface

- Training tasks: `General-Tracking-G1` motion tracking and `G1-Ladder-Climb-RL` RL-only ladder climbing
- Inference observation: `velcmd_history` (167D, dual-input ONNX with `obs` + `obs_history`)
- TemporalCNN actor/critic with scaled dims (2048,1024,512,256,128)
- Realtime inference uses a retargeted-reference timeline before observation build; `reference_steps=[0]` is the default production path

## Directory Structure

```
teleopit/                 # Core inference package
├── interfaces.py         # Protocol definitions: Robot, Controller, InputProvider, Retargeter, etc.
├── pipeline.py           # TeleopPipeline — thin sim runtime facade
├── runtime/              # Shared runtime assembly: config/path resolution, factories, CLI helpers
├── bus/                  # InProcessBus message pub/sub
├── configs/              # Hydra YAML configs
│   ├── default.yaml      # Offline sim2sim
│   ├── sim2real.yaml     # sim2real
│   ├── robot/g1.yaml     # G1 robot: XML path, PD gains, default angles, action dims
│   ├── controller/rl_policy.yaml
│   ├── input/bvh.yaml    # Offline BVH file input
│   └── input/pico4.yaml  # Pico4 realtime input
├── controllers/
│   ├── rl_policy.py      # RLPolicyController — single-input or dual-input ONNX inference with fail-fast dim checks
│   └── observation.py    # VelCmdObservationBuilder
├── inputs/
│   ├── bvh_provider.py       # BVHInputProvider — offline BVH file
│   ├── pico4_provider.py     # Pico4InputProvider — pico_bridge receiver input
│   ├── pico_video.py         # Optional camera preview pushed back to Pico through pico-bridge
│   ├── rot_utils.py          # Quaternion helpers for input-space transforms
│   └── udp_bvh_provider.py   # UDPBVHInputProvider — realtime BVH packet input
├── retargeting/
│   ├── core.py           # RetargetingModule + extract_mimic_obs()
│   └── gmr/              # Self-contained GMR code; heavyweight assets are downloaded into an ignored path
├── robots/
│   └── mujoco_robot.py   # MuJoCoRobot — MuJoCo sim wrapper
├── sim/
│   └── loop.py           # SimulationLoop — PD control at 200Hz, policy at 50Hz
├── sim2real/
│   ├── mp/               # Process-isolated sim2real runtime and IPC
│   └── hands/            # Optional LinkerHand driver/mapper plugins
└── recording/            # Pico motion NPZ recording helpers
scripts/
├── run/run_sim.py        # Offline sim2sim pipeline
├── run/run_sim2real.py   # G1 sim2real control; supports offline BVH playback and Pico4
├── run/record_pico_motion.py # Interactive Pico recording → G1 motion NPZ clips
├── render/render_sim.py  # Render single BVH → 3 MuJoCo videos (mocap input, retarget, sim2sim)
└── dev/compute_ik_offsets.py # Compute IK quaternion offsets for new BVH formats
train_mimic/              # Training package
├── app.py                # Shared app helpers for train/play/benchmark
├── tasks/tracking/config/
│   ├── constants.py      # Public task constants
│   ├── registry.py       # Registers tracking and RL-only ladder tasks
│   ├── env.py            # Tracking and generated ladder env builders
│   └── rl.py             # TemporalCNN tracking and ladder PPO cfgs
├── tasks/tracking/mdp/
│   └── ladder.py         # Ladder targets, simulated grip FSM, rewards, and success term
├── tasks/tracking/rl/
│   ├── runner.py         # Training runner and policy ONNX export wrapper
│   ├── conv1d_encoder.py # 1-D CNN encoder for temporal history groups
│   └── temporal_cnn_model.py # TemporalCNN actor/critic model
└── scripts/
    ├── train.py          # Training entry point
    ├── train_ladder.py   # RL-only G1 ladder-climbing entry point
    ├── play.py           # Checkpoint playback
    ├── benchmark.py      # Policy evaluation with tracking errors
    ├── benchmark_ladder.py # Ladder success metrics and MP4 recording
    ├── record_ladder_video.py # Ladder policy MP4 recording without metrics
    └── save_onnx.py      # Export TemporalCNN ONNX
```

## Key Technical Details

### Sim2Sim Pipeline
- Policy runs at 50Hz, PD control at 200Hz (`decimation=4`, `sim_dt=0.005`)
- Action flow: `compute_action()` returns raw action → `get_target_dof_pos()` applies clip `[-10, 10]`, scale, and `default_dof_pos`
- Must use `assets/robots/unitree_g1/g1_29dof.xml` for training, sim2sim, dataset FK, and retargeting; it is the canonical G1 XML entry point

### Multi-Viewer Support
`SimulationLoop` supports multiple simultaneous viewer windows controlled by the `viewers` config:

```bash
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=sim2sim
python scripts/run/run_sim.py controller.policy_path=policy.onnx 'viewers=[mocap,retarget,sim2sim]'
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
python scripts/run/run_sim.py controller.policy_path=policy.onnx 'viewers=[sim2sim,camera]'
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=none
```

- `sim2sim`: MuJoCo physics result
- `retarget`: kinematic retarget result
- `mocap`: retargeting input skeleton rendered by MuJoCo custom geoms
- `camera`: G1 `d435i_rgb` fixed RGB camera view
- `bvh` viewer naming is removed; use `mocap`
- `viewers=all` opens `mocap`, `retarget`, and `sim2sim`; add `camera` explicitly when needed
- All viewers run in separate subprocesses because GLFW/GLX only supports one window per process
- Simulation exits when all active viewer windows are closed
- sim2real defaults to `viewers=none`; it supports only optional `viewers=retarget`
- `viewers` is the only supported viewer key; legacy `viewer` alias is removed

### default_dof_pos Propagation
RL policy outputs action offsets relative to the default standing pose:

```
target_dof_pos = clip(action, -10, 10) × action_scale + default_dof_pos
```

`default_dof_pos` comes from `robot/g1.yaml` `default_angles`. `TeleopPipeline` automatically propagates `robot_cfg.default_angles` into `controller_cfg.default_dof_pos`. If this propagation is missing, knees and elbows lose their standing offset and the robot cannot balance.

### Offline Playback
- Offline sim2sim and default sim2real both read `input.bvh_file` directly; no UDP relay path remains
- Offline sim2sim playback can be keyboard-controlled: `Space/P` pause/resume, `R` replay from frame 0, `Q` stop
- Offline pause holds the commanded pose; resume resets policy/reference state and reanchors yaw/XY without qpos interpolation or retargeter IK reset
- sim2sim keyboard playback is optional via `playback.keyboard.enabled=true`
- sim2real reuses the Unitree remote: `Start` → `STANDING`, `Y` → playback, `X` → back to `STANDING`, `L1+R1` → `DAMPING`
- `playback.pause_on_end=true` keeps the final pose and waits for manual replay

### Pico4 Realtime Input
- `Pico4InputProvider` reads realtime body tracking from the in-process `pico_bridge.PicoBridge`
- The pico-bridge receiver runs on the Teleopit host, which can be a workstation PC or robot onboard computer; do not maintain a separate onboard Pico input mode
- pico-bridge 0.2.1 is the supported runtime; camera preview uses `PicoBridge(video="frames").push_video_frame(rgb_uint8)`
- Pico video preview is optional and disabled by default; sim2sim uses the MuJoCo `d435i_rgb` camera and sim2real uses RealSense when `input.video.enabled=true`
- Bone naming follows `pico_bridge_to_g1.json`
- The provider applies an input-space transform to match the current retarget config
- Do not hardcode that transform as a public coordinate-system contract; validate against actual retarget/sim2sim behavior when SDK or firmware changes
- Pico4 realtime control uses the same retargeted-reference timeline path as the shared realtime input stack
- Pico sim2sim supports a keyboard-driven top-level mode state machine: `STANDING → MOCAP ↔ ARMS`, `X` returns to `STANDING`
- Default Pico sim2sim keyboard mappings are `Y` → `MOCAP`, `A` → pause/resume mocap, `B` → toggle `MOCAP`/`ARMS`, `X` → back to `STANDING`, `Q` → quit
- Pico4 sim2real pause/resume is handled as a mocap-session control event (`toggle_pause`), not as a mode switch to `STANDING`
- Default Pico pause button is `A`; resume resets policy/reference state and yaw/XY root-offset alignment while the process-isolated realtime reference worker continues its live input timeline
- Pico4 sim2real arms the process-isolated reference worker only when entering `MOCAP`; `STANDING` and `DAMPING` disarm it so cold startup frames do not warm-start GMR before mocap entry
- Pico4 sim2sim/sim2real support `ARMS` mode toggled from `MOCAP` with Pico/controller `B`; retargeting continues, while the control loop sends the motion tracker a composed reference with stand-pose body/legs/waist and live retargeted arms
- `ARMS` entering/exiting/resume resets policy/reference alignment and uses Kp ramp; offline BVH sim2real does not use `ARMS`, and Unitree remote `B` remains BVH replay
- Realtime Pico pause/resume and `MOCAP ↔ ARMS` switches use a retargeter-preserving soft reset: policy/reference state, smoothers, and reference alignment are reset, while the GMR IK warm-start is retained
- Optional LinkerHand control uses `hands.enabled=true`, `hands.driver=linkerhand_l6|linkerhand_o6`, and `hands.mode=gripper|vr_hand_pose`; default is disabled
- Optional Pico sim2real HDF5 recording uses `--config-name sim2real_record` or `recording.enabled=true`; it requires `input.provider=pico4`, `input.video.enabled=true`, `input.video.source=realsense`, an interactive terminal, and the `recording` extra
- Recording is manual only: terminal `R` starts an episode, `S` saves, `D` discards the active episode, and `Q` shuts down; `STANDING`, `MOCAP`, `ARMS`, and paused mocap are recordable
- Recording captures `observation.images.d435i_rgb` RealSense RGB video at 30Hz plus `observation.state(68)`, `observation.mode(1)`, `action(36)`, and `action.hand(12)`; RealSense capture lives in `pico_input` through the normal `input.video` path
- HDF5 recording writes compressed MP4 sidecar videos under `recording.output_dir/videos/<camera_key>/` while HDF5 episodes store `frame_index`, `timestamp`, low-dimensional data, and video sync attributes; raw RGB image datasets are not supported
- `gripper` mode reuses `Pico4InputProvider.get_controller_snapshot()` for Pico grip/trigger open-close control and supports LinkerHand L6 and O6
- `vr_hand_pose` mode reuses `Pico4InputProvider.get_hand_snapshot()` and somehand 0.2.0 public `somehand.api` for continuous Pico hand-pose retargeting; do not start a second `PicoBridge` for hand control
- Teleopit owns Pico 26-joint hand-state to 21-landmark conversion; do not import `somehand.pico_input`
- LinkerHand O6 supports only `hands.mode=gripper`; its default `close_pose` is `[86, 73, 118, 111, 110, 111]`
- L6 `gripper` mode uses the configured `hands.linkerhand_l6.speed` (default `[50]*6`); O6 `gripper` mode uses `hands.linkerhand_o6.speed` (default `[255]*6`); `vr_hand_pose` always sets LinkerHand L6 speed to `[255]*6`
- `vr_hand_pose` defaults to a low-latency somehand path: `hands.somehand.rate_hz=60`, `max_iterations=12`, `temporal_filter_alpha=1.0`, and `output_alpha=1.0`; this prioritizes response speed over smoothing
- LinkerHand control is active in all sim2real modes when `hands.enabled=true`; shutdown and hand-runtime failure must send the configured open pose
- In `vr_hand_pose` mode, missing/inactive hand pose holds the last commanded pose for that side instead of opening the hand

### SimulationLoop Runtime Behavior
- `realtime=true` enforces wall-clock pacing even without a viewer
- `num_steps=0` means infinite loop (`max_steps = 2**63`)
- `KeyboardInterrupt` is handled for clean shutdown
- BVH frame alignment is time-based: `bvh_idx = int(policy_time × input_fps)`
- Realtime reference buffering is controlled by `retarget_buffer_enabled`, `retarget_buffer_window_s`, `retarget_buffer_delay_s`, `reference_steps`, and `realtime_buffer_warmup_steps`
- Realtime inferred `motion_joint_vel`, anchor linear velocity, and anchor angular velocity can be EMA-smoothed via `reference_velocity_smoothing_alpha` and `reference_anchor_velocity_smoothing_alpha`
- Sim2real Pico pause/resume uses mocap-session states `ACTIVE ↔ PAUSED`; resume clears policy/reference state, rebuilds yaw/XY root alignment, and does not interpolate retarget qpos from the paused pose
- Realtime sim2sim with Pico control events uses the same mocap-session pause/resume semantics and rebuilds the realtime reference path on resume, including the configured warmup
- Realtime sim2sim `STANDING ↔ MOCAP` transitions rebuild the realtime reference path on entry; Pico sim2real `STANDING -> MOCAP` additionally rearms and resets the process-isolated reference worker before accepting fresh references
- Realtime Pico sim2sim can start directly in `STANDING` with keyboard mode control enabled via top-level `keyboard.enabled`

### Inference Observation
Observation format: `velcmd_history` (167D, dual-input ONNX)

```
ref_joint_pos(29)
+ ref_joint_vel(29)
+ ref_anchor_ori_b(6)
+ robot_base_ang_vel_b(3)
+ robot_joint_pos_rel(29)
+ robot_joint_vel(29)
+ prev_action(29)
+ robot_projected_gravity_b(3)
+ ref_anchor_lin_vel_b(3)
+ ref_anchor_ang_vel_b(3)
+ ref_projected_gravity_b(3)
+ ref_anchor_height(1)
```

Runtime constraints:
- Public builder is `VelCmdObservationBuilder`
- `RLPolicyController` accepts dual-input `obs` + `obs_history` ONNX
- Startup validates the observation definition against the ONNX signature and raises immediately on mismatch

### Training Tasks
The motion-tracking task is `General-Tracking-G1` (experiment name: `g1_general_tracking`).

- Uses TemporalCNN actor/critic with scaled dims (2048,1024,512,256,128)
- 167D `velcmd_history` observation, dual-input ONNX export
- Training env uses `sampling_mode="rewind"`
- Tracking rewards include root position/orientation/linear velocity/angular velocity, body pose/velocity, joint position/velocity, survival, action-rate, joint-limit, self-collision, and ankle acceleration terms
- Supported motion sampling modes are `uniform`, `start`, and `rewind`; `rewind` restarts failed environments from the same clip after stepping back `rewind_min_steps..rewind_max_steps` with probability `rewind_prob`, otherwise it falls back to uniform sampling
- Playback/benchmark use `play=True`, which switches motion sampling to `start`
- `window_steps=[0]`
- `save_onnx.py` exports dual-input TemporalCNN ONNX

The ladder task is `G1-Ladder-Climb-RL` (experiment name: `g1_ladder_rl`).

- It is trained only with PPO reinforcement learning through `train_mimic/scripts/train_ladder.py`; it does not load a motion dataset or use `MotionTrackingOnPolicyRunner`
- It uses TemporalCNN actor/critic models with scaled dims (2048,1024,512,256,128), separate Conv1d encoders for state history and ladder geometry, and the standard 29D G1 joint-position action
- The base current-frame groups remain 117D actor / 120D clean critic: the 24D ladder command contains a five-state phase one-hot, torso-frame hand/foot target vectors, hand attachment, active-hand grip strength, physical foot-contact state, and normalized hand/foot progress; each side also receives a 10-frame history and a separate `9 x 15` torso-frame rung-token tensor with finite endpoints, validity, and per-limb target/support markers. Only the critic receives an additional current-only 14D privileged reward/FSM group containing body/record heights, cycle/phase ascent, torso/joint stability, two physical foot supports, required-support validity, release progress, and dwell progress; it is intentionally excluded from `critic_history`
- The A-frame ladder, grip sites, and weld anchors are generated onto the canonical `assets/robots/unitree_g1/g1_29dof.xml` at configuration time. Each face uses fixed collidable side rails and separate flat-topped box rungs with physically open gaps and no invisible face blocker. The ladder-only G1 collision overlay adapts G1 rev. 1.0 with its own torso surface and a +10 mm head-frame correction, and replaces coarse body primitives with convex parts from 25 OmniRetarget/Holosoma source links, preserving canonical joints, mass, inertia, visuals and hand capsules. Do not maintain a second G1 XML. Build ignored assets with `pip install -e '.[collision-build]'` and `python scripts/setup/download_assets.py --only g1_collision`; the source revision, source/part checksums and CoACD settings are pinned by `teleopit/runtime/g1_collision_assets.py` and the asset manifest.
- G1's default collision editor disables generated geoms that do not match `.*_collision`; ladder configuration explicitly re-enables rails and rungs afterward. All collision masks use bit 1; rails/rungs use priority 2, stiff `condim=4` contacts and sliding friction 1.4/1.8, overriding robot-foot materials. Robot-only friction randomization does not change ladder contact friction. New convex feet retain priority 1 and canonical ground-contact friction. Missing or corrupt overlay assets must fail fast.
- Rubber-hand attachment is an environment mechanic; feet use physical rung contacts without welds, and the policy still controls only the 29 G1 joints
- Ladder training keeps the deterministic climbing keyframe with both feet physically contacting rung 2 and both hands attached at rung 5, so there is no ground approach phase; after phase-boundary states become available, 50% of resets sample the per-phase GPU bank and the rest keep the deterministic start
- The ladder FSM repeats five ordered phases: stabilize four supports, move the configured first hand, move the second hand, move the configured first foot, and move the second foot; only the phase-selected limb may move, and both hands remain attached throughout foot phases
- Every hand phase begins with an internal feedback-controlled `PRE_RELEASE` substage while keeping the public five-phase one-hot unchanged: after 8 stable preload frames, the selected per-environment weld follows a 20-frame smoothstep ramp from its compiled `solref/solimp` to `timeconst=0.18` and impedance `0.05`; loss of physical support or torso/COM stability reverses the ramp by 2 steps, joint motion remains penalized but cannot veto release, binary detach occurs only after 5 additional stable fully-soft frames, and exceeding 300 `PRE_RELEASE` policy steps terminates the episode as a failure covered by the unified `-50` unsuccessful-termination impulse. Episode metrics separately expose the hand/foot-support, torso-speed, torso-orientation, and support-offset gates, their combined validity, and normalized preload/ramp/final-dwell progress
- `eq_solref` and `eq_solimp` are expanded per environment through the ladder startup event, so parallel environments can release independently. The 24D command exposes active-hand grip strength in place of the post-reset initialization bit, held-hand rung markers fade by that strength, and the torso support centroid weights each hand by grip strength and each foot by physical support so the balance target moves continuously as load transfers
- Stabilization requires a per-environment random 50-to-100-frame continuous hold with both foot supports, both attached hands, torso-COM speed <= 0.20 m/s, joint-speed RMS <= 1.0 rad/s, maximum torso/pelvis angular speed <= 0.40 rad/s, maximum absolute waist-joint speed <= 0.60 rad/s, and support-relative torso offset <= 0.18 m; losing any condition resets the dwell counter. Hand and foot targets require 3 and 5 consecutive valid frames respectively, including physical contact for feet. Stabilization metrics expose every linear, angular, waist, support, offset, combined-gate, and normalized-dwell value separately
- The adaptive prefix curriculum opens the next phase only when the rolling last-100-episode success rate is strictly greater than 80% and the current phase has accumulated its configured minimum: 36,000 environment steps (1,500 PPO iterations) for initial stabilization and 120,000 steps (5,000 iterations) for every later transition. Reaching a locked boundary truncates the completed prefix episode; boundary-started episodes train PPO but are excluded from this promotion window, benchmark unlocks all phases, and `play.py --ladder_phase` can reproduce any fixed curriculum prefix
- Successful transitions populate a bounded per-phase GPU reset bank containing root/joint state, hand-anchor poses, weld state, and hand/foot rung assignments; reset sampling uses only currently unlocked phase banks, and the requested FSM phase is configured after physical state restoration
- `LadderOnPolicyRunner` merges curriculum outcomes across all distributed ranks once per PPO iteration and persists the unlocked phase, phase-start step, recent window, pending outcomes, and phase-boundary bank in every checkpoint; adaptive training must fail fast when resuming a fixed-schedule checkpoint without this state
- The supported `mjlab==1.4.0` / MuJoCo Warp 3.8 training stack pins `warp-lang==1.15.0`; Warp 1.16.0 fails in sensor-kernel code generation with `Referencing undefined symbol: xmat`, so every training/playback entry that imports the training stack validates the Warp version before CUDA environment creation
- Ladder task shaping separates supported novel whole-body height, phase-conditioned active-limb target progress, and physical foot placement. The phase potential contains normalized target distance plus continuous hand-release progress `1 - grip_strength` with coefficient `1.0` during hand phases (all former phase body-height coefficients are zero), while `STABILIZE` pays only increases of the per-phase maximum normalized dwell. Resetting and rebuilding an already reached dwell level therefore pays zero and cannot exploit discounted returns. Reversing the grip ramp produces symmetric negative progress, so a closed soften/regrip cycle has zero net reward. Positive target progress normally retains 25% strength when phase support/posture constraints are invalid and receives full strength when valid; after the selected hand is detached, positive hand-target progress is zero unless the phase support/posture constraints are valid. Negative movement-phase progress is never attenuated
- The fixed ladder reward set is supported novel maximum whole-body height `20`, signed phase-aware foot placement `8`, phase target progress `8`, any ordered phase completion `25`, stabilization-only torso-orientation squared error `-1`, unsuccessful episode termination `-50`, final success `100`, movement-only survival `3`, action rate `-0.1`, missing required foot support `-2` per foot, contact-independent held-foot recovery progress `4`, stabilization threshold violation `-1`, joint limits `-10`, self-collisions `-0.1`, and ankle-joint acceleration `-2.5e-6`; success and a completed locked curriculum prefix are excluded from the terminal failure penalty, while ordinary timeout, stalled `PRE_RELEASE`, falls, and low-root endings receive it. There is no reward-weight schedule or separate dense positive support, upright, joint-velocity, limb-progress, signed torso-ascent, or cycle-completion reward. The orientation term is soft shaping only and never gates a phase transition. The novel-height term uses the mean pelvis/torso COM height, anchors its per-environment record after every reset, always advances the record, and pays `delta(max_height) / step_dt` only while the physical supports required by the active phase are present, so unsupported jumps consume the record without reward and lowering/re-climbing cannot farm it. The foot-placement term uses physical contact and a 0.06 m exponential distance score: stabilization/hand phases evaluate both held foot rungs, foot phases evaluate the support foot on its held rung and the selected foot on its new target, unchanged contact pays zero, and placement/restoration versus contact loss produce symmetric positive/negative potential changes, so a closed detach/reattach cycle has zero net reward. Movement-phase completion requires support-relative COM-offset error <= 0.15 m, torso speed <= 0.20 m/s, and joint-speed RMS <= 1.0 rad/s; torso orientation remains a shaping/diagnostic signal but is not a phase-transition gate. The first foot may not lower whole-body height by more than 0.03 m from its phase start, and the second foot requires at least 0.12 m whole-body ascent from the cycle start
- Ladder PPO starts Gaussian exploration at `0.7` and clamps effective action standard deviation to `[0.25, 1.0]`. The ladder runner projects the raw scalar/log std parameter back into its native bounds after every PPO update and checkpoint load, preventing a below-bound parameter from becoming gradient-dead, and logs it as `Policy/raw_mean_std`. Every curriculum promotion resets all actor action standard deviations to the configured initial `0.7`, clears optimizer state for that parameter, and restores the adaptive PPO learning rate and optimizer groups to the configured `5e-4`
- The ladder-only MuJoCo articulation scales all shoulder, elbow, and wrist effort limits to 70% of the standard G1 configuration (`25 -> 17.5 Nm`, `5 -> 3.5 Nm`); the standard tracking and inference robot configurations remain unchanged
- Ladder success requires the final hand rung and final coordinated foot support; falls and low-root states terminate as failures
- Checkpoints trained with the former 105D/108D or flat 117D/120D MLP contracts, TemporalCNN checkpoints using the former `9 x 7` endpoint-only ladder geometry, checkpoints without the critic-only 14D privileged group, checkpoints predating continuous grip-strength command semantics, and checkpoints trained with the former scheduled multi-term ladder reward are incompatible with standard full resume; the current policy uses multi-group TemporalCNN inputs (`actor|critic`, history, `9 x 15` target-aware ladder geometry, and critic-only reward/FSM state), while the feedback-controlled soft-release mechanic, phase-potential reward and whole-body ascent gates, five-phase command semantics, adaptive curriculum/reset-bank state, ordered FSM, climbing keyframe, active bar contacts, and refined surface collision geometry require retraining from scratch. Reusing an earlier actor requires an explicit actor-only warm start with a newly initialized critic
- `benchmark_ladder.py` evaluates ladder checkpoints without PPO updates, counts success/failure/timeout episodes, and optionally records a single-environment MP4
- `record_ladder_video.py` records one single-environment ladder-policy MP4 without aggregating metrics or writing benchmark reports; an explicit `--ladder_phase` freezes the successfully completed selected phase without transitioning or emitting the curriculum-boundary termination, while omitting it records the complete climb; the recorder stops before other automatic episode resets and defaults to a fixed world-space outside-ladder overview that keeps the complete robot and ladder framed instead of tracking the torso through the rungs
- `play.py --task G1-Ladder-Climb-RL --ladder_phase stabilize|first_hand|second_hand|first_foot|second_foot` runs interactive ladder playback without a motion dataset; the selected phase is the deepest enabled ordered prefix, and the default `second_foot` enables the complete climb
- Ladder scene assembly removes the canonical XML's embedded `floor`, uses one solid-color non-reflective `SceneCfg` plane, removes embedded robot lights, and disables playback shadows/reflections to prevent duplicate contacts, texture moiré, and lighting artifacts
- Tracking-only `train.py`, `benchmark.py`, and `save_onnx.py` do not accept the ladder task; `play.py` accepts both tracking and ladder tasks

### Dataset Pipeline
- Dataset build spec supports a `preprocess` section for root-xy normalization, ground alignment, and basic clip filtering
- Final distributed dataset build outputs are minimal HDF5 shards directly under `data/datasets/<dataset>/` (recursive shard discovery is supported; no train/val split and no manifest file)
- `train_mimic/scripts/data/precompute_dataset.py` converts a minimal dataset into a separate precomputed training dataset directory; `build_dataset.py` must not run precompute
- Each shard stores only `root_pos`, `root_quat_w`, `joint_pos`, `body_names`, and clip-aware window metadata (`clip_starts`, `clip_lengths`, `clip_fps`); long clips are split into overlapping bounded windows
- Training `motion_file` must point to a precomputed training dataset, not the minimal distributed dataset; training reads joint velocities and body FK/velocities from those precomputed shards and must not run MuJoCo FK while loading motion clips
- `MotionLib` loads all discovered precomputed HDF5 motion windows into CPU/GPU memory at startup
- `MotionLib` samples only valid center frames for the configured `window_steps`; default is `window_steps=[0]`
- Training supports `uniform` and `rewind` sampling over the fully loaded precomputed dataset
- `scripts/run/record_pico_motion.py` records Pico live body tracking as retargeted G1 motion NPZ clips in `data/pico_motion/clips/`; it opens a live `Retarget` viewer, uses terminal keys `R/S/D/N/Q`, stores semantic labels in filenames, and intentionally does not write per-clip JSON
- Build Pico-recorded clips into shards with `python train_mimic/scripts/data/build_dataset.py --spec data/pico_motion/pico_recorded.yaml --force`

Quick reference:

```bash
python train_mimic/scripts/data/build_dataset.py --spec train_mimic/configs/datasets/twist2.yaml
python scripts/run/record_pico_motion.py
python train_mimic/scripts/data/build_dataset.py --spec data/pico_motion/pico_recorded.yaml --force
python train_mimic/scripts/data/precompute_dataset.py data/datasets --outdir data/datasets_precomputed --jobs 8
python train_mimic/scripts/train.py --motion_file data/datasets_precomputed
python train_mimic/scripts/train_ladder.py --num_envs 4096 --max_iterations 60000
python train_mimic/scripts/benchmark_ladder.py --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt --num_envs 64
python train_mimic/scripts/record_ladder_video.py --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt --output ladder.mp4 --frames 1000
python train_mimic/scripts/data/precompute_dataset.py data/datasets/twist2 --outdir data/datasets/twist2_precomputed --jobs 8 --force
python train_mimic/scripts/save_onnx.py --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt --output policy.onnx --history_length 10
```

### GMR Retargeting
- Self-contained in `teleopit/retargeting/gmr/`; assets need `scripts/setup/download_assets.py --only robots gmr`
- Supports `lafan1` BVH (22 joints, 30fps, centimeters)
- Supports `hc_mocap` BVH (50 joints, 60fps downsampled to 30fps, meters)
- `lafan1-resolved` still needs an adapter layer and remains unsupported

### External Assets
- Do not commit robot meshes, datasets, checkpoints, or demo media to Git; use `scripts/setup/download_assets.py`
- `assets/robots/unitree_g1/g1_29dof.xml` and its meshes are the canonical G1 robot model assets; they are downloaded from the `robots` asset group and are not tracked in Git
- `teleopit/retargeting/gmr/assets/` is gitignored; downloaded at runtime
- `train_mimic/assets/` is no longer tracked; FK tooling reuses `assets/robots/unitree_g1/g1_29dof.xml`
- `third_party/linkerhand-python-sdk` and `third_party/somehand` support optional LinkerHand sim2real control
- Run `python scripts/dev/check_large_tracked_files.py` before pushing

Assets are split across two ModelScope repos by type:

| Repo | Type | Contents |
|------|------|----------|
| `BingqianWu/Teleopit-models` | model | checkpoints, GMR retargeting assets, sample BVH |
| `BingqianWu/Teleopit-datasets` | dataset | training/validation data shards |

Asset group → repo mapping is defined in `teleopit/runtime/external_assets.py` (`MODEL_REPO_ID` / `DATASET_REPO_ID`).

**Uploading a new release:**

```bash
# 1. Prepare upload directory
python scripts/setup/prepare_modelscope_assets.py --only ckpt robots gmr bvh --clean
python scripts/setup/prepare_modelscope_assets.py --only data

# 2. Upload to each repo
modelscope upload --repo-type model BingqianWu/Teleopit-models \
  data/modelscope_upload/checkpoints checkpoints
modelscope upload --repo-type model BingqianWu/Teleopit-models \
  data/modelscope_upload/archives archives
modelscope upload --repo-type dataset BingqianWu/Teleopit-datasets \
  data/modelscope_upload/data data

# 3. Tag the release on the model repo (match the Git tag; dataset repo does not support tags)
python -c "from modelscope.hub.api import HubApi; api=HubApi(); print(api.create_model_tag('BingqianWu/Teleopit-models', 'vX.Y.Z'))"
```

The old `BingqianWu/Teleopit-assets` repo is deprecated; do not upload to it.

### IK Offset Calibration
For each `(robot_body, human_bone)` pair, IK config stores a quaternion offset `R_offset` (`w,x,y,z`, scalar-first):

```
R_result = R_human * R_offset
R_offset = R_human_tpose^{-1} * R_robot_tpose
```

Critical note: align robot root orientation to the BVH human forward direction before computing `R_robot_tpose`. For `hc_mocap`, G1 default faces `+X` while the BVH human faces `-Y` (`Z-up`), so the robot root must receive a `-90°` Z rotation first.

`scripts/dev/compute_ik_offsets.py` can print or write calibrated offsets.

## Development

### Runtime Validation Policy
- Fail fast for logical mismatches such as observation definition vs. ONNX signature mismatch
- Do not silently pad, trim, clip, or replace invalid data/config to "make it run"
- Error messages should identify the mismatched components and the direct fix path

### Commit Policy
- Do not auto-commit changes
- Use the default git user as commit author
- After major feature changes, update `AGENTS.md` and `README.md` together with the code
- English docs (`docs/docs/`), Chinese docs (`docs/i18n/zh-Hans/`), and code implementation must stay in sync. Chinese docs are translations of the English originals — never generate Chinese content independently; always translate from the corresponding English page
- Documentation updates must be written for users and developers as stable product/development guidance, not as explanations of the current code patch or implementation diff

```bash
pip install -e .
pytest tests/ -v
```

## Known Issues

1. `lafan1-resolved` retargeting is still broken because it uses a different BVH skeleton layout.
2. Legacy downloaded GMR XMLs under `teleopit/retargeting/gmr/assets/unitree_g1/` are not the project entry point; use `assets/robots/unitree_g1/g1_29dof.xml`.

- Ladder progress and foot-placement rewards are registered separately for each of the five phases with weight `8` each. Their histories update for all environments before output masking, preserving the unsplit shaping sum across transitions and resets. Height and event bonuses remain shared; orientation remains stabilization-only. Self-collision sensor settings and ankle acceleration selection reuse tracking helpers.

- Stabilization has no survival reward. Missing required foot support is penalized each step; foot phases exclude the selected moving foot. Held-foot recovery uses signed distance reduction normalized by `2 * 0.35 m`, without a contact multiplier, and reanchors after reset or phase/held-rung changes. Stabilization violation is the mean squared normalized excess over the five existing speed/offset thresholds (torso linear speed, joint RMS speed, torso/pelvis angular speed, waist speed, support offset); valid gates cost zero. Existing dwell and transition gates remain unchanged.
