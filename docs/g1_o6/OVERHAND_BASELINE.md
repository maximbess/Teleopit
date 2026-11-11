> Historical report preserved from f5c7067fb9e640a17501764df4eaf818679a146d. Its commands require that revision. See the [current quickstart](README.md) and [history index](HISTORY.md).

# Physical G1 / O6 ladder prototype

This CPU prototype demonstrates load transmission through two physical O6 hands, but **does not pass unsupported whole-body holding**. The nominal run aborts after 0.50 s without assistance, against a 10 s target. Both hands and feet still carry load; finger saturation, thumb-limit deflection and body drift fail the acceptance gates. Release and reacquisition remain blocked. Do not treat this as successful physical climbing or readiness for learning.

## Reproduce

Run from the repository root on `dex-hand-exp`, using the `teleopit` conda environment. The reviewed baseline was `9c7a16bd8c0c680dfc7aa1ba0cc151a829bfef1b`; the checkout contained no newer commits or changes when work began.

```powershell
# Only if the canonical G1 / collision assets are missing:
conda run --no-capture-output -n teleopit python scripts/setup/install_climb_assets.py

# Headless nominal, interactive viewer (Space pauses/resumes), and video:
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --config docs/g1_o6/nominal.json --output outputs/g1_o6/nominal
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --viewer --output outputs/g1_o6/viewer
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --video --output outputs/g1_o6/video

# Fixed eight-case suite; no search or automatic threshold relaxation:
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --suite --config docs/g1_o6/nominal.json --output outputs/g1_o6/verified_suite
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --video --config docs/g1_o6/friction_low.json --output outputs/g1_o6/video_friction_low
```

Each trial writes `resolved.json`, `scene.xml`, `steps.jsonl`, `summary.json`, and both `left.wrist.json` and `right.wrist.json`. Video mode adds `trial.mp4`, `initial.png`, and `last.png`. Viewer mode uses the same logging runner; closing it records an abort. Contact-force/normal arrows default off. Video rendering uses ImageIO/FFmpeg and Pillow, available in the tested environment.

Measured suite: code `4fe09d98cacd802b23dc634f67ef4c56a47adfe2`, clean tracked checkout. Video camera/terminal-frame improvements are in `5ab9e89`; later changes add validation and this report. Full installed asset hashes, actuator inventory and runtime versions are in [audit.json](audit.json). Resolved parameters are frozen in [nominal.json](nominal.json), and measured numerical results in [results.json](results.json).

## Assets and composition

- Reused the canonical assets already installed by the repository setup workflow. The distribution is [12e21/Teleopit-models](https://modelscope.cn/models/12e21/Teleopit-models); it does not supply an upstream G1 commit identifier. The exact installed XML and mesh hashes are recorded rather than inventing a revision. Missing canonical XML/meshes produce an installation command; the collision overlay rejects missing, corrupt or mismatched assets.
- Collision surfaces use [amazon-far/holosoma](https://github.com/amazon-far/holosoma/tree/bccd4d7451640a2800ddc77e469d911a84f91994), revision `bccd4d7451640a2800ddc77e469d911a84f91994`. The existing pinned convex-decomposition manifest and hashes remain authoritative.
- Both hands use [linker-bot/linkerhand-urdf](https://github.com/linker-bot/linkerhand-urdf/tree/49f26021b0a91ff5c33150df22970c07bb548672/O6), revision `49f26021b0a91ff5c33150df22970c07bb548672`. The authoritative left URDF and collision meshes were downloaded and bundled with their SHA256 manifest. Left thumb axes differ from the right; this is not a renamed right hand. All bundled source hashes were verified.
- `teleopit.sim.o6_hand` owns URDF translation, namespacing, active/passive mappings, mimic constraints, inertia, collision geometry, servos and optional free wrist. It contains no rung, fixture, replay or episode timing. The standalone right-hand scene uses this builder unchanged physically.
- Each O6 base is a jointless child of its corresponding G1 wrist-yaw link. Wrist-to-hand transform: position `(0.0415, 0, 0)` m, quaternion WXYZ `(sqrt(0.5), 0, sqrt(0.5), 0)`, so O6 +Z points along wrist +X. This reuses the existing fixed-O6 asset convention; a physical adapter still needs calibration. Mount, palm and rung frames remain separate.
- One free G1 base, 29 body actuators, 12 active finger actuators, 10 passive mimic joints/constraints, one ground plane, and the full existing ladder arrangement. There are no hand free joints, grip welds, mocap support anchors, standalone wrist fixtures or replay forces in the coupled model.

The old rubber-hand visuals and simplified hand collisions are removed. Each original wrist-yaw body has **0.254576 kg** aggregated wrist/hand mass; the source has no separate rubber-hand inertia. That aggregate is retained, with its inertial frame and principal moments recorded. No guessed mass is subtracted. Thus some original hand mass may be counted twice; adapter/hardware mass is not calibrated.

| Quantity | Canonical G1 | With two O6 hands |
|---|---:|---:|
| Robot mass, excluding fixed ladder | 33.341142 kg | 34.081102 kg |
| COM at the same canonical zero-joint/base pose, world XYZ | (0.02033208, 0.00008226, 0.67133407) m | (0.02654791, 0.00008058, 0.67522897) m |

Added mass is 0.739960 kg. Initial coupled-pose COM is `(-0.76467531, 0.00008003, 1.48882503)` m. Mass comes from the **pelvis subtree**, not the sum including fixed ladder bodies.

## Contacts and control

Physics is 1 ms, control 20 ms, implicit-fast/Newton, elliptic cone, 200 iterations and tolerance `1e-11`. Body gains and effort limits come from the canonical G1 XML and are listed per actuator in the audit. Finger parameters preserve the standalone model, including its provisional ±0.35 Nm servo limit. Joint target changes are limited to 1 rad/s and clipped to physical joint ranges; compiled effort limits remain enabled.

The near/negative-X ladder face and ground collide with robot geometry. The far face remains visual-only, matching the existing ladder task. Robot and O6 collision masks are `contype=2, conaffinity=3`. Visual geoms do not collide. Collision dimensions are explicitly 4 with priority 0; both material partners are scaled in the low-friction case. Actual hand/rung contact friction is 1.8 nominal and 0.25 in the low-friction case. XML round-trip tests verify these settings survive export.

Connected hand links, MuJoCo parent/child pairs, inherited knee/foot-shell pairs and the O6 base against its own wrist roll/pitch housing are excluded. Other nonadjacent self-collisions, hand/robot contacts, rails and nearby front rungs remain active.

Both hands target rung 08 at Y=±0.18 m; both feet target rung 04. Bounded multistart arm IK finds an overhand approach, while leg IK preserves the intended feet. The initial robot/feet are translated 3 mm away from the original pose to remove knee overlap. Maximum hand position error is 1.702 mm and angular error 0.01349 rad; initial penetration is zero. Earlier upright/crouched attempts were rejected as unreachable or penetrating.

`INITIALIZE → ACQUIRE (3 s) → REMOVE_ASSISTANCE (2 s) → HOLD (10 s)` uses a logged pelvis wrench during acquisition: spring 3000 N/m, linear damping 150 Ns/m, rotational spring 200 Nm/rad, damping 20 Nms/rad, plus gravity compensation. Norm limits are 600 N and 150 Nm. A smooth ramp removes it; during HOLD it is exactly zero. Physics alone updates positions/velocities during the scored interval.

Predeclared gates: base drift <30 mm, base angle <0.25 rad, support slip <20 mm/s, penetration <3 mm, joint-limit error <0.02 rad, mimic error <0.015 rad, effort fraction <0.98, base speeds <0.05 m/s and <0.2 rad/s, external-force residual <25 N, no warnings or unintended supports. Every intended support must stay loaded; hands also need opposing normals. A persistent 0.5 s failure aborts; the full 10 s must pass before release.

Release/reacquisition code is guarded but **not mechanically validated**:

`HOLD → UNLOAD_SELECTED_HAND → OPEN → RETRACT → HOLD_REMAINING_SUPPORTS → RETURN → REACQUIRE`.

Unloading uses a bounded 0.04 rad waist-roll candidate, requiring selected-wrist force <8 N and moment <0.4 Nm plus stable remaining supports for 0.5 s. Timeout is 5 s. Opening takes 2 s; a bounded shoulder retraction takes 3 s and must establish at least 10 mm geometric clearance. Remaining-support hold lasts 3 s. Returning/reclosing cannot mark support valid: opposing loaded contacts with bounded slip must persist for 0.5 s. No trial passed the prior static gate, so neither side opened and no release load-redistribution result is claimed.

## Diagnostics and measured outcome

Wrist sensors measure the **parent-on-hand wrench**, in mount axes, about the mount origin. Logs rotate both vectors to world axes without changing that reference point. A controlled 1 kg child, offset COM and known extra force verifies force sign and moment origin, including a rotated sensor. This follows MuJoCo's [force sensor definition](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-force).

Contact rows identify both bodies/geoms. Only contacts with the intended rung count as hand/foot support; contact with another hand, the torso, a rail or another rung cannot masquerade as a grasp. Whole-body balance sums external robot/environment contact forces and gravity, **excluding internal wrist forces and robot self-contacts**. Logs include slip, body/finger effort, COM/base state, hand/rung pose, joint errors, penetration, mimic error, warnings, assistance and unintended support.

At nominal abort (5.54 s), all intended supports remain loaded, with opposing hand contacts and no unintended support. Upward forces are 28.26/27.79 N at the hands and 138.26/142.44 N at the feet. Weight is 334.34 N; the force-balance residual norm is 3.26 N. Wrist-force peaks are 30.88/30.74 N; mount-moment peaks are 1.831/1.824 Nm.

Both thumb-yaw and index servos saturate at 0.35 Nm. Thumb IP limit violations reach 0.03197/0.03406 rad; thumb pitch also deflects past its limit. Base drift reaches 94.71 mm. Peak body effort is 31.45% over the entire trial, and 20.31% during HOLD, so G1 arm torque saturation is not the observed bottleneck. Maximum penetration is 0.331 mm and mimic error 0.00007255 rad. The limiting evidence points to loaded grasp geometry/finger torque and body posture; hardware strength and contact validity cannot be inferred from these provisional parameters.

| Scenario | Base drift at abort | Max penetration | Outcome / failed gates |
|---|---:|---:|---|
| Nominal | 94.71 mm | 0.331 mm | Drift, thumb limits, finger effort, velocity |
| Physics 0.5 ms | 96.59 mm | 0.316 mm | Same failure categories |
| Body X +2 mm, feet preserved by IK | 87.76 mm | 0.337 mm | Drift, thumb limits, finger effort |
| Wrist target X +2 mm | 94.28 mm | 0.334 mm | Drift, thumb limits, finger effort |
| Effective friction 0.25 | 193.69 mm | 0.319 mm | Drift, velocities, force balance, loss/slip of intended supports |
| 10 N pelvis X disturbance for first 0.2 s of HOLD | 77.79 mm | 0.331 mm | Drift, thumb limits, finger effort, velocity |
| Two control-step actuator delay (40 ms) | 94.66 mm | 0.330 mm | Drift, thumb limits, finger effort, velocity |
| Right-hand release selected | 94.71 mm | 0.331 mm | Same nominal hold failure; release not reached |

All eight abort after 0.50 s of zero-assistance HOLD; none has a numerical warning or unintended support during that interval. Nominal and finer-step outcomes are qualitatively the same. This small suite establishes repeatability of a failure, not robustness of a successful controller. Generated local artifacts: [nominal video](../../outputs/g1_o6/video/trial.mp4), [low-friction video](../../outputs/g1_o6/video_friction_low/trial.mp4), and [diagnostic plot](../../outputs/g1_o6/nominal_diagnostics.png).

## Wrist replay

All 16 exported traces load through `WristLoadReplay`, preserve the recorded timestamps and signed components, and declare `source.kind="simulation"`, commit/configuration/side/trial, world axes, application site `wrist_mount`, and zero site-relative application offset. Hand gravity is not added to the exported wrench. Coupled wrist measurements already describe the force exerted by the parent on the hand; the standalone physics still applies its ordinary gravity.

Replay the two right-hand examples with the logged initial overhand pose, translated to the standalone rung. The pose is supplied explicitly in [right_replay_pose.json](right_replay_pose.json); no sign or coordinate-axis guess is needed.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --config docs/g1_o6/right_replay_pose.json --wrist-load outputs/g1_o6/verified_suite/nominal/right.wrist.json --replay-fade 0 --output outputs/g1_o6/replay_matched_nominal
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --config docs/g1_o6/right_replay_pose.json --wrist-load outputs/g1_o6/verified_suite/fine_step/right.wrist.json --replay-fade 0 --output outputs/g1_o6/replay_matched_fine
```

Both replays fail, with zero verified seconds and numerical warnings after losing the grasp; active replay time is 4.88/4.84 s of the 5.54 s traces. They do not validate the entire load history. Using the older upright bench orientation fails still earlier. The standalone entry point currently represents a right hand: left traces are exported and parser-checked, not falsely replayed using a right-hand model. Replay lacks the closed-loop humanoid dynamics; the warning-free coupled failure is the authoritative result.

## Validation and remaining work

- 68 standalone/component tests pass, including the long unsupported gravity holds. The original standalone viewer/Space behavior remains usable.
- 25 coupled tests cover composition, reachable nonpenetrating reset, actuator independence/bounds, sensor sign/origin, effective saved friction, contact classification, internal-force exclusion, signed replay export, release/reacquisition authorization, geometric clearance, symmetric target motion and viewer pause/close.
- Eight relevant training composition/collision checks pass. `test_refined_initial_contacts_and_mesh_budget` fails on the existing knee/rung overlap; executing the complete reviewed-baseline environment implementation in the same pytest runtime reproduces the same contacts. The prototype corrects its own reset; the weld-training reset is intentionally preserved.
- Ruff passes. Combined test/scenario/video coverage is approximately 97%. Named crapper/dryer/mutator tools were unavailable. Transparent substitutes: Radon plus measured line coverage gives maximum CRAP 9.375 and complexity 9; AST function-body token comparison finds no similarity ≥0.8 among bodies of at least 50 tokens; four targeted behavioral fault injections are killed. These are scoped checks, not claims of exhaustive mutation coverage.

```powershell
conda run --no-capture-output -n teleopit python -m pytest tests/test_o6_grasp.py tests/test_o6_ladder_hold.py tests/test_o6_wrist_replay.py tests/test_o6_viewer.py tests/test_o6_component.py tests/test_g1_o6.py tests/test_g1_o6_controller.py tests/test_g1_o6_viewer.py -q --basetemp outputs/pytest_g1_o6
```

Next work should optimize the loaded overhand posture and finger targets while retaining the declared friction/effort limits, then repeat the static gate before attempting release. Measure O6 torque limits, transmission/mimic compliance, contact materials and the adapter transform/inertia; separate original hand inertia from the G1 wrist if authoritative data becomes available. The current controller has no evidence supporting PPO, residual hand policies, a GPU backend port, hardware use or full ladder ascent. See [WORK_LOG.md](WORK_LOG.md) for the implementation and validation record.
