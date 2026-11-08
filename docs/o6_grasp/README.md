# Standalone LinkerHand O6 rung grasp

This CPU MuJoCo experiment runs one floating right O6 hand against one fixed rung.
It does not import the humanoid, hardware runtime, cameras, perception or training
stack. The deterministic baseline acquires, removes wrist support, loads, unloads
and releases. No hand–rung weld, attractive force or state correction is used.

## Run

From the repository root, using the existing `teleopit` conda environment:

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_grasp --viewer
conda run -n teleopit python -m teleopit.sim.o6_grasp --output outputs/o6_grasp/baseline
conda run -n teleopit python -m teleopit.sim.o6_grasp --suite --output outputs/o6_grasp/suite
conda run -n teleopit python -m teleopit.sim.o6_grasp --render --output outputs/o6_grasp/frames
conda run -n teleopit python -m pytest tests/test_o6_grasp.py -q
```

The viewer runs one 9.5-second simulation trial. With the viewer window focused,
press **Space** to pause/resume physics and controller time; camera controls and
the window remain responsive while paused. The wrist label shows `PAUSED`.
Contact points are enabled; contact frames/normals and force arrows are hidden
by default. Enable them manually under Rendering (Frame → Contact, Contact force).
An orange wrist marker means the fixture is enabled; green means its gain is zero.
The marker label shows the command mode, fixture gain and verification status.
Green alone is **not** a successful grasp. RELEASE remains excluded from holding.
`--render` exports four inspection images and does not require an interactive window.

`--config docs/o6_grasp/default.json` loads the complete reference configuration.
A JSON file may contain only the fields to override, for example:

```json
{
  "cross_section": "rectangular",
  "width": 0.024,
  "height": 0.020,
  "load_force": [0, 0, -2],
  "load_torque": [0, 0.1, 0],
  "load_profile": "ramp"
}
```

Unknown fields, invalid vectors, nonfinite values, invalid timing and targets beyond
joint limits are rejected. The bundled assets allow offline use after dependencies
are installed. Requirements are existing project dependencies: MuJoCo, NumPy and
SciPy; image exports additionally use imageio. Tested with MuJoCo 3.8.1 on Windows.

## Model and command interface

Source: [manufacturer right O6 URDF at pinned revision](https://github.com/linker-bot/linkerhand-urdf/tree/49f26021b0a91ff5c33150df22970c07bb548672/O6/right).
See [MODEL.md](MODEL.md) for the per-link mass/inertia and joint audit. The bundle
includes the upstream Apache-2.0 license and a SHA-256 manifest.

There are exactly six position commands, in the
[manufacturer Python SDK order](https://github.com/linker-bot/linkerhand-python-sdk):

| Index | Active joint | Local joint axis | Range (rad) |
|---:|---|---|---|
| 0 | `rh_thumb_cmc_pitch` | +X | [0, 0.58] |
| 1 | `rh_thumb_cmc_yaw` | -Z | [0, 1.36] |
| 2 | `rh_index_mcp_pitch` | +Y | [0, 1.60] |
| 3 | `rh_middle_mcp_pitch` | +Y | [0, 1.60] |
| 4 | `rh_ring_mcp_pitch` | +Y | [0, 1.60] |
| 5 | `rh_pinky_mcp_pitch` | +Y | [0, 1.60] |

The simulator uses **radians**, with lower limits open. The existing TeleopIt
hardware driver uses 0–255 SDK positions. No calibrated radian-to-byte conversion
is claimed or sent to hardware. The manufacturer's linker-sim itself labels its
linear byte conversion a placeholder. This environment preserves channel count
and ordering; a measured conversion is still required for deployment.

The five passive joints have no actuators. Equality constraints copy the URDF:
`thumb_ip = 1.86 * thumb_cmc_pitch`; each `*_dip = 0.89 * *_mcp_pitch`, zero offsets.
These are manufacturer **linear mimic approximations**, not a calibrated nonlinear
linkage model. Solver equality time constant is 4 ms with impedance
`[0.9999, 0.99999, 0.001]`; measured coupling error is logged, never corrected by
overwriting state during stepping.

All original link masses, center-of-mass offsets and full inertia tensors are
retained; total mass is 0.36998 kg. The source collision meshes are used as separate
convex pieces per articulated link. This retains the cavity between fingers and
palm. Only directly adjacent connected links are explicitly excluded; other
self-collisions, finger–rung and palm–rung collisions remain enabled. Rendering
uses separate noncolliding geoms made from the low-resolution source meshes,
not the manufacturer's much larger GLB visuals. Thumb tip and finger tip sites
are approximate diagnostic locations, not measured sensor frames.

**Uncalibrated dynamics:** all six servos use `kp=3 Nm/rad`, `kv=0.02 Nm s/rad`,
force range ±0.35 Nm; joint damping is 0.002 Nm s/rad and reflected armature is
0.0002 kg m². These are configurable simulation parameters selected for finite,
stable actuation. The URDF's uniform `effort=100` values are not adopted as
hardware torque limits. The command rate of 1 rad/s matches its velocity field,
which also needs physical confirmation. Friction 0.8, torsional friction 0.001 m,
contact time constant 6 ms and collision shape fidelity need identification.
No manufacturer payload claim is used as a climbing-load rating.

## Frames, fixture and loads

All lengths, times, forces and moments use SI units. World Z is upward; gravity
is `[0, 0, -9.81]`. Quaternions are **wxyz**. `wrist_mount` is the source hand base
origin; `palm_frame` has the same axes and an offset `[0, 0, 0.06]` m. The initial
`palm_pos` and `palm_quat` configure the **mounting origin**, despite their short
configuration names. Default mount position is `[0, 0, 0.5]` m.

The rung's local X is its longitudinal axis. Default rotation maps local X to
world Y, with center `[0.032, 0, 0.608]` m. Circular radius is 12 mm, length 300 mm.
For rectangular rungs, width is along local Y, height along local Z. `rung_frame`
is the rung center. All poses and dimensions are configurable.

The six-dimensional virtual fixture acts at `palm_frame` toward a fixed target.
Optional `fixture_target_pos` and `fixture_target_quat` specify its world pose;
otherwise the target equals the reset palm pose. Translation stiffness/damping
are 1500 N/m and 35 N s/m; rotation stiffness/damping are 8 Nm/rad and
0.3 Nm s/rad. Force and moment norms are capped at 30 N and 2 Nm. Rotation error
uses the shortest world-axis rotation vector. There is no fixture gravity
compensation, hidden attachment, fixed wrist or mocap body.

The OPEN/CLOSE/HOLD/RELEASE durations are 0.5/2.5/4/2.5 seconds. OPEN maintains the
open posture. CLOSE follows cubic smoothstep to the reference posture, then rate
limits each command. HOLD keeps bounded targets. In `evaluation_mode=unsupported`,
fixture stiffness and damping smoothly decrease to exactly zero during the first
second of HOLD. A control interval is marked unsupported only when **every**
physics substep has zero fixture gain in HOLD. `evaluation_mode=acquisition`
retains the fixture; those results can never count as unsupported success.

Loads start after the fixture-removal interval. `load_force` and `load_torque`
are expressed in `load_frame` (`world` or `palm`). `load_point` is an offset from
the palm site expressed in palm axes; its `r × F` moment is included. Constant,
cubic ramp and finite sin² pulse profiles are available. Pulse values are peak
N/Nm, not impulse integrals; integrate the logs for Ns/Nms. All profiles unload
smoothly during the final 0.5 seconds of HOLD. RELEASE starts with zero applied
load, smoothly opens, and ramps the fixture back as a catch. It never contributes
to load-bearing scores. Very severe failures can exceed the catch's capabilities.

Every physics substep clears and recomputes applied forces. Wrenches in logs are
`[Fx, Fy, Fz, Mx, My, Mz]`, in **world axes about the current palm_frame origin**.
The generalized free-joint moment also includes the palm-to-mount lever arm.

## Residual interface and actor observation

```python
import numpy as np
from teleopit.sim.o6_grasp import O6GraspEnv
from teleopit.sim.o6_grasp.observations import History

env = O6GraspEnv()
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(np.zeros(6))
# Optional: history_env = History(env), returning (10, 48).
```

This is a Gym-style tuple API with no Gym dependency. Completion is a time-limit
truncation; failed grasps continue through unloading and release for diagnosis.
Call `reset()` before stepping a completed episode. Each environment owns its
model, state, command queue, observation queue and random generator.

Actions are six finite values in [-1, 1]. Targets are
`rate_limit(clip(q_nominal + residual_bounds * action, limits))`. Offsets never
accumulate. Corrections operate only in CLOSE/HOLD; their last effective value
decays to zero during RELEASE, independent of new policy commands. The controller
alone authorizes release. The reference open/closed postures are fitted to the
nominal simulated rung; sizes are not automatically extrapolated. Actuator delay
is an integer number of control steps applied after target rate limiting.

`o6_actor_v1` is exactly 48 values; offsets are zero-based, end-exclusive:

| Slice | Field | Normalization |
|---|---|---|
| 0:6 | Active positions | Joint limits map to [-1, 1] |
| 6:12 | Active velocities | / 5 rad/s |
| 12:18 | Nominal minus measured position | / 1 rad |
| 18:24 | Previous normalized policy output | Unscaled; zero at reset |
| 24:27 | Rung position in palm frame | / 0.1 m |
| 27:33 | Rung orientation in palm frame | First column then second column |
| 33:36 | Palm linear velocity in palm axes | / 1 m/s |
| 36:39 | Palm angular velocity in palm axes | / 5 rad/s |
| 39:41 | Diameter/diameter or width/height | / 0.1 m |
| 41:43 | Circular/rectangular one-hot | Unscaled |
| 43:47 | OPEN/CLOSE/HOLD/RELEASE one-hot | Unscaled |
| 47:48 | Phase progress | Clipped [0, 1] |

Relative orientation is `R_palm.T @ R_rung`; flatten the first two columns in
Fortran order. Raw values are retained in `info['raw_observation']`; normalized
values before sensing effects are also logged. Soft physical limit violations
can put normalized positions outside [-1, 1]; these remain visible for diagnosis.

Relative pose and palm velocity are simulation truth and **require a sensing
and estimation plan on hardware**. Configurable Gaussian noise affects normalized
joint states, tracking errors, relative pose and velocities. Delay holds the
whole observation for N control steps; dropout holds the previous delivered
sample. These are simple synthetic corruption models, not hardware sensor models.
Reset clears all queues and previous actions. History is oldest-to-newest,
includes current observation, and initially repeats reset observation ten times.

Friction, contact state, slip labels, unmeasured load, passive states and fixture
wrench never enter the default actor. They are available through model/config or
diagnostic logs for a future privileged critic. Start a later RL experiment with
an MLP `[128, 128, 64]` and six bounded outputs; compare history/recurrent models
later. No training is needed here. Named reward terms cover acquisition, verified
support, slip, effort, action changes and authorized release. Holding/slip shaping
is disabled during RELEASE; release clearance is rewarded instead.

## Verification, numerical checks and outputs

HOLD never implies VERIFIED. Verification requires force-bearing contact on at
least two hand links including a distal finger, limited translation/rotation from
the end of acquisition, low tangential contact slip, bounded effort and limited
penetration. Defaults are 15 mm translation, 0.35 rad rotation, 0.02 m/s slip,
98% actuator force saturation, and 3 mm penetration, persisting for 0.3 seconds
without fixture support. Plausible regions are a coarse diagnostic, not a force
closure proof. Tangential slip uses the hand contact-point velocity relative to
the fixed rung. Small settling is allowed by these thresholds; sustained sliding
is separately labeled. Persistent violations or 80 mm pullout mark failure.
Failure reasons are included in every row.

An unsupported trial passes only with at least 1 second after verification,
no unsupported failure interval, no numerical warnings, no initial penetration,
and bounded penetration, coupling error and physical joint-limit violation.
Assisted contact duration and unsupported verified duration are reported separately.
Release requires an open measured posture, no rung contacts and at least 2 mm
geometric clearance; it does not count as support.

Default physics/control periods are 0.002/0.02 seconds, Newton solver, elliptic
friction cone, 100 iterations, tolerance 1e-10. These are simulation settings,
not verified hardware rates. The suite repeats the nominal trial at 0.001 seconds,
200 iterations and tolerance 1e-11; compare numerical and contact metrics, not
just a binary result. It also evaluates free motion, assisted acquisition,
fixture removal, small/large loads, moments along/across the rung, combined loads,
impulses, release, alignment, dimensions, friction, actuator delay and finite edges.

Each run writes resolved configuration, generated standalone XML, one JSONL row
per control step, and a summary JSON. The suite adds `summary.json`/`summary.md`.
Rows include measured/target joint states, efforts, palm and relative pose/velocity,
contact identity/location, contact-frame normal/tangential forces, world force,
net contact wrench, applied load and fixture wrench/gains, mode and verification.
MuJoCo contact forces act on geom2; the logger flips them as needed to report the
force **on the hand**. Contact tangential components retain the MuJoCo contact
frame convention. Wrench samples are from the final physics substep, not interval
averages. Mode/nominal/progress describe the issued control command; state time
is after its integration interval.

See [RESULTS.md](RESULTS.md) for measured outcomes and limitations, and
[WORK_LOG.md](WORK_LOG.md) for implementation and validation actions. Full logs and
images live under ignored `outputs/o6_grasp/`; compact results are committed.
