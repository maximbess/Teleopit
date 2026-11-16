# Native G6 pose, load and hand-transfer experiments

This opt-in campaign implements [the supplied plan](PLAN.md) on `dex-hand-exp`.
It starts from reviewed commit `298cd2f555dcf48637384987ef0668e85f219392`.
The existing default remains unchanged. Physical milestones are reported separately
from software checks in `outputs/g1_o6/g6_stability/milestones.json`.

## Run

Use the repository root and the `teleopit` Conda environment. Native MuJoCo on
this Windows machine requires environment activation; use `conda run`.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_stability status
# Fresh output directory: reproduce/import the reviewed control, then freeze code.
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_stability prepare
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_stability import-control outputs/g1_o6/g6_stability/reference
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_stability --workers 3 batch A --parent control
# Choose the next frozen parent from measured results; repeat for B, C and D.
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_stability promote
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_stability downstream
```

`search` runs A–D automatically, then eligible promotions and downstream stages.
`trial <candidate> <stage>` resumes an identical completed request without
repeating it. `retry <candidate> <stage> --reason "..."` permits only technical
ERROR/interruption recovery with unchanged source identity and a new run ID.
A failed physical trial cannot be retried through that path. All invocations
hold the existing single-writer campaign lock. The exact actual batch commands,
parents, source bytes, asset hashes and run checksums are retained in the ledger.
Use its recorded source revision and byte snapshot when resuming after later
repository edits; changing code or assets intentionally invalidates resume.

## Frozen search

The native control is hands 8 / feet 3. G6 dimensions, material, gravity, masses,
inertias, actuator limits, collision masks and solver settings remain unchanged.
All search candidates use the original `HoldController`, 1 ms physics and 20 ms
control. Scored assistance is zero. A 0.5 ms winner confirmation retains 20 ms
control. The maximum is 40 distinct configurations including the control.

| Batch | Count | Changes relative to the preceding selected parent |
|---|---:|---|
| A | 12 | Root X or Z ±20/40 mm; waist ±0.05/0.10 rad |
| B | 8 | Hands/feet 7/2, 7/3, 6/1, 6/2; ±10 mm foot X on 7/2 and 6/1 |
| C | 12 | Tilt ±5/10°; grip X/Z ±5 mm; per-hand lateral offset ±20 mm; right grip Z ±5 mm |
| D | 7 | Preload 0.10/0.14 Nm; thumb closure ±0.1/0.2 rad; finger closure +0.1 rad |

Each batch freezes its parent and full values before execution. Reset IK,
physical joint margins and initial interpenetration are checked first. Feasible
candidates get acquisition plus a 5 s probe; only full strict/contact passes
proceed to a fresh 30 s screen. Promotions are bounded to eight 60 s and four
120 s trials. A 120 s convergence pass earns a fresh nominal repeat and finer
step confirmation. Ranking uses physical/contact validity, measured short-window
base slope, clearance, effort and next-tread endpoint reachability. Endpoint IK
is diagnostic; it does not certify a swept transfer.

## Evidence contracts

- **Stable initial pose:** 30 → 60 → 120 s, then 120 s nominal repeat and 0.5 ms
  confirmation; existing strict/contact/final-30-second convergence gates.
- **Capacity:** actual sampled G6 state and source tread; free wrist, gravity,
  zero-fixture retention, force/moment/combined ramps, plateaus and recovery.
  Simultaneous sample directions require 1.2 reserve. Bounds are declared from
  measured demands and planned 16% per-hand support; no actuator strength changes.
  Each side has a declared 192-trial budget. Untested states and out-of-bound
  cases remain uncovered. Failed larger diagnostic loads do not invalidate a
  smaller proven reserve; the report separates trial outcomes from capacity.
- **Loading:** signed external hand Fz divided by compiled robot weight, in
  ordered 5/10/20/30% trials; total tolerance ±1 percentage point, symmetric
  per-hand tolerance ±1 point and independently tested upper bounds. The final
  30% plateau progresses 10 → 30 → 60 s, then nominal/fine repeats. New measured
  critical states require capacity tests before promotion.
- **Release:** both independent closed-finger unloads, then 1/3/10 s detached
  three-support holds, then one/three/alternating same-tread cycles. Lost hand
  load is assigned to both feet, subject to measured compensation guards.
- **Adjacent transfer:** explicit higher G6 tread, first-hand mixed-height
  qualification before the second hand, both starting orders, three nominal
  repeats per order and one fine confirmation per order. Final recovery is
  30 s at measured 30% with convergence and capacity coverage.

Moving-hand phases are OPEN, RETRACT, THREE_SUPPORT_HOLD, REACH, APPROACH, CLOSE
and VERIFY_GRASP. Only the selected hand is exempt from stationary anatomical
opposition during those phases. Both feet and the other hand remain fixed;
global RESET drift, physical limits and assistance checks remain active. Full
opposition returns on reload. Reach planning uses scratch data at at most 5 mm
Cartesian spacing, checks hand/wrist/forearm against the scene, and executes
through arm target increments bounded to 0.05 rad/s. It never writes live qpos.

RESET and HOLD-entry references are retained independently. Per-physics-step
contacts include effective friction, and actual applied-force arrays are saved
for every scored step. Body-clearance diagnostics cover all 20 treads/rails for
each knee, shin, housing, forearm and torso. Authentic knee-link mesh centroids
split knee/shin regions at local Z = −0.1 m; this is an observation convention,
not a geometry change. Distant clearances are capped at 100 mm, unknown exact
zero distances are explicitly uncertified. Dynamic balance uses COM Jacobian
velocity differences and external contact force; internal wrist forces are
not double-counted.

FAIL is a measured physical failure, BLOCKED means a prerequisite or coverage
is missing, NOT RUN means no eligible attempt, and ERROR is a technical failure.
Recorded-state videos use separate display data and never qualify physics.
See [WORK_LOG.md](WORK_LOG.md) for implementation and validation history.
