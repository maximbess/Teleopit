# Native G6 pose, load and hand-transfer experiments

The latest right-thumb study includes native-step measurements, a baseline comparison and the current milestone results. [G6 thumb retention](../g6_thumb_retention/README.md)

This opt-in campaign implements [the supplied plan](PLAN.md) on `dex-hand-exp`.
It starts from reviewed commit `298cd2f555dcf48637384987ef0668e85f219392`.
The existing default remains unchanged. Physical milestones are reported separately
from software checks in `outputs/g1_o6/g6_stability/milestones.json`.
The completed 48-configuration search found no qualifying pose; see
[OUTCOMES.md](https://github.com/maximbess/Teleopit/blob/f27fa4c/docs/g1_o6/g6_stability/OUTCOMES.md) and [CANDIDATES.csv](https://github.com/maximbess/Teleopit/blob/f27fa4c/docs/g1_o6/g6_stability/CANDIDATES.csv).

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
control. The initial campaign maximum is 40 distinct configurations including
the control. A separate eight-case follow-up is recorded under `second_batch`;
it does not alter the original budget or failed runs.

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
See [WORK_LOG.md](https://github.com/maximbess/Teleopit/blob/f27fa4c/docs/g1_o6/g6_stability/WORK_LOG.md) for implementation and validation history.

## Delivered evidence and reproduction

The local ZIP and its SHA-256 are listed in [artifact.json](https://github.com/maximbess/Teleopit/blob/f27fa4c/docs/g1_o6/g6_stability/artifact.json).
That outer receipt is committed separately from the ZIP; the archive itself
contains `evidence_manifest.json` with every included file checksum.
Extract it at one root so relative asset paths remain intact. Each run contains
immutable `scene.xml` / `scene.mjb`, resolved configuration, summaries and raw
traces. Use the additional `scene_portable.xml` after extraction. The bundle in
`outputs/g1_o6/g6_stability/source/dex-hand-exp.bundle` contains the source
history; exact runtime byte snapshots preserve line endings used by the hashes.
Do not modify or resume the completed ledgers with a later source revision.

For an exact physical reproduction, use a separate checkout of the recorded
runtime revision with the archived physical assets and its matching source ZIP.
The native control was run at `298cd2f`; the search at `180b0da`. The archive's
`reproduce_reference_portable.py` reads the complete frozen reference payload
without requiring the previous ladder/material campaign directory. Choose a
fresh output path:

```powershell
$env:PYTHONPATH = (Get-Location).Path
# At the reviewed control source, with the archived driver and payload available:
conda run --no-capture-output -n teleopit python outputs/g1_o6/g6_stability/reproduce_reference_portable.py --source outputs/g1_o6/g6_stability/reference_source.json --output outputs/g1_o6/g6_reproduction/reference
# At the frozen search source, use the fresh campaign path for all commands:
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_reproduction prepare
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_reproduction import-control outputs/g1_o6/g6_reproduction/reference
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_reproduction --workers 3 batch A --parent control
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_reproduction --workers 3 batch B --parent a_03
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_reproduction --workers 3 batch C --parent b_00
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_experiment --output outputs/g1_o6/g6_reproduction --workers 3 batch D --parent c_05
```

The archived `run_second_batch.py` records all eight E combinations, their
hypothesis and an explicit cap of eight before execution. Its `ROOT` points to
the original campaign; set that data path to the fresh reproduction campaign
when repeating it. `second_batch/campaign_state.json` retains the exact values.
No stage is credited merely because a replay or resume command succeeds.

The following commands were used for reporting and independent validation at
the delivery source. Verification needs the bundled Git history for controller
identity, the assets, and the reviewed numeric model snapshot in each `source`
directory. The verifier uses the additional portable XML when present, so it
can load relocated scenes without changing raw evidence. The archive manifest verifier
checks raw file bytes without loading any model or depending on those paths.

```powershell
$env:PYTHONPATH = (Get-Location).Path
conda run --no-capture-output -n teleopit python scripts/g1_o6_g6_report.py --root outputs/g1_o6/g6_stability
conda run --no-capture-output -n teleopit python scripts/g1_o6_g6_report.py --root outputs/g1_o6/g6_stability/second_batch
conda run --no-capture-output -n teleopit python scripts/verify_g1_o6_g6.py --root outputs/g1_o6/g6_stability
conda run --no-capture-output -n teleopit python scripts/verify_g1_o6_g6.py --root outputs/g1_o6/g6_stability/second_batch
conda run --no-capture-output -n teleopit python -m pytest tests/test_g1_o6_g6_evidence.py tests/test_g1_o6_ladder_evidence.py -q
conda run --no-capture-output -n teleopit python -c "from pathlib import Path; from scripts.package_g1_o6_support import verify; print(len(verify(Path('outputs/g1_o6/g6_stability_evidence.zip'))['files_sha256']))"
```

The ZIP stores every source file needed by these
commands in its provenance snapshots and Git bundle, rather than relying on an
external GitHub download. Restore the delivery source in a separate workspace
before running the analysis scripts. Exact runtime versions are recorded in
the campaign source provenance. `validation/mutations` retains both the initial
uncollected mutation and its corrected assertion failure; the consolidated
receipt explicitly chooses the corrected result.
