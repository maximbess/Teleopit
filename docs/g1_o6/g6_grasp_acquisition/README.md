# Bounded native G6 grasp acquisition

Completed outcome: **thumb opposition established; stable G1 pose not qualified**.
See [measured outcomes and evidence](OUTCOMES.md) and [work log](WORK_LOG.md).

This experiment keeps hands on tread 7, feet on tread 2, root X −40 mm and
the original G6 geometry/material. It changes named thumb targets and, for
selected variants, closure timing. Original physical/contact/convergence
gates, preload, actuator limits, collision settings and assistance schedule
remain in force. Previous experiments and global defaults are preserved.

The native hand command order is **thumb pitch, thumb yaw, index, middle,
ring, pinky**. Targets are assigned by `ACTIVE_NAMES`, with thumb values
checked against the native URDF. The previous control uses pitch 0.05 rad
and yaw 0.5 rad. Its actual pitch at removal is about 0.041 rad. The finite
scratch scan is geometric evidence only; dynamic contact forces decide success.

| Pair / candidate | Thumb pitch (rad) | Thumb yaw (rad) | Acquisition timing |
|---|---:|---:|---|
| control | 0.05 | 0.5 | Original simultaneous closure |
| pitch_025 | 0.25 | 0.5 | Original simultaneous closure |
| pitch_035 | 0.35 | 0.5 | Original simultaneous closure |
| first_025 | 0.25 | 0.5 | Thumb first |
| first_035 | 0.35 | 0.5 | Thumb first |
| yaw_020 | 0.20 | 1.0 | Thumb first |
| yaw_030 | 0.30 | 1.0 | Thumb first |

All four fingers retain 1.35 rad targets, the original 0.12 Nm bounded closing
law and the 1 rad/s servo target limiter. Thumb-first timing ramps thumbs over
the first 1.5 s; fingers start at 0.75 s and finish at 3 s. Both schedules
retain the original 3 s acquisition and 2 s assistance removal. Scored HOLD
always has zero applied assistance.

Each later pair requires two failed candidates from its preceding pair and
an immutable audit of the premise, measured actors and removed assignment.
There are at most six acquisition variants, not an open parameter sweep.
If an acquired grasp needs stabilization, select its parent once and create
at most two additional candidates at 0.005/0.020 rad/s. These reuse the original
5 mm translation, 0.03 rad rotation and 0.03 rad joint-offset bounds, and latch
the measured wrist reference **at HOLD entry**. They do not retain an invalid
earlier removal-entry reference.

The acquisition observer records both-hand anatomical validity, actual thumb
forces, structural/slip violations and continuous valid tails at each phase.
Readiness for removal and HOLD are separate observations. The stabilizer
parent must have either a continuous valid final 0.5 s of removal or an
original contact PASS over at least 0.5 s of unsupported HOLD, plus native
friction, no applied assistance and no housing/prohibited support. A single
contact or a requested target cannot establish this prerequisite.

Physical qualification is unchanged: fresh **5 s probe → 30 → 60 → 120 s**,
then a fresh nominal 120 s repeat and a 0.5 ms confirmation. All 120 s runs
must also pass the original final-30-second convergence checks. Budgets are
9 probes including control, 8 screens, 4 sixty-second and 2 initial
120-second trials, then one repeat and one fine-step trial. Unrun cases remain
explicitly unrun. Failed physical trials cannot be silently repeated.

From the repository root, with the `teleopit` Conda environment:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment prepare
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment trial control probe
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment pair pitch
# Inspect completed evidence before authoring a premise/actor audit.
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment audit pitch path/to/audit_pitch.json
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment pair first
# The yaw pair similarly requires audit first. Stabilization requires an acquired parent.
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment stabilize
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment trial candidate_id 60s
python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment status
```

Use `--output <fresh-directory>` before the subcommand for a new campaign.
The ledger rejects changed source, assets, candidates and invalid promotion.
Technical ERROR/interruption recovery uses `retry <candidate> <stage> --reason
"..."` with identical source identity. Reproduction requires the recorded
revision, exact source byte snapshot and runtime, not a later checkout.

`scripts/verify_g1_o6_g6.py` checks original physics, material, forces, files
and scene/source identity. `scripts/g1_o6_g6_acquisition_report.py` independently
reconstructs thumb/finger opposition and world-height ordering from the native
physics contact stream, checks actual target rates and bounds, and produces
full-phase plots. See [WORK_LOG.md](WORK_LOG.md) for implementation and outcomes.
