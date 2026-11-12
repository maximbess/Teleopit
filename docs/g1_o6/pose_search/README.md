# Bounded long-HOLD pose experiments

This campaign tests a coupled G1/O6 four-support stance: both feet stay on rung 04, with two physical O6 hands on their declared rungs. A successful stance is not independent hand capacity, hanging, climbing, or hardware qualification. Capacity and transfer prerequisites remain unchanged.

The requested plan is implemented on `dex-hand-exp`, using conda `teleopit`. The original working default, physical limits, solver settings and strict evaluator remain unchanged. The campaign specification was frozen before candidate dynamics in [campaign.json](campaign.json). Full results and interpretation are recorded in [COMPARISON.md](COMPARISON.md) and [results.json](results.json); the final interpretation follows completion of the bounded runs.

## Reproduction

Run from the repository root. Use a new output directory for an independent campaign; an existing campaign resumes completed trials without overwriting them or reselecting frozen parents.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search mandatory
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search screen
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search adaptive
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search controllers
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search promote-60
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search promote-120
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search qualify
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search robustness
conda run --no-capture-output -n teleopit python scripts/g1_o6_pose_report.py
```

The default output is `outputs/g1_o6/pose_search`. To reproduce one saved adaptive/controller candidate independently, register its full configuration first, then request eligible stages:

Use global option `--workers 1` through `--workers 6` before the subcommand to set independent trial concurrency; the default is three, with one ledger writer. This campaign uses four workers for the longer batches. Scheduling does not alter physics, timing or candidate identity.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search --output outputs/g1_o6/reproduce_pose register --candidate-config docs/g1_o6/pose_search/candidates/reference_hand_spacing_m_0p16.json
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search --output outputs/g1_o6/reproduce_pose trial reference_hand_spacing_m_0p16 screen
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search --output outputs/g1_o6/reproduce_pose trial reference_hand_spacing_m_0p16 60s
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.pose_search --output outputs/g1_o6/reproduce_pose trial reference_hand_spacing_m_0p16 120s
```

Stages `repeat` and `fine` require a qualified preceding 120 s result; `fine` uses 0.5 ms physics with unchanged 20 ms control. Robustness offsets require a matching nominal/repeat/fine qualified parent and change exactly one reset coordinate by +/-2 mm. Failed runs consume budget. A documented `--retry-reason` is accepted only after a technical `ERROR`, retains the original directory, and cannot retry a physical failure.

For the mandatory geometry-only comparison through the original runner:

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support long-hold --seconds 30 --grasp-config docs/g1_o6/pose_search/rung10.json --output outputs/g1_o6/rung10_independent
```

This last command uses the original telemetry. Use the campaign runner for full anatomical/event diagnostics.

## What is measured

- `resolved.json`: exact pose/controller/seed, requested timing, actual rung transforms, arm IK, actual mount separation, physical asset hashes, code revision and dirty diff. Spacing is a signed offset per hand; grip offset is world XYZ; tilt changes the target translation as well as orientation.
- `steps.jsonl`: full control-rate state, raw/clipped/rate-limited/applied commands, active/passive angles, travel/work, contacts, all four signed reactions, wrist wrench, scene clearances and arm/controller state.
- `physics_events.csv`, `all_phase_events.json`, `contact_episodes.json`: complete event times, durations and extrema. Anatomical contact groups distinguish geometric presence, loaded contact (0.05 N) and valid thumb-specific opposition. Solver contact-ID changes do not define a support loss.
- `physics_windows.jsonl.gz`: detailed 1 ms (or 0.5 ms) samples spanning 0.25 s before/after HOLD entry and first contact/threshold events. Actual applied-force arrays are audited at every scored physics step. Native contact forces can be one physics interval old; telemetry never advances or writes the live simulation.
- `summary.json`: separate strict, late-motion, contact-quality and interval-qualification verdicts. The RESET reference remains fixed. Final qualification additionally requires distinct fresh nominal and fine-step runs with the same candidate identity.

Wrist forces/moments are parent-on-hand in world axes, with the moment at the current wrist mount. Generalized actuator work uses `qfrc_actuator[dof] * qvel[dof]`. Tangential utilization is the translational elliptical ratio; torsional utilization is separate, and neither is claimed as the full contact-cone margin. Contact-plus-gravity residuals are static residuals, not inertial-corrected errors.

Contact quality requires every scored control sample to preserve thumb opposition and thumb-below/all-fingers-above ordering, with no physics-rate invalid episode lasting 0.5 s. Shorter interruptions and contact minima are still reported. Comparisons flag a hand's signed load below 50% of the matched reference 20–30 s window; short aborted trials have no matched-window claim.

## Controller identity

Reference/preload/closing-target variants retain the original measured-angle finger law and serialize their different parameters. Finger latches/caps use `o6_hold_reference_v1`; bounded arm feedback uses `g1_o6_bounded_arm_reference_v1`. Both new laws are explicitly rejected by isolated diagnostic runners that do not implement them. No pose result grants capacity or transfer eligibility.

Conditional arm feedback replaces one higher-preload slot only when saved reference evidence shows wrist movement after the thumb target becomes stationary. It tracks the HOLD-entry rung-relative mount pose using scratch-data Jacobians and original arm actuators, bounded to 0.03 rad and 0.005 rad/s with limit freezing. It adds no Cartesian support force. This is a coupled arm-and-hand intervention, not independent retention.

## Evidence and verification

[WORK_LOG.md](WORK_LOG.md) records actions, hypotheses, tests and refactoring receipts. [baseline_verification.json](baseline_verification.json) pins all 1,753 original physical baseline rows and the strict summary; [projection_verification.json](projection_verification.json) pins strict/late-motion results after reducing in-memory telemetry.

Representative synchronized plots show complete trials, HOLD entry and first threshold windows. Videos under the archive's `outputs/g1_o6/pose_search/videos` are explicitly labeled **offline recorded-state replay**: they use saved positions and never step or qualify dynamics. The portable archive retains failed/infeasible results, original scenes/manifests, compiled-equivalent portable scenes, exact source revisions, assets and SHA256 checksums. Previous archives remain intact.
