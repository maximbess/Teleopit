# Posture consolidation validation

The single default reproduces the qualified balanced grasp exactly. All four required holding trials pass with zero numerical differences in complete trajectories, summaries, phase timing, support forces, wrist wrenches, drift and effort. This remains a four-contact stance with most weight on the feet; release, hanging, climbing and hardware capability are unvalidated.

## Revisions and runtime

References were captured before edits at `b34ba73683986ef4252d953def71569b77a22eac`, using the explicit then-current preset. The default/initialization change is `8f8f16683684d0c451568799ede4cf1153a457d2`; diagnostic migration is `402a326eecfc31677989bc93265823f5990000c1`. All four qualifications and the full nominal diagnostic ran from clean `ac5b96702ed7cafaa6f08b8f9ef31e2d656c4b29`.

The later comparator-only correction is `39f36f5d943cf0be2bb9876030f5e9b382caf41c`. Final implementation `959624fd92a1c467dc3c51c2cc5def8f0d2e1138` additionally rejects incomplete source geometry before loading; it does not change coupled dynamics. Work stayed on `dex-hand-exp`, using conda `teleopit`: Python 3.10.20, MuJoCo 3.8.1, NumPy 2.2.6 and SciPy 1.15.2. Canonical assets were present and their resolved hashes match exactly before/after.

## Equivalence

[Tolerances](tolerances.json) were declared before refactoring: zero for all compared numerical fields. [Machine-readable validation](validation.json) records the complete comparisons and limitations. Fresh references remain in `outputs/g1_o6/consolidation/before_10s`, `before_30s` and `before_fine_30s`; final runs are under `qualification`.

| Trial | Physics step | Zero-assistance HOLD | Max HOLD drift | Max wrist bend | Minimum housing clearance | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Default | 1 ms | 10.02 s | 9.955491 mm | 22.132430° | 18.598124 mm | HOLD_COMPLETE |
| Fresh repeat | 1 ms | 10.02 s | 9.955491 mm | 22.132430° | 18.598124 mm | HOLD_COMPLETE |
| Long | 1 ms | 30.00 s | 15.721970 mm | 22.132430° | 18.598124 mm | HOLD_COMPLETE |
| Fine long | 0.5 ms | 30.02 s | 15.718745 mm | 22.120341° | 18.637182 mm | HOLD_COMPLETE |

Trajectory and summary files are byte-identical to the corresponding pre-refactor runs; both exported wrist-sample arrays are identical. All 472 captured model/reset arrays match at each timestep, including initial qpos and actuator targets. Assets, model audits, initial snapshots, resolved grasp, support targets and retained configuration fields match. Revision/schema fields and removed release-only configuration fields intentionally differ.

All **110,080 HOLD physics steps** have exactly zero `qfrc_applied` and `xfrc_applied`, and no wrist-housing/ladder contact. Angle, clearance and maximum-drift statistics are sampled at the unchanged 20 ms control interval. Peak effort over the entire trial is 0.565806 of the physical limit (0.565798 for the fine run). No threshold, mass/inertia, collision setting, gain, mimic or actuator bound was relaxed.

The XML files have one nonphysical serialization difference: MuJoCo's warm mesh cache omits 180 redundant `content_type="model/obj"` attributes on OBJ assets. Every other XML node and attribute matches. The first raw-byte comparator failed on this difference; its result remains preserved in `qualification/comparison.json`. The verified result is `qualification/comparison_verified.json`. The corrected comparator retains the raw-byte mismatch and compares scene structure without those MIME hints; a regression ensures changed friction still fails. Numerical tolerances remain zero.

To repeat the physical comparison against the retained references, choose a new output directory:

```powershell
conda run --no-capture-output -n teleopit python scripts/validate_g1_o6_consolidation.py --reference-root outputs/g1_o6/consolidation --output outputs/g1_o6/consolidation/fresh_verification
```

## Diagnostics and interfaces

The real 10 s nominal diagnostic produces the exact ordinary strict summary, reaches HOLD_COMPLETE, has `abort_time_s=null`, and never continues. Behavioral tests also compare every shared ordinary/instrumented row on success and on a failed continuation; the latter remains failed, never releases, and has zero assistance.

The documented both-hand free-close CLI produced four passes: left/right with gravity on/off. Source targets, current closing angles, bounded preload, rate and delay are validated and propagated. The two isolated free-wrist load smoke trials failed through grasp loss after achieving only 2.52/2.46 s of the required plateau. Their signed demands come from simultaneous samples of the qualified source run. Fixed/free-base differences, fixtures, origins and ordinary gravity remain explicit in metadata. No new capacity matrix was run or inferred from these failures.

The native viewer was exercised through its actual Space callback: time stayed at 0.14 s while paused, resumed to 0.28 s, contact arrows stayed off, and closing logged `viewer_closed`. A real video export completed a deliberately short 0.04 s HOLD; the terminal frame was inspected. That is an interface smoke test, separate from the 10/30 s qualifications above. CLI tests verify common config propagation and clear errors for removed selectors or missing source runs.

## Tests and quality

The final affected controller, component, posture, diagnostic, source, CLI, comparison and viewer suite passes **118 tests**. Standalone/component/training-collision regressions pass **74 tests**, with one pre-existing training failure: `test_refined_initial_contacts_and_mesh_budget` finds knee/rung overlap up to 0.454 mm. Its test, training configuration and collision-asset code have no diff from the starting revision; the same failure was documented in the [original report](../OVERHAND_BASELINE.md). The training reset was not changed to manufacture a green result.

Ruff and diff checks pass. The named ub-mode crapper/dryer/mutator tools were unavailable. Transparent scoped substitutes measured 46 changed functions: maximum CRAP 9.7971 using Radon and statement coverage; no AST-token similarity >=0.8 among compared bodies of at least 50 AST nodes. Six isolated in-memory mutations were killed: finger bound, command rate, whole-interval gate, successful terminal state, diagnostic completion and source controller validation. These targeted checks are not exhaustive mutation coverage. No obsolete selector, installer, release branch or hardcoded hand-rung-08 fallback remains in either active package.

The work introduced no worktree and removed no raw experiment directory or user archive. [History](../HISTORY.md) retains exact revisions, commands and asset/archive provenance; both existing local ZIP hashes were reverified. See the [action log](WORK_LOG.md), [evidence manifest](evidence_manifest.json) and [local archive manifest](artifact.json). No consolidation blocker remains; the isolated load failures and existing training-reset regression remain explicitly documented limits.
