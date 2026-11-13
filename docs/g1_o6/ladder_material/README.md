# G1/O6 ladder geometry and tread-friction campaign

This campaign keeps the reviewed balanced grasp and original `HoldController`.
The default scene is unchanged. `PLAN.md` is the user-supplied experiment brief;
`campaign.json` freezes the 25 screen conditions, budgets, reset shortlist and
promotion rule. Software verification and physical qualification are separate.

The compiled scene uses full rung dimensions: lateral length (Y), approach depth
(X), vertical thickness (Z). Front means the negative-X approach side and keeps
the historical `left_ladder_*` geom names. G6 is a fixed, simplified commercial
stepladder approximation. Only its 80 mm depth and 265 mm **vertical** interval
come from the [ZARGES family specification](https://www.zarges.com/de/produkte/r13step-b).
Other dimensions are explicit modeling assumptions. G5 uses the pinned
[LadderMan task](https://github.com/amazon-far/LadderMan/blob/9a8d5ed2b44621e84cc40e8efee7abe4b1543025/cfg/task/base/ladder.yaml)
and [generator](https://github.com/amazon-far/LadderMan/blob/9a8d5ed2b44621e84cc40e8efee7abe4b1543025/active_adaptation/envs/ladder_env.py).

The adapted reset preserves the reference ankle and hand clearances from the
compiled front/top rung faces. The root follows the foot frame; both legs and
arms are solved within original limits. Candidate ranking uses displacement
error plus 0.1 times foot-height error, with the declared preferred pairs first.
The first feasible candidate is frozen before dynamics. Rejected candidates,
exact-pose inspections, complete qpos and joint/clearance evidence are retained.
The authoritative exact-pose evidence is `exact_reference_qpos_inspections` in
the ledger and `exact_reference_inspection` in each new run. Initial preflight
records labeled `exact_reference` are retained as an explicitly corrected
unadapted-target diagnostic; see the work log. They do not select the primary reset.

Scope H changes O6 palm/finger–tread tangential friction. Scope M changes all
eligible robot–tread contacts, including feet and housings. Explicit symmetric
pairs preserve the original effective solver/contact parameters and masks.
Native-equivalent 1.8 contacts remain implicit to preserve contact ordering.
All other friction components, rails, ground and self-contact are unchanged.
These are sensitivity levels, not calibrated commercial materials. See
[MuJoCo mixing rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).

Use the `teleopit` conda environment on Windows; direct unactivated Python can
miss its native DLL paths. Commands below run from the repository root.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.ladder_experiment freeze
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.ladder_experiment prepare
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.ladder_experiment --workers 3 all
```

`all` requires the pre-refactor reference reproduction and verification under
`outputs/g1_o6/ladder_material/reference`. The original source ZIP and model/reset
snapshot are preserved there. `reference_verification.json` records exact
agreement with the reviewed 30 s and 120 s traces. Import rechecks all 471 saved
compiled arrays and reset fields, then uses each reproduction once in the ledger.

Individual stages are `A`, `B`, `C`, `promote-60`, `promote-120`, and `confirm`.
Promotion identities and rankings are saved before launching fresh resets.
An absent final motion window is NOT RUN. Nonconvergent shorter passes may enter
a declared long diagnostic, but cannot qualify as stable.

Resume requires the same implementation revision, dirty diff, source files,
assets, runtime, geometry, reset, controller, material, duration and timestep.
Completed evidence is checksummed before reuse. An interrupted technical attempt
can be repeated with `retry CONDITION STAGE --reason "reason"`; its original
directory remains immutable, and the retry starts with acquisition from reset.
Physical failures cannot be retried to search for a pass. Do not edit simulation
code while a campaign is running.

The campaign writes raw streamed traces, per-step actual applied arrays, effective
contact coefficients, support wrenches, anatomical episodes, event windows, scenes,
HOLD-entry observations and terminal causes. `geometry_views/` contains compiled
front/side reset views. Raw evidence stays outside Git under the separate campaign
directory; earlier pose-search evidence is preserved.

The completed campaign has 25 screens, eight 60 s attempts and six nominal long
attempts. Nothing qualifies; see [OUTCOMES.md](OUTCOMES.md). The physical campaign
ran at `5886277`. Later changes fix scene export and add acquisition-window
triggers without changing the controller or compiled physics. Existing immutable
campaign identities intentionally reject resuming under that later source.

An auxiliary compilation exposed MuJoCo's global-last-XML export behavior. All
original scene files are retained. Use `reconstructions/<run_id>/scene.mjb` or the
archive's corresponding `scene_portable.xml`; each has a verification receipt.
The final runner saves environment-owned XML and an exact binary model. Detailed
acquisition windows for the historical runs are supplied separately by 25
deterministic prefixes with 266 identical recorded control states each. The exact
prefix program used is retained under `source/acquisition_windows_program.py`.
The imported pre-refactor reference has a per-step boolean force audit and event
window arrays, rather than the newer full-step array stream; see the limitation
in the outcomes report.

Verify and regenerate derived reports from the repository root:

```powershell
conda run --no-capture-output -n teleopit python scripts/verify_g1_o6_ladder_campaign.py
conda run --no-capture-output -n teleopit python scripts/g1_o6_ladder_report.py --output outputs/g1_o6/ladder_material/report
conda run --no-capture-output -n teleopit python scripts/g1_o6_ladder_reconstruct.py
```

The reconstruction command verifies existing recovered files. To rebuild missing
reconstructions, use a separate checkout of `5886277` with the archived assets,
ledger and reconstruction helper; the helper checks all frozen simulation-source
hashes, asset hashes and runtime versions before building. It performs no physics
integration. Restore `source/frozen_simulation_source.zip` over that separate
checkout before rebuilding: it contains all 91 simulation files with the exact
recorded bytes, including 20 files with CRLF line endings. A normal Git checkout
can normalize those bytes and fail the deliberate source-hash guard. The source
ZIP and every contained file are checked by `source/frozen_source_receipt.json`.
The complete branch history is included in `source/dex-hand-exp.bundle`.
The current development checkout remains on `dex-hand-exp`.

The archive keeps every original file byte-for-byte and adds portable scene copies
instead of rewriting the originals or their ledger hashes. `artifact.json` gives
its location and checksum; the ZIP also contains `evidence_manifest.json`, physical
assets, source snapshots, the Git bundle, raw traces, recovered models, prefix
windows, figures, videos and validation receipts. It is a local artifact, not an
uploaded release. To rebuild an archive after verification, choose unused paths:

```powershell
conda run --no-capture-output -n teleopit python scripts/package_g1_o6_ladder.py --output outputs/g1_o6/ladder_material_evidence.zip --staging outputs/g1_o6/ladder_material_portable
```

Run focused checks with:

```powershell
conda run --no-capture-output -n teleopit python -m pytest tests/test_g1_o6_ladder_geometry.py tests/test_g1_o6_ladder_material.py tests/test_g1_o6_ladder_campaign.py tests/test_g1_o6_ladder_ledger.py tests/test_g1_o6_ladder_runtime.py --basetemp outputs/g1_o6/ladder_material/pytest_check -q
```

`WORK_LOG.md` records implementation decisions, actual tests, numerical comparisons
and execution results. No result from this campaign grants load capacity, transfer,
release, disturbance robustness or a new project default.

Final software checks: **226 tests passed**; all 39 evidence records verified;
ten targeted mutants killed. [test_results.json](test_results.json) records the
commands, source/receipt hashes and limitations. [quality.json](quality.json)
reports 156 measured changed functions, maximum CRAP 9.648 and no qualifying
duplication pair. Named crapper/dryer/mutator tools were unavailable, so these
use Radon with measured execution coverage, AST comparison and targeted
behavioral mutations. [evidence_verification.json](evidence_verification.json)
contains per-run checks. These software results do not change the negative
physical qualification result.
