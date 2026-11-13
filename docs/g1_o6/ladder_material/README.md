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

Run focused checks with:

```powershell
conda run --no-capture-output -n teleopit python -m pytest tests/test_g1_o6_ladder_geometry.py tests/test_g1_o6_ladder_material.py tests/test_g1_o6_ladder_campaign.py tests/test_g1_o6_ladder_ledger.py tests/test_g1_o6_ladder_runtime.py --basetemp outputs/g1_o6/ladder_material/pytest_check -q
```

`WORK_LOG.md` records implementation decisions, actual tests, numerical comparisons
and execution results. No result from this campaign grants load capacity, transfer,
release, disturbance robustness or a new project default.
