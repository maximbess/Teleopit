Implement and run a bounded campaign on Teleopit branch `dex-hand-exp` to determine whether the original G1 plus O6 holding controller can complete a physically valid 120-second HOLD on other ladder geometries and contact materials.

Start with the current `thumb-below-balanced` reference. Keep its original controller throughout this campaign. Include a commercial stepladder approximation and several friction values. Do not include an edge-rounding experiment, thumb-freeze variants, thumb caps, new feedback laws, learning, load transfer, or hand release.

The primary deliverable is a reproducible comparison that separates geometry effects, hand-contact friction effects, and changes in foot support. A failed campaign with complete evidence is a valid outcome; a lower drift value alone is not success.

# 1 Establish and preserve the reference

Use reviewed Teleopit revision `cc0cd5393c670e7801bb7d242c700e6af4d47303` as the reference. Inspect the current branch before implementing. If HEAD has advanced, document the relevant differences and retain an executable reference at the reviewed revision.

The original variant means the current balanced pose and its original HoldController, not an archived failed posture:

- Hand rungs: 11 and 11; foot rungs: 4 and 4.

- Waist lean: 0.30 rad; hand tilt: 50 degrees.

- Per-hand lateral offset: 0.18 m, approximately 0.36 m between mount targets.

- Grip offset: (-0.023, 0, -0.020) m.

- Finger targets in the existing named active-joint order: (0.05, 0.50, 1.35, 1.35, 1.35, 1.35) rad.

- Finger preload: 0.12 Nm.

- Physics step: 0.001 s; control period: 0.02 s.

- Preserve assistance during acquisition and its removal schedule. Applied external assistance must be zero throughout scored HOLD.

Preserve robot and hand masses, inertias, collision geometry, joint ranges, mimic relationships, actuator limits, gains, controller rate limits, gravity, solver configuration and acceptance thresholds. Only declared ladder geometry, initial support placement and selected contact-friction coefficients may differ.

Before the refactor, reproduce the original 30-second trial and its 120-second attempt. The archived long attempt fails drift after approximately 88.02 seconds of HOLD. Compare the failure mechanism, timing and relevant traces, allowing explicitly documented numerical differences between platforms. Do not force numerical agreement by tuning physics. Reuse this long attempt in the campaign ledger rather than counting it twice.

# 2 Implement a resolved ladder configuration

Introduce an explicit ladder configuration and pass it through scene construction, reset placement, diagnostics, serialization and replay. Keep the existing default scene unchanged.

Represent:

- Full rung length along the lateral axis, depth along the approach axis, and vertical thickness.

- Inclination relative to the horizontal floor.

- Rung pitch with an explicit convention: along-rail or vertical.

- First-rung center, rung count, rail dimensions and transforms.

- Front and rear collision participation, declared support targets and any top structure.

- A geometry identity and source or assumption for every dimension.

Compute and save all actual rung centers, rotations and full dimensions after compilation. Derive vertical and horizontal pitch from the resolved transforms. Do not confuse full sizes with MuJoCo half-sizes.

Relevant implementation locations include `teleopit/sim/ladder_geometry.py`, `g1_o6/model.py`, `g1_o6/initialize.py`, `g1_o6/grasp_pose.py`, and `g1_o6_support/pose_trial.py`. Reuse the existing long-HOLD runner, contact observer, immutable ledger and reports.

Remove experiment-facing assumptions that feet always use rung 4, that every ladder has 14 rungs, or that diagnostics can always query rungs 4, 10 and 11. The current mount target also derives its translation from a standalone preset with the original rung dimensions: make that dependency explicit and geometry-aware.

Preserve the historical compiled model and trajectory when no new configuration is supplied. Separate unchanged robot/controller hashes from intentionally changed ladder/material hashes; do not weaken the original qualification checks to admit arbitrary physical changes.

# 3 Geometry matrix

All dimensions below are full dimensions. First run every geometry with the original controller and reference contact material. All rung collision shapes remain unrounded boxes.

| ID | Geometry | Rung depth × thickness | Placement |
| - | - | - | - |
| G0 | Original reference | 110 × 70 mm | Original 700 mm length, 69.44-degree incline and 199.36 mm along-rail pitch |
| G1 | Thickness only | 110 × 38.1 mm | Preserve G0 rung centers, length, rails and angle |
| G2 | Depth only | 80 × 70 mm | Preserve G0 rung centers, length, rails and angle |
| G3 | Depth and thickness | 80 × 38.1 mm | Preserve G0 rung centers, length, rails and angle |
| G4 | Pitch only | 110 × 70 mm | Increase along-rail pitch to 250 mm; preserve angle, length and rail cross-section |
| G5 | LadderMan code reference | 80 × 38.1 mm | 600 mm length, 65.5-degree incline, 250 mm along-rail pitch, six rungs |
| G6 | Commercial stepladder approximation | 80 × 38.1 mm | 540 mm length, 70-degree incline, 265 mm vertical pitch, eight rungs on each side |

For G4, anchor the original foot-rung center and redistribute neighboring rungs along the ladder axis. Extend rail endpoints as needed and report the resulting overall dimensions. This is a controlled pitch diagnostic, not a proposed commercial product.

For G5, use the published generator and task configuration as the geometry reference, including first-rung center (-0.58, 0, 0.19095) m and its rail construction. Retain O6 hands and the Teleopit controller. This is a geometry comparison, not a reproduction of LadderMan's policy or rubber-hand hardware.

G1–G3 preserve rung centers to isolate dimensions; their top surfaces therefore move when thickness changes. Save this displacement and use the common reset adaptation rule below rather than silently re-centering a rung to improve the result.

# 4 Commercial stepladder approximation

G6 must visibly and physically resemble a double-sided commercial stepladder, while remaining an explicitly simplified fixed environment.

Use the ZARGES R13step B family as the dimensional reference for an 80 mm tread depth and a 265 mm vertical tread interval. The manufacturer's German specification explicitly describes this interval as vertical. Do not interpret it as along-rail pitch.

Use the following declared approximation:

- Eight horizontal treads on each side.

- Tread centers at z = 0.265 × i m for i = 1 through 8.

- Front and rear rail inclination: 70 degrees to the floor.

- Rail apex height: 2.35 m; symmetric A-frame footprint in the approach direction.

- Tread length: 0.54 m; tread depth: 0.08 m; thickness: 0.0381 m.

- Rectangular rails with a 0.068 m dimension in the ladder plane and a 0.025 m lateral thickness. Position their inner faces at the tread ends, giving an approximate overall lateral width of 0.59 m.

- Constant lateral width; no taper, perforations, tread texture, rounded edges or compliant joints.

- Front and rear rails and treads participate in collision detection.

- Keep the complete ladder fixed in the world. Do not add a dynamic tip-over model in this campaign.

Only the 80 mm depth and 265 mm vertical interval are directly adopted as product-family specifications. The other values, including thickness, apex height and simplified rail profile, are modeling assumptions. Label G6 as a commercial stepladder approximation, not an exact digital twin or a certification-compliant reconstruction. Its 70-degree incline implies approximately 282 mm along-rail pitch.

Build rails from the same resolved geometry as the treads; inspect front and side views and verify that no rung intersects the floor, that rail/tread placement is consistent, and that rear structures cannot provide unreported support. Contact between robot housings and any tread or rail remains prohibited.

G6 is a system-level geometry comparison. Its result cannot by itself identify whether thickness, width, pitch, rails or initial posture caused an improvement; use G1–G4 for that distinction.

# 5 Adapt reset placement without changing the controller

Changing rung dimensions or pitch can invalidate a previously valid initial pose. Distinguish exact pose reuse from a deterministic geometry-adapted reset.

First inspect the unchanged reference reset wherever its target rung indices exist. Record reachability, penetration, joint margins and foot placement. Do not run a known invalid reset as if it were a valid HOLD experiment.

For the primary comparison:

1. Preserve the balanced grasp family, wrist orientation, waist lean, finger targets, preload and control law.

2. Derive hand and foot target transforms from the actual compiled rung faces and original geometric clearances. Keep this mapping deterministic and verify that it reproduces G0.

3. Translate the robot root to the selected foot-support frame and solve both legs and arms within their original limits. The old initializer's fixed world foot positions are not valid for an arbitrary ladder.

4. Select hand and foot rung pairs from a predeclared short list based on similarity to the reference hand-to-foot displacement and reachable support heights. Keep both hands on one rung and both feet on one rung for this campaign.

5. Preserve the original 0.36 m hand-target separation where feasible. Any required lateral relocation is an explicit geometry adaptation, not a controller improvement.

Allow at most 12 deterministic reset candidates per geometry. Predeclare the rung-pair shortlist, ordering and geometric score before dynamic trials. Choose the first feasible candidate or the best geometric score under that rule, never the candidate with the longest subsequent HOLD. Store rejected candidates and reasons. If none is feasible, report INFEASIBLE and stop that geometry.

For G6, foot rung 3 and hand rung 8 are a useful first geometric candidate: their vertical separation is 1.325 m, close to the original 1.307 m. This does not guarantee IK or collision feasibility.

For each geometry, freeze the selected reset specification before any friction sweep. Different friction values may produce different acquisition trajectories; that is an experimental outcome. Do not retune the initial pose or controller per friction value. State whether a comparison used exact pose reuse or an adapted reset.

# 6 Friction experiment and contact implementation

Test effective tangential friction values:

**mu = 0.4, 0.6, 0.8, 1.2 and 1.8.**

These are sensitivity-test levels, not measured coefficients of named materials. The 1.8 case is the current reference. Hold torsional friction, rolling parameters, contact dimensionality, solver reference/impedance and all nonselected contact properties constant.

MuJoCo uses the element-wise maximum of geom friction for equal-priority contacts. Reducing only the rung coefficient below a hand coefficient of 0.8 will not produce an effective coefficient of 0.4 or 0.6. The existing global friction_scale also changes more than the intended experiment.

Implement explicit, symmetric contact-pair overrides or an equivalent verified mechanism. Override only the selected tangential components. If creating explicit pairs, preserve the original effective condim, solref, solimp, margin, gap and other coefficients. Restrict pairs to physically eligible robot–rung pairs: do not accidentally create collisions for visual geoms or bypass intentional masks/exclusions.

Read actual friction values from active MuJoCo contacts. Verify both left and right hands, multiple finger groups and every potential target rung. Store requested and observed coefficients separately.

Two scopes are required:

| Scope | What changes | What remains fixed | Purpose |
| - | - | - | - |
| H: hand-contact isolation | O6 collision geoms against ladder treads | Feet, G1 housings, rails, ground and self-contact | Determine the effect on the grasp itself |
| M: whole-tread material | All eligible robot collision geoms against ladder treads, including feet and housings | Rails, ground and robot self-contact | Determine the coupled consequence of changing tread contact friction |

H includes the palm and fingers, but cannot turn palm-only or housing contact into an accepted grasp. M applies to unintended robot–tread contacts too, while those contacts remain failures. Neither scope changes the accepted support list.

For M, assigning the same effective coefficient to different robot contact surfaces is a controlled simplification. A physically calibrated material model would require separate rubber–tread and foot–tread measurements. Do not label this sweep as such a calibration.

# 7 Execute the campaign in stages

**Stage A — Geometry with the original controller.** Run G0–G6 at reference friction. Use 30-second HOLD screens after successful acquisition. Preserve early failures as failures and classify reset infeasibility separately. Complete all feasible screens before ranking.

**Stage B — Isolated hand friction.** Run scope H at all five mu values for G0, G5 and G6. Do not omit the commercial geometry because another geometry scores better. Reuse an existing mu = 1.8 result only when the compiled physical state and effective contact model are verified equivalent. This adds at most 12 screens.

**Stage C — Whole-tread friction.** Run scope M at mu = 0.6, 0.8 and 1.2 for G0 and G6, adding six screens. Compare these to scope H at the same geometry and mu to reveal changes caused by foot support. A lower drift achieved by unloading the hands is not evidence of a stronger grasp.

The primary screen budget is therefore **25 distinct 30-second conditions**: seven geometry screens, 12 additional H screens and six M screens. Reset inspections are separate from dynamic trials. Do not assume behavior is monotonic in friction.

**Stage D — Longer holds.** Promote at most eight distinct conditions to 60 seconds and at most six to a first 120-second attempt, including the original long reference. Require the preceding duration's strict HOLD and thumb-specific contact gates to pass. Record the late-motion result separately; a shorter trial with nonconvergent motion may receive a declared long diagnostic, but cannot be called stable.

Reserve promotion opportunities for G5 and G6 when eligible, a lower-friction H case, and a G6 whole-tread M case. Fill remaining slots by a predeclared ranking: valid duration and contact integrity first, late motion second, actuator margin third. Publish the ranked list and exact selected identities before launching promotions. Do not promote a small-drift trial with failed opposition or prohibited support.

**Stage E — Confirmation.** For at most two conditions that pass a complete 120-second nominal trial and the existing late-motion assessment, run one fresh 120-second repeat and one 120-second trial at 0.0005 s physics step. Preserve the 0.02 s control period and express controller updates in physical units so that halving the physics step does not alter the control law.

Every longer trial starts from a fresh reset and includes acquisition and assistance removal. Do not extend a saved 30-second state. Deterministic repeats establish repeatability, not disturbance robustness. Broader robustness, capacity and transfer campaigns remain subsequent work.

# 8 Diagnostics and comparisons

Retain all existing strict evaluation and thumb-specific opposition checks. Extend the evidence so each result can be interpreted in its own ladder geometry.

At reset and HOLD entry, save:

- Full ladder configuration, compiled transforms, sizes and rail geometry.

- Selected support names and world positions; hand-to-foot separation.

- Root pose, achieved IK errors, wrist/elbow angles and joint margins.

- Palm, thumb and finger positions relative to the actual target rung.

- Minimum housing clearance against all front and rear obstacles.

- Controller identity, named targets, preload and resolved friction scope.

During acquisition and scored HOLD, record:

- Base drift relative to RESET and HOLD entry; base orientation.

- Rung-relative palm/wrist translation and rotation and late-motion slopes/excursions.

- Per-finger normal force, contact centroid, slip and persistent contact interruptions.

- Thumb-specific opposition normals and opposition quality, independently of force magnitude.

- Signed support forces and moments for each hand and foot, force/moment balance and hand load fractions.

- Effective contact friction, tangential friction utilization and torsional utilization separately.

- Joint-limit errors, actuator saturation, power/work and minimum housing clearance.

- Actual external-force arrays at every scored physics step.

- First threshold crossing, persistence onset and terminal cause, including the phase.

Reuse physics event windows around acquisition transitions and the first failure. An absent late-motion window is NOT RUN, never PASS.

Compare matched time windows only: 20–30 s for screens, 50–60 s for completed 60-second trials, and 90–120 s for completed long trials. Report early failures without inventing matched-window averages.

For G1–G3, explicitly relate outcomes to the changed surface positions and measured contact migration. For H versus M, compare foot slip, foot forces and hand load fractions. If hand load differs materially, qualify the causal interpretation instead of silently adding a new load controller.

Add a geometry-by-friction summary, failure timelines, drift/contact/opposition plots, support-load plots and front/side images with true dimensions. Preserve the original failure even if an optional unscored continuation is used for diagnosis.

# 9 Acceptance and software verification

A qualified condition requires all of the following:

- A feasible, nonpenetrating reset and completed acquisition.

- A full 120 seconds of unsupported scored HOLD.

- The original strict limits and thumb-specific contact-quality gates pass.

- No hand-to-rung weld, hidden fixture, replayed support wrench, state reset or prohibited housing/rail/rear-rung support.

- The existing final-window late-motion criteria pass.

- A fresh repeat and the smaller-physics-step trial also pass.

If a condition completes 120 seconds but fails late motion, label it “duration passed, stability not qualified.” Do not change evaluator thresholds to fit results. A pass at mu = 1.8 applies to that condition; it does not establish a lower-friction or commercial-material pass. A pass with only a small fraction of weight on the hands does not establish hand load capacity.

Add focused behavioral checks for:

1. G0 model and controller preservation after parameterization.

2. Correct vertical/along-rail pitch conversion and compiled dimensions.

3. Dynamic support indexing on six- and eight-rung ladders.

4. Geometry-derived hand/foot placement and rejection of invalid resets.

5. Actual mu = 0.4, 0.6, 0.8, 1.2 and 1.8 contacts despite MuJoCo mixing.

6. H leaves foot contacts unchanged; M changes feet while preserving rails and self-contact.

7. Explicit-pair implementation preserves non-friction parameters and does not activate unintended collisions.

8. Failure detection for lost opposition, prohibited supports and nonzero assistance.

9. Resume/cache identity includes geometry, reset, controller, material scope, effective coefficients, timestep, duration, code and assets.

Test a small controlled contact scene as well as compiled coupled contacts. Run affected existing tests and report their actual outcomes. Separate software PASS from physical qualification.

# 10 Evidence and final handoff

Create a separate ladder/material campaign directory and preserve previous pose-search evidence. Use immutable per-run records with exact commands, code revision and dirty diff, configuration hashes, asset hashes, environment versions, seeds, status and failure reasons. Resume only exact identities; interrupted or numerically invalid runs do not count as passes.

Archive resolved scenes and assets, raw traces, event windows, contact histories, summary tables, plots and recorded-state videos. Include checksums and a reconstruction command. Keep long logging bounded in memory and write incrementally.

The final report must answer:

- Does a thinner or shallower rung improve the original grasp, and under which reset adaptation?

- Does increased pitch reduce housing collisions or merely change load distribution?

- Does the commercial stepladder approximation complete the long hold?

- How do hand-only and whole-tread friction changes differ?

- At which tested coefficients is each geometry qualified, duration-only, failed, infeasible or not run?

- Which failure mechanism remains: opposition loss, drift, joint limits, foot slip, unintended support or numerical failure?

Deliver the implementation, frozen campaign configuration, resumable runner, test results, evidence archive and a concise recommendation. Do not replace the project default automatically. If nothing qualifies, preserve the negative result and identify the next single hypothesis supported by the evidence. Controller redesign belongs to a later campaign.

Source references:

- [Teleopit reference geometry](https://github.com/maximbess/Teleopit/blob/cc0cd5393c670e7801bb7d242c700e6af4d47303/teleopit/sim/ladder_geometry.py)

- [Teleopit measured pose-search outcomes](https://github.com/maximbess/Teleopit/blob/cc0cd5393c670e7801bb7d242c700e6af4d47303/docs/g1_o6/pose_search/OUTCOMES.md)

- [LadderMan base task configuration](https://github.com/amazon-far/LadderMan/blob/9a8d5ed2b44621e84cc40e8efee7abe4b1543025/cfg/task/base/ladder.yaml)

- [LadderMan geometry generator](https://github.com/amazon-far/LadderMan/blob/9a8d5ed2b44621e84cc40e8efee7abe4b1543025/active_adaptation/envs/ladder_env.py)

- [ZARGES R13step B product-family dimensions](https://www.zarges.com/de/produkte/r13step-b-epim23831)

- [MuJoCo contact parameter mixing](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
