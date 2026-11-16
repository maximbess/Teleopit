Implement and run a staged G1 + Linkerhand O6 validation campaign on the existing G6 commercial stepladder approximation. Find a physically valid pose that completes sequential 30, 60 and 120 second holds at native friction, demonstrate controlled transfer of 30% of robot weight to the two hands, and only then validate hand transfers to an adjacent higher tread.

Use the existing thumb-below-balanced grasp as the starting configuration. Deliver reproducible code, configurations and physical evidence. A failed search is a valid experimental outcome; software completion, a commanded load target or a visually plausible video cannot substitute for a physical pass.

# 1 Scope and starting evidence

Start from branch `dex-hand-exp`. The reviewed reference commit is `298cd2f555dcf48637384987ef0668e85f219392`. Inspect the current branch before editing, record the actual starting commit and reconcile any newer changes with this task.

The recorded native G6 case uses hand tread 8 and foot tread 3. It fails through right-knee contact with front tread 4; the reported 0.50 seconds of HOLD is an early abort interval, not a successful hold. Native G6 also fails the anatomical contact-quality gate. Address both acquisition clearance and grasp quality.

G6 has eight treads per side. A pose with both hands on tread 8 has no adjacent higher tread. Retain 8/3 as a reproduction control, but prioritize feasible starting poses with hands on tread 7 or lower for the complete task. A terminal pose on tread 8 may qualify for hold/load testing only.

This campaign ends after validated hand transfers. Full climbing, foot stepping, PPO training and hardware validation are outside its scope.

# 2 Fixed experimental conditions

Use `geometry_matrix()["G6"]` without modifying its dimensions: 540 mm tread length, 80 mm depth, 38.1 mm thickness, 265 mm vertical pitch, 70 degree inclination and eight treads on each side. Preserve front and rear collision geometry and the fixed-world ladder.

Use native material settings with `friction_scale=1.0` and no H/M material override. Preserve the complete friction vectors, priorities, contact dimensions, solver settings and contact-pair rules. The native effective sliding coefficient at the intended hand–tread contact is 1.8; verify it in the compiled model and actual contacts. The O6 geom coefficient of 0.8 alone does not identify the effective pair coefficient.

Keep robot/hand masses, inertias, gravity, joint ranges, actuator limits and collision masks unchanged. Keep the original controller for the first pose-search batch. Do not add rounded edges, adhesive contacts, hand–tread welds, virtual supports or persistent acquisition assistance.

All scored coupled phases must have zero externally applied assistance. Initialization assistance is permitted only under the existing acquisition/removal schedule. Any isolated capacity-test wrench must be explicitly declared and audited separately.

Use 1 ms physics steps and the existing 20 ms control period. Confirm qualifying results at 0.5 ms physics steps with the control period unchanged. These remain simulation results under the current provisional actuator model.

# 3 Repair experiment plumbing before the campaign

Reuse the existing modules under `teleopit/sim/g1_o6_support`, the G6 ladder campaign and the O6 diagnostic harness. Avoid a separate simulator or duplicate evaluator.

1. Propagate the resolved ladder, material, pose, controller and reset configuration through hold, capacity, loading, unloading and reacquisition runners. At the reviewed commit, `execute_transfer` constructs `Environment` with only config and pose, omitting the source G6 ladder/material. Fix this path and verify the compiled scene identity before stepping.

2. Use the environment-owned scene export method. The transfer runner still calls `mj_saveLastXML`; replace that path with the established owned-model export and round-trip checks.

3. Extend target-share parsing, request validation, controller configuration, reporting and prerequisite checks from 5/10/20% to **5/10/20/30%**. A 30% request must require qualified 20% evidence.

4. Extend plateau-duration progression where needed. Keep 10 → 30 second tests and add a 60 second loaded-hold confirmation.

5. Review the existing 15% maximum share per hand. It leaves no upward tolerance for a symmetric 30% total target. Express per-hand operating bounds using tested capacity and explicit balance tolerances; preserve physical actuator limits. Do not simply remove the bound.

6. Strengthen capacity-evidence matching. Include geometry/material fingerprints, effective contact properties, source pose/state and controller identity. The current `capacity_matches` comparison does not explicitly compare the ladder/material fields.

7. Make support targets phase-dependent for real transfers. Distinguish the old tread, declared destination tread and prohibited scene geometry. Missing G6 identity or required contact diagnostics must fail validation rather than default to PASS.

Add focused regression tests for G6 preservation, native friction, owned-scene export, 30% prerequisites, capacity mismatch rejection, destination-tread selection and phase-aware contact checks. Run the relevant existing tests. Preserve old experiment records and defaults while this candidate is unqualified.

# 4 Diagnostics and acceptance rules

Record simulator time and phase-relative time. Keep RESET and HOLD-entry reference frames separately; never reset drift references to hide accumulated motion. Log all transitions through acquisition, assistance removal, HOLD, loading, unloading, reaching and regrasping.

Extend the existing diagnostics with:

- Minimum signed distances and actual contacts for each knee, shin, wrist housing, forearm and torso against every tread and rail. Record first contact, persistence onset, phase and geom pair.

- Per-hand thumb/finger opposition, intended-tread contact, normal and tangential forces, slip velocity, contact-centroid migration and wrist six-dimensional wrench.

- Signed world-frame vertical support reactions for each hand and foot; base pose, COM, velocities, support motion, joint-limit margins, actual actuator efforts and saturation duration.

- Effective contact friction and actual `qfrc_applied`/`xfrc_applied` at every scored physics step.

- First-failure timelines and plots that include acquisition/removal, not only HOLD.

Apply the current strict physical thresholds, including 30 mm RESET drift, 0.25 rad orientation deviation, 20 mm/s slip, 3 mm penetration and the existing joint, mimic and effort limits. Require valid anatomical grasp contacts, no wrist-housing support and no prohibited support.

For 120 second stationary holds, also require the existing final-30-second convergence limits: translation slope at most 0.05 mm/s and excursion at most 2 mm; rotation slope at most 0.0005 rad/s and excursion at most 0.01 rad for the currently evaluated bodies. Report COM and both wrists alongside these metrics.

A 30 or 60 second strict/contact pass may be promoted to a longer **diagnostic** run despite a failing late-drift metric, but cannot be called a stable qualified pose. The final 120 second gate requires strict, contact and convergence passes.

During a declared hand-reach phase, evaluate the moving hand against its commanded trajectory and destination; stationary-hand drift/opposition predicates do not apply to that intentionally released hand. Preserve physical limits, stationary-support checks and base/COM safety criteria. Define this phase contract before testing, retain global traces, and restore full stationary grasp checks on reacquisition. Do not globally disable the evaluator.

# 5 Search for an initial G6 pose

First reproduce the native G6 8/3 failure and verify the knee/tread pair, contact-quality failures and zero assistance after removal. Inspect the full acquisition trajectory. A collision-free reset snapshot alone is insufficient.

Then run small, hypothesis-driven pose batches:

| Batch | Parameters to explore | Main question |
| - | - | - |
| A | Root distance from ladder, pelvis height, waist lean and leg configuration | Can the knee/shin clear the treads throughout assistance removal? |
| B | Hand tread 7 or lower; feasible foot treads; bounded foot placement on tread | Can a lower pose retain clearance and leave a reachable next tread? |
| C | Hand tilt, tread-relative grip offset, lateral hand spacing and limited left/right asymmetry | Can both hands establish genuine opposition without housing support? |
| D | Bounded thumb/finger closure targets and preload within existing actuator limits | Can a valid contact arrangement stop slow migration and drift? |

Use the previous batch winner as a reference. Start with one parameter family at a time, then test a small number of combinations supported by measured results. Reject unreachable poses, invalid joint margins and scene interpenetration before dynamic testing. Parameter ranges must be recorded before each batch and clipped to actual kinematic feasibility.

A practical first local grid is root offsets around ±20 and ±40 mm, waist changes around ±0.05 and ±0.10 rad, hand tilt changes around ±5 and ±10 degrees, and grip-offset changes around ±5 and ±10 mm. These are proposed search increments, not known feasible ranges; avoid a full Cartesian sweep.

Screen feasible candidates through acquisition and a short unassisted interval, then run full 30 second HOLD tests. Rank by physical/contact validity first, followed by drift trend, clearance, effort margin and reachability of the next tread. Do not rank by duration alone.

Suggested initial budget: up to 40 distinct dynamic screens, eight 60 second promotions and four 120 second promotions, excluding required winner confirmations. Use the failure diagnosis to propose a bounded second batch if necessary; preserve all trials. If pose/grasp tuning is exhausted, a bounded feedback-controller correction may be tested as a separately identified experiment, with the complete qualification sequence restarted.

For each unchanged candidate, run fresh-reset **30 → 60 → 120 second** trials. Count only fully unassisted HOLD time. After a complete 120 second strict/contact/convergence pass, perform a fresh 120 second repeat at 1 ms and a 120 second confirmation at 0.5 ms. Freeze the qualifying configuration and evidence identity before load-transfer qualification.

# 6 Qualify grasp capacity for the selected pose

Generate new capacity evidence from the qualified G6 configuration. Old G0 evidence cannot qualify G6. Test both hands independently using the actual source tread geometry, contact material, wrist orientation and finger state.

Follow the existing zero-fixture retention, force-only, moment-only, combined-wrench, ramp, plateau and recovery protocol. Preserve gravity and physical actuator limits. Use simultaneous measured force/moment samples, including critical transient demands, with the existing 1.2 reserve criterion; do not construct artificial component-wise peaks.

The current isolated matrix limits test cases to 50 N and 3 Nm. Those are experimental search bounds, not measured hardware capacities. A symmetric 30% load split is approximately 50 N of vertical support per hand for the current approximately 34.08 kg model, before wrench-direction effects and reserve. Therefore the existing 50 N search ceiling may exclude necessary cases. Parameterize the experimental bounds from planned/measured six-dimensional demands and their reserve, record them before running, and report out-of-budget demands as uncovered. Never silently omit a required case or increase actuator strength.

At each new loading level, distinguish a physical run from capacity-qualified evidence. Use measured demands to run additional reserve tests before promoting that result to the next level. Requalify destination-grasp states and critical transfer demands where their contact arrangement differs.

If zero-fixture retention or reserve fails, preserve the failed gate and return to diagnosis or candidate correction. An arm-mounted diagnostic can investigate the different wrist boundary condition, but it must not be reported as a free-wrist pass or silently replace the existing evidence contract.

# 7 Transfer 30 percent of weight to the hands

Compute the target from external support forces acting on the robot:

```text
hand_share = (Fz_left_hand + Fz_right_hand) / (robot_mass * abs(gravity_z))
```

Use signed upward world-frame forces. Normal-force magnitude, grip squeeze force, internal wrist reaction or an actuator command is not the fraction of body weight supported by the hands. Log both per-hand shares and the total.

For a mass near 34.08 kg, 30% corresponds to approximately 100.3 N total, or 50.2 N per hand under an equal split. Recompute these values from each compiled model. In quasi-static holds, verify the total external support force against weight; during motion, include COM acceleration in the balance residual.

Start with a symmetric allocation. Preserve the existing total-share tolerance of **±1 percentage point**. At the final plateau, target approximately 15% per hand with a predeclared balance tolerance, suggested ±1 percentage point per hand, and capacity-qualified upper bounds. Report continuous traces, extrema and time within tolerance so a short peak or plateau average cannot pass.

Proceed through 5 → 10 → 20 → 30%. If the qualified initial pose naturally exceeds a lower target, measure and document that starting load, validate controlled settling at the relevant levels and do not count an unperformed step as passed.

For each target:

1. Start a fresh run from the frozen pose, reproduce acquisition and establish unassisted support.

2. Use the existing bounded joint-target load controller to ramp the measured share slowly. Begin with its 10 second ramp and existing offset/rate limits; pause or back off on slip, drift, effort saturation or contact degradation.

3. Require the target band continuously for at least the existing 0.5 second settling guard before starting the plateau timer.

4. Complete a 10 second plateau, recover toward the original allocation, and verify stable support. Qualify the newly measured wrist demands before advancing.

5. At 30%, extend plateau duration through **10 → 30 → 60 seconds**, each from a fresh reset. Require the final 30 seconds of the 60 second plateau to pass the stationary convergence checks.

Repeat the final 60 second loaded hold from a fresh reset and confirm it at 0.5 ms. This loaded-hold duration is a proposed acceptance criterion in addition to the required 120 second initial-pose test. A 120 second hold at 30% load is a useful optional extension.

Do not begin hand-transfer experiments until both the 120 second pose qualification and the measured 30% loaded-hold qualification pass with relevant capacity coverage.

# 8 Validate hand unloading and same tread reacquisition

Begin each trial from the qualified 30% support state. Test left and right independently.

First unload a selected hand while keeping its fingers closed. Preserve the current measured guards: at least 80% reduction from its initial vertical support; wrist-force norm below 8 N; wrist-moment norm below 0.4 Nm; compensating support reactions within the existing 2 N tolerance; and a continuous 0.5 second guard followed by the existing unloaded hold. Verify support compensation from measurements.

Then open only the selected hand, retract using bounded arm commands, and verify measured opening, absence of scene contacts and at least 5 mm clearance. Progress through 1, 3 and 10 second three-support holds. The remaining hand and both feet must stay on their intended supports within tested wrench and actuator limits.

It is acceptable for the total hand share to fall below 30% during unloading and release as the feet take more load. Do not force the remaining hand to carry the entire 30% unless that allocation has been independently qualified.

Reacquire the original tread, restore anatomical opposition, reload gradually and restore the 30% total share. Complete the existing one-cycle, three-cycle and alternating-side progression before attempting an adjacent-tread transfer. Treat these as release/reacquisition calibration; they do not establish upward hand transfer.

# 9 Transfer each hand to the adjacent higher tread

Implement an explicit adjacent-tread mode; the existing same-tread reacquisition is insufficient. Select the destination from the unchanged G6 geometry and reject nonexistent or unreachable targets.

Use this sequence: qualified loaded hold → controlled unloading → opening and clearance → collision-checked reach → destination grasp → measured reloading → stable recovery.

Plan with scratch model/data, then execute through bounded joint/actuator commands in the live simulation. Never overwrite live qpos, teleport the hand or create an attachment. Check the swept hand, wrist and forearm against all treads, rails, the body and the other hand.

Test left-first and right-first orders in separate fresh-reset trials. After the first hand reaches the next tread, verify the mixed-height support state before moving the second hand. Transfer the second hand only after the first grasp, force allocation and relevant capacity evidence are qualified.

A successful two-hand transfer requires:

- Both hands end on the declared adjacent higher tread, with valid opposition and no unintended support.

- Both feet retain their declared supports throughout; no unplanned foot step is credited.

- The remaining supports stay valid during each release/reach, and no external assistance returns.

- Each reacquisition is verified from actual contacts and finger state before reloading.

- After both transfers, the robot restores **30% ±1 percentage point** total hand support and completes a 30 second stable recovery with contact, physical and convergence checks.

- The complete sequence succeeds in three independent nominal trials for each starting-hand order; confirm at least one successful sequence for each order at 0.5 ms.

Record first-hand success, second-hand success and complete-sequence success separately. If the fixed-foot mixed-height or final state is infeasible, report the specific kinematic, contact or load limit. Return to the initial-pose search rather than adding a tread, permitting knee support or claiming that same-tread reacquisition completed this stage.

# 10 Evidence and completion report

Save a new campaign ledger with immutable run IDs, exact commands, code commit/diff, resolved configurations, source and asset hashes, scene fingerprints, model exports and per-stage qualification evidence. Save interrupted/failed cases and distinguish FAIL, BLOCKED, NOT RUN and technical ERROR.

Provide:

1. A reproducible G6 pose configuration and implementation changes with focused test results.

2. A candidate table showing pose parameters, achieved duration, first failure, contact validity, drift, minimum clearance and actuator margin.

3. Initial-pose 30/60/120 second evidence, plus nominal repeat and finer-step confirmation.

4. Capacity results for each hand, covered simultaneous wrench demands and reserve limits.

5. Load-share plots for 5/10/20/30%, final loaded-hold repeats and measured force-balance checks.

6. Per-hand unloading, three-support and same-tread reacquisition results.

7. Adjacent-tread transfer results for both orders, including final recovery and capacity coverage.

8. Recorded-state videos of the baseline failure, best qualified hold, 30% loading and completed hand transfers, where achieved.

9. A checksum-verified portable evidence archive and a short README with exact reproduction commands.

Use a final status table:

| Milestone | Required evidence |
| - | - |
| Stable initial G6 pose | Sequential 30/60/120 s; 120 s repeat and fine-step pass |
| Load-bearing grasp | Measured 30% total share; qualified reserve; 60 s plateau, repeat and fine-step pass |
| Controlled release | Both hands independently pass unloading and 1/3/10 s three-support holds |
| Same-tread reacquisition | Existing cycle progression passes with restored loading |
| Adjacent-tread hand transfer | Both hands move upward; both orders repeat successfully; stable 30% recovery |

Only report complete task success when all milestones pass. Otherwise report the highest qualified stage, the measured limiting mechanism and the next bounded experiment. Keep the current project default until the replacement configuration has earned the relevant qualification.

# Repository references

- [Recorded ladder and friction outcomes](https://github.com/maximbess/Teleopit/blob/298cd2f555dcf48637384987ef0668e85f219392/docs/g1_o6/ladder_material/OUTCOMES.md)

- [G6 geometry definition](https://github.com/maximbess/Teleopit/blob/298cd2f555dcf48637384987ef0668e85f219392/teleopit/sim/ladder_config.py)

- [Existing support transfer protocol and results](https://github.com/maximbess/Teleopit/blob/298cd2f555dcf48637384987ef0668e85f219392/docs/g1_o6/support_transfer/README.md)

- [Transfer implementation](https://github.com/maximbess/Teleopit/blob/298cd2f555dcf48637384987ef0668e85f219392/teleopit/sim/g1_o6_support/transfer.py)

- [Qualification checks](https://github.com/maximbess/Teleopit/blob/298cd2f555dcf48637384987ef0668e85f219392/teleopit/sim/g1_o6_support/qualification.py)
