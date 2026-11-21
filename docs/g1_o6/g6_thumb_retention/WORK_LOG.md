# G6 thumb-retention work log

| Step | Action and evidence |
| --- | --- |
| Preserve and retire | Compared the local and remote `dex-hand-exp` tips at `f27fa4c`. Commit `b7130a5` retired 158 superseded result files and redirected surviving documentation links to that revision. Moved 150 historical local output entries to a sibling recovery archive, including five clean simulator copies and 19 old test directories. Checked the active viewer and retained scene dependencies before moving their neighbors. The branch retains reusable configuration, the wrist baseline and its media. |
| Reconstruct | Wrote failing tests for signed edge margins, tread-relative rotation and single-sample events. Reconstructed the captured distal point from the contact centroid and distal pose, then verified its local coordinates against the recorded controller. Paired each solver contact with its pre-integration pose. Knee clearance uses the integrated pose. |
| Select | Measured the retreat sequence, signed rotation deficit and proximal effort. Selected the negative-x face as the nearest tread edge and shifted the right acquisition target in positive x. Pre-oriented along the normalized median unserved rotation in tread axes, `[0.631418, 0.545911, 0.550720]`. Used a right-only shoulder/elbow correction as the third intervention. |
| Freeze and run | Executed the five fresh correction-campaign runs from clean simulator revision `b96c2c8` on `dex-hand-exp`. Compared every control-probe command/state row with the archived right-wrist probe. Both acquisition changes failed the force gate. The premise that right acquisition alone could resolve the loss failed twice, with the dominant vertical force deficit at the feet. Reviewed both failures before running proximal recentering. The 4 mm branch stayed closed because measured clearance decreased. |
| Score | Independently reconstructed opposition, individual left-finger loads, signed force residual, actual actuator commands, wrist projection, and right-knee clearance for all five runs. The independent verifier passed each recording, including the measured failures. Every scoring step retained native friction and zero external assistance. |
| Check downstream wiring | Executed the actual load and adjacent wrappers with their sourced controllers and idle transfer commands. A fresh cross-checkout restore exposed path-dependent mesh filename offsets. The restore now verifies the archived binary and the rebuilt physical fields, then initializes the actual saved model. Tested rejection of changed mass and a mismatched qualification identity. A legacy free-wrist bench cannot supply the bilateral controller's live feedback. A blocked result preserves that distinction in the stage ledger. |
| Repeat the reference | Ran independent 5, 30 and 60 s requests with the original right wrist, a fresh reset at every stage and promotion only after a passed preceding stage. The prior measured trial hashes also ship in the repository so the reference can be reproduced from a clean checkout. |
| Verify and publish | Verified all 160 raw-file hashes and independently replayed all eight trials, comprising 135,420 unsupported native steps. The fresh reference reproduced the archived command/state trace and the decoded contact, position/knee and external-force streams at each duration. Published the full-resolution comparison traces, figure, per-run gate times and separate milestone results. |

The G1/O6 and O6 regression run passed 517 tests. The final focused run passed 31 tests, including the later confirmation and native-stream checks. An earlier run exposed a redundant source-file read in the capacity preflight. Capability rejection now comes from the original source validator. The archive portability test needs its temporary directory inside the checkout, which was used for the full passing run. Completed scratch fixtures were moved to local temporary storage.

The standalone `crapper`, `dryer` and `mutator` executables were unavailable. The repository's Radon and coverage check measured 84 changed functions, with a maximum CRAP score of 9.375. Its AST comparison found zero matches at or above 0.8. All six deliberate mutations failed their behavioral assertions in an isolated checkout. They tested the edge sign, pre-orientation direction, recentering direction, conditional 4 mm admission, 120 s promotion and strict force threshold. [Verification receipt](handoff.json)

## Reproduction

Use the repository's `teleopit` conda environment and canonical G1/O6 assets. The three output directories below are separate from the archived evidence.

```powershell
python -m teleopit.sim.g1_o6_support.g6_thumb_retention_experiment --output outputs/g1_o6/retention_new run
python -m scripts.g1_o6_g6_reference_replay docs/g1_o6/g6_wrist_grip/results.json outputs/g1_o6/reference_new
python -m scripts.verify_g1_o6_g6_wrist_grip --root outputs/g1_o6/retention_new
python -m scripts.g1_o6_g6_retention_handoff outputs/g1_o6/retention_new/runs/recenter_screen outputs/g1_o6/retention_handoff
python -m scripts.g1_o6_g6_thumb_report outputs/g1_o6/retention_new outputs/g1_o6/reference_new outputs/g1_o6/retention_cache outputs/g1_o6/retention_report
```

Passing the original raw campaign directory as the first reference argument additionally verifies the decoded contact, knee/position and applied-force streams against the archive. The `reference` configuration and the three corrections preserve the original 5 mm Cartesian translation bound, 0.030 rad rotation and per-joint travel bounds, 0.020 rad/s joint slew and 98% effort freeze. At 120 s the original final-window convergence, fresh nominal repeat and finer timestep gates apply.

The captured thumb point is a fixed point on the distal link. The centroid is the normal-force-weighted mean of loaded thumb contacts on the target tread. Clearance is the signed distance to the nearest edge of the tread's usable rectangle in its local plane. Requested wrist rotation is measured after joint, slew and effort constraints and before Cartesian projection. Applied rotation is reconstructed from the executed actuator command. Both are measured relative to the reference arm on the same body pose at that decision.
