# Native G6 grasp acquisition — measured outcome

**Thumb opposition was established. A stable G1 pose was not qualified.**
The bounded campaign completed 15 physical trials on `dex-hand-exp`, using
the exact simulator bytes frozen at `8134ee06f595cd064dfcecc08cbdbd71cef3ab77`.
The robot retained the lowered pose: hands on tread 7, feet on tread 2,
root X offset −40 mm. Native friction and every original physical,
anatomical-contact and convergence gate remained unchanged.

The native command order is thumb **pitch**, thumb yaw, then four fingers.
Increasing the named pitch target from 0.05 to 0.25 or 0.35 rad, with yaw
remaining 0.5 rad, established loaded thumbs below the tread opposing all
four loaded fingers above it. This developed during the original assistance
removal phase; acquisition assistance was zero throughout every scored HOLD.
The simultaneous-closing pair succeeded at grasp acquisition, so the four
predeclared thumb-first/yaw variants were left unrun.

The table's PASS means the original physical **and** anatomical-contact checks
passed for that duration. It does not mean stable-pose qualification. All
completed 30 s and 60 s motion windows failed the original convergence check.

| Candidate | 5 s probe | 30 s screen | 60 s hold | 120 s attempt |
|---|---|---|---|---|
| Original control | FAIL at 0.50 s; no thumb load | Gated off | Gated off | Gated off |
| Pitch 0.25 | PASS | PASS | PASS, 60.02 s | FAIL at 87.96 s: drift and contact validity |
| Pitch 0.35 | PASS | PASS | Physical FAIL, contact PASS, 60.02 s | Gated off |
| Pitch 0.35 + slow feedback | PASS | PASS | PASS, 60.02 s | Not selected |
| Pitch 0.35 + fast feedback | PASS | PASS | PASS, 60.02 s | FAIL at 107.02 s: drift and contact validity |

The control reproduced the prior lowered-control trajectory byte for byte.
Its thumbs carried zero normal force and it failed both hand gates at 0.5 s.
For the acquired pitch-0.25 30 s screen, minimum thumb loads were 0.681 N
(left) and 0.415 N (right), above the unchanged 0.05 N loaded-contact threshold.
Both hands remained anatomically valid throughout that screen and its fresh
60 s hold.

Pitch 0.35's 60 s physical failure was a single original control-sample slip
violation: left-hand slip reached 0.02098146 m/s at simulator time 64.50 s
(about 59.46 s into HOLD), above the 0.02 m/s limit. The full-interval gate
correctly rejected the trial despite retained anatomical contact.

After the two pitch variants failed the early convergence check, an immutable
premise audit recorded the actual actors: two loaded thumbs, eight loaded
fingers and two feet, with continuing wrist/body motion. The original ranking
selected `pitch_035_screen` as the parent of the two reserved stabilizers.
They latched the measured wrist reference at HOLD entry and used only the
predeclared 0.005/0.020 rad/s rates, 5 mm translation bound, 0.03 rad rotation
bound and 0.03 rad joint-offset bound. All applied commands were checked
independently using the saved model.

Both stabilizers reached command bounds. The fast variant reduced 30–60 s
left/right wrist rotation slopes to 0.001047/0.000884 rad/s; these remained
above the original 0.0005 rad/s threshold. It received the last long-trial
slot because it ranked ahead of the slow variant, without changing its
settings. In its fresh long run, left-hand order/validity was continuously
lost for 27.761 s near the end: the left pinky unloaded while the thumb
remained loaded and opposed. By the final abort, the left middle and ring
fingers were unloaded too. The original drift gate stopped the run at
107.02 s, with 30.625 mm displacement. The fixed-arm pitch-0.25 long run had
stopped earlier at 87.96 s and 31.331 mm, with 24.331 s continuous left-hand
validity/order loss.

Neither attempt completed the required 90–120 s convergence window. No
nominal repeat, 0.5 ms confirmation, load transfer, release or hand-transfer
qualification was promoted. The remaining coupled motion and finger
unloading are unresolved. Duration and visible thumb contact are not treated
as a stable-pose pass.

The experiment used 5/9 probe slots, 4/8 screen slots, 4/4 sixty-second slots
and 2/2 initial long-trial slots. A scratch Windows batch-launch error produced
two ERROR records before simulation started. Both are preserved, with explicit
identical technical recoveries in separate directories. These are not physical
retries. The corrected launcher was subsequently exercised by the two-worker
stabilization batch. Physical closure verified that the simulator/runtime
identity still matched the clean freeze before report-only changes began.

The complete physics audit passed all 651,240 physics rows, including 575,640
scored HOLD steps with zero applied assistance. The observed hand-contact
friction vector was `[1.8, 1.8, 0.02, 0.002, 0.002]`. Each of the 15 saved
MJB/XML scene pairs matched across 584 numeric model fields, and the reviewed
default model/reset matched across 471 fields. No housing or unintended
support was used in scored HOLD. The exact 110-file simulator snapshot and
all recorded run-file checksums passed verification.
These audit PASS results certify evidence integrity. The physical
stable-pose outcome remains FAIL.

The implementation passed 299 G1/O6 regressions before the physical campaign.
The report recovery changes passed 18 targeted tests, followed by 16 final
history/anatomy tests including all 12 recovery/report cases. All 16 targeted fault
injections were killed by behavioral assertions. The initial axis-swap probe
first failed through a range check; its test was strengthened and the repeated
probe failed through the intended mapping assertion. Initial logs are retained.
Measured quality covered 63 changed production/report functions: maximum
CRAP 9.0, none at or above 10, and no AST duplication at or above 0.8. Named
crapper/dryer/mutator tools were unavailable; the recorded substitutes use
Radon, measured current-source coverage, AST matching and targeted behavioral
faults. Report coverage collected before report changes was excluded from
the final measurement.

Evidence and reproduction entry points:

- [Protocol and commands](README.md), [complete work log](WORK_LOG.md).
- [Immutable campaign ledger](../../../outputs/g1_o6/g6_grasp_acquisition/campaign_state.json), [closure proof](../../../outputs/g1_o6/g6_grasp_acquisition/closure_verification.json).
- [Physics/source verification](../../../outputs/g1_o6/g6_grasp_acquisition/verification.json), [independent contact and command verification](../../../outputs/g1_o6/g6_grasp_acquisition/report/acquisition_verification.json).
- [Candidate table](../../../outputs/g1_o6/g6_grasp_acquisition/report/candidates.csv), [actor analysis](../../../outputs/g1_o6/g6_grasp_acquisition/actor_analysis.json), [original control-sample failure analysis](../../../outputs/g1_o6/g6_grasp_acquisition/control_failure_analysis.json).
- [Acquired 60 s replay](../../../outputs/g1_o6/g6_grasp_acquisition/videos/acquired_60s/replay.mp4), [final failed long replay](../../../outputs/g1_o6/g6_grasp_acquisition/videos/final_bounded_hold/replay.mp4). Replays use recorded states and display-only forward kinematics; they do not rerun physics.
- [Exact runtime byte snapshot](../../../outputs/g1_o6/g6_grasp_acquisition/source/runtime_source.zip), [snapshot hashes](../../../outputs/g1_o6/g6_grasp_acquisition/source/runtime_source.json).

Use the recorded runtime revision and exact source byte snapshot for physical
reproduction; Windows checkout line endings can differ from archived bytes.
Use a fresh output directory for a new campaign. The closed ledger rejects
changed source or settings. The later report-only revision validates recovered
attempt history without changing the physical experiment.
