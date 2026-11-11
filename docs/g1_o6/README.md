# Physical G1 / O6 hold

The default CPU MuJoCo experiment uses both authentic O6 hands on rung 11, thumbs below and fingers above, with both feet on rung 04. It completes the qualified 10 s and 30 s four-contact holds, including a 0.5 ms timestep trial. Most weight is carried by the feet. This does not establish hanging, release/reacquisition, climbing or hardware capability.

The consolidation preserves the successful trajectory exactly. See the [before/after validation](consolidation/README.md), [work log](consolidation/WORK_LOG.md) and [historical experiments](HISTORY.md).

## Run

Use conda `teleopit` from the repository root on `dex-hand-exp`. Install missing canonical assets with `python scripts/setup/install_climb_assets.py`. Choose fresh output paths for each experiment.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --output outputs/g1_o6/current
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --viewer --output outputs/g1_o6/current_viewer
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --video --output outputs/g1_o6/current_video
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --config docs/g1_o6/posture/hold_30s.json --output outputs/g1_o6/current_30s
```

Space pauses/resumes the viewer. Contact normals default off. Closing records `viewer_closed`; completing the strict hold ends at `HOLD_COMPLETE`. Release success always remains false. Every mode uses the same environment, settings and logging contract. The old `--posture` flag is an argument error; historical commands require their archived revision.

Runs write `resolved.json`, `scene.xml`, `steps.jsonl`, `summary.json` and signed left/right wrist traces. Video adds a labeled MP4 and initial/final PNGs. Wrist exports are simulation measurements: world axes, parent-on-hand sign, moments about the wrist mount, with no extra gravity term.

## Configure the current grasp

`--config PATH` overrides simulation timing, perturbations and acceptance settings. `--grasp-config PATH` separately overrides the single `GraspPose`; omitted fields retain its defaults. Both options mean the same thing in headless, viewer, video and nominal diagnostics. Save this JSON and pass its path to `--grasp-config`:

```json
{
  "hand_rungs": [11, 11],
  "waist_lean_rad": 0.3,
  "hand_tilt_deg": 50.0,
  "grip_offset_m": [-0.023, 0.0, -0.020],
  "hand_spacing_m": 0.18,
  "closed_pose": [0.05, 0.5, 1.35, 1.35, 1.35, 1.35],
  "finger_preload_Nm": 0.12
}
```

Hand rungs are left/right integers in 1..14; feet remain on rung 04. Closing targets are radians in the six named active-joint order recorded for each hand in `command_joint_mapping`. Unknown fields, nonfinite values, invalid sizes/rungs, out-of-range closing targets and invalid preload fail before stepping. Changed geometry must still pass the unchanged reachability, penetration and HOLD gates; arbitrary parameter choices are not qualified.

`wrist_offset_x` adds world X displacement to the current mount target, in addition to `grip_offset_m`. It no longer changes a discarded IK solve. The optional `--suite` contains nominal, fine-step, body offset, wrist offset, low friction, disturbance and delay cases. It has no release case. Only trials listed in the validation report were executed for this consolidation; the suite is not a robustness claim.

`resolved.json` uses `g1_o6_run_v2` and records full pose/controller settings, units, named command mapping, targets, initial state, assets and runtime. New runs never infer rung 08 or legacy finger targets from missing metadata.

## Control and physical limits

The sequence is `INITIALIZE → ACQUIRE → REMOVE_ASSISTANCE → HOLD → HOLD_COMPLETE`, or `ABORT` on failure. Reset preloads existing body servos using reference gravity torque minus estimated foot-load torque. Acquisition uses a bounded pelvis pose stabilizer without the former gravity lift. HOLD has exactly zero external assistance and no state overwrite.

The shared finger law is `min(amount * closed_target, measured_angle + preload / kp)`, followed by physical target clipping, the 1 rad/s command limit and configured delay. **Preload** is commanded squeeze of 0.12 Nm; the separate **physical actuator cap** remains the provisional 0.35 Nm. Neither is a manufacturer payload claim.

Masses, inertias, meshes, contact masks/friction, mimics, ranges, gains, actuator limits, solver settings and strict HOLD thresholds are unchanged. The shared hand builder and training environment are unchanged. The preserved [asset audit](audit.json) and resolved run metadata identify the physical model. No hand-to-rung weld, mocap support or isolated fixture exists in the coupled scene.

## Diagnostics

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_diagnostics nominal --output outputs/g1_o6/current_diagnostics
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_diagnostics free-close --side both --source-run outputs/g1_o6/current --output outputs/g1_o6/current_free_close
```

Nominal diagnostics accept both configuration options and retain physical-step events, actuator/constraint telemetry, force/moment balance, both drift references and rung-relative transforms. `--continue-after-failure SECONDS` permits only a bounded, unscored continuation of failed HOLD; the original failure stays failed. A successful HOLD never continues and has no abort time.

Free closing is a fixed-base mechanism test. The load-response runner retains a free wrist, applies signed loads at the mount and scores only zero-fixture intervals. Its `--source-run` is also required. It copies current source geometry, closing targets, preload law, rate and delay; it reports its independent acquisition schedule and removed G1/housing geometry. Ordinary MuJoCo gravity acts once. Historical logs remain readable/plotable; running an old controller requires its original revision.

Both hands passed fixed-base gravity-on/off free-close smoke runs. Two isolated loaded smoke trials lost their grasps before completing the required load plateau and remain failures. No new load-capacity matrix was run. Increased hand loading, three-support balance and rung-to-rung climbing remain subsequent experiments.
