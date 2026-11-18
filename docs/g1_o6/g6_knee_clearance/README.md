# Keeping the right knee clear of rung 3 on G6

The matched parent is the foot-specific 16 N ground-reaction controller. The G1 starts at the acquired grasp with both hands on rung 7 and both feet on rung 2. A fresh five-second parent probe must reproduce its archived control trace. The experiment retains the native MuJoCo scene, friction, 20 ms commands, grip and original physical gates.

The soft stance configuration passed 60.02 s. All four refinements exhausted their qualification stages. [Measured outcomes and trajectories](OUTCOMES.md) · [Per-trial measurements](results.json)

The controller measures the position and pitch of each foot at HOLD entry. After two seconds of settling it opposes fore–aft motion and pitch through the leg position servos. The right-foot correction also grows when the measured gap from the right knee to rung 3 falls below 40 mm. The two stance configurations use gains of 0.75 and 1.5 on the same 800 N/m and 24 Nm/rad feedback. Each foot's horizontal correction is capped at 8 N and its pitch correction at 2 Nm. The net commanded pelvis wrench stays within 30 N and 5 Nm, including the parent's native-contact preview alternatives. The original joint position, rate, effort and inhibition limits apply.

If both stance configurations fail, the campaign records their actual support forces, knee clearance, finger loading and motion before testing a third configuration. It uses the firmer stance feedback and integrates the mean of the two measured palm pose errors into the pelvis reference. The translation and rotation corrections are capped at 12 mm and 0.020 rad. It uses the same arm and grip commands.

**Qualification** requires strictly positive clearance between rung 3 and all seven collision meshes in the native right-knee group. Any solver or control-endpoint contact between that pair disqualifies a hold, including a contact below the original unintended-support force threshold. Each native step records the integrated joint pose and the closest signed gap after forward kinematics on a private state. The force residual must also stay strictly below 25 N at both sampling rates, and each of the four left non-thumb fingers must carry at least 0.05 N on the intended tread at every native step. The original gates score thumb opposition, the other supports, physical limits and convergence.

Each promoted configuration starts a fresh 5, 30, 60 and 120 s hold at a 1 ms timestep. A qualified 120 s hold then needs a fresh nominal repeat and a 120 s confirmation at 0.5 ms. The final 30 s convergence window uses the floating base, COM and both palms relative to the tread. The campaign stops after the first confirmed pose or after testing the three declared configurations.

Run from the repository root with the native assets installed. Use a fresh output directory for an independent campaign.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_knee_experiment --output outputs/g1_o6/g6_knee_clearance run
```

Reconstruct the signed gap from every saved joint configuration and cross-check the native contacts, control endpoints, finger loads, servo bounds and convergence with the following command. Add `--run-id` to verify one completed trial.

```powershell
conda run --no-capture-output -n teleopit python -m scripts.verify_g1_o6_g6_knee --root outputs/g1_o6/g6_knee_clearance
```

## Conditional settled-grip trial

The initial 30 s screens showed that stronger stance feedback increased knee approach while all eight non-thumb finger targets continued advancing. A separate follow-up ledger reserves one final configuration if the three first-ledger configurations all fail. It checks their terminal summaries against their original hashes before starting. The total search bound is four refinements.

The final configuration uses the matched foot-specific ground-reaction parent. When clearance falls below 35 mm it requests a pelvis retreat along the measured approach axis, capped at 8 mm. After five seconds of HOLD, it captures the twelve applied finger and thumb servo targets at an update with both anatomically valid grasps, at least 0.35 N on each non-thumb finger, and no structural or slip inhibition. Each subsequent hand command takes the smaller of the native preload command and its captured target. The original opening response, joint limits, delay and command slew still apply. The parent's native force preview receives the complete capped command before choosing leg targets. The new ledger uses the same force, clearance, finger and pose gates and the same fresh stages.

Keep the first ledger at `g6_knee_clearance` beside the follow-up output directory when reproducing the conditional trial.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_knee_retention_experiment --output outputs/g1_o6/g6_knee_retention run
conda run --no-capture-output -n teleopit python -m scripts.verify_g1_o6_g6_knee_retention --root outputs/g1_o6/g6_knee_retention
```
