# Keeping the right knee clear of rung 3 on G6

The matched parent is the foot-specific 16 N ground-reaction controller. The G1 starts at the acquired grasp with both hands on rung 7 and both feet on rung 2. A fresh five-second parent probe must reproduce its archived control trace. The experiment retains the native MuJoCo scene, friction, 20 ms commands, grip and original physical gates.

The controller measures the position and pitch of each foot at HOLD entry. After two seconds of settling it opposes fore–aft motion and pitch through the leg position servos. The right-foot correction also grows when the measured gap from the right knee to rung 3 falls below 40 mm. The two stance configurations use gains of 0.75 and 1.5 on the same 800 N/m and 24 Nm/rad feedback. Each foot's horizontal correction is capped at 8 N and its pitch correction at 2 Nm. The net commanded pelvis wrench stays within 30 N and 5 Nm, including the parent's native-contact preview alternatives. The original joint position, rate, effort and inhibition limits apply.

If both stance configurations fail, the campaign records their actual support forces, knee clearance, finger loading and motion before testing a third configuration. It uses the firmer stance feedback and integrates the mean of the two measured palm pose errors into the pelvis reference. The translation and rotation corrections are capped at 12 mm and 0.020 rad. It uses the same arm and grip commands.

**Qualification** requires strictly positive clearance between rung 3 and all seven collision meshes in the native right-knee group. Any solver or control-endpoint contact between that pair disqualifies a hold, including a contact below the original unintended-support force threshold. Each native step records the integrated joint pose and the closest signed gap after forward kinematics on a private state. The force residual must also stay strictly below 25 N at both sampling rates, and each of the four left non-thumb fingers must carry at least 0.05 N on the intended tread at every native step. The original gates score thumb opposition, the other supports, physical limits and convergence.

Each promoted configuration starts a fresh 5, 30, 60 and 120 s hold at a 1 ms timestep. A qualified 120 s hold then needs a fresh nominal repeat and a 120 s confirmation at 0.5 ms. The final 30 s convergence window uses the floating base, COM and both palms relative to the tread. The campaign stops after the first confirmed pose or after testing the three declared configurations.

Run from the repository root with the native assets installed. Use a fresh output directory for an independent campaign.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_knee_experiment --output outputs/g1_o6/g6_knee_clearance run
```
