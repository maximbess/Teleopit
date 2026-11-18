# G6 whole-body stabilization experiment

The starting pose is the previous synchronous 0.35 rad thumb-pitch grasp, with 0.5 rad thumb yaw and 0.12 Nm finger preload. Both hands target rung 7 and both feet target rung 2. The pelvis reset offset is −40 mm in x. A fresh arm-only control must reproduce the previous five-second applied trace byte for byte.

All candidates use the native G6 scene, gravity, collision pairs, contact material, existing 20 ms controller and original finite assistance removal. At unsupported HOLD entry they latch the measured world pelvis pose and torso orientation. A waist orientation servo acts alongside the parent's bounded wrist controller. The additional twelve leg and three waist position targets can move at most 0.12 rad from their native clipped baseline and 0.06 rad/s. The existing joint limits, force limits, structural and slip inhibition apply.

The first pair uses damped differential inverse kinematics on each stance leg and tests gains 0.8 and 1.6. If both fail, record the measured support actors and terminal gates, then try a second pair at the same gains. That pair distributes a pelvis position and attitude correction between the two feet and maps it to leg servo offsets via the contact Jacobians. The internal desired net force and moment are capped at 30 N and 5 Nm. Only the existing position servos actuate the robot.

Every candidate starts fresh at each stage. Promotion requires the original physical and anatomical gates and a minimum of 0.05 N on **each** left non-thumb finger on its intended tread at **every** unsupported physics step. The stages are 5, 30, 60 and 120 seconds at 1 ms. A qualified 120-second result must also pass a fresh nominal repeat and a 120-second run at 0.5 ms. Each long run retains the original final 30-second convergence criteria on the base, COM and both wrists. The search ends on the first fully confirmed candidate or after exhausting the four configurations.

Run from a clean checkout of the experiment revision with the G1 assets installed:

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_whole_body_experiment --output outputs/g1_o6/g6_whole_body run
```
