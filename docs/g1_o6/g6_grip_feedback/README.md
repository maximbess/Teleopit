# Bounded grip feedback on G6

The G1 starts with both hands on rung 7 and both feet on rung 2. The matched control is the preceding settled-grip controller with the foot-specific 16 N force preview and a knee-clearance guard. The new configurations use its acquisition, native friction, arm and leg limits, finite 0.12 Nm finger preload and physical gates. [Matched control measurements](../g6_knee_clearance/OUTCOMES.md)

The initial experiment permits three refinements. After five seconds of unsupported HOLD, capture requires both anatomically valid grasps, at least 0.35 N on every non-thumb finger and no structural or slip inhibition. The controller captures the twelve applied servo targets and each palm's height above its tread. It adjusts the closure limits for thumb pitch and the four fingers of each hand. Thumb yaw retains its captured limit. Every limit stays within 0.12 rad of capture and changes by at most 0.04 rad/s. The native closing command limits the applied preload. Inhibition freezes the feedback limits, and the existing backoff still acts on the body.

The first pair regulates target-tread normal loads with gains of 0.04 and 0.12 rad/(N·s), a 0.08 N deadband, a 0.70 N thumb target and a 1.40 N target for each finger. Loss of thumb opposition requests bounded thumb closure. If both configurations fail, a third configuration regulates each palm's captured height with a shared finger closure rate of 6 rad/(m·s). Each finger also responds to loads below 1.00 N. The thumbs retain the higher-gain opposition feedback.

The runner tests a fresh 5 s matched control and requires its complete control trace to match the archived probe. Each candidate advances through fresh 5, 30, 60 and 120 s holds. A stable pose requires a second nominal 120 s hold and a 120 s hold at 0.5 ms. Both thumbs must have loaded opposing contact normals on their target tread. All four left fingers must carry at least 0.05 N on their tread. These checks run at each native physics step and 20 ms control endpoint, alongside positive knee clearance, zero right-knee/rung-3 contact, a force residual strictly below 25 N, and all original physical and anatomical gates. Shorter holds screen physical support and record any available drift window. Each 120 s hold must pass convergence over its final 30 s. Translation excursion and slope limits are 2 mm and 0.05 mm/s. Rotation excursion and slope limits are 0.01 rad and 0.0005 rad/s. The limits apply to the floating base, center of mass and palms wherever the corresponding signal is defined.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_grip_feedback_experiment --output outputs/g1_o6/g6_grip_feedback run
```

The verifier reconstructs thumb opposition from native contact normals on each intended tread, replays knee geometry from the recorded joint positions and checks force balance at both cadences. It also reconstructs each commanded closure limit from the previous contact loads and palm heights, then checks the applied targets against native preload, slew and delay.

```powershell
conda run --no-capture-output -n teleopit python -m scripts.verify_g1_o6_g6_grip_feedback --root outputs/g1_o6/g6_grip_feedback
```

Both pressure gains lost left-thumb opposition between controller observations. One additional configuration is reserved if all three initial configurations fail. Its runner checks the initial specification, terminal decisions and raw evidence hashes before starting. It repeats the palm-height control probe, then uses the same controller with a one-interval preview of both thumb contacts. After selecting a leg command, it can evaluate up to three hand alternatives within the original closure limits. Each forecast measures the weakest left finger, both opposed thumbs and the largest force residual through the native solver steps and the interval endpoint. The controller prefers a command that keeps 0.50 N on each opposed thumb, 0.35 N on every left finger and the residual below 25 N. The 0.05 N minimum contact load, opposing-normal test and all promotion gates also apply to this configuration.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_grip_preview_experiment --output outputs/g1_o6/g6_grip_preview run
conda run --no-capture-output -n teleopit python -m scripts.verify_g1_o6_g6_grip_preview --root outputs/g1_o6/g6_grip_preview
```

The measured outcomes include the stage decisions, native minima, first failures and both full-source revisions. [Experiment results](OUTCOMES.md)
