# Run and view the floating redistribution trials

From the repository root on `dex-hand-exp`, activate the `teleopit` environment. The runner verifies the G6 reference files before restoring the measured four-support state. If the reference is missing, rebuild it using the [G6 reproduction command](../rung7_regrasp/COMMANDS.md).

## Replay the recorded cycle in Viser

Run the same command from the repository root in WSL or PowerShell.

```sh
conda activate teleopit
python -m scripts.view.view_rung7_regrasp --run outputs/g1_o6/rung7_redistribution_verified/free_left_1/release
```

Open the [local viewer](http://127.0.0.1:8012). For the right side, use `outputs/g1_o6/rung7_redistribution_verified/free_right_1/release`. The replay uses the measured joint states from the first open request through the final two-second hold. The speed and frame controls let you inspect either regrasp.

The left-hand recording includes a close-up with the measured contact state and time since the first release. [Watch the MP4](media/left-release-regrasp.mp4)

## Rerun the matched experiment

The following commands write both repeats to a fresh local directory. Each trial runs a unilateral release followed by its both-hands-holding control from another reset. Each side uses its own release clock and its own pre-opening force baseline.

```powershell
conda activate teleopit
$runRoot = 'outputs/g1_o6/rung7_redistribution_repeat'
foreach ($hand in 'left','right') {
    foreach ($repeat in 1,2) {
        python -m teleopit.sim.g1_o6_support.regrasp trial --pelvis free --free-controller unassisted --hand $hand --output "$runRoot/free_${hand}_$repeat"
        if ($LASTEXITCODE -ne 0) { throw "Run failed for $hand repeat $repeat" }
    }
}
python -m scripts.g1_o6_regrasp_report $runRoot "$runRoot/report"
python -m scripts.g1_o6_free_regrasp_audit $runRoot "$runRoot/report/native_audit.json"
```

`resolved.json` records the launch positions, velocities and servo commands, model and runner hashes, seed, gains and disabled assistance limits. `commands.npz` contains the measured joint and wrist states at 20 ms intervals. `physics.npz` records every 1 ms physics interval. `contacts.npz` gives each contributing contact and its force direction. The publication script checks matched pre-command states and stance commands. The native audit checks the full episode, measured clearances, regrasp slip and the force sign on each robot geom.

The earlier fixed-pelvis comparison and its commands are recorded separately. [Fixed-pelvis experiment](../rung7_regrasp/COMMANDS.md)
