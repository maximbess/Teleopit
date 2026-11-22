# Reproduce and watch a floating rung-7 regrasp

Run these PowerShell commands from the repository root in the `teleopit` Conda environment. The runner verifies the G6 scene, source configuration and reference command log at reset. The archived reference, the fixed-pelvis experiments and their reproduction commands are available in [the original experiment](../rung7_regrasp/COMMANDS.md).

```powershell
conda activate teleopit
$root = 'outputs/g1_o6/rung7_free_regrasp'
$report = 'docs/g1_o6/rung7_free_recovery'
foreach ($hand in 'left','right') {
    foreach ($repeat in 1,2) {
        python -m teleopit.sim.g1_o6_support.regrasp trial --pelvis free --free-controller balanced --hand $hand --output "$root/free_${hand}_$repeat"
    }
}
python -m scripts.g1_o6_regrasp_report $root $report
python -m scripts.g1_o6_free_regrasp_audit $root "$report/native_audit.json"
```

Each trial directory contains a release and a both-hands-holding control from independent resets. Choose a fresh `$root` when repeating an existing campaign. The force report checks the full precommand state and all 15 applied stance channels. The native audit checks the 1 ms contacts, commanded wrist arrivals, confirmed clearance, both two-second holds and signed force on all four supports.

## Watch the measured motion in Viser

```powershell
conda activate teleopit
python -m scripts.view.view_rung7_regrasp --hand left
```

Open [the local viewer](http://127.0.0.1:8012). The loop starts at the first open request and ends after the second recovered hold. The camera starts on the selected wrist. Use **Play**, **Speed** and **Frame** to inspect finger opening, wrist clearance and regrasp. Select the other side with `--hand right`. To use a different recording, supply its release directory.

```powershell
python -m scripts.view.view_rung7_regrasp --run outputs/g1_o6/rung7_free_regrasp/free_right_2/release
python -m scripts.view.view_rung7_regrasp --run outputs/g1_o6/rung7_regrasp/fixed_left_1/release
```

The viewer uses saved joint positions and forward kinematics. The four force measurements come from the original simulation's signed contact records.
