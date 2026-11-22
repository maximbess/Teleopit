# Reproduce the rung-7 trials

Run these PowerShell commands from the repository root on `dex-hand-exp`, with the canonical G1 assets installed. Each trial output directory must be fresh. The default source is `outputs/g1_o6/g6_thumb_reference/runs/control_screen`. If it is absent, first rebuild it with the archived right-wrist replay.

```powershell
conda activate teleopit
python -m scripts.g1_o6_g6_reference_replay docs/g1_o6/g6_wrist_grip/results.json outputs/g1_o6/g6_thumb_reference
```

With the source available, reproduce the 30 s floating reference, run both nominal repeats for each hand and pelvis mode, and run the two fixed-pelvis approach offsets. Each free trial command runs its release and hold-control branches from independent resets and matches their clocks.

```powershell
conda activate teleopit
$root = 'outputs/g1_o6/rung7_new'
$runner = 'teleopit.sim.g1_o6_support.regrasp'
python -m $runner reference --output "$root/reference"
foreach ($hand in 'left','right') {
    foreach ($repeat in 1,2) {
        python -m $runner trial --pelvis fixed --hand $hand --output "$root/fixed_${hand}_$repeat"
        python -m $runner trial --pelvis free --free-controller reference --hand $hand --output "$root/free_${hand}_$repeat"
    }
    python -m $runner trial --pelvis fixed --hand $hand --offset-mm -2 --output "$root/fixed_${hand}_minus2"
    python -m $runner trial --pelvis fixed --hand $hand --offset-mm  2 --output "$root/fixed_${hand}_plus2"
}
python -m $runner probe --hand left --probe air --output "$root/neg_air"
python -m $runner probe --hand right --probe wrong_surface --output "$root/neg_wrong_rung"
python -m scripts.g1_o6_regrasp_report $root "$root/report"
```

For a different reference directory, put `--source <run-directory>` before the subcommand. The scoring script verifies identical pre-command states, controller targets, signed reactions and stance commands for each pair. A failed physical transition is a recorded experimental outcome in `outcome.json`. Missing settled windows carry the reason in `matched_events.json`.

## Inspect a trial

The published runs are in `outputs/g1_o6/rung7_regrasp`. The following example prints one floating pair's native force range and its matched effects. The ordering of the four force columns is left hand, right hand, left foot, right foot.

```python
from pathlib import Path
import numpy as np
from teleopit.sim.g1_o6_support.regrasp_results import score_pair

pair = Path("outputs/g1_o6/rung7_regrasp/free_left_1")
with np.load(pair / "release/physics.npz") as native:
    clear = native["phase"] == "clear1"
    print(native["fz"][clear].min(axis=0), native["fz"][clear].max(axis=0))
for row in score_pair(pair)["forces"]:
    if row["window"] == "clear1":
        print(row["support"], row["effect_N"], row["effect_pp"], row["status"])
```

| File | Contents |
| --- | --- |
| `resolved.json` | Seed, physical model and controller identities, geometry names, actuator mapping, step sizes and thresholds |
| `commands.npz` | 20 ms control endpoints, requested arm and grip channels, desired targets, slew-limited targets, applied controls, joint state, rung-frame wrist poses and velocities, internal wrist wrenches |
| `physics.npz` | 1 ms solver interval starts, four signed reactions, other external and fixture reactions, pelvis pose, vertical momentum, assistance, grip and support measurements, slip and limit signals |
| `contacts.npz` | Per-step contact pairs by geom ID, force sign on geom 2, world normals, position height, slip and signed distance |
| `outcome.json` | Event times, completed or interrupted windows, handoffs, first missing stage condition and final momentum |
| `fixture.xml` | Fixed-mode scene with its six-axis pelvis weld |

In the control trace, a NaN arm or grip request indicates delegation to the reference controller. Clearance is capped at 30 mm. A NaN gap marks a phase where no distance query was requested and no external contact occurred. Each solver force belongs to the interval beginning at its recorded time. The report derives pelvis rotation from the recorded quaternions normalized against the initial pose.
