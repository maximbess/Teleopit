# O6 unsupported hold on the climbing rung

The `ladder-rung` preset now holds the repository's **700 × 110 × 70 mm** rung
under gravity and releases cleanly. The default trial and 31-second unsupported
holds pass at both 1 ms and 0.5 ms physics steps with the existing verification
thresholds. This is a simulation result for the provisional O6 model, not a
hardware payload or climbing certification.

## Run

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --viewer
conda run -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --output outputs/o6_grasp/ladder_hold
```

Space pauses/resumes the viewer. Normals and force arrows remain hidden by
default. The default trial still lasts 9.5 seconds, including fixture-assisted
acquisition, 3 seconds unsupported, and release. To inspect the longer hold:

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --config docs/o6_grasp/ladder_hold_long.json --viewer
```

The long config extends HOLD to 32 seconds: 1 second removes the wrist fixture,
then 31 seconds are unsupported. It does not change the verification thresholds.
Run the physical regressions, including the finer-step long hold, with:

```powershell
conda run -n teleopit python -m pytest tests/test_o6_ladder_hold.py -q --basetemp outputs/o6_grasp/pytest-ladder-hold
```

## What changed

The former upright wrist and small-cylinder curl saturated all six actuators
against the broad rung. Simply reducing closure produced short-lived holds that
eventually slipped off the edge. The new wrist placement lets the fingers curl
over the top while the thumb opposes below; joint targets near that loaded
configuration provide finite preload.

| Setting | Ladder preset value |
|---|---|
| Mount origin, world metres (`palm_pos`) | `[0.016, 0, 0.535]` |
| Mount orientation, wxyz (`palm_quat`) | `[0.96360256, -0.22279641, -0.14719825, -0.01282755]` |
| Approximate XYZ Euler angles | roll −26.4°, pitch −16.8°, yaw +2.45° |
| Closed commands, radians | `[0.05, 0.14, 0.77, 0.98, 1.04, 1.07]` |

Command order is thumb pitch, thumb yaw, index, middle, ring, pinky. `palm_pos`
names the mounting origin, not the palm diagnostic site. The open commands remain
zero. Source values live in `ladder_rung_config()` in `presets.py`.

Rung geometry/contact properties, masses, actuator gains and limits, mimic
couplings, control timing, fixture removal and success criteria are unchanged.
The nominal round-rung preset is unchanged. No constraint attaches the hand to the
rung, and no fixture or extra external wrench supports the gravity-only hold.

## Measured results

MuJoCo 3.8.1, Windows, conda `teleopit`, seed 0:

| Trial | Unsupported / verified time | Peak translation | Peak rotation | Release |
|---|---:|---:|---:|---|
| Original gravity baseline | 3 / 0 s | 26.85 mm | 0.472 rad | Clear |
| New default, 1 ms | 3 / 2.72 s | 1.29 mm | 0.0232 rad | Clear |
| New long hold, 1 ms | 31 / 30.72 s | 4.10 mm | 0.115 rad | Clear |
| New long hold, 0.5 ms | 31 / 30.72 s | 4.05 mm | 0.118 rad | Clear |
| Bundled synthetic replay, 1 ms | 3 / 2.72 s | 1.69 mm | 0.0212 rad | Clear |

All new trials start without penetration and pass unsupported verification without
persistent failure or numerical warnings. Across the gravity trials, peak
penetration is at most 0.115 mm, mimic error 0.000263 rad, and joint-limit violation
0.0141 rad (below the existing 0.02 rad gate). Peak actuator effort is below 74% of
the unchanged 0.35 Nm limit. The 0.5 ms check also doubles solver iterations to 400
and tightens tolerance to 1e-12.

The regression checks loaded opposing surfaces including the thumb, zero fixture
and external wrench during holding, and mean vertical contact force matching the
hand's approximately 3.63 N weight. Exported full logs, resolved configs, scene
XML, summaries and inspection images are in `outputs/o6_grasp/ladder_posture/`.

These results cover the specified placement and finite hold durations. Candidate
searches showed sensitivity to placement and initial overlap; this is not yet a
validated approach region for a moving humanoid. Next steps are approach/release
tolerance testing, disturbances and recorded climbing loads. The bundled replay
remains synthetic; see [wrist replay](WRIST_REPLAY.md) for its contract.
