# Ladder rung and wrist-load replay

The `ladder-rung` preset uses the climbing task's rectangular rung: **700 mm
long, 110 mm deep, 70 mm high**, with length along world Y. Dimensions and rung
friction/solver properties come from `teleopit/sim/ladder_geometry.py`, shared with
the training scene. These are repository model values, not measurements of a
physical ladder. The standalone scene translates one fixed rung to the hand bench;
it does not load the humanoid or the rest of the ladder.

There is no wrist-load recording yet. The supplied example is explicitly
**synthetic** and only exercises the replay interface. A successful synthetic
trial would not establish a climbing load rating.

## Run

From the repository root, in the `teleopit` conda environment:

```powershell
# Actual model rung, gravity only, no extra wrist load
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --viewer

# Synthetic wrist-load example, interactive
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --wrist-load docs/o6_grasp/wrist_load_example.json --viewer

# Save diagnostics and an exact copy of the load trace
conda run -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --wrist-load docs/o6_grasp/wrist_load_example.json --output outputs/o6_grasp/ladder_replay

# Export four inspection images
conda run -n teleopit python -m teleopit.sim.o6_grasp --preset ladder-rung --wrist-load docs/o6_grasp/wrist_load_example.json --render --output outputs/o6_grasp/ladder_replay/frames
```

Space pauses/resumes both physics and replay time. Contact normals and force
arrows remain hidden by default. The camera looks from the hand side of the broad
rung. The nominal small round-rung experiment remains the default without
`--preset`.

The preset uses a 1 ms physics step, 20 ms control step, 200 solver iterations and
1e-11 solver tolerance. It places the hand at the near upper edge without initial
penetration. The original closing posture and provisional actuator limits remain
uncalibrated for this larger cross-section.

## Trace contract

Use JSON in this exact top-level shape (the bundled example has a longer trace):

```json
{
  "schema": "o6_wrist_load_v1",
  "source": {"kind": "synthetic", "description": "Hand-authored interface test"},
  "units": {"time": "s", "force": "N", "torque": "N*m"},
  "frame": "world",
  "application_site": "wrist_mount",
  "point_m": [0, 0, 0],
  "samples": [
    {"time_s": 0, "force_N": [0, 0, 0], "torque_Nm": [0, 0, 0]},
    {"time_s": 1, "force_N": [0, 0, -1], "torque_Nm": [0, 0.02, 0]},
    {"time_s": 2, "force_N": [0, 0, 0], "torque_Nm": [0, 0, 0]}
  ]
}
```

- `source.kind`: `synthetic`, `simulation` or `hardware`. A nonempty description
  identifies the generator/recording and its provenance. Label a converted
  recording truthfully; do not rename the synthetic example as recorded data.
- `force_N` and `torque_Nm`: external force and free moment **on the hand** at the
  application point, in XYZ order. Reverse the sign if your recording describes
  the hand's force on the arm. Gravity is already simulated; do not add it twice.
- `frame`: axes for both vectors, either `world`, `palm` (`palm_frame`) or `rung`
  (`rung_frame`). Palm axes follow the current simulated hand. Rung axes follow
  its fixed orientation. A trace recorded in another sensor frame needs explicit
  conversion before loading.
- `application_site`: `wrist_mount` or `palm_frame`. `point_m` is an offset in
  **that site's axes**, independently of the vector frame. The two sites have
  the same orientation; the palm origin is 60 mm along the wrist's local +Z.
- The simulator rotates both vectors to world axes and shifts the moment to the
  palm origin using `M_palm = M_point + (p_point - p_palm) × F`. Logged
  `external_wrench` is `[Fx,Fy,Fz,Mx,My,Mz]` in world axes about the palm.
- There must be at least two finite samples, beginning at exactly zero, with
  strictly increasing timestamps. Each sample has exactly the three fields
  shown. Units, vector sizes, metadata and timestamps are checked at load time.

## Timing and configuration

The environment reads the file once and owns an immutable snapshot. Reset rewinds
the simulation without rereading the file. Wrenches interpolate linearly at every
physics step using simulation time. Time zero is the end of fixture removal in
HOLD. OPEN, CLOSE, the fixture ramp, and RELEASE receive no replay load. In the
default `unsupported` evaluation, fixture gain is zero throughout replay. An
explicit `acquisition` evaluation retains support and cannot certify unsupported
holding.

There is no looping, extrapolation or tail hold. Outside the trace the extra load
is zero. By default a smooth 0.1 s fade at each end prevents discontinuities;
`--replay-fade 0` disables fading for exact piecewise-linear amplitudes. The two
fades must fit within the trace. `--replay-scale 0.5` explicitly halves both force
and moment; the default is 1. Loads are not automatically normalized or clipped.
Replay replaces the analytic load, with no residual force accumulation.

HOLD extends when necessary to contain fixture removal, the entire recording and
the unload interval. The normal RELEASE/catch phase follows. As with the nominal
evaluation, a failed grasp is logged but the trial continues through release.

`--config` applies JSON overrides to the chosen preset. Avoid passing the complete
nominal `default.json` when you intend to retain ladder dimensions. Config files
can set `load_profile: "replay"`, `wrist_load_path`, `replay_scale` and
`replay_fade_duration`; paths in a config resolve relative to that config file.
CLI replay options take precedence. The analytic `--suite` cannot be combined
with replay because its scenarios substitute their own loads. Without replay,
the ladder suite scales both rectangular width and height in its size cases.

## Outputs and observed baseline

A headless trial writes `ladder_rung.jsonl`, `.summary.json`, `.config.json`,
`.xml`, and `.replay.json`. The last file preserves the exact input bytes. Its
SHA-256, source metadata, frames, duration and sample count are in the summary.
The saved config points to this adjacent snapshot and includes the extended HOLD
duration, so it remains runnable after the original trace is moved:

```powershell
conda run -n teleopit python -m teleopit.sim.o6_grasp --config outputs/o6_grasp/ladder_replay/ladder_rung.config.json --output outputs/o6_grasp/reproduced
```

Logs include requested/scaled trace vectors and the world wrench actually applied
at the palm. The summary reports the first persistent failure and its wrench,
replay-window verification duration, and sampled force/moment peaks. Diagnostics
are recorded once per control step (last physics substep); these peaks are not
continuous-time extrema. Replay values remain diagnostics, outside the actor's
unchanged 48-value observation.

The bundled synthetic trace, seed 0, MuJoCo 3.8.1 on Windows produced:

| Metric | Result |
|---|---:|
| Initial penetration | 0 mm |
| Replay duration / unsupported interval | 2.0 s / 3.0 s |
| Verified unsupported holding | 0 s; **failed** |
| First persistent failure | 4.30 s (translation, rotation, actuator loading) |
| Maximum translation / rotation drift | 27.06 mm / 0.473 rad |
| Peak applied force / moment about palm | 1.413 N / 0.1071 Nm |
| Peak penetration / mimic error | 0.309 mm / 0.000207 rad |
| Peak joint-limit violation | 0.0749 rad (above the 0.02 rad limit) |
| Numerical warnings / release clear | None / yes |

This is a reproducible failing grasp benchmark. The next controller work is to
find an opposing-contact posture for this broad rung and verify unsupported
holding under gravity before increasing loads. Once a climbing simulation or
hardware test yields wrist wrenches with known sign, axes and origin, convert
them to this schema and replace the synthetic input. The present result does not
validate O6 ladder climbing or physical hand strength.

The full gravity-only preset was also run: 0 s verified holding, first persistent
failure at 4.30 s with the same reasons, maximum drift 26.85 mm / 0.472 rad,
no numerical warnings and a clear release. Its logged external wrench is zero.
This establishes that the posture needs work even before additional wrist loads.
