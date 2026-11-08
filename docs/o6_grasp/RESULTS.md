# Deterministic evaluation — 2026-11-08

Run in the `teleopit` conda environment, Windows, Python 3.10.20, MuJoCo 3.8.1,
seed 0. Configuration is [default.json](default.json). Full trajectories, resolved
per-trial configurations and generated scenes are in `outputs/o6_grasp/final/`.
The committed [results.json](results.json) and [scenario table](results_table.md)
are compact records of that run. These are simulation findings, not physical
O6 climbing-load ratings.

## Findings

- Nominal 24 mm circular rung: **2.72 seconds VERIFIED** during 3.0 seconds with
  exactly zero fixture support. The load ramps to 1 N downward, in addition to
  the 0.36998 kg hand's gravity. Measured contact support agrees with gravity plus
  load to within the test tolerance of 0.12 N. End-of-acquisition to holding drift
  stays below 10.24 mm and 0.233 rad. Release opens and clears the rung.
- A 30 N ramp along the rung produces sliding, pullout and failure. It verifies
  for only 0.78 seconds before losing the grasp; the whole trial is a failure.
  The simulation deliberately continues the prescribed load/unload/release
  sequence after failure, so later free-flight displacement is large. This is
  not a measured 30 N failure threshold. There is no floor or hidden catch
  constraint; the finite fixture cannot recover every severe failure. Release
  clearance after a fall does not establish controlled re-acquisition.
- All 18 scenarios finish with measured open-posture clearance. Two are
  fixture-assisted checks and cannot claim unsupported success. Of the remaining
  16, eleven pass the configured verification criteria and five fail.
- The 10 mm-radius rung fails translation/rotation limits. The 14 mm-radius rung
  fails actuator-loading limits. The 24 × 24 mm rectangle fails loading and
  translation limits. The short, offset rung fails translation/rotation limits.
  The reference posture therefore does not generalize across these geometries.
- Ramped moments of 0.15 Nm along/across the rung, combined loading, the short
  force pulse, friction 0.25, a 4 mm alignment shift, and 60 ms actuator delay
  pass this limited one-factor sweep. This is not a statistical robustness claim.

## Numerical sensitivity

| Metric | 2 ms / 100 iterations | 1 ms / 200 iterations |
|---|---:|---:|
| Unsupported verified duration | 2.72 s | 2.72 s |
| Maximum translation during unsupported HOLD | 10.233 mm | 10.189 mm |
| Maximum rotation during unsupported HOLD | 0.2327 rad | 0.2297 rad |
| Maximum penetration during unsupported HOLD | 0.2062 mm | 0.2062 mm |
| Maximum penetration over the entire trial | 2.3003 mm | 0.4070 mm |
| Maximum passive-coupling error | 0.001460 rad | 0.000200 rad |
| Maximum soft joint-limit violation | 0.007372 rad | 0.001990 rad |
| MuJoCo numerical warnings | None | None |

The 2 ms run's largest penetration occurs transiently on the pinky during CLOSE,
not during verified support. The 1 ms comparison preserves holding with much
smaller transient penetration. Thus holding does not depend on that large
closure transient, but closure/contact dynamics remain timestep-sensitive.
Use the 1 ms configuration for closer contact analysis and continue convergence
and physical identification work before interpreting loads as hardware capacity.
Actuator targets remain within limits; actual joint constraints in MuJoCo are
soft, so measured violations are logged and bounded separately. Actuator efforts
reach saturation transiently during closure; persistent unsupported saturation
is rejected by verification.

## Validation

- **94 tests passed** across the new environment and existing hand, runtime,
  interface, simulation-loop and controller tests. The conda environment has an
  unrelated anyio/pytest deprecation warning.
- Native interactive viewer completed a full trial. OPEN, closed, unsupported
  and released images were rendered and inspected. Contact-force display and
  the labeled fixture-status marker are available in the viewer.
- The wheel built with `--no-deps --no-build-isolation`; extracting it and loading
  the scene with an isolated interpreter confirms the packaged URDF and meshes
  are sufficient outside the checkout.
- Source asset SHA-256 checks pass. The existing `SimulationLoop` public import
  remains available, while importing the experiment no longer eagerly loads
  the humanoid runtime.
- Five isolated fault injections were all caught: fixture support leakage,
  accumulating generalized forces, missing off-center moments, wrong rotation
  flattening order and removed command-rate limiting. These are targeted manual
  mutation checks, not a claim of exhaustive mutation coverage.
- Ruff checks pass. Radon and coverage are used for local complexity review;
  the requested named crapper/dryer/mutator tools were unavailable. See the work
  log and local quality report for the checks actually performed.
- Cumulative line coverage from tests plus CLI/viewer runs is 99.45%. The local
  Radon/coverage CRAP estimate is at most 9.0 per function. A production-function
  token comparison found no similarity pairs at or above 0.8 (minimum 40 tokens).

## Remaining physical identification

Identify actuator effort/current curves, response delays, reflected inertia,
damping, nonlinear linkage coupling, friction/contact compliance and collision
shape accuracy. The low-resolution convex link meshes and approximate fingertip
sites need comparison with the real hand. Hardware use additionally needs a
radian/byte calibration and estimators for rung-relative pose and palm velocity.
No RL training or physical grasp validation is claimed by this deliverable.
