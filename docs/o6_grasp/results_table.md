# Deterministic O6 evaluation

| Scenario | Unsupported verified (s) | Success | Failure/sliding | Release clear | Peak penetration (mm) |
|---|---:|---|---|---|---:|
| free_motion | 0.00 | False | False | True | 0.00 |
| acquisition | 0.00 | False | False | True | 2.30 |
| fixture_removal | 2.72 | True | False | True | 2.30 |
| pullout_small | 2.72 | True | False | True | 2.30 |
| pullout_large | 0.78 | False | True | True | 2.30 |
| twist_along | 2.72 | True | False | True | 2.30 |
| twist_across | 2.72 | True | False | True | 2.30 |
| combined | 2.72 | True | False | True | 2.30 |
| impulse | 2.72 | True | False | True | 2.30 |
| release | 2.72 | True | False | True | 2.70 |
| friction_low | 2.72 | True | False | True | 1.87 |
| alignment_offset | 2.72 | True | False | True | 2.25 |
| rung_small | 0.00 | False | True | True | 1.79 |
| rung_large | 0.00 | False | True | True | 1.91 |
| rectangular | 0.00 | False | True | True | 1.78 |
| actuator_delay | 2.72 | True | False | True | 1.98 |
| finite_edge | 0.00 | False | True | True | 1.06 |
| fine_timestep | 2.72 | True | False | True | 0.41 |
