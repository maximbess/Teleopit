# Controlled 30-second comparisons

All rows use 30 s requested HOLD and pass strict/contact criteria. Every row fails late motion. Maximum RESET drift alone does not determine ranking. The late-motion cost is the maximum ratio to the unchanged limit across base, COM and both hand translations/orientations; <=1 is required.

C1/C2/C3 and C6 have exactly identical physical acquisition rows to their parent through HOLD entry (253 rows). Preload and closing-target comparisons alter acquisition as well as HOLD and therefore do not isolate only post-entry reference advancement.

| Parent spacing (m per hand) | Change | RESET drift max (mm) | Worst late-motion ratio | Palm migration L/R (mm) | Thumb target advance L/R (rad) | Matched 20–30 s hand share L/R (%) |
|---:|---|---:|---:|---:|---:|---:|
| 0.18 | reference law | 15.722 | 7.347 | 10.610/10.553 | 0.204/0.203 | 0.980/0.984 |
| 0.18 | freeze_all | 9.814 | 3.483 | 5.386/5.351 | 0.000/0.000 | 0.864/0.866 |
| 0.18 | freeze_thumb | 8.568 | 4.893 | 7.220/7.300 | 0.000/0.000 | 0.843/0.845 |
| 0.18 | cap_thumb | 9.769 | 4.967 | 7.627/7.677 | 0.029/0.029 | 0.864/0.865 |
| 0.18 | thumb target 0.40 rad | 12.609 | 5.789 | 9.066/8.995 | 0.124/0.123 | 0.916/0.919 |
| 0.18 | preload 0.10 Nm | 15.003 | 6.395 | 9.205/8.962 | 0.214/0.213 | 0.983/0.992 |
| 0.18 | preload 0.14 Nm | 16.183 | 8.076 | 12.005/11.910 | 0.198/0.194 | 0.980/0.985 |
| 0.16 | reference law | 15.926 | 7.130 | 10.255/10.444 | 0.210/0.203 | 1.015/1.017 |
| 0.16 | freeze_all | 10.102 | 3.531 | 5.447/5.460 | 0.000/0.000 | 0.903/0.902 |
| 0.16 | freeze_thumb | 8.830 | 4.786 | 7.044/7.173 | 0.000/0.000 | 0.882/0.881 |
| 0.16 | cap_thumb | 10.054 | 4.850 | 7.385/7.527 | 0.029/0.029 | 0.903/0.901 |
| 0.16 | preload 0.10 Nm | 14.872 | 5.534 | 8.291/8.310 | 0.219/0.214 | 1.007/1.014 |
| 0.16 | thumb target 0.40 rad | 12.966 | 5.678 | 8.871/9.017 | 0.127/0.125 | 0.950/0.956 |
| 0.16 | arm_feedback | 20.941 | 9.398 | 12.137/12.187 | 0.210/0.203 | 1.110/1.147 |

Freezing thumb targets reduces early drift and palm migration under matched acquisition. Freezing all finger targets improves the worst motion ratio further, while base displacement alone favors thumb-only freezing. This supports continued reference advancement as a contributor, not the sole cause: motion persists with stationary references. The declared arm correction worsens this parent's screen. No result here establishes independent hand capacity or long-duration qualification.

[Full duration results and separate qualification verdicts](COMPARISON.md).
