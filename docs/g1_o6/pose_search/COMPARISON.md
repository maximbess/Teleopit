# Bounded long-HOLD pose campaign

This report is generated from saved results. A screen is not 120 s qualification.

| Candidate / stage | Strict / late / contact | HOLD achieved (s) | Max RESET drift (mm) | Mean hand shares L/R (%) | First terminal reason |
|---|---|---:|---:|---:|---|
| reference / 60s | PASS / FAIL / PASS | 60.02 | 21.403 | 0.994/0.997 | none |
| reference_hand_spacing_m_0p16 / screen | PASS / FAIL / PASS | 30.00 | 15.926 | 0.963/0.966 | none |
| reference / screen | PASS / FAIL / PASS | 30.00 | 15.722 | 0.925/0.928 | none |
| reference_hand_spacing_m_0p2 / screen | PASS / FAIL / PASS | 30.00 | 15.454 | 0.888/0.891 | none |
| reference_waist_lean_rad_0p2 / screen | PASS / FAIL / PASS | 30.00 | 24.793 | 1.254/1.257 | none |
| reference_waist_lean_rad_0p4 / screen | PASS / FAIL / PASS | 30.00 | 9.906 | 0.407/0.423 | none |
| reference / 120s | FAIL / NOT RUN / PASS | 88.02 | 30.546 | 1.064/1.067 | drift |
| split_11_10 / screen | FAIL / NOT RUN / PASS | 8.00 | 7.719 | -0.074/0.005 | joint_limits |
| split_10_11 / screen | FAIL / NOT RUN / PASS | 7.76 | 7.966 | 0.001/-0.076 | joint_limits |
| rung10_hand_tilt_deg_40 / screen | FAIL / NOT RUN / PASS | 3.02 | 12.609 | -0.216/-0.213 | joint_limits |
| reference_hand_tilt_deg_40 / screen | FAIL / NOT RUN / PASS | 2.90 | 4.197 | 0.573/0.586 | unintended_support |
| rung10_waist_lean_rad_0p2 / screen | FAIL / NOT RUN / PASS | 0.50 | 7.947 | -0.165/-0.166 | joint_limits |
| rung10_waist_lean_rad_0p4 / screen | FAIL / NOT RUN / PASS | 0.50 | 6.059 | -0.156/-0.155 | joint_limits |
| rung10 / screen | FAIL / NOT RUN / PASS | 0.50 | 6.828 | -0.159/-0.156 | joint_limits |
| rung10_hand_spacing_m_0p16 / screen | FAIL / NOT RUN / PASS | 0.50 | 7.020 | -0.161/-0.160 | joint_limits |
| rung10_hand_spacing_m_0p2 / screen | FAIL / NOT RUN / PASS | 0.50 | 6.687 | -0.158/-0.154 | joint_limits |
| reference_hand_tilt_deg_60 / screen | FAIL / NOT RUN / FAIL | 0.50 | 7.349 | 0.601/0.601 | left_hand, right_hand |
| rung10_hand_tilt_deg_60 / screen | FAIL / NOT RUN / FAIL | 0.50 | 19.151 | -0.059/-0.077 | joint_limits, unintended_support, left_hand, right_hand |

Qualified nominal/repeat/fine candidates: NONE.

[Complete configurations, per-hand metrics, late windows, first crossings and qualification status](results.json).

A >50% reduction in either hand's signed 20–30 s share is explicitly flagged in the JSON; it does not establish stronger grasp capacity.

Contact quality requires all scored control samples to retain thumb-specific opposition and accepted ordering, with no physics-rate quality-loss episode lasting 0.5 s. Shorter interruptions remain reported.
