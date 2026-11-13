# Ladder/material measured results

Software checks and physical qualification are separate. Every promoted run starts from a fresh reset.

| Condition | Stage | HOLD s | Strict | Contact | Late motion | Terminal cause |
|---|---|---:|---|---|---|---|
| G0_reference | screen | 30.00 | PASS | PASS | FAIL |  |
| G0_reference | 120s | 88.02 | FAIL | PASS | NOT RUN | drift |
| G1_reference | screen | 0.50 | FAIL | FAIL | NOT RUN | left_hand, right_hand |
| G2_reference | screen | 0.50 | FAIL | PASS | NOT RUN | joint_limits, unintended_support |
| G3_reference | screen | 0.50 | FAIL | FAIL | NOT RUN | unintended_support, left_hand, right_hand |
| G4_reference | screen | 0.50 | FAIL | PASS | NOT RUN | joint_limits |
| G5_reference | screen | 0.50 | FAIL | FAIL | NOT RUN | drift |
| G6_reference | screen | 0.50 | FAIL | FAIL | NOT RUN | unintended_support |
| G0_H_0p4 | screen | 30.00 | PASS | PASS | FAIL |  |
| G0_H_0p6 | screen | 30.00 | PASS | PASS | FAIL |  |
| G0_H_0p8 | screen | 30.00 | PASS | PASS | FAIL |  |
| G0_H_1p2 | screen | 30.00 | PASS | PASS | FAIL |  |
| G5_H_0p4 | screen | 0.50 | FAIL | FAIL | NOT RUN | drift |
| G5_H_0p6 | screen | 0.50 | FAIL | FAIL | NOT RUN | drift |
| G5_H_0p8 | screen | 0.50 | FAIL | FAIL | NOT RUN | drift |
| G5_H_1p2 | screen | 0.50 | FAIL | FAIL | NOT RUN | drift |
| G6_H_0p4 | screen | 0.50 | FAIL | FAIL | NOT RUN | unintended_support |
| G6_H_0p6 | screen | 0.50 | FAIL | PASS | NOT RUN | unintended_support |
| G6_H_0p8 | screen | 0.50 | FAIL | PASS | NOT RUN | unintended_support |
| G6_H_1p2 | screen | 0.50 | FAIL | PASS | NOT RUN | unintended_support |
| G0_M_0p6 | screen | 30.00 | PASS | PASS | FAIL |  |
| G0_M_0p8 | screen | 30.00 | PASS | PASS | FAIL |  |
| G0_M_1p2 | screen | 30.00 | PASS | PASS | FAIL |  |
| G6_M_0p6 | screen | 0.50 | FAIL | PASS | NOT RUN | unintended_support |
| G6_M_0p8 | screen | 0.50 | FAIL | PASS | NOT RUN | unintended_support |
| G6_M_1p2 | screen | 0.50 | FAIL | PASS | NOT RUN | unintended_support |
| G0_H_1p2 | 60s | 60.02 | PASS | PASS | FAIL |  |
| G0_reference | 60s | 60.02 | PASS | PASS | FAIL |  |
| G0_M_1p2 | 60s | 60.02 | PASS | PASS | FAIL |  |
| G0_H_0p6 | 60s | 60.02 | PASS | PASS | FAIL |  |
| G0_M_0p6 | 60s | 60.02 | PASS | PASS | FAIL |  |
| G0_M_0p8 | 60s | 60.02 | PASS | PASS | FAIL |  |
| G0_H_0p8 | 60s | 60.02 | PASS | PASS | FAIL |  |
| G0_H_0p4 | 60s | 49.02 | FAIL | FAIL | NOT RUN | drift, left_hand, right_hand |
| G0_H_0p6 | 120s | 83.78 | FAIL | FAIL | NOT RUN | drift, left_hand, right_hand |
| G0_M_0p6 | 120s | 83.02 | FAIL | FAIL | NOT RUN | drift, left_hand, right_hand |
| G0_M_0p8 | 120s | 85.76 | FAIL | FAIL | NOT RUN | drift |
| G0_H_0p8 | 120s | 86.24 | FAIL | PASS | NOT RUN | drift |
| G0_M_1p2 | 120s | 87.50 | FAIL | FAIL | NOT RUN | drift |

## Qualification by tested condition

| Condition | Status |
|---|---|
| G0_reference | FAILED |
| G1_reference | FAILED |
| G2_reference | FAILED |
| G3_reference | FAILED |
| G4_reference | FAILED |
| G5_reference | FAILED |
| G6_reference | FAILED |
| G0_H_0p4 | FAILED |
| G0_H_0p6 | FAILED |
| G0_H_0p8 | FAILED |
| G0_H_1p2 | NOT RUN (long qualification) |
| G5_H_0p4 | FAILED |
| G5_H_0p6 | FAILED |
| G5_H_0p8 | FAILED |
| G5_H_1p2 | FAILED |
| G6_H_0p4 | FAILED |
| G6_H_0p6 | FAILED |
| G6_H_0p8 | FAILED |
| G6_H_1p2 | FAILED |
| G0_M_0p6 | FAILED |
| G0_M_0p8 | FAILED |
| G0_M_1p2 | FAILED |
| G6_M_0p6 | FAILED |
| G6_M_0p8 | FAILED |
| G6_M_1p2 | FAILED |

H at mu=1.8 aliases each geometry's native reference only where compiled equivalence was verified. Other unlisted geometry/friction combinations are NOT RUN. Incomplete runs have no matched-window average. A small hand-load fraction is not evidence of hand capacity.
