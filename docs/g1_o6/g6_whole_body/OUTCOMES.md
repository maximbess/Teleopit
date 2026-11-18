# Whole-body stabilization on G6

**No stable G1 pose qualified.** All four whole-body configurations kept each left finger loaded for the five-second MuJoCo probe. All four exceeded the original 25 N force-balance limit during unsupported HOLD. The search therefore ended before the 30-second screen. The previously observed left-finger unloading near 79 seconds remains unresolved.

Both hands grasped rung 7 and both feet stood on rung 2. Each fresh run recorded 5.02 seconds of unsupported HOLD, or 5,020 steps at 1 ms. The minimum load below is the smallest intended-tread normal force across the four left fingers and all those steps. The force-balance residual is the magnitude of the signed robot–environment contact forces plus gravity at each 20 ms control update.

| Controller | Gain | Minimum left finger (N) | Peak residual (N) | First balance violation (s) | Violating updates | Strict probe |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Matched arm-only control | — | 0.709 | 21.72 | — | 0 | Pass |
| Stance-leg inverse kinematics | 0.8 | 0.719 | 30.41 | 4.64 | 1 | Fail |
| Stance-leg inverse kinematics | 1.6 | 0.739 | 25.87 | 0.60 | 2 | Fail |
| Ground reaction through leg servos | 0.8 | 0.958 | 31.44 | 1.02 | 2 | Fail |
| Ground reaction through leg servos | 1.6 | 1.026 | 25.67 | 0.92 | 1 | Fail |

The higher-gain ground-reaction controller had the largest minimum left-finger load. At 0.92 s, the left and right feet supplied 145.0 and 155.1 N vertically. The hands supplied 4.6 and 4.0 N. Against the robot's 334.3 N weight, the vertical residual was −25.66 N. This one update invalidated the full five-second interval. Both hands passed the original anatomical contact criterion in all five runs.

The matched control reproduced the prior arm-only probe byte for byte. Verification checked 50,300 physics contact records and 25,100 unsupported steps across the five runs. All scored applied-force arrays were zero. The observed hand–tread friction vector was `[1.8, 1.8, 0.02, 0.002, 0.002]`, and scored housing and unintended support counts were zero. All 471 reviewed default model and reset fields matched. The independent command audit also checked native joint limits, arm command kinematics and the 15 additional leg and waist targets.

The simulation revision is `f59ae0af8cc24ce0214aa63073c8a9c681a25884`. The local campaign directory is `outputs/g1_o6/g6_whole_body`. Its ledger records every trial and the conditional decision to start the second pair. The machine-readable results include the raw trace checksums, the per-finger minima and every control update that violated force balance. [Results](results.json) · [Protocol](README.md) · [Work log](WORK_LOG.md)
