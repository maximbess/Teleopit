# Reproduction commands

Use `teleopit` on `dex-hand-exp` from the repository root. Extract the evidence archive first when using its source runs. Choose a new output root for every repetition; runners reject nonempty output directories.

## Stage 0 and 1

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6 --config docs/g1_o6/support_transfer/hold_30s.json --output outputs/g1_o6/reproduction/control_30s
conda run --no-capture-output -n teleopit python -m teleopit.sim.o6_diagnostics free-close --source-run outputs/g1_o6/support_transfer/control_before --side both --output outputs/g1_o6/reproduction/free_close
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support long-hold --seconds 60 --output outputs/g1_o6/reproduction/hold_60s
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support long-hold --seconds 120 --output outputs/g1_o6/reproduction/hold_120s
```

Only extend after the preceding strict duration passes. Only after a successful 120 s attempt, repeat with a fresh output path and compare at the finer timestep:

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support long-hold --seconds 120 --output outputs/g1_o6/reproduction/hold_120s_repeat
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support long-hold --seconds 120 --physics-dt 0.0005 --output outputs/g1_o6/reproduction/hold_120s_fine
```

The single tested correction is selected with `--grasp-config docs/g1_o6/support_transfer/correction_thumb_yaw.json`, starting at `--seconds 30`. It is a separate failed candidate, not a qualified default. Its repeat/fine 120 s qualification is not eligible in the delivered evidence.

## Stage 2

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support capacity-case --source-run outputs/g1_o6/support_transfer/control_before --state initial --side left --output outputs/g1_o6/reproduction/left_zero
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support capacity-case --source-run outputs/g1_o6/support_transfer/control_before --state settled --side right --output outputs/g1_o6/reproduction/right_settled_zero
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support capacity-matrix --source-run outputs/g1_o6/support_transfer/control_before --state settled --prior-trials 2 --output outputs/g1_o6/reproduction/capacity
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support capacity-matrix --source-run outputs/g1_o6/support_transfer/control_before --state settled --physics-dt 0.0005 --prior-trials 3 --output outputs/g1_o6/reproduction/capacity_fine
```

Use both sides for independent zero-load checks. `capacity-matrix` selects the actual simultaneous source samples, runs force-only/moment-only/combined families and reserve conditions, then bounded refinement/repeats and selected replay only when eligible. The supplied grasp fails unsupported settling, so the matrix records a zero-load failure and blocks loading/replay. `--prior-trials` charges already-used trials against the declared budget. No component-wise synthetic peak or hardware recording is used.

## Stages 3–6

All commands below currently return `BLOCKED`; they do not run a simulation with the supplied evidence book. After future valid evidence is added, each remains a fresh-reset experiment. Original RESET drift thresholds apply across its complete sequence.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support transfer --mode load --target-share 0.05 --evidence docs/g1_o6/support_transfer/evidence.json --output outputs/g1_o6/reproduction/load_05
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support transfer --mode unload --side left --target-share 0.05 --evidence docs/g1_o6/support_transfer/evidence.json --output outputs/g1_o6/reproduction/unload_left
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support transfer --mode release --side left --three-seconds 1 --evidence docs/g1_o6/support_transfer/evidence.json --output outputs/g1_o6/reproduction/release_left_1s
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support transfer --mode reacquire --side left --cycles 1 --three-seconds 10 --evidence docs/g1_o6/support_transfer/evidence.json --output outputs/g1_o6/reproduction/reacquire_left_1
```

The same commands with `--side right` test the right hand. Stage 3 targets are `--target-share 0.05`, then `0.10`, then `0.20`; extend the highest qualified target with `--plateau-seconds 30`. Release durations are `--three-seconds 1`, then `3`, then `10`, independently on each side. Reacquisition uses `--cycles 1`, then `3`, then `--cycles 3 --alternating`; add `--physics-dt 0.0005` for the selected finer comparison. Name every output uniquely. Prerequisite checks enforce preceding levels, both sides' unloading/three-support holds and cycle progression. Actual new wrist demands must have tested isolated reserve before a physical transfer result becomes a qualified prerequisite.

## Reports, offline videos, packaging and tests

```powershell
conda run --no-capture-output -n teleopit python scripts/g1_o6_support_report.py
conda run --no-capture-output -n teleopit python scripts/g1_o6_state_video.py --run outputs/g1_o6/support_transfer/hold_120s --output outputs/g1_o6/reproduction/original_video
conda run --no-capture-output -n teleopit python scripts/package_g1_o6_support.py --output C:/Users/maxim/Downloads/G1_O6_Support_Reproduction.zip --staging outputs/g1_o6/reproduction/archive_staging
$testFiles = rg --files tests | Where-Object { $_ -match 'test_(g1_o6|o6).*\.py$' }
conda run --no-capture-output -n teleopit python -m pytest @testFiles -q --basetemp outputs/g1_o6/reproduction/pytest
```

Video replays recorded positions on separate display data, without physics stepping or new qualification. Packaging verifies every archived file hash and exact equality of public numeric model/solver fields after replacing absolute file references with relative paths; filename offsets and path/buffer storage sizes are excluded. Keep the original asset/model licenses distributed with the files.
