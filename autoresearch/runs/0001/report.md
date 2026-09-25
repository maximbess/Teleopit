# Task 0001

Engineering cleanup of the climbing work on this branch. Independent validation passed after the fixes below. No cluster training was run.

The two newest commits by maximbess, `d8cfb2f` and `9772e19`, are only on `origin/ladder_dev`. This worktree is based on his earlier climbing commits, `ab84379` and `0896b0f`. The cleanup applies to the code that is here. The later reward redesign was read and not merged.

## What changed and why

Unused artifacts from the climbing commits were removed and ignored: Jupyter notebooks and `.ipynb_checkpoints/`, the copied scene under `assets/ladder_g1/` (about 63 MB) and `assets/ladders/`, `show_scene.py`, `generate_double_ladder.py`, `PROJECT_CONTEXT.md`, and the unused crawl-filter scripts plus `seed_crawl.yaml` and its metadata CSV. The live ladder task builds the ladder onto the canonical G1 XML, and nothing else imported those files. `PROJECT_CONTEXT.md` was scratch notes that `d8cfb2f` had already deleted on `ladder_dev`.

`train_mimic/tasks/tracking/mdp/ladder.py` went from 3443 lines to 3059. The deleted helpers were not registered by the ladder environment: target-progress and torso-ascent rewards, support, posture, upright, and stability terms, and the separate rung, foot, stabilize, and cycle impulses. The registered reward set and the FSM flags the live rewards still read are unchanged.

Three defects in the current tree were fixed.

- Box rungs and the trunk blocker used `margin=0.002`, including the collision-editor entries that overwrite the generated spec. MuJoCo Warp multi-CCD rejects a nonzero margin on box and mesh contacts, so ladder training cannot start on the pinned Warp 1.15 stack. Those margins are now 0. Capsule rails stay at 2 mm.
- Ladder PPO z-scored the leading five phase one-hot values, including history, so a phase switch was no longer a clean categorical input. Ladder actor and critic now use `LadderTemporalCNNModel`, which copies those five values through. The tracking policy still normalizes every feature. A checkpoint trained with the old normalizer can load, but it is not equivalent and should be retrained.
- `LadderOnPolicyRunner.load` projected the actor standard deviation before noticing that a checkpoint had no curriculum state. The unit test then failed on a missing `alg` attribute and never reached the intended error. The runner now rejects that checkpoint before the projection.

`pyproject.toml` sets pytest `pythonpath` to the repository root. The controller runs the `pytest` script, which does not put the root on `sys.path`, and `autoresearch` is not an installed package. Without that setting, collection failed with `No module named 'autoresearch'`.

## Deliberately left alone

`teleopit/robots/automatic_grip.py`, `teleopit/configs/robot/g1_ladder.yaml`, and root `test_automatic_grip.py` stay. `MuJoCoRobot` still loads that grip config. The root test stubs `mujoco` at import time, so it is not part of the pytest collection.

These `ladder_dev` changes were not ported. They sit on intermediate commits this branch does not contain, and they change the current training contract:

- Deadline handling that sets `time_out=False`, plus a critic-only `remaining_time` input. This tree still uses `time_out=True`. The failure penalty already uses `dones`, so a deadline still receives the `-50` impulse, but PPO can bootstrap it as a truncation.
- Stabilization-pose, joint-velocity, and unwanted-contact rewards.
- Narrowing the shared self-collision sensor from `.*` to `pelvis|.*_link`. That helper is also used by tracking.

The 14D critic observation and the registered ladder reward weights were not changed. No cluster job was launched.

## Checks

`python -m compileall -q teleopit train_mimic autoresearch` succeeded.

`scripts/dev/check_large_tracked_files.py` reported no blocked or oversized tracked files after the mesh removal.

`test_automatic_grip.py` passed 6 tests.

`pytest` was not green on the first attempts:

- With `mjlab` temporarily installed and the G1 XML absent, `pytest tests/ -q` was 398 passed, 45 skipped, 2 failed. The failures were missing `g1_29dof.xml` and `g1_29dof_dex3.xml`.
- Uninstalling `mjlab` afterward made a later controller run exit 2: 14 modules failed collection with `No module named 'mjlab'`.
- Reinstalling `mjlab==1.4.0`, `warp-lang==1.15.0`, and `rsl-rl-lib==5.2.0` fixed collection. That stack pins MuJoCo 3.8.1 in the `teleopit` conda env. The robot archive was then downloaded with `python scripts/setup/download_assets.py --only robots --source huggingface` into gitignored `assets/robots/`.
- `python -m pytest tests/ -q` was then 444 passed, 1 skipped.
- The controller's `pytest` script still failed collection with `No module named 'autoresearch'`. After the `pythonpath` setting, that same `pytest tests/ -q` command was 444 passed, 1 skipped.

Independent validation of this final tree has passed. The box-margin test builds a ladder on a minimal spec, runs the ladder collision editor, and checks the compiled margins. The phase-normalizer test checks a phase switch, a state-dict round trip, and TorchScript. No climbing policy was trained or played back in simulation.

## Remaining uncertainty

A ladder checkpoint trained before the phase-preserving normalizer will not see the same inputs if it is resumed. That was not measured in a training run.

Robot mesh margins on the downloaded canonical XML were not separately audited beyond the existing full-spec test. The Warp failure that motivated the margin change is specifically nonzero margin on box and mesh contacts. Capsule rails remain at 2 mm.

The deadline still bootstraps as a truncation while also taking the failure penalty. Whether that should become a hard terminal failure is the `ladder_dev` choice that was left unmerged.

`mjlab==1.4.0` and MuJoCo 3.14 cannot be installed together. The test env is on MuJoCo 3.8.1, which does not satisfy `mink`'s `mujoco>=3.10` requirement.
