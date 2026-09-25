# Implementation

The two newest maximbess commits are `d8cfb2f` and `9772e19` on `origin/ladder_dev`. This branch does not contain them. It contains his earlier climbing commits, `ab84379` and `0896b0f`. The cleanup below is of the artifacts and ladder code in this worktree. The later commits were read for bugs that still apply here. Their reward redesign was not merged.

## Removed artifacts

Deleted from git and ignored where they would otherwise come back:

- `.ipynb_checkpoints/`, `sim2sim_test.ipynb`, `train_scripts.ipynb`
- `assets/ladder_g1/` (about 63 MB of copied meshes and experimental XML) and `assets/ladders/`
- `show_scene.py`, `generate_double_ladder.py`
- `PROJECT_CONTEXT.md`
- `train_mimic/crawl_creator.py`, `train_mimic/crawl_length_count.py`, `train_mimic/configs/datasets/seed_crawl.yaml`, `train_mimic/data/seed/seed_crawl_metadata.csv`

The active ladder task builds the ladder onto `assets/robots/unitree_g1/g1_29dof.xml`. Nothing else imported those notebooks, the copied scene, or the crawl filter. `PROJECT_CONTEXT.md` was scratch notes; `d8cfb2f` already deleted it on `ladder_dev`.

Kept `teleopit/robots/automatic_grip.py`, `teleopit/configs/robot/g1_ladder.yaml`, and root `test_automatic_grip.py`. `MuJoCoRobot` still loads that grip config. The root test stubs `mujoco` at import time, so it stays out of the pytest collection.

`scripts/dev/check_large_tracked_files.py` reports no blocked or oversized tracked files after the mesh removal.

## Ladder code

`train_mimic/tasks/tracking/mdp/ladder.py` went from 3443 lines to 3059. The deleted helpers were not registered by `make_g1_ladder_rl_env_cfg`: target-progress and torso-ascent rewards, support/posture/upright/stability terms, and the separate rung, foot, stabilize, and cycle impulses. The FSM flags those live rewards still read are unchanged. The registered reward set is the same.

## Bugs fixed

**Box contact margin.** Generated rung and trunk-blocker boxes, and the collision-editor entries that overwrite them, used `margin=0.002`. MuJoCo Warp multi-CCD rejects a nonzero margin on box/mesh contacts, so ladder training cannot start on the pinned Warp 1.15 stack. Those margins are now 0. Capsule rails stay at 2 mm. A unit test builds the ladder on a minimal spec, runs the ladder collision editor, and checks the compiled margins. The full canonical G1 XML is not in this checkout, so `test_ladder_robot_augments_canonical_g1_spec` still cannot compile that file.

**Phase one-hot normalization.** The ladder command starts with a 5-way phase indicator, and that command is the first actor/critic block. `EmpiricalNormalization` was z-scoring it, including history, so a phase switch was no longer a clean categorical input. Ladder PPO now uses `LadderTemporalCNNModel`, which copies those five values through. The tracking policy is unchanged. A checkpoint trained with the old normalizer can still load, but it is not equivalent and should be retrained. `tests/test_ladder_model.py` checks a phase switch, a state-dict round trip, and TorchScript.

**Curriculum checkpoint rejection.** `LadderOnPolicyRunner.load` projected the actor standard deviation before noticing that a checkpoint had no curriculum state. The unit test therefore died on a missing `alg` attribute and never reached the intended error. The runner now rejects that checkpoint before the projection.

## Bugs inspected and left unchanged

`9772e19` also marks the episode deadline as a real termination (`time_out=False`) and adds a critic-only `remaining_time` feature. This tree still marks `time_out=True`. The failure penalty already uses `dones`, so a deadline still receives the `-50` impulse, but PPO can bootstrap the value as a truncation. Porting the later commit would also add an observation the current 14D critic contract does not have. That change stays on `ladder_dev`.

`d8cfb2f` adds stabilization-pose, joint-velocity, and unwanted-contact rewards on top of an intermediate ladder reward set this branch does not contain. Those terms change the training objective. They were not copied.

That same line of work narrows the shared self-collision sensor from `.*` to `pelvis|.*_link`. That helper is also used by tracking, and the change was not verified against this branch's scene, so the pattern is unchanged.

## Checks

`python -m compileall -q teleopit train_mimic autoresearch` succeeded.

The first full `pytest tests/ -q` in the `teleopit` conda env, after a temporary `mjlab` install, was **398 passed, 45 skipped, 2 failed**. Both failures were missing robot XML. `test_automatic_grip.py` passed 6 tests. `mjlab` was then uninstalled so MuJoCo could return to 3.14.0. That uninstall made the controller's later `pytest tests/ -q` exit 2: collection failed with `ModuleNotFoundError: No module named 'mjlab'` in 14 modules, including `tests/test_ladder_model.py`.

`mjlab==1.4.0`, `warp-lang==1.15.0`, and `rsl-rl-lib==5.2.0` are installed again in `/opt/anaconda3/envs/teleopit`. `mjlab` requires MuJoCo 3.8, so that env is on MuJoCo 3.8.1. The canonical G1 XML was downloaded with `python scripts/setup/download_assets.py --only robots --source huggingface` into gitignored `assets/robots/`. After that, `python -m pytest tests/ -q` was **444 passed, 1 skipped**.

The controller invokes the `pytest` script, not `python -m pytest`. The script does not put the repository root on `sys.path`, and `autoresearch` is not an installed package, so collection failed with `No module named 'autoresearch'`. `pyproject.toml` now sets `pythonpath = ["."]` for pytest. The same `pytest tests/ -q` command then passed: **444 passed, 1 skipped**.

No cluster training was launched. The ladder reward definition and the 14D critic observation were not changed.
