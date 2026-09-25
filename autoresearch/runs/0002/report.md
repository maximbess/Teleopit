# Task 0002

Engineering cleanup of the G1 ladder climber. Independent validation has passed. No cluster training was run, and the registered ladder reward weights were not changed.

## What changed and why

The climbing reset pose now lives in `train_mimic/tasks/tracking/config/ladder_init.py`, which does not import MuJoCo or mjlab. The mjlab robot config still copies that same root position and joint pose into the simulator reset.

`train_mimic/tasks/tracking/rl/isolated.py` adds `train(model, reward_fn, init_state, transition_fn)`. The model maps state to an action. The reward takes `(state, action, next_state)` and must be differentiable in the action. `transition_fn` is the only stand-in for a physics simulator. Each update starts from `init_state` and returns the same module. Production ladder PPO remains `train_mimic/scripts/train_ladder.py`.

`train_mimic/tasks/tracking/mdp/ladder.py` went from 3458 lines to 3217. These helpers were not registered by `make_g1_ladder_rl_env_cfg` and had no tests, so they were removed: `LadderTargetProgressReward`, `LadderTorsoAscentReward`, `ladder_support_hand`, `ladder_rung_advance`, `ladder_foot_support`, `ladder_torso_stability_exp`, and `ladder_torso_posture_exp`.

Capsule rails are 2 mm again in both the generated spec and the rail collision editor. Box rungs stay at margin 0. MuJoCo Warp rejects a nonzero margin only when both geoms in a pair are boxes or meshes. Commit `d8cfb2f` had zeroed the rails as well, which disagreed with `test_ladder_box_contacts_keep_zero_margin`.

The ladder builder already deletes the canonical XML `floor` and the two XML lights, then the scene adds one terrain plane. `test_composed_ladder_scene_has_one_ground_plane` locks that. The compiled scene had plane names `["terrain"]` and no robot-side lights.

The shared self-collision helper had been narrowed to `pelvis|.*_link` for every task. Tracking requires `.*`. Tracking uses `.*` again. The ladder environment passes `pelvis|.*_link`, so ladder bodies and grip anchors in the same entity stay out of that sensor.

The English and Chinese training notes now say box rungs use zero margin and capsule rails keep 2 mm. The English note no longer mentions a trunk blocker.

## Deliberately left alone

Production training is still `train_ladder.py` and `LadderOnPolicyRunner`. The isolated loop is a differentiable reward ascent from a fixed initial state. It does not run PPO, the ladder FSM, or MuJoCo.

Phase-progress body-height weights (`first_hand_body_weight`, `second_hand_body_weight`, `foot_body_weight`) stay at 0. The live potential still pays target distance and hand release. Changing those weights would change the training objective.

The two ladder faces still meet at the top rail endpoints. Both bodies are fixed to the world, so that overlap is not a simulated contact.

## Checks

`python -m compileall -q teleopit train_mimic autoresearch` succeeded.

The first isolated-trainer test used a transition that added a 2-wide action to a 3-wide state and raised `RuntimeError`. That was a bad test. The transition now leaves the state unchanged, because that reward depends on the action. After the fix, the focused ladder and prototype tests were **79 passed, 1 deselected**. The deselected test was `test_ladder_robot_augments_canonical_g1_spec`. Run alone, it failed with `FileNotFoundError` because `omniretarget_collision/manifest.json` was missing.

The first full `pytest tests/ -q` was **9 failed, 469 passed, 1 skipped**. Eight failures were that missing manifest. The ninth was `test_general_tracking_task_is_registered`: the self-collision pattern was `pelvis|.*_link` where the tracking test requires `.*`.

`pip install -e '.[collision-build]'` installed `coacd==1.0.14` and `shapely==2.1.2` (`trimesh==5.1.0` was already present). It also replaced MuJoCo 3.8.1 with 3.14.0. `mjlab==1.4.0` requires the 3.8 series, so MuJoCo was installed back at 3.8.1. `mink` still wants `mujoco>=3.10`. That conflict was already present on 3.8.1.

`python scripts/setup/download_assets.py --only g1_collision` exited 0 in about 28 seconds and wrote 182 convex parts under gitignored `assets/robots/`.

`pytest tests/ -q` was then **478 passed, 1 skipped**. Independent validation of this tree has passed.

## Remaining uncertainty

The isolated trainer will fail at `backward` if the reward does not depend differentiably on the action. It is not a substitute for a ladder training run. No climbing policy was trained.

The 2 mm rail margin was checked on the generated ladder spec plus the ladder collision editor. The full-suite geometry tests compiled the OmniRetarget overlay and checked that box and mesh margins are 0. They did not call `mjwarp.put_model` on that model.

The collision meshes are gitignored. A clean checkout needs `python scripts/setup/download_assets.py --only g1_collision` before those geometry tests can compile the overlay. Installing the `collision-build` extra can upgrade MuJoCo past 3.8 unless that package is pinned afterward. This environment is on MuJoCo 3.8.1.
