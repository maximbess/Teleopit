# Implementation

The climbing task stays the G1 ladder PPO pipeline. This change gives it a simulator-free training step, removes reward helpers that nothing registers, and puts the capsule-rail contact margin back where the unit test and MuJoCo Warp's actual restriction put it. No cluster training was launched. The registered reward weights were not changed.

## Simulator-free training step

`train_mimic/tasks/tracking/rl/isolated.py` defines `train(model, reward_fn, init_state, transition_fn, *, steps, horizon=1, lr=1e-2)`. The model maps a state batch to an action tensor. The reward receives `(state, action, next_state)` and must be differentiable in the action. `transition_fn` is the only stand-in for a physics simulator. Every update starts again from `init_state`, and the same module is returned in eval mode so its `state_dict` is the artifact to save. The module imports neither MuJoCo nor mjlab.

The climbing reset pose moved to `train_mimic/tasks/tracking/config/ladder_init.py`, which also has no simulator import. `make_g1_ladder_training_robot_cfg` still copies those same root and joint values into the mjlab robot. Production PPO is still `train_mimic/scripts/train_ladder.py`.

Expected observation before running: a linear policy whose reward is the squared distance of the action from a fixed target should reduce that distance, and the saved weights should reload to the same action.

The first test transition added a 2-wide action to a 3-wide state and raised `RuntimeError` on the shape mismatch. That was a bad test, not a trainer failure. The test now keeps the state unchanged, because that reward depends on the action directly. After the fix, `tests/test_isolated_train.py` passed, including a batched initial state over a two-step horizon and a state-dict reload.

## Compact rewards

`train_mimic/tasks/tracking/mdp/ladder.py` went from 3458 lines to 3217. The deleted helpers were not registered by `make_g1_ladder_rl_env_cfg` and had no tests: `LadderTargetProgressReward`, `LadderTorsoAscentReward`, `ladder_support_hand`, `ladder_rung_advance`, `ladder_foot_support`, `ladder_torso_stability_exp`, and `ladder_torso_posture_exp`. The live reward set is unchanged.

## Physics

MuJoCo Warp rejects a nonzero margin only when both sides of a contact are boxes or meshes (`mujoco_warp/_src/io.py`, around the MULTICCD check). `d8cfb2f` had also set capsule rails to margin 0. The generated rail spec and the rail collision editor are 2 mm again. Box rungs stay at 0. `test_ladder_box_contacts_keep_zero_margin` passed: compiled box margins are 0 and compiled capsule margins are 0.002.

The canonical G1 XML still contains one geom named `floor` plus two lights. The ladder builder already deletes that floor and those lights, and the scene adds one terrain plane. `test_composed_ladder_scene_has_one_ground_plane` compiles the canonical robot, applies that cleanup, adds the ladder, attaches it beside one terrain plane, and checks the result. Observed plane names: `["terrain"]`. Observed robot-side light count: 0. A second floor was not present.

The two ladder faces still meet at the top rail endpoints. Both bodies are fixed to the world, so MuJoCo does not generate a static-static contact there. That geometry was left as it is.

## Left unchanged

`make_g1_ladder_rl_env_cfg` still sets the phase-progress body-height weights (`first_hand_body_weight`, `second_hand_body_weight`, `foot_body_weight`) to 0. The live potential still pays target distance and hand release. Changing those weights would change the training objective, so they were not edited.

The full robot compile in `test_ladder_robot_augments_canonical_g1_spec` still raises `FileNotFoundError` because `assets/robots/unitree_g1/omniretarget_collision/manifest.json` is not in this checkout. That failure is the missing collision-asset download, and it happens before any assertion this task added. The overlay was not rebuilt here.

## Checks

`python -m compileall -q teleopit train_mimic autoresearch` succeeded.

`pytest` on `tests/test_isolated_train.py`, `tests/test_ladder_posture_rewards.py`, `tests/test_ladder_model.py`, `tests/test_ladder_training_fixes.py`, and `tests/test_ladder_training.py`, excluding `test_ladder_robot_augments_canonical_g1_spec`, was **79 passed, 1 deselected**.

The excluded canonical-spec test was run on its own and failed with the missing collision-asset error above.

## Validation failure

`pytest tests/ -q` then failed with 9 errors. Eight tests called `apply_g1_collision_overlay` and raised `FileNotFoundError` because `assets/robots/unitree_g1/omniretarget_collision/manifest.json` was absent. `test_general_tracking_task_is_registered` failed because the shared self-collision helper used `pelvis|.*_link`. The tracking contract is `.*`.

Tracking now keeps `.*`. The ladder env passes `pelvis|.*_link` into the same helper, so ladder and grip-anchor bodies stay out of that sensor. `test_ladder_task_is_rl_only` checks the ladder pattern. Both of those tests passed before the mesh build finished.

`pip install -e '.[collision-build]'` installed `coacd==1.0.14`, `shapely==2.1.2`, and the already present `trimesh==5.1.0`. That command also replaced MuJoCo 3.8.1 with 3.14.0, which `mjlab==1.4.0` rejects. MuJoCo was installed back at 3.8.1. `mink` still reports that it wants `mujoco>=3.10`; that conflict was already present when this environment was on 3.8.1.

`python scripts/setup/download_assets.py --only g1_collision` then finished with exit code 0 in about 28 seconds and wrote 182 convex parts under the gitignored `assets/robots/` tree.

After that, `pytest tests/ -q` was **478 passed, 1 skipped**.

## Remaining uncertainty

The isolated loop is a differentiable reward ascent from a fixed initial state. It is not PPO, and it does not step the ladder FSM or MuJoCo. A prototype reward that is not differentiable in the action will fail at `backward`.

The 2 mm rail margin was checked on the generated ladder spec plus the ladder collision editor. It was not checked on a model that also includes the OmniRetarget meshes, because those assets are absent. Warp's margin rejection is specifically box/mesh pairs, so a capsule rail at 2 mm should still load; that was not executed through `mjwarp.put_model` on the full G1.

No climbing policy was trained.

The collision meshes are gitignored under `assets/robots/`. A clean checkout still needs `python scripts/setup/download_assets.py --only g1_collision` before those geometry tests can compile the overlay. Installing the `collision-build` extra with an unpinned MuJoCo dependency will upgrade past the 3.8 series that mjlab 1.4 requires; this environment was put back on MuJoCo 3.8.1 after that happened.
