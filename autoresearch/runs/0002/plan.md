# Plan

The live climbing task is the G1 ladder PPO pipeline. After the `ladder_dev` merge it is still one large command module, the training entry can only run inside mjlab, and the capsule-rail contact margin disagrees with the unit test. This task separates a prototype training step from MuJoCo, deletes reward helpers that nothing registers, and corrects the contact margin. It does not change the registered ladder reward weights or launch training.

## Simulator-free training step

Add `train(model, reward_fn, init_state, transition_fn)` under `train_mimic/tasks/tracking/rl/isolated.py`. The first three arguments are the prototype contract. `transition_fn(state, action) -> next_state` is the only stand-in for a physics simulator, so the function imports neither MuJoCo nor mjlab. Each update rolls out from `init_state`, maximizes the mean reward, and returns the same module. Move the climbing root pose and joint pose into `ladder_init.py`, which also imports neither simulator, and keep the mjlab robot config on those same values.

Production ladder PPO stays `train_ladder.py`. This step is for prototyping a model, reward, and initial state without compiling a scene.

## Compact the ladder rewards

`LadderTargetProgressReward`, `LadderTorsoAscentReward`, `ladder_support_hand`, `ladder_rung_advance`, `ladder_foot_support`, `ladder_torso_stability_exp`, and `ladder_torso_posture_exp` are not registered and have no tests. Delete them. Leave the reward terms that `make_g1_ladder_rl_env_cfg` still uses.

## Physics

MuJoCo Warp rejects a nonzero margin only when both geoms in a pair are boxes or meshes. Capsule rails were zeroed along with the box rungs, while `test_ladder_box_contacts_keep_zero_margin` still requires a 2 mm rail margin. Restore that rail margin in the generated spec and in the collision editor. Leave box and mesh margins at 0.

The canonical G1 XML has a `floor` plane and the ladder scene adds a terrain plane. The builder already deletes the XML floor. Add a composed-scene check that the attached robot plus one terrain plane compiles to exactly one plane, so a second floor cannot come back unnoticed.

## Checks

Compile the packages. Run the isolated-trainer test, the rail-margin test, the composed-floor test, and the ladder tests that do not need the OmniRetarget collision meshes. Do not start cluster training.
