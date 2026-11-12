import numpy as np
import pytest


@pytest.fixture
def env():
    from teleopit.sim.g1_o6.env import Environment

    return Environment()


@pytest.mark.parametrize("mode", ["freeze_thumb", "freeze_all", "cap_thumb"])
def test_reference_latches_actual_side_targets_without_jump_and_clears(env, mode):
    from teleopit.sim.g1_o6_support.pose_controller import FingerExperiment

    controller = FingerExperiment(env, mode)
    env.phase = "HOLD"
    indices = {s: controller.thumb[s] for s in env.hands}
    env.data.ctrl[indices["left"]] = 0.21
    env.data.ctrl[indices["right"]] = 0.24
    first = controller.desired(env, 1)
    for side, index in indices.items():
        if mode != "cap_thumb":
            assert first[index] == env.data.ctrl[index]
    latch = {s: v.copy() for s, v in controller.latch.items()}
    env.data.qpos[env.act_qpos[list(indices.values())]] += 0.2
    second = controller.desired(env, 1)
    for side, index in indices.items():
        expected = latch[side][list(env.hand_act[side]).index(index)]
        assert second[index] <= expected + (0.03 if mode == "cap_thumb" else 0) + 1e-12
        np.testing.assert_array_equal(controller.latch[side], latch[side])
    controller.reset(env)
    assert controller.latch == {}
    assert controller.state()["latch_time_s"] is None


def test_freezing_one_thumb_does_not_change_body_or_other_fingers(env):
    from teleopit.sim.g1_o6_support.pose_controller import FingerExperiment

    controller = FingerExperiment(env, "freeze_thumb")
    env.phase = "HOLD"
    original = controller.base.desired(env, 1)
    result = controller.desired(env, 1)
    keep = np.ones(env.model.nu, dtype=bool)
    keep[list(controller.thumb.values())] = False
    np.testing.assert_array_equal(original[keep], result[keep])


def test_controller_mode_identity_and_invalid_source_are_not_silent(env):
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.metadata import resolved
    from teleopit.sim.g1_o6_support.pose_controller import FingerExperiment
    from teleopit.sim.g1_o6_support.qualification import identity

    original = resolved(env, snapshot(env))
    controller = FingerExperiment(env, "freeze_all")
    changed = {**original, "controller": controller.metadata(env)}
    assert changed["controller"]["schema"] != original["controller"]["schema"]
    assert identity(changed) != identity(original)
    with pytest.raises(ValueError):
        FingerExperiment(env, "bad")
