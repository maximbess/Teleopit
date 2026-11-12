import numpy as np


def test_arm_feedback_is_bounded_and_freezes_at_effort_or_command_limits():
    from teleopit.sim.g1_o6_support.pose_arm import bounded_correction

    offset = np.zeros(3)
    bounds = np.array([[-1, 1]] * 3)
    step = bounded_correction(
        offset, np.ones(3), np.array([0, 1, 0]), bounds, np.array([0, 0, 0.98]), 0.02
    )
    np.testing.assert_allclose(step, [0.0001, 0, 0])
    for _ in range(400):
        offset = bounded_correction(
            offset, np.ones(3), np.zeros(3), bounds, np.zeros(3), 0.02
        )
    np.testing.assert_allclose(offset, 0.03)


def test_arm_feedback_uses_scratch_state_and_preserves_fingers_feet_and_forces():
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.pose_arm import ArmReference

    env = Environment()
    controller = ArmReference(env)
    env.phase = "HOLD"
    original = controller.base.desired(env, 1)
    first = controller.desired(env, 1)
    np.testing.assert_allclose(first, original, atol=1e-14)
    # A synthetic tracking error for the command-law test; no physics evidence.
    env.data.qpos[0] += 0.002
    # Leave derived arrays stale deliberately: only scratch forward may refresh them.
    arrays = (
        "qpos",
        "qvel",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "qacc",
        "qacc_warmstart",
        "site_xpos",
        "actuator_force",
    )
    before = {n: getattr(env.data, n).copy() for n in arrays}
    next_target = controller.desired(env, 1)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env.data, name), value)
    changed = np.abs(next_target - controller.base.desired(env, 1))
    arm_indices = np.concatenate(list(controller.indices.values()))
    assert 0 < changed.max() <= 0.0001 + 1e-12
    assert np.all(changed[np.setdiff1d(np.arange(env.model.nu), arm_indices)] == 0)
    assert controller.metadata(env)["schema"] != "o6_bounded_closing_v1"
    assert controller.state()["reference_at_hold_entry"]
    controller.reset(env)
    assert not controller.state()["reference_at_hold_entry"]
    assert all(np.all(x == 0) for x in controller.offset.values())
