"""Motor-only gravity compensation must keep the original actuator contract."""

import numpy as np
import pytest

from teleopit.sim.g1_o6.env import Environment


@pytest.fixture(scope="module")
def env():
    return Environment()


def test_gravity_preload_moves_targets_without_external_support(env):
    from teleopit.sim.g1_o6.holding import HoldController

    before = {
        name: getattr(env.model, name).copy()
        for name in (
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "jnt_range",
            "geom_friction",
            "body_mass",
        )
    }
    applied = env.data.qfrc_applied.copy()
    control = HoldController(
        env, closed_pose=(0.3, 0.14, 1.35, 1.35, 1.35, 1.35), preload_Nm=0.18
    )
    env.phase = "HOLD"
    desired = control.desired(env, 1.0)
    assert np.linalg.norm(desired[:29] - env.body_target[:29]) > 0.01
    np.testing.assert_array_equal(env.data.qfrc_applied, applied)
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(env.model, key), value)
    for ids in env.hand_act.values():
        assert (
            np.max(desired[ids] - env.data.qpos[env.act_qpos[ids]]) <= 0.18 / 3 + 1e-12
        )


def test_feet_carry_reference_weight_during_acquisition(env):
    from teleopit.sim.g1_o6.holding import HoldController

    control = HoldController(
        env, closed_pose=(0.3, 0.14, 1.35, 1.35, 1.35, 1.35), preload_Nm=0.18
    )
    env.phase = "ACQUIRE"
    before = control.desired(env, 1.0)
    env.phase = "HOLD"
    after = control.desired(env, 1.0)
    np.testing.assert_allclose(before[:29], after[:29])
    assert np.linalg.norm(after[:29] - env.body_target[:29]) > 0.01


def test_balanced_reset_preloads_motors_and_does_not_lift_body():
    from teleopit.sim.g1_o6.grasp_pose import GraspPose

    env = Environment(pose=GraspPose())
    assert np.linalg.norm(env.target[:29] - env.body_target[:29]) > 0.01
    env.apply_forces()
    np.testing.assert_array_equal(env.assistance, np.zeros(6))
    env.phase = "HOLD"
    env.data.qpos[0] += 0.01
    env.apply_forces()
    np.testing.assert_array_equal(env.data.qfrc_applied, np.zeros(env.model.nv))
    before = env.target.copy()
    env.commands()
    assert np.max(np.abs(env.target - before)) <= env.config.control_dt + 1e-12


@pytest.mark.parametrize("preload", [0.0, float("nan"), 0.36])
def test_preload_must_fit_existing_hand_effort_limits(env, preload):
    from teleopit.sim.g1_o6.holding import HoldController

    with pytest.raises(ValueError, match="preload"):
        HoldController(
            env, closed_pose=(0.3, 0.14, 1.35, 1.35, 1.35, 1.35), preload_Nm=preload
        )
