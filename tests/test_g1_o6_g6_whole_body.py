"""A grounded correction must go through the native leg and waist servos."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize("law", ["stance", "wrench"])
def test_world_disturbance_drives_grounded_legs_and_waist_only_in_hold(law):
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.g6_whole_body import WholeBody

    candidate = replace(candidates()[2], controller=WholeBody(balance=law, gain=.8))
    env = make_environment(candidate, 5, .001)
    control = env.hold_controller
    env.phase = 'REMOVE_ASSISTANCE'
    np.testing.assert_array_equal(control.desired(env, 1), control.parent.desired(env, 1))
    assert control.anchor is None
    env.phase = 'HOLD'
    control.desired(env, 1)
    # Displace the pelvis in +x and roll the waist. At fixed feet the
    # correcting leg motion must move the ankles +x relative to the pelvis.
    env.data.qpos[0] += .004
    env.data.joint('waist_roll_joint').qpos[0] += .006
    mujoco.mj_forward(env.model, env.data)
    before = {key: getattr(env.data, key).copy() for key in ('qpos','qvel','ctrl','xfrc_applied','qfrc_applied')}
    command = control.desired(env, 1)
    for k, original in before.items():
        np.testing.assert_array_equal(getattr(env.data, k), original)
    limits = env.model.jnt_range[env.model.actuator_trnid[:, 0]]
    parent_target = np.clip(control.parent.raw, limits[:, 0], limits[:, 1])
    offset = np.clip(command, limits[:, 0], limits[:, 1]) - parent_target
    for side in env.hands:
        leg = control.chains[side]
        assert np.linalg.norm(offset[leg.indices]) > 0
        if law == 'stance':
            translation = leg.jacobian(env)[:3] @ offset[leg.indices]
            assert translation[0] > 0
    waist = control.chains['waist']
    assert (waist.jacobian(env)[3:] @ offset[waist.indices])[0] < 0
    other = np.setdiff1d(np.arange(env.model.nu), control.indices)
    np.testing.assert_array_equal(command[other], control.parent.raw[other])
    assert np.max(np.abs(offset)) <= .06 * env.config.control_dt + 1e-12
    env.reset()
    assert env.hold_controller.anchor is None
    np.testing.assert_array_equal(env.hold_controller.offset, 0)


def test_body_command_bounds_include_effort_joint_limits_and_backoff():
    from teleopit.sim.g1_o6_support.g6_whole_body import bounded_offset

    bounds = np.array([[-.12, .12], [-.002, .001], [-.12, .12]])
    previous = np.array([.1199, 0., .02])
    for _ in range(130):
        current = bounded_offset(previous, np.ones(3), bounds, [False, False, True], .02)
        assert np.max(np.abs(current - previous)) <= .0012 + 1e-12
        assert np.all(current <= bounds[:, 1])
        previous = current
    np.testing.assert_allclose(current, [.12, .001, .02])
    backed_off = bounded_offset(current, -current, bounds, [False]*3, .02)
    assert np.all(np.abs(backed_off) < np.abs(current))


def test_commanded_foot_reactions_sum_to_requested_pelvis_force_and_moment():
    from teleopit.sim.g1_o6_support.g6_whole_body import foot_wrenches

    force, moment = np.array([4., -3., 10.]), np.array([.6, 1.8, -2.])
    left, right = np.array([.03, .11, -.75]), np.array([.01, -.12, -.73])
    legs = foot_wrenches(force, moment, left, right)
    np.testing.assert_allclose(legs['left'][:3] + legs['right'][:3], force)
    achieved = (np.cross(left, legs['left'][:3]) + np.cross(right, legs['right'][:3])
                + legs['left'][3:] + legs['right'][3:])
    np.testing.assert_allclose(achieved, moment)
    # The two foot forces supply the moment perpendicular to their separation.
    # Only its parallel component must be commanded as an ankle moment.
    np.testing.assert_allclose(np.cross(legs['left'][3:], left - right), 0., atol=1e-12)


def test_whole_body_profile_round_trips_without_changing_parent_physics():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.g6_whole_body import WholeBody
    from teleopit.sim.g1_o6_support.g6_whole_body_experiment import candidates
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import canonical, restore_environment

    parent, candidate = candidates()[:2]
    assert G6Candidate.from_dict(candidate.record()) == candidate
    assert parent.digest() != candidate.digest()
    base = make_environment(parent, 5, .001)
    env = make_environment(candidate, 5, .001)
    np.testing.assert_array_equal(env.data.qpos, base.data.qpos)
    for field in ('geom_friction', 'body_mass', 'jnt_range', 'actuator_forcerange', 'actuator_gainprm'):
        np.testing.assert_array_equal(getattr(env.model, field), getattr(base.model, field))
    resolved = canonical(experiment_metadata(env, snapshot(env), {'mode': 'long-hold'}))
    restored = restore_environment(resolved, .0005)
    assert restored.grasp_acquisition == candidate.controller
    assert restored.config.control_dt == .02
    assert restored.config.friction_scale == 1
    assert resolved['controller']['mode'] == 'g6_whole_body'
    for kwargs in ({'balance':'free'}, {'gain':float('nan')}, {'arm_rate_rad_s':.5}, {'order':'thumb_first'}):
        with pytest.raises(ValueError):
            WholeBody(**kwargs)
