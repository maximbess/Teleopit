"""Behavioral checks for the independent O6 experiment (no robot assets needed)."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from teleopit.sim.o6_grasp import GraspConfig, O6GraspEnv
from teleopit.sim.o6_grasp.controller import Mode
from teleopit.sim.o6_grasp.observations import History, SPEC


def test_reset_action_contract_and_history():
    env = O6GraspEnv()
    obs, info = env.reset(seed=12)
    assert obs.shape == (48,) and np.isfinite(obs).all()
    assert sum(field.size for field in SPEC) == 48
    assert env.model.nu == 6 and env.model.neq == 5
    assert env.active_names[:2] == ('rh_thumb_cmc_pitch', 'rh_thumb_cmc_yaw')
    assert env.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert np.all(env.model.actuator_forcelimited)
    assert info['raw_observation'][18:24] == [0.0] * 6
    history = History(env)
    stacked, _ = history.reset(seed=12)
    assert stacked.shape == (10, 48)
    np.testing.assert_array_equal(stacked[0], stacked[-1])
    for action in ([0] * 5, [2] * 6, [np.nan] * 6):
        with pytest.raises(ValueError):
            env.step(action)
    for _ in range(5):
        env.step(np.zeros(6))
    again, _ = env.reset(seed=12)
    np.testing.assert_array_equal(obs, again)


def test_relative_frame_and_orientation_column_order():
    env = O6GraspEnv()
    _, info = env.reset()
    raw = np.asarray(info['raw_observation'])
    palm = env.data.site('palm_frame')
    rung = env.data.site('rung_frame')
    rotation = palm.xmat.reshape(3, 3)
    np.testing.assert_allclose(raw[24:27], rotation.T @ (rung.xpos - palm.xpos))
    relative = rotation.T @ rung.xmat.reshape(3, 3)
    np.testing.assert_allclose(raw[27:33], relative[:, :2].flatten(order='F'))


def test_initial_clearance_and_finite_actuation():
    env = O6GraspEnv()
    _, info = env.reset()
    assert info['max_penetration'] < 1e-6
    initial = env.data.qpos[env.active_qpos].copy()
    last_target = initial.copy()
    for _ in range(180):
        _, _, _, _, info = env.step(np.ones(6))
        target = np.array(info['target'])
        assert np.max(np.abs(target - last_target)) <= env.config.command_rate * env.config.control_dt + 1e-9
        assert np.all(target >= env.limits[:, 0]) and np.all(target <= env.limits[:, 1])
        assert np.max(np.abs(info['actuator_effort'])) <= env.config.effort_limit + 1e-8
        last_target = target
    assert np.linalg.norm(env.data.qpos[env.active_qpos] - initial) > 0.2


def test_free_motion_coupling_and_release():
    env = O6GraspEnv(replace(GraspConfig(), rung_enabled=False, evaluation_mode='acquisition'))
    env.reset()
    maximum = 0.0
    modes = set()
    while True:
        _, _, _, done, info = env.step(np.zeros(6))
        modes.add(info['mode'])
        maximum = max(maximum, info['coupling_error'])
        if done:
            break
    assert modes == {mode.name for mode in Mode}
    assert maximum < 0.015
    assert info['release_clear']
    assert np.max(np.abs(env.data.qpos[env.active_qpos] - env.config.open_pose)) < 0.05


def test_unsupported_means_no_fixture_and_no_false_success():
    cfg = replace(GraspConfig(), rung_enabled=False)
    env = O6GraspEnv(cfg)
    env.reset()
    unsupported = []
    while True:
        _, _, _, done, info = env.step(np.zeros(6))
        if info['unsupported']:
            unsupported.append(info)
            assert info['fixture_gain'] == 0
            np.testing.assert_array_equal(info['fixture_wrench'], np.zeros(6))
            assert info['verification'] != 'VERIFIED'
        if done:
            break
    assert unsupported
    assert max(row['translation_drift'] for row in unsupported) > 0.1


def test_invalid_config_is_rejected():
    for change in ({'physics_dt': 0}, {'control_dt': 0.021}, {'friction': -1}, {'cross_section': 'triangle'}):
        with pytest.raises(ValueError):
            O6GraspEnv(replace(GraspConfig(), **change))


def test_wrenches_use_world_palm_origin_and_do_not_accumulate():
    from teleopit.sim.o6_grasp.wrenches import apply_wrenches, external_wrench

    cfg = replace(GraspConfig(), load_force=(0, 2, 0), load_torque=(1, 0, 0),
                  load_point=(0.1, 0, 0), load_profile='constant')
    env = O6GraspEnv(cfg)
    np.testing.assert_allclose(external_wrench(env, 1), [0, 2, 0, 1, 0, 0.2])
    fixture, load, gain = apply_wrenches(env, Mode.HOLD, 1.5)
    first = env.data.qfrc_applied.copy()
    apply_wrenches(env, Mode.HOLD, 1.5)
    assert gain == 0
    np.testing.assert_array_equal(fixture, np.zeros(6))
    np.testing.assert_allclose(load, [0, 2, 0, 1, 0, 0.2])
    np.testing.assert_array_equal(env.data.qfrc_applied, first)
    # Root force equals the specified force; the generalized root moment includes
    # both palm offset (z=.06) and the configured off-centre point.
    np.testing.assert_allclose(first[:3], [0, 2, 0])
    np.testing.assert_allclose(first[3:6], [0.88, 0, 0.2])


def test_residual_offsets_delay_and_release_authority():
    from teleopit.sim.o6_grasp.controller import Controller, nominal_grasp

    env = O6GraspEnv()
    cfg = env.config
    controller = Controller(cfg, env.limits)
    nominal = nominal_grasp(cfg, Mode.HOLD, 0)
    for _ in range(100):
        applied, *_ = controller.command(4, np.ones(6))
    expected = np.clip(nominal + cfg.residual_bounds, env.limits[:, 0], env.limits[:, 1])
    np.testing.assert_allclose(applied, expected)
    # At zero delay the applied and current targets are identical.
    np.testing.assert_array_equal(applied, controller.target)
    for _ in range(100):
        applied, *_ = controller.command(10, -np.ones(6))
    np.testing.assert_allclose(applied, cfg.open_pose, atol=1e-12)
    delayed = Controller(replace(cfg, actuator_delay_steps=2), env.limits)
    for _ in range(2):
        applied, *_ = delayed.command(4, np.zeros(6))
        np.testing.assert_array_equal(applied, cfg.open_pose)
    applied, *_ = delayed.command(4, np.zeros(6))
    assert np.max(applied) > 0


def test_sensor_delay_dropout_seed_and_current_history():
    from teleopit.sim.o6_grasp.observations import ObservationSensor

    zero, one, two = np.zeros(48), np.ones(48), np.full(48, 2)
    sensor = ObservationSensor(GraspConfig(), 2)
    np.testing.assert_array_equal(sensor.sample(zero), zero)
    np.testing.assert_array_equal(sensor.sample(one), one)
    sensor = ObservationSensor(replace(GraspConfig(), observation_delay_steps=1), 2)
    sensor.sample(zero)
    np.testing.assert_array_equal(sensor.sample(one), zero)
    np.testing.assert_array_equal(sensor.sample(two), one)
    sensor = ObservationSensor(replace(GraspConfig(), observation_dropout=1), 2)
    sensor.sample(zero)
    np.testing.assert_array_equal(sensor.sample(one), zero)
    cfg = replace(GraspConfig(), observation_noise=0.02)
    a, b = O6GraspEnv(cfg), O6GraspEnv(cfg)
    np.testing.assert_array_equal(a.reset(seed=42)[0], b.reset(seed=42)[0])
    np.testing.assert_array_equal(a.step(np.zeros(6))[0], b.step(np.zeros(6))[0])
    history = History(a)
    old, _ = history.reset(seed=42)
    new, *_ = history.step(np.zeros(6))
    np.testing.assert_array_equal(new[:-1], old[1:])
    assert not np.array_equal(new[-1], old[-1])


@pytest.mark.parametrize('profile', ['constant', 'ramp', 'impulse'])
def test_load_is_removed_before_release(profile):
    from teleopit.sim.o6_grasp.controller import load_gain, fixture_gain

    cfg = replace(GraspConfig(), load_profile=profile)
    assert load_gain(cfg, Mode.OPEN, 0) == 0
    assert load_gain(cfg, Mode.HOLD, 0.5) == 0
    assert load_gain(cfg, Mode.HOLD, 1.05) > 0
    assert load_gain(cfg, Mode.HOLD, cfg.phase_durations[2]) == 0
    assert load_gain(cfg, Mode.RELEASE, 0.2) == 0
    assert fixture_gain(cfg, Mode.RELEASE, 0) == 0
    assert fixture_gain(cfg, Mode.RELEASE, 0.2) > 0


def test_rectangular_rung_axis_and_dimensions():
    cfg = replace(GraspConfig(), cross_section='rectangular', width=0.018, height=0.026)
    env = O6GraspEnv(cfg)
    _, info = env.reset()
    np.testing.assert_allclose(env.model.geom_size[env.rung_geom], [cfg.length / 2, 0.009, 0.013])
    np.testing.assert_allclose(info['raw_observation'][39:43], [0.018, 0.026, 0, 1])
    np.testing.assert_allclose(env.data.site('rung_frame').xmat.reshape(3, 3)[:, 0], [0, 1, 0], atol=1e-15)


def test_release_rewards_never_reward_holding():
    from teleopit.sim.o6_grasp.diagnostics import reward_terms

    info = dict(plausible_contacts=True, verification='VERIFIED', contact_slip_speed=1,
                effort_fraction=0.5, release_clear=True)
    terms = reward_terms(Mode.RELEASE, info, 0)
    assert terms['stable_support'] == 0 and terms['acquisition'] == 0
    assert terms['authorized_release'] == 1 and terms['relative_slip'] == 0
