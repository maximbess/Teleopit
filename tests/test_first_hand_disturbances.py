"""Force lifetime, per-world resets and mandatory post-grasp recovery."""
from types import SimpleNamespace

import torch

from train_mimic.tasks.tracking.mdp.first_hand_disturbances import FirstHandDisturbances
from train_mimic.tasks.tracking.config.first_hand import make_first_hand_env_cfg
from test_first_hand_motion import gate_command, tick, GateCommand, Stage


def disturbance(c):
    d = object.__new__(FirstHandDisturbances)
    d.c, d.cfg, d.device, d.n = c, c.cfg, 'cpu', 2
    c.cfg.clean_episode_probability = 0.
    c.cfg.gravity_scale_range = (1.03, 1.03)
    c.cfg.push_weight_fraction = (.03, .03)
    c.cfg.push_duration_s = (.04, .04)
    c.cfg.push_interval_s = (.12, .12)
    d.nominal_gravity = torch.tensor([0., 0., -9.81])
    d.gravity = d.nominal_gravity.repeat(2, 1)
    d.weight = 300.
    d.force = torch.zeros(2, 3)
    d.gravity_scale = torch.ones(2)
    for name in ('remaining', 'cooldown', 'push_count', 'peak_force'):
        setattr(d, name, torch.zeros(2))
    for name in ('clean', 'challenge_started', 'challenge_done', 'challenge_active'):
        setattr(d, name, torch.zeros(2, dtype=torch.bool))
    c._torso_body_id = 1
    c._env.sim = SimpleNamespace(data=SimpleNamespace(xfrc_applied=torch.zeros(2, 3, 6)))
    d.reset(torch.arange(2))
    return d


def test_gravity_is_constant_during_episode_and_partial_reset_is_independent():
    c = gate_command(Stage.TRANSFER)
    d = disturbance(c)
    torch.testing.assert_close(d.gravity[:, 2], torch.full((2,), -9.81*1.03))
    d.cooldown[:] = 0
    d.tick(torch.full((2,), .02))
    before = {name: getattr(d, name)[1].clone() for name in ('force', 'remaining', 'cooldown', 'gravity', 'push_count')}
    c.cfg.clean_episode_probability = 1.
    d.reset(torch.tensor([0]))
    torch.testing.assert_close(d.gravity[0], d.nominal_gravity)
    assert not d.force[0].any()
    assert not c._env.sim.data.xfrc_applied[0].any()
    for name, value in before.items():
        torch.testing.assert_close(getattr(d, name)[1], value)
    d.tick(torch.full((2,), .02))
    torch.testing.assert_close(d.gravity[1], before['gravity'])


def test_push_has_bounded_horizontal_force_and_expires_without_teleport():
    c = gate_command(Stage.TRANSFER)
    d = disturbance(c)
    d.cooldown[:] = 0
    d.tick(torch.full((2,), .02))
    torch.testing.assert_close(d.force.norm(dim=-1), torch.full((2,), 9.))
    assert d.force[:, 2].eq(0).all()
    torch.testing.assert_close(c._env.sim.data.xfrc_applied[:, 1, :3], d.force)
    assert not c._env.sim.data.xfrc_applied[:, [0, 2]].any()
    d.tick(torch.full((2,), .02))
    assert d.remaining.gt(0).all()
    d.tick(torch.full((2,), .02))
    assert not d.force.any()
    assert not c._env.sim.data.xfrc_applied.any()
    assert d.push_count.tolist() == [1., 1.]


def test_stabilization_and_clean_episodes_have_no_push_and_done_clears_force():
    c = gate_command(Stage.STABILIZE)
    d = disturbance(c)
    d.cooldown[:] = 0
    d.tick(torch.full((2,), .02))
    assert not d.force.any()
    c.motion_stage[:] = int(Stage.TRANSFER)
    d.clean[0] = True
    d.tick(torch.full((2,), .02))
    assert not d.force[0].any()
    assert d.force[1].norm() > 0
    c.finished[1] = True
    d.tick(torch.full((2,), .02))
    assert not d.force.any()


def test_hold_waits_for_test_pulse_then_continuous_recovery(monkeypatch):
    c = gate_command(Stage.HOLD)
    c.attached[:] = True
    c.held_rung[:, 0] += 1
    c.stage_elapsed[:] = 0
    c.cfg.stable_hold_s = 2.
    c.cfg.minimum_hold_s = 3.
    d = disturbance(c)
    c.disturbances = d
    c.robot = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(2, 3)))
    c._waist_joint_ids = torch.arange(3)
    angular = torch.zeros(2, 3)
    monkeypatch.setattr(GateCommand, 'torso_ang_vel_w', property(lambda self: angular))
    monkeypatch.setattr(GateCommand, 'pelvis_ang_vel_w', property(lambda self: angular))
    for _ in range(24):
        tick(c); d.tick(c._dt)
    assert c.hold_elapsed.eq(0).all()
    assert not d.challenge_started.any()
    for _ in range(8):
        tick(c); d.tick(c._dt)
    assert d.challenge_done.all()
    assert not c.finished.any()
    for _ in range(80):
        tick(c); d.tick(c._dt)
    angular[0, 0] = .5
    tick(c); d.tick(c._dt)
    assert c.hold_elapsed[0] == 0
    angular.zero_()
    for _ in range(45):
        tick(c); d.tick(c._dt)
    assert c.finished.tolist() == [False, True]
    for _ in range(60):
        tick(c); d.tick(c._dt)
    assert c.finished.all()
    assert d.push_count.tolist() == [1., 1.]


def test_clean_hold_requires_full_dwell_and_low_waist_speed(monkeypatch):
    c = gate_command(Stage.HOLD)
    c.attached[:] = True
    c.held_rung[:, 0] += 1
    c.stage_elapsed[:] = 0
    c.cfg.stable_hold_s = 2.
    c.cfg.minimum_hold_s = 3.
    d = disturbance(c)
    c.disturbances = d
    d.clean[:] = True
    c.robot = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(2, 3)))
    c._waist_joint_ids = torch.arange(3)
    monkeypatch.setattr(GateCommand, 'torso_ang_vel_w', property(lambda self: torch.zeros(2, 3)))
    monkeypatch.setattr(GateCommand, 'pelvis_ang_vel_w', property(lambda self: torch.zeros(2, 3)))
    c.robot.data.joint_vel[0, 0] = .7
    for _ in range(130):
        tick(c); d.tick(c._dt)
    assert not c.finished.any()  # Even the stable environment must wait 3 seconds.
    for _ in range(25):
        tick(c); d.tick(c._dt)
    assert c.finished.tolist() == [False, True]
    assert c.hold_elapsed[0] == 0
    assert not d.push_count.any()


def test_robust_task_configuration_allows_recovery_timeout():
    cfg = make_first_hand_env_cfg()
    c = cfg.commands['ladder']
    assert c.disturbances_enabled
    assert c.stable_hold_s == 2. and c.minimum_hold_s == 3.
    assert c.hold_timeout_s == 8.
    assert 'motion_reference' in cfg.observations['actor'].terms
    assert cfg.episode_length_s == 20.
