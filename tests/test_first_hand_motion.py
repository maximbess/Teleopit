"""Residual control, contact gates and soft limit margins for a single transfer."""
from types import SimpleNamespace

import pytest
import torch

from train_mimic.tasks.tracking.mdp.first_hand import (
    FirstHandCommand, FirstHandCommandCfg, FirstHandStage as Stage,
    FirstHandFailure as Failure, ResidualReferenceAction, event_reward, proximity_cost,
    body_angular_velocity_cost, waist_velocity_cost, release_orientation_cost,
)
from train_mimic.tasks.tracking.config.first_hand import make_first_hand_env_cfg


def test_limit_margin_is_symmetric_and_increases_towards_both_limits():
    q = torch.tensor([[0.], [.70], [.85], [1.], [1.15]], requires_grad=True)
    limits = torch.tensor([[[-1., 1.]]]).expand(5, -1, -1)
    cost = proximity_cost(q, limits)
    torch.testing.assert_close(cost, torch.tensor([0., 0., .25, 1., 2.25]))
    torch.testing.assert_close(proximity_cost(-q, limits), cost)
    cost.sum().backward()
    assert q.grad[2:].gt(0).all()


def test_unlimited_and_degenerate_joints_do_not_dilute_limit_cost():
    q = torch.tensor([[1., 0., 0.]])
    limits = torch.tensor([[[-1., 1.], [-float('inf'), float('inf')], [0., 0.]]])
    torch.testing.assert_close(proximity_cost(q, limits), torch.ones(1))
    with pytest.raises(ValueError):
        proximity_cost(q, limits, .5)


def test_residual_actions_follow_reference_joint_order_and_hard_limits():
    action = object.__new__(ResidualReferenceAction)
    action.cfg = SimpleNamespace(residual_bound=1.)
    action._target_ids = torch.tensor([2, 0])
    action._raw_actions = torch.zeros(1, 2)
    action._processed_actions = torch.zeros(1, 2)
    action._scale = .2
    action._reference = SimpleNamespace(reference_joint_pos=torch.tensor([[.9, -.5, .3]]))
    action._entity = SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-1., 1.]]*3])))
    action.process_actions(torch.zeros(1, 2))
    torch.testing.assert_close(action._processed_actions, torch.tensor([[.3, .9]]))
    action.process_actions(torch.tensor([[-50., 50.]]))
    torch.testing.assert_close(action._processed_actions, torch.tensor([[.1, 1.]]))


class GateCommand(FirstHandCommand):
    """Use actual stage logic with measured contacts/poses supplied by each test."""
    @property
    def hand_pos_w(self): return self.hand
    @property
    def foot_support(self): return self.feet
    @property
    def active_hand_pos_w(self): return self.hand[:, 0]
    @property
    def target_pos_w(self): return self._position_reference[-1, 0].expand(2, -1)
    @property
    def active_hand_vel_w(self): return self.velocity
    @property
    def phase_completion_stable(self): return self.stable
    @property
    def torso_orientation_error(self): return torch.zeros(2)
    def _release_is_stable(self): return self.stable
    def _begin_release(self, ids, hand): self._release_active[ids] = True
    def _advance_release_ramp(self, mask): pass
    def _attach(self, ids, hand, rungs):
        self.attached[ids, hand] = True
        self.held_rung[ids, hand] = rungs


def gate_command(stage=Stage.TRANSFER):
    c = object.__new__(GateCommand)
    c.cfg = FirstHandCommandCfg(entity_name='robot', resampling_time_range=(1e9, 1e9))
    c._env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02)
    c.motion_stage = torch.full((2,), int(stage), dtype=torch.long)
    c.max_motion_stage = c.motion_stage.clone()
    c.failure_reason = torch.zeros(2, dtype=torch.long)
    c.failure_stage = torch.full((2,), -1, dtype=torch.long)
    c._stage_at_start = c.motion_stage.clone()
    for name in ('motion_failed', 'finished', '_fresh_reset', '_release_active', 'release_pulse', 'grasp_pulse', 'success_pulse', 'just_advanced'):
        setattr(c, name, torch.zeros(2, dtype=torch.bool))
    c.reference_time = torch.full((2,), 3.9)
    c.stage_elapsed = torch.ones(2)
    c.hold_elapsed = torch.zeros(2)
    c.progress_pulse = torch.zeros(2)
    c._phase_dwell_count = torch.zeros(2, dtype=torch.long)
    c._dt = torch.full((2,), .02)
    c._position_reference = torch.zeros(2, 6, 3)
    c._hold_hand_offset = torch.zeros(2, 3)
    c.hand = torch.zeros(2, 2, 3)
    c.velocity = torch.zeros(2, 3)
    c.feet = torch.ones(2, 2, dtype=torch.bool)
    c.attached = torch.tensor([[False, True], [False, True]])
    c.stable = torch.ones(2, dtype=torch.bool)
    c.target_rung = torch.full((2,), c.cfg.start_rung + 1, dtype=torch.long)
    c.held_rung = torch.full((2, 2), c.cfg.start_rung, dtype=torch.long)
    return c


def tick(c):
    c._stage_at_start = c.motion_stage.clone()
    c.stage_elapsed += c._dt
    c.release_pulse.zero_()
    c.grasp_pulse.zero_()
    c.success_pulse.zero_()
    c._advance_hand_phase(None)


def test_free_hand_is_expected_only_during_transfer():
    c = gate_command()
    assert c.required_supports.all()
    c.motion_stage[0] = int(Stage.PREPARE)
    c.feet[1, 0] = False
    assert not c.required_supports.any()


def test_endpoint_waits_for_contact_speed_and_continuous_dwell():
    c = gate_command()
    c.reference_time[:] = 2.
    for _ in range(5): tick(c)
    assert not c.attached[:, 0].any()  # Endpoint proximity alone cannot skip motion.
    c.reference_time[:] = 3.9
    c.velocity[0, 0] = 2.
    c.feet[1, 0] = False
    for _ in range(5): tick(c)
    assert (c.motion_stage == int(Stage.TRANSFER)).all()
    assert (c.reference_time == 3.9).all()
    c.velocity.zero_()
    c.feet[:] = True
    for _ in range(c.cfg.hand_target_dwell_steps-1): tick(c)
    assert not c.attached[:, 0].any()
    c.feet[:] = False
    tick(c)
    c.feet[:] = True
    for _ in range(c.cfg.hand_target_dwell_steps): tick(c)
    assert c.attached.all()
    assert (c.motion_stage == int(Stage.HOLD)).all()
    assert c.grasp_pulse.all()
    assert not c.finished.any()


def test_success_needs_continuous_stable_hold_and_is_one_shot():
    c = gate_command(Stage.HOLD)
    c.attached[:] = True
    c.held_rung[:, 0] += 1
    c.stage_elapsed.zero_()
    for _ in range(10): tick(c)
    assert not c.finished.any()
    c.feet[0, 0] = False
    tick(c)
    assert c.hold_elapsed[0] == 0
    c.feet[:] = True
    for _ in range(20): tick(c)
    assert c.finished.all()
    assert (c.motion_stage == int(Stage.DONE)).all()
    tick(c)
    assert not c.success_pulse.any()
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: c), step_dt=.02)
    assert not event_reward(env, event='success').any()
    c.success_pulse[0] = True
    torch.testing.assert_close(event_reward(env, event='success') * env.step_dt, torch.tensor([1., 0.]))


def test_preparation_waits_for_safe_release_and_keeps_reference_at_boundary():
    c = gate_command(Stage.PREPARE)
    c.attached[:] = True
    c.reference_time[:] = 1.69
    c.stable[0] = False
    tick(c)
    assert c.motion_stage.tolist() == [int(Stage.PREPARE), int(Stage.RELEASE)]
    torch.testing.assert_close(c.reference_time, torch.full((2,), 1.7))
    assert c._release_active.tolist() == [False, True]
    assert not c.release_pulse.any()


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_release_reward_requires_detach_and_fires_once_per_episode(dt):
    c = gate_command(Stage.RELEASE)
    c.reference_time[:] = 1.7
    c.attached[:] = True
    c._release_active[:] = True
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: c), step_dt=dt)
    tick(c)
    assert not event_reward(env, event='release').any()
    c.attached[0, 0] = False
    c._release_active[0] = False
    tick(c)
    assert c.motion_stage.tolist() == [int(Stage.TRANSFER), int(Stage.RELEASE)]
    torch.testing.assert_close(event_reward(env, event='release')*dt, torch.tensor([1., 0.]))
    tick(c)
    assert not event_reward(env, event='release').any()


def test_first_hand_task_is_fixed_and_has_terminal_success_and_soft_limit_cost():
    cfg = make_first_hand_env_cfg()
    assert not cfg.commands['ladder'].curriculum_enabled
    assert not cfg.commands['ladder'].freeze_at_max_unlocked_phase
    assert cfg.actions['joint_pos'].scale == .2
    assert cfg.rewards['joint_limit_proximity'].weight < 0
    assert cfg.rewards['joint_limits'].weight < 0
    assert not cfg.terminations['success'].time_out
    assert 'curriculum_stage_complete' not in cfg.terminations


@pytest.mark.parametrize('side', ['actor', 'critic'])
def test_unseen_motion_stage_stays_one_hot_after_normalization_and_export(side):
    from tensordict import TensorDict
    from train_mimic.tasks.tracking.rl.first_hand_model import FirstHandTemporalCNNModel
    current = torch.zeros(3, 14)
    current[:, 0] = current[:, 6] = 1
    current[:, 11:] = torch.arange(3).float()[:, None]
    obs = TensorDict({side: current, side+'_history': current[:, None].repeat(1, 10, 1)}, [3])
    model = FirstHandTemporalCNNModel(obs, {side: [side, side+'_history']}, side, 2,
                                     hidden_dims=(8,), obs_normalization=True,
                                     cnn_cfg={'output_channels': (4,), 'kernel_size': 3, 'global_pool': 'avg'})
    model.update_normalization(obs)
    model.eval()
    for value in obs.values():
        value[..., 0] = 0
        value[..., 3] = 1
    torch.testing.assert_close(model.obs_normalizer(current)[..., :11], current[..., :11])
    history = obs[side+'_history']
    torch.testing.assert_close(model.obs_normalizers_3d[side+'_history'](history)[..., :11], history[..., :11])
    torch.testing.assert_close(model.as_onnx()(current, history), model(obs))


def test_partial_reset_does_not_change_another_environments_reference(monkeypatch):
    from train_mimic.tasks.tracking.mdp.ladder import LadderClimbCommand
    monkeypatch.setattr(LadderClimbCommand, '_resample_command', lambda self, ids: None)
    c = gate_command()
    c.support_loss_elapsed = torch.ones(2)
    c.best_reach = torch.ones(2)
    c._joint_reference = torch.zeros(3, 4)
    c._quaternion_reference = torch.zeros(3, 6, 4)
    c._quaternion_reference[..., 0] = 1
    c.reference_joint_pos = torch.ones(2, 4)
    c.reference_joint_vel = torch.ones(2, 4)
    c.reference_position = torch.ones(2, 6, 3)
    c.reference_quaternion = torch.ones(2, 6, 4)
    c.failure_reason[:] = int(Failure.SUPPORT_LOST)
    c.failure_stage[:] = int(Stage.TRANSFER)
    c.release_pulse[:] = True
    c._resample_command(torch.tensor([0]))
    assert c.reference_time.tolist() == [0., pytest.approx(3.9)]
    assert c.reference_joint_vel[0].eq(0).all()
    assert c.reference_joint_vel[1].eq(1).all()
    assert c.reference_position[1].eq(1).all()
    assert c.reference_joint_pos[1].eq(1).all()
    assert c.failure_reason.tolist() == [int(Failure.NONE), int(Failure.SUPPORT_LOST)]
    assert c.failure_stage.tolist() == [-1, int(Stage.TRANSFER)]
    assert c.max_motion_stage.tolist() == [0, int(Stage.TRANSFER)]
    assert c.release_pulse.tolist() == [False, True]


def updating_command(monkeypatch):
    from train_mimic.tasks.tracking.mdp.ladder import LadderClimbCommand
    monkeypatch.setattr(LadderClimbCommand, '_update_command', lambda self: None)
    c = gate_command()
    c._sample_reference = lambda rate: None
    c._compute_dt = .02
    c._pending_start_pose_init = torch.zeros(2, dtype=torch.bool)
    c.initialized = torch.ones(2, dtype=torch.bool)
    c.support_loss_elapsed = torch.zeros(2)
    c.pre_release_stalled = torch.zeros(2, dtype=torch.bool)
    c.best_reach = torch.zeros(2)
    c._position_reference[-1, 0, 0] = 1.
    return c


def test_progress_rewards_only_new_reach_and_only_with_required_supports(monkeypatch):
    c = updating_command(monkeypatch)
    c.hand[:, 0, 0] = .5
    c._update_command()
    torch.testing.assert_close(c.progress_pulse, torch.full((2,), .5))
    c.hand[:, 0, 0] = .2
    c._update_command()
    assert c.progress_pulse.eq(0).all()
    c.hand[:, 0, 0] = .6
    c.feet[1, 0] = False
    c._update_command()
    torch.testing.assert_close(c.progress_pulse, torch.tensor([.1, 0.]))


@pytest.mark.parametrize('lost_support', ['foot', 'right_hand'])
def test_release_event_suppressed_when_remaining_support_is_lost(monkeypatch, lost_support):
    from train_mimic.tasks.tracking.mdp.ladder import LadderClimbCommand
    c = updating_command(monkeypatch)
    c.motion_stage[:] = int(Stage.RELEASE)
    c.reference_time[:] = 1.7
    if lost_support == 'foot':
        c.feet[0, 0] = False
    else:
        c.attached[0, 1] = False
    monkeypatch.setattr(LadderClimbCommand, '_update_command', lambda self: self._advance_hand_phase(None))
    c._update_command()
    assert c.motion_stage.tolist() == [int(Stage.TRANSFER)]*2
    assert c.release_pulse.tolist() == [False, True]
    c._update_command()
    assert not c.release_pulse.any()


def test_support_loss_grace_and_transfer_timeout_are_terminal(monkeypatch):
    c = updating_command(monkeypatch)
    c.feet[0, 0] = False
    for _ in range(7): c._update_command()
    assert not c.motion_failed.any()
    c._update_command()
    assert c.motion_failed.tolist() == [True, False]
    c.stage_elapsed[1] = 100.
    c._update_command()
    assert c.motion_failed.all()


def test_stationary_hand_weld_loss_is_immediately_terminal(monkeypatch):
    c = updating_command(monkeypatch)
    c.attached[0, 1] = False
    c._update_command()
    assert c.motion_failed.tolist() == [True, False]
    assert c.failure_reason.tolist() == [int(Failure.RIGHT_HAND_LOST), int(Failure.NONE)]
    assert c.failure_stage.tolist() == [int(Stage.TRANSFER), -1]


@pytest.mark.parametrize('stage', list(Stage)[:-1])
def test_timeout_records_exact_stage_and_resets_independently(monkeypatch, stage):
    c = updating_command(monkeypatch)
    c.motion_stage[:] = int(stage)
    c.attached[:] = True
    c.stage_elapsed[0] = 100.
    c._update_command()
    assert c.failure_reason.tolist() == [int(Failure.STAGE_TIMEOUT), int(Failure.NONE)]
    assert c.failure_stage.tolist() == [int(stage), -1]
    # A later loss of the right weld must not overwrite the original timeout.
    c.attached[0, 1] = False
    c._update_command()
    assert c.failure_reason[0] == int(Failure.STAGE_TIMEOUT)


def test_failure_priority_and_release_stall_are_unambiguous(monkeypatch):
    c = updating_command(monkeypatch)
    c.motion_stage[:] = int(Stage.RELEASE)
    c.attached[:] = True
    c.attached[0, 1] = False
    c.feet[0, 0] = False
    c.support_loss_elapsed[0] = 1.
    c.stage_elapsed[:] = 100.
    c.pre_release_stalled[:] = True
    c._update_command()
    assert c.failure_reason.tolist() == [int(Failure.RIGHT_HAND_LOST), int(Failure.RELEASE_STALLED)]
    assert c.failure_stage.tolist() == [int(Stage.RELEASE)]*2


def test_support_reason_requires_continuous_loss(monkeypatch):
    c = updating_command(monkeypatch)
    c.feet[0, 0] = False
    for _ in range(7): c._update_command()
    c.feet[0, 0] = True
    c._update_command()
    assert c.support_loss_elapsed[0] == 0
    c.feet[0, 0] = False
    for _ in range(8): c._update_command()
    assert c.failure_reason.tolist() == [int(Failure.SUPPORT_LOST), int(Failure.NONE)]


def test_terminal_metrics_are_captured_after_transition_before_partial_reset(monkeypatch):
    from train_mimic.tasks.tracking.mdp.ladder import LadderClimbCommand
    monkeypatch.setattr(LadderClimbCommand, '_update_metrics', lambda self: None)
    c = gate_command(Stage.HOLD)
    names = ['motion_stage', 'reference_time', 'hand_tracking_error', 'hold_progress',
             'motion_failed', 'first_hand_success', 'max_motion_stage']
    names += [f'failure/{stage.name.lower()}/{reason.name.lower()}'
              for stage in list(Stage)[:-1] for reason in list(Failure)[1:]]
    c.metrics = {name: torch.zeros(2) for name in names}
    c.reference_position = torch.zeros(2, 6, 3)
    c.time_left = torch.ones(2)
    c.command_counter = torch.zeros(2, dtype=torch.long)

    def transition():
        c._fail(torch.tensor([True, False]), Failure.STAGE_TIMEOUT)
        c.finished[1] = True
        c._set_stage(torch.tensor([1]), Stage.DONE)
    c._update_command = transition
    c.compute(.02)
    assert c.metrics['motion_failed'].tolist() == [1., 0.]
    assert c.metrics['first_hand_success'].tolist() == [0., 1.]
    assert c.metrics['failure/hold/stage_timeout'].tolist() == [1., 0.]
    assert c.metrics['max_motion_stage'].tolist() == [4., 5.]
    torch.testing.assert_close(c.time_left, torch.full((2,), .98))

    # Real CommandTerm.reset exports snapshots before resampling and only clears
    # the selected environment. Physics is intentionally absent from this test.
    c._resample_command = lambda ids: None
    failure_log = c.reset(torch.tensor([0]))
    assert failure_log['motion_failed'] == 1.
    assert failure_log['failure/hold/stage_timeout'] == 1.
    assert failure_log['first_hand_success'] == 0.
    success_log = c.reset(torch.tensor([1]))
    assert success_log['first_hand_success'] == 1.
    assert success_log['motion_stage'] == 5.
    assert all(value == 0. for name, value in success_log.items() if name.startswith('failure/'))


def test_command_compute_preserves_due_resampling_before_update():
    c = gate_command()
    c.time_left = torch.tensor([.01, 2.])
    calls = []
    c._resample = lambda ids: calls.append(('resample', ids.tolist()))
    c._update_command = lambda: calls.append(('update', c._compute_dt))
    c._update_metrics = lambda: calls.append(('metrics', None))
    c.compute(.02)
    assert calls == [('resample', [0]), ('update', .02), ('metrics', None)]


def test_body_and_waist_speed_penalties_match_gates_and_ignore_arm_speed():
    c = SimpleNamespace(torso_ang_vel_w=torch.tensor([[0., 0., 0.], [.4, 0., 0.], [.8, 0., 0.]]),
                        pelvis_ang_vel_w=torch.tensor([[0., 0., .4], [0., .8, 0.], [0., 0., 0.]]),
                        _waist_joint_ids=torch.tensor([1, 2, 3]),
                        robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.tensor([
                            [100., 0., 0., 0.], [0., -.6, .3, .1], [0., .2, 1.2, .4]]))))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: c))
    torch.testing.assert_close(body_angular_velocity_cost(env), torch.tensor([0., 1., 4.]))
    torch.testing.assert_close(body_angular_velocity_cost(env, body='pelvis'), torch.tensor([1., 4., 0.]))
    torch.testing.assert_close(waist_velocity_cost(env), torch.tensor([0., 1., 4.]))
    with pytest.raises(ValueError): body_angular_velocity_cost(env, body='arm')
    with pytest.raises(ValueError): waist_velocity_cost(env, std=0.)


def test_first_hand_reduces_exploration_without_changing_legacy_ladder():
    from train_mimic.tasks.tracking.config.first_hand import make_first_hand_runner_cfg
    from train_mimic.tasks.tracking.config.rl import make_g1_ladder_ppo_runner_cfg
    cfg = make_first_hand_runner_cfg()
    assert cfg.actor.distribution_cfg['init_std'] == .1
    assert cfg.actor.distribution_cfg['std_range'] == (.03, .2)
    assert cfg.algorithm.entropy_coef == .0005
    assert make_g1_ladder_ppo_runner_cfg().algorithm.entropy_coef == .005
    env = make_first_hand_env_cfg()
    assert env.rewards['torso_angular_velocity'].weight == -.5
    assert env.rewards['pelvis_angular_velocity'].weight == -.5
    assert env.rewards['waist_velocity'].weight == -.25
    assert env.commands['ladder'].stabilization_dwell_steps == 50


def test_release_orientation_cost_uses_gate_margin_and_only_preparation_stages():
    errors = torch.tensor([.15, .20, .225, .25, .30, .30, .30, .30], requires_grad=True)
    c = SimpleNamespace(
        cfg=SimpleNamespace(max_release_torso_orientation_error=.25),
        motion_stage=torch.tensor([int(s) for s in (
            Stage.PREPARE, Stage.RELEASE, Stage.PREPARE, Stage.RELEASE,
            Stage.STABILIZE, Stage.TRANSFER, Stage.HOLD, Stage.DONE)]),
        torso_orientation_error=errors,
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: c))
    cost = release_orientation_cost(env)
    torch.testing.assert_close(cost, torch.tensor([0., 0., .25, 1., 0., 0., 0., 0.]))
    cost.sum().backward()
    assert errors.grad[2:4].gt(0).all()
    assert errors.grad[4:].eq(0).all()
    # The target follows the configured gate, rather than a second fixed angle.
    c.cfg.max_release_torso_orientation_error = .35
    assert release_orientation_cost(env).eq(0).all()
    with pytest.raises(ValueError): release_orientation_cost(env, margin=.35)
    with pytest.raises(ValueError): release_orientation_cost(env, std=0.)


def test_first_hand_release_rewards_and_checkpoint_interval():
    from train_mimic.tasks.tracking.config.first_hand import make_first_hand_runner_cfg
    cfg = make_first_hand_env_cfg()
    assert cfg.rewards['release_orientation'].weight == -2.
    assert cfg.rewards['successful_release'].weight == 10.
    assert cfg.rewards['successful_release'].params['event'] == 'release'
    assert make_first_hand_runner_cfg().save_interval == 1000


def test_resumed_large_std_is_projected_to_new_upper_bound():
    from rsl_rl.modules.distribution import GaussianDistribution
    from train_mimic.tasks.tracking.rl.runner import _project_distribution_std
    old = GaussianDistribution(3, init_std=.6, std_range=(.05, .6), std_type='scalar')
    new = GaussianDistribution(3, init_std=.1, std_range=(.03, .2), std_type='scalar')
    new.load_state_dict(old.state_dict())
    _, projected = _project_distribution_std(new)
    assert projected
    torch.testing.assert_close(new.std_param, torch.full_like(new.std_param, .2))


def test_checkpoint_rejects_different_reference_before_loading_policy(monkeypatch):
    from train_mimic.tasks.tracking.rl.first_hand_runner import FirstHandOnPolicyRunner
    from train_mimic.tasks.tracking.rl.runner import LadderOnPolicyRunner
    runner = object.__new__(FirstHandOnPolicyRunner)
    runner._ladder_command = lambda: SimpleNamespace(reference_signature='current')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'infos': {'first_hand_reference_signature': 'other'}})
    monkeypatch.setattr(LadderOnPolicyRunner, 'load', lambda *args, **kwargs: pytest.fail('policy must not be loaded'))
    with pytest.raises(ValueError, match='different first-hand reference'):
        runner.load('old.pt')


def test_playback_reference_override_warns_and_keeps_strict_weight_loading(monkeypatch):
    from train_mimic.tasks.tracking.rl.first_hand_runner import FirstHandOnPolicyRunner
    from train_mimic.tasks.tracking.rl.runner import LadderOnPolicyRunner
    runner = object.__new__(FirstHandOnPolicyRunner)
    runner._ladder_command = lambda: SimpleNamespace(reference_signature='current')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'infos': {'first_hand_reference_signature': 'other'}})
    called = []
    monkeypatch.setattr(LadderOnPolicyRunner, 'load', lambda self, *args: called.append(args))
    with pytest.warns(RuntimeWarning, match='CURRENT'):
        runner.load('old.pt', map_location='cpu', allow_reference_mismatch=True)
    assert called == [('old.pt', None, True, 'cpu')]
