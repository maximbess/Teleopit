"""Residual control, contact gates and soft limit margins for a single transfer."""
from types import SimpleNamespace

import pytest
import torch

from train_mimic.tasks.tracking.mdp.first_hand import (
    FirstHandCommand, FirstHandCommandCfg, FirstHandStage as Stage,
    ResidualReferenceAction, event_reward, proximity_cost,
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
    c._stage_at_start = c.motion_stage.clone()
    for name in ('motion_failed', 'finished', '_fresh_reset', '_release_active', 'grasp_pulse', 'success_pulse', 'just_advanced'):
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
    c._resample_command(torch.tensor([0]))
    assert c.reference_time.tolist() == [0., pytest.approx(3.9)]
    assert c.reference_joint_vel[0].eq(0).all()
    assert c.reference_joint_vel[1].eq(1).all()
    assert c.reference_position[1].eq(1).all()
    assert c.reference_joint_pos[1].eq(1).all()


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


def test_checkpoint_rejects_different_reference_before_loading_policy(monkeypatch):
    from train_mimic.tasks.tracking.rl.first_hand_runner import FirstHandOnPolicyRunner
    from train_mimic.tasks.tracking.rl.runner import LadderOnPolicyRunner
    runner = object.__new__(FirstHandOnPolicyRunner)
    runner._ladder_command = lambda: SimpleNamespace(reference_signature='current')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'infos': {'first_hand_reference_signature': 'other'}})
    monkeypatch.setattr(LadderOnPolicyRunner, 'load', lambda *args, **kwargs: pytest.fail('policy must not be loaded'))
    with pytest.raises(ValueError, match='different first-hand reference'):
        runner.load('old.pt')
