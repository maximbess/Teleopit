import json
from types import SimpleNamespace as NS

import pytest
import torch

from train_mimic.tasks.tracking.rl.nonfinite import NonfiniteDiagnostics
from train_mimic.tasks.tracking.rl.runner import LadderOnPolicyRunner


def fixture_env():
    command = NS(motion_stage=torch.tensor([2, 3, 4]), reference_time=torch.tensor([1.7, 2.1, 3.9]),
                 attached=torch.ones(3, 2, dtype=torch.bool), disturbances=None)
    env = NS(num_envs=3, device='cpu',
             sim=NS(data=NS(qpos=torch.arange(12.).reshape(3, 4),
                            qvel=torch.ones(3, 3), qacc=torch.zeros(3, 3),
                            ctrl=torch.zeros(3, 2), xfrc_applied=torch.zeros(3, 5, 6),
                            eq_active=torch.ones(3, 2))),
             command_manager=NS(get_term=lambda _: command),
             observation_manager=NS(active_terms={'actor': ['joint_pos', 'joint_vel']},
                                    group_obs_term_dim={'actor': [(2,), (3,)]}),
             cfg=NS(observations={'actor': NS(concatenate_dim=-1)}),
             action_manager=NS(action=torch.zeros(3, 2)), episode_length_buf=torch.arange(3))
    env.unwrapped = env
    return env, command


def check(guard, obs, **kwargs):
    guard.check({f'observations/{k}': v for k, v in obs.items()}, iteration=2629,
                rollout_step=7, when='after_env_step', observations=obs, **kwargs)


def test_finite_values_do_not_create_files(tmp_path):
    env, _ = fixture_env()
    check(NonfiniteDiagnostics(env, tmp_path, 2), {'actor': torch.zeros(3, 5)})
    assert not list(tmp_path.iterdir())


def test_first_rank_local_dump_contains_exact_component_and_state(tmp_path):
    env, command = fixture_env()
    guard = NonfiniteDiagnostics(env, tmp_path, 2)
    before = guard.context()
    command.motion_stage[1] = 0  # A reset must not erase the pre-step stage.
    obs = {'actor': torch.zeros(3, 5)}
    obs['actor'][1, 3] = float('nan')
    obs['actor'][2, 0] = float('inf')
    actions = torch.arange(6.).reshape(3, 2)
    with pytest.raises(ValueError, match='rank=2'):
        check(guard, obs, actions=actions, before=before)
    path, = (tmp_path/'nonfinite_diagnostics').glob('rank2*.pt')
    saved = torch.load(path, weights_only=True)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['bad_env_ids'] == [1, 2]
    assert meta['iteration'] == 2629 and meta['rollout_step'] == 7
    assert meta['findings'][0]['components'][0]['term'] == 'joint_vel'
    assert meta['findings'][0]['components'][0]['term_axis_index'] == 1
    torch.testing.assert_close(saved['tensors']['physics/qpos'], env.sim.data.qpos[[1, 2]])
    torch.testing.assert_close(saved['tensors']['physics/qvel'], env.sim.data.qvel[[1, 2]])
    torch.testing.assert_close(saved['tensors']['physics/qacc'], env.sim.data.qacc[[1, 2]])
    torch.testing.assert_close(saved['tensors']['actions'], actions[[1, 2]])
    assert saved['tensors']['before/motion_stage'].tolist() == [3, 4]
    assert saved['tensors']['command/motion_stage'].tolist() == [0, 4]
    with pytest.raises(ValueError):
        check(guard, obs)
    assert len(list(path.parent.glob('*.pt'))) == 1
    with pytest.raises(ValueError):
        check(NonfiniteDiagnostics(env, tmp_path, 3), obs)
    assert len(list(path.parent.glob('*.pt'))) == 2


def test_unavailable_physics_field_does_not_lose_observation_evidence(tmp_path):
    env, _ = fixture_env()
    del env.sim.data.qacc
    with pytest.raises(ValueError):
        check(NonfiniteDiagnostics(env, tmp_path, 0), {'actor': torch.full((3, 5), float('nan'))})
    path, = (tmp_path/'nonfinite_diagnostics').glob('*.json')
    assert 'physics/qacc' in json.loads(path.read_text())['capture_errors']
    assert path.with_suffix('.pt').exists()


def test_physics_tensor_proxies_are_captured(tmp_path):
    env, _ = fixture_env()
    qpos = env.sim.data.qpos.clone()
    env.sim.data.qpos = NS(detach=lambda: qpos.detach())
    with pytest.raises(ValueError):
        check(NonfiniteDiagnostics(env, tmp_path, 0), {'actor': torch.full((3, 5), float('nan'))})
    path, = (tmp_path/'nonfinite_diagnostics').glob('*.pt')
    saved = torch.load(path, weights_only=True)
    torch.testing.assert_close(saved['tensors']['physics/qpos'], qpos)


def test_io_error_preserves_nonfinite_training_error(tmp_path, monkeypatch):
    env, _ = fixture_env()
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(ValueError, match='Non-finite.*diagnostic save failed: OSError: disk full'):
        check(NonfiniteDiagnostics(env, tmp_path, 0), {'actor': torch.full((3, 5), float('nan'))})


@pytest.mark.parametrize('bad_action', [False, True])
def test_runner_stops_before_ppo_and_dumps_without_tensorboard(tmp_path, bad_action):
    env, _ = fixture_env()
    obs = {'actor': torch.zeros(3, 5)}
    class Observations(dict):
        def to(self, device):
            return self
    env.get_observations = lambda: Observations(obs)
    steps = []
    def step(actions):
        steps.append(1)
        bad = obs['actor'].clone()
        bad[1, 2] = float('nan')
        return Observations(actor=bad), torch.zeros(3), torch.zeros(3, dtype=torch.bool), {}
    env.step = step
    actions = torch.zeros(3, 2)
    if bad_action:
        actions[1, 0] = float('inf')
    runner = object.__new__(LadderOnPolicyRunner)
    runner.env, runner.device, runner.gpu_global_rank = env, 'cpu', 2
    runner.is_distributed = False
    runner.current_learning_iteration = 2628
    runner.cfg = {'num_steps_per_env': 24}
    runner.logger = NS(log_dir=str(tmp_path), writer=None, init_logging_writer=lambda: None)
    runner.alg = NS(train_mode=lambda: None, act=lambda _: actions)
    with pytest.raises(ValueError, match='rank=2, iteration=2629'):
        runner.learn(1)
    assert len(steps) == (0 if bad_action else 1)
    path, = (tmp_path/'nonfinite_diagnostics').glob('*.json')
    meta = json.loads(path.read_text())
    assert meta['when'] == ('before_env_step' if bad_action else 'after_env_step')
