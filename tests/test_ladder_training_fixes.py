"""Deadline and normalization regressions without stepping physics."""
from types import SimpleNamespace

import mujoco
import pytest
import torch
from tensordict import TensorDict
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from train_mimic.tasks.tracking.config.env import make_g1_ladder_rl_env_cfg
from train_mimic.tasks.tracking.mdp.ladder import ladder_remaining_time, ladder_failure_penalty
from train_mimic.tasks.tracking.rl.ladder_model import LadderTemporalCNNModel


@pytest.mark.parametrize("side", ["actor", "critic"])
def test_phase_switch_stays_one_hot_in_current_history_and_export(side):
    current = torch.zeros(4, 8)
    current[:, 0] = 1
    current[:, 5:] = torch.arange(4).float()[:, None]
    obs = TensorDict({side: current, side + "_history": current[:, None].repeat(1, 10, 1)}, [4])
    groups = {side: [side, side + "_history"]}
    model = LadderTemporalCNNModel(obs, groups, side, 2, hidden_dims=(8,),
                                  obs_normalization=True,
                                  cnn_cfg={"output_channels": (4,), "kernel_size": 3, "global_pool": "avg"})
    model.update_normalization(obs)
    # Reproduce the all-STABILIZE checkpoint, including a state-dict round trip.
    model.load_state_dict(model.state_dict())
    model.eval()
    obs[side][:, 0] = 0
    obs[side][:, 1] = 1
    obs[side + "_history"][:, :, 0] = 0
    obs[side + "_history"][:, :, 1] = 1
    normalized = model.obs_normalizer(obs[side])
    torch.testing.assert_close(normalized[:, :5], obs[side][:, :5])
    assert not torch.equal(normalized[:, 5:], obs[side][:, 5:])
    history_norm = model.obs_normalizers_3d[side + "_history"]
    torch.testing.assert_close(history_norm(obs[side + "_history"])[..., :5], obs[side + "_history"][..., :5])
    exported = model.as_onnx()
    torch.testing.assert_close(exported(obs[side], obs[side + "_history"]), model(obs))
    scripted = torch.jit.script(model.obs_normalizer)
    torch.testing.assert_close(scripted(obs[side]), normalized)


def test_deadline_is_terminal_but_completed_prefix_is_truncation():
    cfg = make_g1_ladder_rl_env_cfg()
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=0.02,
                          episode_length_buf=torch.tensor([1000, 500, 0]), max_episode_length=1000)
    torch.testing.assert_close(ladder_remaining_time(env), torch.tensor([[0.], [.5], [1.]]))
    # Real termination-manager masks are the masks passed to the PPO wrapper.
    manager = TerminationManager({
        "time_out": cfg.terminations["time_out"],
        "curriculum_stage_complete": TerminationTermCfg(
            func=lambda env: torch.tensor([False, True, False]), time_out=True),
        "success": TerminationTermCfg(func=lambda env: torch.zeros(3, dtype=torch.bool)),
    }, env)
    env.termination_manager = manager
    manager.compute()
    assert manager.terminated.tolist() == [True, False, False]
    assert manager.time_outs.tolist() == [False, True, False]
    torch.testing.assert_close(-50 * env.step_dt * ladder_failure_penalty(env, "ladder"),
                               torch.tensor([-50., 0., 0.]))
    assert cfg.terminations["curriculum_stage_complete"].time_out
    assert "remaining_time" not in cfg.observations["actor"].terms
    assert "remaining_time" not in cfg.observations["critic_history"].terms
