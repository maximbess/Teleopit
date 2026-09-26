"""Deadline handling without stepping physics."""
from types import SimpleNamespace

import torch
from mjlab.managers.termination_manager import TerminationManager

from train_mimic.tasks.tracking.config.env import make_g1_ladder_rl_env_cfg
from train_mimic.tasks.tracking.mdp.ladder import ladder_remaining_time


def test_deadline_is_terminal_and_has_no_failure_reward() -> None:
    cfg = make_g1_ladder_rl_env_cfg()
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([1000, 500, 0]),
        max_episode_length=1000,
    )
    torch.testing.assert_close(
        ladder_remaining_time(env),
        torch.tensor([[0.0], [0.5], [1.0]]),
    )
    manager = TerminationManager(
        {
            "time_out": cfg.terminations["time_out"],
            "success": cfg.terminations["success"],
        },
        env,
    )
    env.termination_manager = manager
    env.command_manager = SimpleNamespace(
        get_term=lambda _name: SimpleNamespace(finished=torch.zeros(3, dtype=torch.bool))
    )
    manager.compute()
    assert manager.terminated.tolist() == [True, False, False]
    assert manager.time_outs.tolist() == [False, False, False]
    assert cfg.terminations["time_out"].time_out is False
    assert set(cfg.rewards) == {
        "ladder_climb",
        "ladder_hold",
        "ladder_hold_completed",
        "ladder_flight",
    }
    assert "remaining_time" not in cfg.observations["actor"].terms
    assert "remaining_time" not in cfg.observations["critic_history"].terms
