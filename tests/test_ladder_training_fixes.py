"""Deadline handling without stepping physics."""
from types import SimpleNamespace

import torch
from mjlab.managers.termination_manager import TerminationManager

from train_mimic.tasks.tracking.config.env import make_g1_ladder_rl_env_cfg
from train_mimic.tasks.tracking.mdp.ladder import ladder_remaining_time


def test_deadline_bootstraps_and_the_reward_is_the_body_targets() -> None:
    cfg = make_g1_ladder_rl_env_cfg()
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([250, 125, 0]),
        max_episode_length=250,
    )
    torch.testing.assert_close(
        ladder_remaining_time(env),
        torch.tensor([[0.0], [0.5], [1.0]]),
    )
    manager = TerminationManager({"time_out": cfg.terminations["time_out"]}, env)
    env.termination_manager = manager
    manager.compute()
    assert manager.terminated.tolist() == [False, False, False]
    assert manager.time_outs.tolist() == [True, False, False]
    assert cfg.terminations["time_out"].time_out is True
    assert cfg.episode_length_s == 8.0
    assert set(cfg.rewards) == {
        "target_pos",
        "target_gain",
        "action_rate_l2",
        "joint_limit",
        "feet_acc",
        "self_collisions",
    }
    assert cfg.commands["ladder"].latches_enabled is False
    assert "remaining_time" not in cfg.observations["actor"].terms
    assert "remaining_time" not in cfg.observations["critic_history"].terms