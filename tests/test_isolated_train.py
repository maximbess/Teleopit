"""The prototype trainer must run without MuJoCo and improve a known reward."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from train_mimic.tasks.tracking.config.ladder_init import (
    LADDER_INITIAL_JOINT_POS,
    LADDER_INITIAL_ROOT_POS,
)
from train_mimic.tasks.tracking.rl.isolated import train


def test_isolated_module_does_not_import_a_simulator() -> None:
    source = Path("train_mimic/tasks/tracking/rl/isolated.py").read_text()
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "mujoco" not in imported
    assert "mjlab" not in imported


def test_train_moves_the_action_toward_the_reward_and_can_be_reloaded() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    init_state = torch.tensor([1.0, -0.5, 0.25])
    target = torch.tensor([0.4, -0.2])

    def transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return state

    def reward(
        state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        del state, next_state
        return -torch.sum((action - target) ** 2, dim=-1)

    before = torch.sum((model(init_state) - target) ** 2).item()
    trained = train(
        model,
        reward,
        init_state,
        transition,
        steps=40,
        lr=0.2,
    )
    after = torch.sum((trained(init_state) - target) ** 2).item()

    assert trained is model
    assert after < before
    reloaded = torch.nn.Linear(3, 2)
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded(init_state), model(init_state))


def test_train_accepts_a_batch_initial_state() -> None:
    torch.manual_seed(1)
    model = torch.nn.Linear(2, 1)
    init_state = torch.ones(4, 2)

    def transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + action

    def reward(
        state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        del state, action
        return next_state.sum(dim=-1)

    trained = train(model, reward, init_state, transition, steps=5, horizon=2, lr=0.05)
    assert trained is model


def test_ladder_init_pose_is_simulator_free_data() -> None:
    assert LADDER_INITIAL_ROOT_POS[2] > 1.0
    assert LADDER_INITIAL_JOINT_POS[".*_knee_joint"] > 0.0
