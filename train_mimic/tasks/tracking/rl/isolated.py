"""Prototype a policy without a physics simulator.

``train(model, reward_fn, init_state, transition_fn)`` is the seam between the
learning step and whatever produces the next state. MuJoCo is one possible
transition function. This module does not import it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

RewardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TransitionFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train(
    model: nn.Module,
    reward_fn: RewardFn,
    init_state: torch.Tensor,
    transition_fn: TransitionFn,
    *,
    steps: int,
    horizon: int = 1,
    lr: float = 1e-2,
) -> nn.Module:
    """Update ``model`` from ``init_state`` and return that same module.

    ``model(state)`` must return an action tensor. ``reward_fn`` receives
    ``(state, action, next_state)`` and must be differentiable in the action.
    ``transition_fn(state, action)`` is the simulator. Every update starts
    again from ``init_state``; the returned module is the model to save.
    """

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if init_state.ndim == 1:
        state0 = init_state.detach().unsqueeze(0)
    elif init_state.ndim == 2:
        state0 = init_state.detach()
    else:
        raise ValueError(
            "init_state must have shape (state_dim,) or (batch, state_dim), "
            f"got {tuple(init_state.shape)}"
        )

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    device = parameters[0].device
    state0 = state0.to(device=device, dtype=parameters[0].dtype)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    model.train()
    for _ in range(steps):
        state = state0
        rewards: list[torch.Tensor] = []
        for _ in range(horizon):
            action = model(state)
            if not isinstance(action, torch.Tensor):
                raise TypeError("model must return an action tensor")
            next_state = transition_fn(state, action)
            reward = reward_fn(state, action, next_state)
            if not isinstance(reward, torch.Tensor):
                raise TypeError("reward_fn must return a tensor")
            rewards.append(reward)
            state = next_state
        loss = -torch.stack([reward.reshape(-1).mean() for reward in rewards]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model
