"""Ladder normalization keeps the phase one-hot out of the running statistics."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from train_mimic.tasks.tracking.rl.ladder_model import LadderTemporalCNNModel


def test_phase_switch_stays_one_hot_after_normalization() -> None:
    current = torch.zeros(4, 8)
    current[:, 0] = 1.0
    current[:, 5:] = torch.arange(4, dtype=torch.float32)[:, None]
    history = current[:, None].repeat(1, 10, 1)
    obs = TensorDict(
        {"actor": current, "actor_history": history},
        batch_size=[4],
    )
    model = LadderTemporalCNNModel(
        obs,
        {"actor": ["actor", "actor_history"]},
        "actor",
        2,
        hidden_dims=(8,),
        obs_normalization=True,
        cnn_cfg={
            "output_channels": (4,),
            "kernel_size": 3,
            "global_pool": "avg",
        },
    )
    model.update_normalization(obs)
    model.load_state_dict(model.state_dict())
    model.eval()

    obs["actor"][:, 0] = 0.0
    obs["actor"][:, 1] = 1.0
    obs["actor_history"][:, :, 0] = 0.0
    obs["actor_history"][:, :, 1] = 1.0

    normalized = model.obs_normalizer(obs["actor"])
    torch.testing.assert_close(normalized[:, :5], obs["actor"][:, :5])
    assert not torch.equal(normalized[:, 5:], obs["actor"][:, 5:])

    history_norm = model.obs_normalizers_3d["actor_history"]
    torch.testing.assert_close(
        history_norm(obs["actor_history"])[..., :5],
        obs["actor_history"][..., :5],
    )
    scripted = torch.jit.script(model.obs_normalizer)
    torch.testing.assert_close(scripted(obs["actor"]), normalized)
