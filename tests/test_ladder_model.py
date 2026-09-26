"""Ladder observations are normalized like any other continuous features."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from train_mimic.tasks.tracking.rl.ladder_model import LadderTemporalCNNModel


def test_ladder_normalizer_scales_every_feature() -> None:
    current = torch.zeros(8, 6)
    current[:, 0] = torch.arange(8, dtype=torch.float32)
    current[:, 1:] = 1.0
    history = current[:, None].repeat(1, 4, 1)
    obs = TensorDict(
        {"actor": current, "actor_history": history},
        batch_size=[8],
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
    model.eval()

    probe = torch.zeros(8, 6)
    probe[:, 0] = 100.0
    normalized = model.obs_normalizer(probe)
    assert not torch.allclose(normalized[:, 0], probe[:, 0])

    history_probe = probe[:, None].repeat(1, 4, 1)
    history_norm = model.obs_normalizers_3d["actor_history"](history_probe)
    assert not torch.allclose(history_norm[..., 0], history_probe[..., 0])
