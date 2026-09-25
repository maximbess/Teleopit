"""Ladder policy normalization keeps the phase one-hot unchanged."""

from __future__ import annotations

import torch
from rsl_rl.modules import EmpiricalNormalization

from .temporal_cnn_model import TemporalCNNModel

class PhasePreservingNormalization(EmpiricalNormalization):
    """Normalize continuous features and copy the leading phase one-hot through.

    Running statistics are still updated for every dimension. The forward pass
    ignores those statistics for the first five values, which are the ladder
    phase indicator in both the current frame and its history. The slice is a
    literal so TorchScript can compile ``forward``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self._mean) / (self._std + self.eps)
        return torch.cat((x[..., :5], normalized[..., 5:]), dim=-1)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        restored = y * (self._std + self.eps) + self._mean
        return torch.cat((y[..., :5], restored[..., 5:]), dim=-1)


class LadderTemporalCNNModel(TemporalCNNModel):
    """TemporalCNN whose actor and critic inputs keep the phase one-hot raw.

    The tracking policy is unchanged. Ladder observations concatenate the 24D
    command first, so the phase indicator occupies the leading five features of
    the current frame and of ``actor_history`` / ``critic_history``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.obs_groups_1d or self.obs_groups_1d[0] not in ("actor", "critic"):
            raise ValueError(
                "Ladder observations must start with the actor or critic group, "
                "whose first five features are the phase one-hot"
            )
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer = PhasePreservingNormalization(self.obs_dim)
        for name in self.obs_groups_3d:
            if name not in ("actor_history", "critic_history"):
                continue
            old = self.obs_normalizers_3d[name]
            if isinstance(old, EmpiricalNormalization):
                self.obs_normalizers_3d[name] = PhasePreservingNormalization(
                    old.mean.numel()
                )
