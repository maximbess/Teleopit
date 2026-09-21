"""Ladder policy normalization keeps categorical phase inputs unchanged."""
import torch
from rsl_rl.modules import EmpiricalNormalization

from .temporal_cnn_model import TemporalCNNModel


class PhasePreservingNormalization(EmpiricalNormalization):
    """Continuous features use running statistics; the first five are one-hot."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self._mean) / (self._std + self.eps)
        return torch.cat((x[..., :5], normalized[..., 5:]), dim=-1)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        restored = y * (self._std + self.eps) + self._mean
        return torch.cat((y[..., :5], restored[..., 5:]), dim=-1)


class LadderTemporalCNNModel(TemporalCNNModel):
    """Preserve phase indicators in current and historical actor/critic inputs.

    Export wrappers copy these normalizers. Tracking models are unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.obs_groups_1d[0] not in ("actor", "critic"):
            raise ValueError("Ladder observations must start with the phase command")
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer = PhasePreservingNormalization(self.obs_dim)
        for name in self.obs_groups_3d:
            if name in ("actor_history", "critic_history"):
                old = self.obs_normalizers_3d[name]
                if isinstance(old, EmpiricalNormalization):
                    self.obs_normalizers_3d[name] = PhasePreservingNormalization(old.mean.numel())
