"""Keep the six motion stages and five ladder phases categorical during PPO."""
import torch
from rsl_rl.modules import EmpiricalNormalization
from .temporal_cnn_model import TemporalCNNModel


class FirstHandNormalization(EmpiricalNormalization):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self._mean) / (self._std + self.eps)
        return torch.cat((x[..., :11], normalized[..., 11:]), dim=-1)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.cat((y[..., :11], (y * (self._std + self.eps) + self._mean)[..., 11:]), dim=-1)


class FirstHandTemporalCNNModel(TemporalCNNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer = FirstHandNormalization(self.obs_dim)
        for name in self.obs_groups_3d:
            if name in ('actor_history', 'critic_history'):
                old = self.obs_normalizers_3d[name]
                if isinstance(old, EmpiricalNormalization):
                    self.obs_normalizers_3d[name] = FirstHandNormalization(old.mean.numel())
