"""Checkpoint compatibility for the authored reference and residual control task."""
from .runner import LadderOnPolicyRunner


class FirstHandOnPolicyRunner(LadderOnPolicyRunner):
    def save(self, path, infos=None):
        command = self._ladder_command()
        super().save(path, infos={**(infos or {}),
                                  "first_hand_reference_signature": command.reference_signature})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # Validate before mutating the optimizer/policy with an incompatible checkpoint.
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        if infos.get("first_hand_reference_signature") != self._ladder_command().reference_signature:
            raise ValueError("Checkpoint belongs to a different first-hand reference. Start a fresh run.")
        return super().load(path, load_cfg, strict, map_location)
