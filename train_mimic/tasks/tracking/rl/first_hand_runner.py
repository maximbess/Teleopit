"""Checkpoint compatibility for the authored reference and residual control task."""
from .runner import LadderOnPolicyRunner


class FirstHandOnPolicyRunner(LadderOnPolicyRunner):
    def save(self, path, infos=None):
        command = self._ladder_command()
        super().save(path, infos={**(infos or {}),
                                  "first_hand_reference_signature": command.reference_signature})

    def load(self, path, load_cfg=None, strict=True, map_location=None, *, allow_reference_mismatch=False):
        # Validate before mutating the optimizer/policy with an incompatible checkpoint.
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        saved_signature = infos.get("first_hand_reference_signature")
        current_signature = self._ladder_command().reference_signature
        if saved_signature != current_signature:
            if not allow_reference_mismatch:
                raise ValueError(
                    "Checkpoint belongs to a different first-hand reference or IK runtime. "
                    "Use the training reference, robot XML and runtime. For video playback "
                    "with the current reference, explicitly pass --allow_reference_mismatch "
                    "to record_ladder_video.py."
                )
            import warnings
            warnings.warn(
                "Reference mismatch explicitly allowed for playback: using the CURRENT "
                "reference, which is not verified to match training. "
                f"Checkpoint signature: {saved_signature}; current: {current_signature}",
                RuntimeWarning, stacklevel=2,
            )
        return super().load(path, load_cfg, strict, map_location)
