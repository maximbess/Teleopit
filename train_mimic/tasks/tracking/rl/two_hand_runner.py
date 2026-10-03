"""Strict sequence resume and explicit weights-only transfer from the first hand."""
import warnings
import torch
from .first_hand_runner import FirstHandOnPolicyRunner


class TwoHandOnPolicyRunner(FirstHandOnPolicyRunner):
    def save(self, path, infos=None):
        c = self._ladder_command()
        super().save(path,infos={**(infos or {}), 'two_hand_reference_signature':c.reference_signature,
                              'two_hand_bank_signature':c.bank_signature})

    def load(self, path, load_cfg=None, strict=True, map_location=None, **kwargs):
        infos = torch.load(path,map_location='cpu',weights_only=False).get('infos') or {}
        if infos.get('two_hand_reference_signature') != self._ladder_command().reference_signature:
            raise ValueError('Two-hand resume requires the same sequence and bank. Use --warmstart for first-hand weights.')
        return super().load(path,load_cfg,strict,map_location,**kwargs)

    def warmstart(self, path, *, allow_reference_mismatch=False):
        saved = torch.load(path,map_location=self.device,weights_only=False)
        infos = saved.get('infos') or {}
        if 'first_hand_reference_signature' not in infos or 'two_hand_reference_signature' in infos:
            raise ValueError('Warmstart expects a first-hand motion checkpoint')
        if infos['first_hand_reference_signature'] != self._ladder_command().first_reference_signature:
            if not allow_reference_mismatch:
                raise ValueError('First-hand reference differs. Verify provenance and explicitly use --allow_warmstart_reference_mismatch if intended.')
            warnings.warn('Warmstart uses weights trained with a different first-hand reference; validate the resulting policy.',RuntimeWarning)
        # Preflight both models before mutating either. Do not load old PPO,
        # iteration, environment counters or curriculum state into the new task.
        for model,key in ((self.alg._raw_actor,'actor_state_dict'),(self.alg._raw_critic,'critic_state_dict')):
            current = model.state_dict()
            if current.keys() != saved[key].keys() or any(current[k].shape != saved[key][k].shape for k in current):
                raise ValueError(f'Incompatible warmstart model: {key}')
        self.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
        self._project_actor_std()
