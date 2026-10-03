"""Contact-gated left/right sequence with a portable bank of measured starts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_mul, quat_conjugate, quat_apply

from teleopit.runtime.assets import PROJECT_ROOT
from train_mimic.data.two_hand_bank import validate_model
from .first_hand import FirstHandCommand, FirstHandCommandCfg, FirstHandStage
from .ladder import LadderPhase


@dataclass(kw_only=True)
class TwoHandCommandCfg(FirstHandCommandCfg):
    bank_file: str = str(PROJECT_ROOT/'assets/motions/ladder_two_hand_bank.npz')
    second_hand_reset_probability: float = .5

    def build(self, env):
        return TwoHandCommand(self, env)


class TwoHandCommand(FirstHandCommand):
    def __init__(self, cfg, env):
        if not 0 <= cfg.second_hand_reset_probability <= 1:
            raise ValueError('second_hand_reset_probability must be in [0,1]')
        super().__init__(cfg, env)
        path = Path(cfg.bank_file)
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f['metadata']))
            order = [meta['joint_names'].index(n) for n in self.robot.joint_names]
            physics_names = [env.sim.mj_model.joint(j).name for j in range(env.sim.mj_model.njnt)]
            if physics_names != meta['physics_joint_names'] or meta['fps'] != self.ref_fps:
                raise ValueError('Pose-bank model topology or sampling frequency mismatch')
            validate_model(env.sim.mj_model, f, meta)
            self.bank = {k:torch.tensor(f[k],device=self.device) for k in
                         ('qpos','qvel','position','quaternion','joint_pos','joint_vel')}
            self.bank['joint_pos'] = self.bank['joint_pos'][:,:,order]
            self.bank['joint_vel'] = self.bank['joint_vel'][:,:,order]
        if any(not torch.isfinite(v).all() for v in self.bank.values()):
            raise ValueError('Nonfinite pose bank')
        self.bank_signature = hashlib.sha256(path.read_bytes()).hexdigest()
        self.first_reference_signature = self.reference_signature
        self.reference_signature = hashlib.sha256((self.reference_signature+self.bank_signature+'two-hand-v2').encode()).hexdigest()
        self._torso_local_com = torch.tensor(env.sim.mj_model.body_ipos[self._torso_body_id].copy(),device=self.device,dtype=torch.float32)
        schedule = meta['contact_schedule']['right_hand']
        self.second_prepare_end = float(schedule['release_start_s'])
        self.second_transfer_start = float(schedule['detached_by_s'])
        self.second_end = float(schedule['attach_not_before_s'])
        self.bank_index = torch.zeros_like(self.motion_stage)
        self._bank_reset = torch.zeros_like(self.finished)
        self.first_hand_completed = torch.zeros_like(self.finished)
        self.started_at_second = torch.zeros_like(self.finished)
        self._joint_blend_offset = torch.zeros_like(self.reference_joint_pos)
        self._position_offset = torch.zeros_like(self.reference_position)
        self._rotation_offset = torch.zeros_like(self.reference_quaternion)
        self._rotation_offset[...,0] = 1
        for name in ('first_transfer_complete','two_hand_success','second_hand_success','started_at_second','active_transfer'):
            self.metrics[name] = torch.zeros_like(self.reference_time)
        for hand in ('left','right'):
            self.metrics[f'failure/{hand}/stationary_hand_lost'] = torch.zeros_like(self.reference_time)
            for stage in FirstHandStage:
                self.metrics[f'failure/{hand}/{stage.name.lower()}'] = torch.zeros_like(self.reference_time)

    def _resample_command(self, ids):
        super()._resample_command(ids)
        self.first_hand_completed[ids] = False
        self._joint_blend_offset[ids] = 0
        self._position_offset[ids] = 0
        self._rotation_offset[ids] = 0
        self._rotation_offset[ids,:,0] = 1
        self._bank_reset[ids] = torch.rand(len(ids), device=self.device) < self.cfg.second_hand_reset_probability
        self.started_at_second[ids] = self._bank_reset[ids]
        selected = ids[self._bank_reset[ids]]
        indices = torch.randint(len(self.bank['qpos']), (len(selected),),device=self.device)
        self.bank_index[selected] = indices
        # This runs during reset, before the environment's forward() and observations.
        self._env.sim.data.qpos[selected] = self.bank['qpos'][indices]
        self._env.sim.data.qvel[selected] = self.bank['qvel'][indices]
        self.robot.clear_state(env_ids=selected)

    def _initialize_from_start_pose(self):
        ids = torch.where(self._pending_start_pose_init & self._bank_reset)[0]
        super()._initialize_from_start_pose()
        if len(ids):
            self._attach(ids,0,torch.full_like(ids,self.cfg.start_rung+1))
            self.reference_joint_pos[ids] = self.robot.data.joint_pos[ids]
            self._start_second(ids, self.bank_index[ids])
            self._bank_reset[ids] = False

    def _start_second(self, ids, indices=None):
        if not len(ids):
            return
        if indices is None:
            distance = (self.robot.data.joint_pos[ids,None,:]-self.bank['joint_pos'][None,:,0,:]).square().mean(-1)
            indices = distance.argmin(-1)
        self.bank_index[ids] = indices
        self._joint_blend_offset[ids] = self.reference_joint_pos[ids]-self.bank['joint_pos'][indices,0]
        self._position_offset[ids] = self.actual_position[ids]-self.bank['position'][indices,0]
        self._rotation_offset[ids] = quat_mul(self.actual_quaternion[ids],quat_conjugate(self.bank['quaternion'][indices,0]))
        self.active_hand[ids] = 1
        self.phase[ids] = int(LadderPhase.SECOND_HAND)
        self.target_rung[ids] = self.cfg.start_rung+1
        self.reference_time[ids] = 0
        self.hold_elapsed[ids] = 0
        self.support_loss_elapsed[ids] = 0
        self.best_reach[ids] = 0
        self._hold_hand_offset[ids] = 0
        self._reset_release_state(ids)
        self._set_stage(ids,FirstHandStage.PREPARE)
        if self.disturbances is not None:
            # Each transfer gets its own recovery challenge; retain episode gravity.
            for buf in (self.disturbances.challenge_started,self.disturbances.challenge_done,self.disturbances.challenge_active):
                buf[ids] = False

    def _complete_motion(self, ids):
        first = ids[self.active_hand[ids] == 0]
        second = ids[self.active_hand[ids] == 1]
        self.first_hand_completed[first] = True
        self._start_second(first)
        super()._complete_motion(second)

    @property
    def torso_support_offset_error(self):
        original = super().torso_support_offset_error
        if not hasattr(self,'_torso_local_com'):
            return original
        # The second transfer has a different support geometry. Compare with
        # the authored torso COM and the same currently load-bearing supports,
        # rather than the four-contact geometry at construction of the scene.
        weights = torch.cat((self.grip_strength*self.attached.float(),self.foot_support.float()),dim=1)
        centroid = (self.reference_position[:,:4]*weights[:,:,None]).sum(1)/weights.sum(1).clamp_min(1)[:,None]
        com = self.reference_position[:,5]+quat_apply(self.reference_quaternion[:,5],self._torso_local_com.expand(self.num_envs,-1))
        error = torch.linalg.vector_norm(self.torso_support_offset_w-(com-centroid),dim=-1)
        return torch.where(self.active_hand == 1,error,original)

    @property
    def reference_prepare_end(self):
        return torch.where(self.active_hand == 1,self.second_prepare_end,1.7)

    @property
    def reference_transfer_start(self):
        return torch.where(self.active_hand == 1,self.second_transfer_start,1.7)

    @property
    def reference_end_time(self):
        return torch.where(self.active_hand == 1,self.second_end,3.9)

    @property
    def endpoint_position(self):
        return torch.where((self.active_hand == 1)[:,None],
                           self.bank['position'][self.bank_index,-1,1]+self._position_offset[:,1],
                           super().endpoint_position)

    @property
    def initial_hand_position(self):
        return torch.where((self.active_hand == 1)[:,None],
                           self.bank['position'][self.bank_index,0,1]+self._position_offset[:,1],
                           super().initial_hand_position)

    def _bank_sample(self, name, times):
        table = self.bank[name]
        index = (times*self.ref_fps).clamp(0,table.shape[1]-1)
        lo = index.long()
        hi = (lo+1).clamp(max=table.shape[1]-1)
        alpha = (index-lo).reshape((-1,)+(1,)*(table.ndim-2))
        return torch.lerp(table[self.bank_index,lo],table[self.bank_index,hi],alpha)

    def future_joint_reference(self, times):
        u = (times/self.second_prepare_end).clamp(0,1)
        blend = 1-u**3*(10-15*u+6*u*u)
        second = self._bank_sample('joint_pos',times)+blend[:,None]*self._joint_blend_offset
        return torch.where((self.active_hand == 1)[:,None],second,super().future_joint_reference(times))

    def _sample_reference(self, rate=None):
        super()._sample_reference(rate)
        ids = torch.where(self.active_hand == 1)[0]
        self.reference_joint_pos[ids] = self.future_joint_reference(self.reference_time)[ids]
        u = (self.reference_time/self.second_prepare_end).clamp(0,1)
        derivative = -30*u*u*(1-u)**2/self.second_prepare_end
        speed = self._bank_sample('joint_vel',self.reference_time)+derivative[:,None]*self._joint_blend_offset
        self.reference_joint_vel[ids] = speed[ids]*(0. if rate is None else rate[ids,None])
        self.reference_position[ids] = (self._bank_sample('position',self.reference_time)+self._position_offset)[ids]
        self.reference_position[ids,1] += self._hold_hand_offset[ids]
        q = torch.nn.functional.normalize(self._bank_sample('quaternion',self.reference_time),dim=-1)
        self.reference_quaternion[ids] = quat_mul(self._rotation_offset[ids],q[ids])

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics['first_hand_success'][:] = self.first_hand_completed
        self.metrics['first_transfer_complete'][:] = self.first_hand_completed
        self.metrics['two_hand_success'][:] = self.finished & ~self.started_at_second
        self.metrics['second_hand_success'][:] = self.finished & self.started_at_second
        self.metrics['started_at_second'][:] = self.started_at_second
        self.metrics['active_transfer'][:] = self.active_hand
        for h,hand in enumerate(('left','right')):
            self.metrics[f'failure/{hand}/stationary_hand_lost'][:] = self.motion_failed & (self.active_hand == h) & (self.failure_reason == 1)
            for stage in FirstHandStage:
                self.metrics[f'failure/{hand}/{stage.name.lower()}'][:] = self.motion_failed & (self.active_hand == h) & (self.failure_stage == int(stage))


def active_position_cost(env, command_name='ladder', supports=False, std=.04):
    c = env.command_manager.get_term(command_name)
    error = (c.actual_position-c.reference_position).square().sum(-1)/std**2
    hand = error[c._all_env_ids,1-c.active_hand if supports else c.active_hand]
    return torch.cat((hand[:,None],error[:,2:4]),dim=1).clamp(max=9).mean(-1) if supports else hand.clamp(max=9)
