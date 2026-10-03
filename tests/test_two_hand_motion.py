"""Regression checks for mixed left/right phase gates and checkpoint transfer."""
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import torch
import pytest
from test_first_hand_motion import GateCommand, gate_command, tick, Stage
from train_mimic.tasks.tracking.mdp.two_hand import TwoHandCommand
from train_mimic.tasks.tracking.config.two_hand import make_two_hand_env_cfg
from train_mimic.scripts.record_ladder_video import _resolve_recording_task
from train_mimic.tasks.tracking.rl.two_hand_runner import TwoHandOnPolicyRunner


class SequenceGate(GateCommand, TwoHandCommand):
    @property
    def actual_position(self):
        return self.reference_position.clone()

    @property
    def actual_quaternion(self):
        return self.reference_quaternion.clone()

    def _reset_release_state(self, ids):
        self._release_active[ids] = False


def sequence_command():
    c = gate_command(Stage.HOLD)
    c.__class__ = SequenceGate
    c.first_hand_completed = torch.zeros(2,dtype=torch.bool)
    c.bank_index = torch.zeros(2,dtype=torch.long)
    c.phase = torch.ones(2,dtype=torch.long)
    c.reference_joint_pos = torch.zeros(2,3)
    c.reference_position = torch.zeros(2,6,3)
    c.reference_quaternion = torch.zeros(2,6,4)
    c.reference_quaternion[...,0] = 1
    c._joint_blend_offset = torch.zeros(2,3)
    c._position_offset = torch.zeros(2,6,3)
    c._rotation_offset = torch.zeros(2,6,4)
    c.bank = {'joint_pos':torch.ones(2,2,3)*.1,
              'position':torch.zeros(2,2,6,3),
              'quaternion':c.reference_quaternion[:,None].expand(-1,2,-1,-1).clone()}
    c.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2,3)))
    c.second_prepare_end,c.second_transfer_start,c.second_end = 1.7,2.3,7.7
    c.support_loss_elapsed = torch.zeros(2)
    c.best_reach = torch.zeros(2)
    c.attached[:] = True
    c.held_rung[:,0] += 1
    return c


def test_first_completion_preserves_supports_and_starts_second_without_success():
    c=sequence_command()
    supports=c.attached.clone()
    joints=c.robot.data.joint_pos.clone()
    c._complete_motion(torch.tensor([0]))
    assert c.active_hand.tolist() == [1,0]
    assert not c.finished.any() and not c.success_pulse.any()
    assert c.first_hand_completed.tolist() == [True,False]
    torch.testing.assert_close(c.attached,supports)
    torch.testing.assert_close(c.robot.data.joint_pos,joints)
    torch.testing.assert_close(c.bank['joint_pos'][c.bank_index[0],0]+c._joint_blend_offset[0],c.reference_joint_pos[0])
    assert c.motion_stage[0] == Stage.PREPARE


def test_handoff_rearms_recovery_without_resampling_gravity():
    c=sequence_command()
    c.disturbances=SimpleNamespace(challenge_started=torch.ones(2,dtype=torch.bool),
        challenge_done=torch.ones(2,dtype=torch.bool),challenge_active=torch.zeros(2,dtype=torch.bool),
        gravity_scale=torch.tensor([.97,1.03]))
    c._complete_motion(torch.tensor([0]))
    assert c.disturbances.challenge_done.tolist() == [False,True]
    assert c.disturbances.challenge_started.tolist() == [False,True]
    torch.testing.assert_close(c.disturbances.gravity_scale,torch.tensor([.97,1.03]))


def test_right_transfer_requires_left_support_and_attaches_only_right():
    c=sequence_command()
    c._start_second(torch.arange(2))
    c.motion_stage[:] = int(Stage.TRANSFER)
    c.reference_time[:] = c.second_end
    c.attached[:,1] = False
    c.attached[0,0] = False
    for _ in range(c.cfg.hand_target_dwell_steps+1): tick(c)
    assert not c.attached[0,1]
    assert c.attached[1].all()
    assert c.motion_stage[1] == Stage.HOLD
    assert not c.finished.any()


def test_second_success_requires_continuous_hold_and_does_not_restart_first():
    c=sequence_command()
    c._start_second(torch.arange(2))
    c.motion_stage[:] = int(Stage.HOLD)
    c.held_rung[:,1] += 1
    c.stage_elapsed.zero_()
    c.cfg.stable_hold_s=2.
    c.cfg.minimum_hold_s=3.
    for _ in range(120): tick(c)
    assert not c.finished.any()
    c.feet[0,0]=False
    tick(c)
    c.feet[:]=True
    for _ in range(35): tick(c)
    assert c.finished.tolist() == [False,True]
    for _ in range(70): tick(c)
    assert c.finished.all() and c.active_hand.eq(1).all()


def test_portable_bank_and_play_configuration():
    cfg=make_two_hand_env_cfg(play=True)
    assert cfg.commands['ladder'].second_hand_reset_probability == 0
    assert cfg.episode_length_s == 40
    with np.load(cfg.commands['ladder'].bank_file,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']))
        assert len(meta['pose_ids']) == 8
        assert f['joint_pos'].shape == (8,436,29)
        assert np.isfinite(f['joint_pos']).all()
        np.testing.assert_allclose(f['joint_pos'][:,0],f['qpos'][:,7:],atol=1e-6)
    assert _resolve_recording_task('auto',{'infos':{'two_hand_reference_signature':'x','first_hand_reference_signature':'y'}}) == 'G1-Ladder-TwoHand-Motion'


def test_warmstart_loads_weights_without_optimizer_or_iteration_and_rejects_mismatch(tmp_path):
    runner=object.__new__(TwoHandOnPolicyRunner)
    runner.device='cpu'
    actor,critic=torch.nn.Linear(2,1),torch.nn.Linear(2,1)
    calls=[]
    runner.alg=SimpleNamespace(_raw_actor=actor,_raw_critic=critic,
        load=lambda saved,cfg,strict:calls.append(cfg))
    runner._ladder_command=lambda:SimpleNamespace(first_reference_signature='first',reference_signature='both')
    runner._project_actor_std=lambda:None
    path=tmp_path/'checkpoint.pt'
    saved={'actor_state_dict':actor.state_dict(),'critic_state_dict':critic.state_dict(),
           'infos':{'first_hand_reference_signature':'other'},'iter':10000}
    torch.save(saved,path)
    with pytest.raises(ValueError,match='differs'):
        runner.warmstart(path)
    assert not calls
    with pytest.warns(RuntimeWarning):
        runner.warmstart(path,allow_reference_mismatch=True)
    assert calls == [{'actor':True,'critic':True,'optimizer':False,'iteration':False}]
    with pytest.raises(ValueError,match='resume'):
        runner.load(path)


def test_second_support_gate_tracks_authored_geometry_not_original_centroid(monkeypatch):
    c=sequence_command()
    c.active_hand[:]=1
    c._torso_local_com=torch.zeros(3)
    c.grip_strength=torch.ones(2,2)
    c._reference_torso_support_offset_w=torch.full((2,3),10.)
    monkeypatch.setattr(SequenceGate,'torso_support_offset_w',property(lambda self:torch.zeros(2,3)))
    # A large difference from the original climbing geometry is harmless when
    # the body matches the new authored support geometry.
    torch.testing.assert_close(c.torso_support_offset_error,torch.zeros(2))
    c.reference_position[:,5,0]=.2
    torch.testing.assert_close(c.torso_support_offset_error,torch.full((2,),.2))
    c.active_hand[0]=0
    assert c.torso_support_offset_error[0] > 1
