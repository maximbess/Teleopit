"""Collect successful, stable first-hand states before automatic episode reset."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from train_mimic.data.terminal_pose_bank import pose_distances, representatives, stable_history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--representatives', type=int, default=8)
    p.add_argument('--num_envs', type=int, default=16)
    p.add_argument('--max_steps', type=int, default=10000)
    p.add_argument('--history_s', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--allow_reference_mismatch', action='store_true')
    args = p.parse_args(argv)
    if min(args.samples, args.representatives, args.num_envs, args.max_steps) <= 0:
        p.error('Counts must be positive')
    if args.representatives > args.samples or not 0 < args.history_s <= 2:
        p.error('representatives must not exceed samples; history_s must be in (0, 2]')
    return args


def snapshot(env, c, actions):
    import torch
    d = c.disturbances
    torso_speed = torch.linalg.vector_norm(c.torso_com_vel_w, dim=-1)
    torso_omega = torch.linalg.vector_norm(c.torso_ang_vel_w, dim=-1)
    pelvis_omega = torch.linalg.vector_norm(c.pelvis_ang_vel_w, dim=-1)
    waist_speed = c.robot.data.joint_vel[:, c._waist_joint_ids].abs().amax(dim=-1)
    joint_speed = c.robot.data.joint_vel.square().mean(dim=-1).sqrt()
    # Independently recheck the measured state; phase_completion_stable becomes
    # false in DONE because its parent implementation excludes finished envs.
    stable = (c.attached.all(dim=1) & c.foot_support.all(dim=1)
              & (c.held_rung[:, 0] == c.cfg.start_rung+1)
              & (c.held_rung[:, 1] == c.cfg.start_rung)
              & (c.torso_orientation_error <= c.cfg.max_phase_torso_orientation_error)
              & (c.torso_support_offset_error <= c.cfg.max_phase_support_offset_error)
              & (torso_speed <= c.cfg.max_phase_completion_torso_speed)
              & (joint_speed <= c.cfg.max_phase_completion_joint_speed)
              & (torso_omega <= c.cfg.max_stabilization_body_angular_speed)
              & (pelvis_omega <= c.cfg.max_stabilization_body_angular_speed)
              & (waist_speed <= c.cfg.max_stabilization_waist_joint_speed)
              & ~c.motion_failed & d.recovery_ready & (d.remaining <= 0))
    values = dict(qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, qacc=env.sim.data.qacc,
                  ctrl=env.sim.data.ctrl, actions=actions, joint_pos=c.robot.data.joint_pos,
                  track_pos=c.actual_position, track_quat=c.actual_quaternion,
                  motion_stage=c.motion_stage, reference_time=c.reference_time,
                  hold_elapsed=c.hold_elapsed, attached=c.attached, held_rung=c.held_rung,
                  foot_rung=c.foot_rung, foot_contact=c.foot_contact, foot_support=c.foot_support,
                  gravity=d.gravity, gravity_scale=d.gravity_scale, clean=d.clean,
                  push_force=d.force, push_count=d.push_count, peak_push_force=d.peak_force,
                  hold_push_completed=d.challenge_done, stable=stable,
                  torso_speed=torso_speed, torso_angular_speed=torso_omega,
                  pelvis_angular_speed=pelvis_omega, waist_speed=waist_speed,
                  joint_speed_rms=joint_speed, torso_orientation_error=c.torso_orientation_error,
                  torso_support_offset_error=c.torso_support_offset_error,
                  elapsed=env.episode_length_buf*env.step_dt)
    return {k: v.detach().cpu().numpy().copy() for k, v in values.items()}


def main(argv=None):
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {out}')
    # Import torch before MuJoCo on Windows.
    import torch
    import mujoco
    from train_mimic.app import import_training_stack, build_runner_cfg_dict
    from train_mimic.tasks.tracking.config.first_hand import make_first_hand_env_cfg, make_first_hand_runner_cfg
    from train_mimic.tasks.tracking.rl.first_hand_runner import FirstHandOnPolicyRunner
    _, Env, Wrapper, *_ = import_training_stack()
    cfg = make_first_hand_env_cfg(play=True)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    env = Env(cfg=cfg, device=args.device)
    try:
        wrapped = Wrapper(env)
        c = env.command_manager.get_term('ladder')
        runner = FirstHandOnPolicyRunner(wrapped, build_runner_cfg_dict(make_first_hand_runner_cfg()),
                                        log_dir=None, device=args.device)
        runner.load(str(checkpoint), map_location=args.device,
                    allow_reference_mismatch=args.allow_reference_mismatch)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = wrapped.reset()
        out.mkdir(parents=True, exist_ok=True)
        (out/'states').mkdir()
        mujoco.mj_saveModel(env.sim.mj_model, str(out/'scene.mjb'))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        history_frames = int(np.ceil(args.history_s/env.step_dt))+1
        manifest = dict(version=1, checkpoint=str(checkpoint),
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        checkpoint_iteration=saved['iter'], seed=args.seed, deterministic_policy=True,
                        checkpoint_reference_signature=saved['infos']['first_hand_reference_signature'],
                        runtime_reference_signature=c.reference_signature,
                        reference_signature_matches=saved['infos']['first_hand_reference_signature']==c.reference_signature,
                        mujoco_version=mujoco.__version__, torch_version=str(torch.__version__),
                        step_dt=env.step_dt, history_s=args.history_s, history_frames=history_frames,
                        track_names=['left_hand','right_hand','left_foot','right_foot','pelvis','torso'],
                        quaternion_order='wxyz', joint_names=list(c.robot.joint_names),
                        physics_joint_names=[env.sim.mj_model.joint(i).name for i in range(env.sim.mj_model.njnt)],
                        hand_site_ids=c._hand_site_ids.cpu().tolist(), foot_site_ids=c._foot_site_ids.cpu().tolist(),
                        pelvis_body_id=c._pelvis_body_id, torso_body_id=c._torso_body_id,
                        command_cfg={k: getattr(c.cfg,k) for k in ('gravity_scale_range','clean_episode_probability',
                        'push_weight_fraction','push_duration_s','push_interval_s','stable_hold_s','minimum_hold_s')},
                        requested_samples=args.samples, records=[], rejected_unstable=0, completed_episodes=0,
                        completed_failures=0)
        del saved
        buffers = [deque(maxlen=history_frames) for _ in range(args.num_envs)]
        episodes = np.zeros(args.num_envs,dtype=np.int64)
        def save_manifest():
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        save_manifest()
        for step in range(args.max_steps):
            with torch.inference_mode():
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise ValueError('Policy produced nonfinite actions')
                obs, _, dones, _ = wrapped.step(actions)
                if any(not torch.isfinite(v).all() for v in obs.values()):
                    raise ValueError('Environment produced nonfinite observations')
                done_ids = dones.nonzero().flatten().cpu().tolist()
                manifest['completed_episodes'] += len(done_ids)
                if done_ids:
                    manifest['completed_failures'] += int((~env.termination_manager.get_term('success')[done_ids]).sum())
                for i in done_ids:
                    buffers[i].clear()
                    episodes[i] += 1
                frame = snapshot(env,c,actions)
                for i in range(args.num_envs):
                    buffers[i].append({**{k:v[i].copy() for k,v in frame.items()},'episode_id':episodes[i]})
                # This pulse is set on HOLD -> DONE, one control step BEFORE the
                # environment's next termination check and automatic reset.
                for i in c.success_pulse.nonzero().flatten().cpu().tolist():
                    if len(manifest['records']) >= args.samples:
                        break
                    if not stable_history(buffers[i],history_frames):
                        manifest['rejected_unstable'] += 1
                        continue
                    index=len(manifest['records']); name=f'states/state_{index:04d}.npz'
                    arrays={k:np.stack([f[k] for f in buffers[i]]) for k in frame}
                    for k in ('eq_data','eq_solref','eq_solimp'):
                        value=getattr(env.sim.model,k)
                        arrays['model_'+k]=value[i if value.shape[0]>1 else 0].detach().cpu().numpy().copy()
                    arrays['eq_active']=env.sim.data.eq_active[i].detach().cpu().numpy().copy()
                    np.savez_compressed(out/name,**arrays)
                    manifest['records'].append(dict(id=index,file=name,env_id=i,episode_id=int(episodes[i]),
                        collection_step=step,clean=bool(frame['clean'][i]),
                        gravity_scale=float(frame['gravity_scale'][i]),push_count=int(frame['push_count'][i]),
                        terminal_torso_error=float(frame['torso_orientation_error'][i]),
                        pelvis_drift_m=float(np.linalg.norm(arrays['track_pos'][:,4]-arrays['track_pos'][-1,4],axis=-1).max())))
                    save_manifest()
            if step % 100 == 0:
                print(f"Step {step}: collected {len(manifest['records'])}/{args.samples}; failures={manifest['completed_failures']}",flush=True)
            if len(manifest['records']) >= args.samples:
                break
        manifest['collection_steps']=step+1
        manifest['complete']=len(manifest['records'])==args.samples
        if manifest['records']:
            finals=[]
            for r in manifest['records']:
                with np.load(out/r['file']) as f:
                    finals.append({k:f[k][-1].copy() for k in ('track_pos','track_quat','joint_pos')})
            distances=pose_distances(*[np.stack([f[k] for f in finals]) for k in ('track_pos','track_quat','joint_pos')])
            chosen=representatives(distances,min(args.representatives,len(finals)))
            manifest['representatives']=chosen
            manifest['nominal_id']=chosen[0]
            manifest['selection']='Real medoid, then farthest-point coverage; no averaged or synthesized pose.'
        save_manifest()
        print(f"Saved {len(manifest['records'])} states to {out}; complete={manifest['complete']}",flush=True)
        return 0 if manifest['complete'] else 2
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
