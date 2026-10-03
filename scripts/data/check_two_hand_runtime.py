"""Short physical rollout; reports failures honestly, without claiming trained success."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from train_mimic.app import import_training_stack, build_runner_cfg_dict
from train_mimic.tasks.tracking.config.two_hand import make_two_hand_env_cfg, make_two_hand_runner_cfg
from train_mimic.tasks.tracking.rl.two_hand_runner import TwoHandOnPolicyRunner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint')
    p.add_argument('--allow_reference_mismatch',action='store_true')
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--num_envs',type=int,default=8)
    p.add_argument('--second_start',type=float,default=.5)
    p.add_argument('--no_disturbances',action='store_true')
    p.add_argument('--output',type=Path,default=Path('outputs/two_hand_runtime.json'))
    a=p.parse_args()
    _,Env,Wrapper,*_=import_training_stack()
    cfg=make_two_hand_env_cfg(play=True)
    cfg.scene.num_envs=a.num_envs
    cfg.commands['ladder'].second_hand_reset_probability=a.second_start
    if a.no_disturbances:
        cfg.commands['ladder'].disturbances_enabled=False
    env=Env(cfg=cfg,device='cuda:0')
    try:
        wrapped=Wrapper(env)
        runner=TwoHandOnPolicyRunner(wrapped,build_runner_cfg_dict(make_two_hand_runner_cfg()),log_dir=None,device='cuda:0')
        if a.checkpoint:
            runner.warmstart(a.checkpoint,allow_reference_mismatch=a.allow_reference_mismatch)
        policy=runner.get_inference_policy(device='cuda:0')
        obs,_=wrapped.reset()
        c=env.command_manager.get_term('ladder')
        stages=set(); transitions=successes=failures=0
        reasons={}
        failures_detail=[]
        for step in range(a.steps):
            previous=c.active_hand.clone()
            with torch.inference_mode():
                action=policy(obs) if a.checkpoint else torch.zeros_like(c.reference_joint_pos)
                obs,_,done,_=wrapped.step(action)
            if not torch.isfinite(env.sim.data.qpos).all() or not torch.isfinite(env.sim.data.qvel).all():
                raise RuntimeError('Nonfinite physical state')
            transitions+=int(((previous==0)&(c.active_hand==1)&~done.bool()).sum())
            successes+=int(c.success_pulse.sum())
            for h,s in zip(c.active_hand.cpu().tolist(),c.motion_stage.cpu().tolist()):
                stages.add((h,s))
            for h,r in zip(c.active_hand[c.motion_failed].cpu().tolist(),c.failure_reason[c.motion_failed].cpu().tolist()):
                key=f'hand{h}/reason{r}'
                reasons[key]=reasons.get(key,0)+1
            failures+=int(c.motion_failed.sum())
            if len(failures_detail)<8 and c.motion_failed.any():
                i=int(c.motion_failed.nonzero()[0])
                failures_detail.append(dict(hand=int(c.active_hand[i]),stage=int(c.motion_stage[i]),reason=int(c.failure_reason[i]),
                    release_gates={k:(bool(v[i]) if v.dtype == torch.bool else float(v[i])) for k,v in c._release_stability_conditions().items()},
                    torso_orientation=float(c.torso_orientation_error[i]),support_offset=float(c.torso_support_offset_error[i]),
                    torso_speed=float(c.torso_com_vel_w[i].norm()), reference_time=float(c.reference_time[i])))
            if (step+1)%200 == 0:
                a.output.parent.mkdir(parents=True,exist_ok=True)
                a.output.with_suffix('.progress.json').write_text(json.dumps(dict(step=step+1,stages=sorted(stages),transitions=transitions,failures=failures)))
        result=dict(mode='warmstarted policy' if a.checkpoint else 'zero residual / physical PD execution',
                    steps=a.steps,envs=a.num_envs,stages=sorted(stages),
                    transitions_without_reset=transitions,success_pulses=successes,
                    failure_frames=failures,failure_frames_by_reason=reasons,failure_samples=failures_detail,finite=True)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__=='__main__':
    main()
