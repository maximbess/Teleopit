"""Exercise the first-hand task in physics and optionally run a tiny PPO update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from train_mimic.app import import_training_stack, build_runner_cfg_dict
from train_mimic.tasks.tracking.config.first_hand import make_first_hand_env_cfg, make_first_hand_runner_cfg
from train_mimic.tasks.tracking.rl.first_hand_runner import FirstHandOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--ppo-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/first_hand_physics")
    args = parser.parse_args()
    _, Env, Wrapper, *_ = import_training_stack()
    cfg = make_first_hand_env_cfg(play=True)
    cfg.scene.num_envs = args.num_envs
    args.output.mkdir(parents=True, exist_ok=True)
    env = Env(cfg=cfg, device=args.device)
    samples, episodes = [], []
    try:
        obs, _ = env.reset()
        command = env.command_manager.get_term("ladder")
        for step in range(args.steps):
            actions = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
            before_stage = command.motion_stage.detach().cpu().tolist()
            before_time = command.reference_time.detach().cpu().tolist()
            obs, reward, terminated, truncated, extras = env.step(actions)
            assert torch.isfinite(reward).all()
            for value in obs.values():
                assert torch.isfinite(value).all()
            if step % 10 == 0:
                sample = {"step": step, "stage": before_stage, "reference_time": before_time,
                          "supports": command.required_supports.detach().cpu().tolist(),
                          "attached": command.attached.detach().cpu().tolist(),
                          "reward": reward.detach().cpu().tolist()}
                sample["release_conditions"] = {name: value.detach().cpu().tolist()
                                                for name, value in command._release_stability_conditions().items()}
                sample["support_offset"] = command.torso_support_offset_error.detach().cpu().tolist()
                sample["torso_orientation_error"] = command.torso_orientation_error.detach().cpu().tolist()
                samples.append(sample)
                if step % 50 == 0:
                    print(json.dumps(sample), flush=True)
            if (terminated | truncated).any():
                terminal_log = {key: float(value) for key, value in extras["log"].items()
                                if key.startswith("Metrics/ladder/")}
                failed_fraction = terminal_log["Metrics/ladder/motion_failed"]
                reason_sum = sum(value for key, value in terminal_log.items()
                                 if key.startswith("Metrics/ladder/failure/"))
                assert abs(reason_sum - failed_fraction) < 1e-6
                done = terminated | truncated
                expected_failed = env.termination_manager.get_term("motion_failed")[done].float().mean().item()
                expected_success = env.termination_manager.get_term("success")[done].float().mean().item()
                assert abs(failed_fraction - expected_failed) < 1e-6
                assert abs(terminal_log["Metrics/ladder/first_hand_success"] - expected_success) < 1e-6
                episodes.append({"step": step, "stage": before_stage, "reference_time": before_time,
                                 "terminal_metrics": terminal_log,
                                 "success": env.termination_manager.get_term("success").cpu().tolist(),
                                 "terminations": {name: env.termination_manager.get_term(name).cpu().tolist()
                                                  for name in env.termination_manager.active_terms}})
        result = {"zero_residual_physics": True, "steps": args.steps, "samples": samples, "episodes": episodes}
        (args.output / "rollout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("EPISODES " + json.dumps(episodes), flush=True)
        if args.ppo_iterations:
            wrapped = Wrapper(env)
            runner_cfg = make_first_hand_runner_cfg()
            runner_cfg.actor.hidden_dims = (64, 32)
            runner_cfg.critic.hidden_dims = (64, 32)
            runner_cfg.algorithm.num_mini_batches = 1
            runner_cfg.algorithm.num_learning_epochs = 1
            runner_cfg.num_steps_per_env = 8
            runner = FirstHandOnPolicyRunner(wrapped, build_runner_cfg_dict(runner_cfg),
                                            log_dir=str(args.output / "ppo_smoke"), device=args.device)
            runner.learn(args.ppo_iterations, init_at_random_ep_len=False)
            checkpoint = args.output / "ppo_smoke.pt"
            runner.save(str(checkpoint))
            restored = FirstHandOnPolicyRunner(wrapped, build_runner_cfg_dict(runner_cfg),
                                              log_dir=None, device=args.device)
            restored.load(str(checkpoint), map_location=args.device)
            print("PPO update and checkpoint round trip passed", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
