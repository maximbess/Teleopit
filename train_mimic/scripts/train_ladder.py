#!/usr/bin/env python3
"""Train the G1 ladder-climbing policy with plain PPO reinforcement learning.

This entry point intentionally has no motion dataset, sampling mode, imitation
reward, or motion-tracking runner.  The policy learns directly from the ladder
command, robot state, sparse rung progress, and dense reach/height rewards.

Example:
    python train_mimic/scripts/train_ladder.py \
        --num_envs 4096 --max_iterations 60000
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
from datetime import datetime
from typing import Any, Sequence

from train_mimic.app import (
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    validate_checkpoint_path,
)
from train_mimic.scripts import train as shared_train
from train_mimic.tasks.tracking.config.constants import LADDER_RL_TASK
from train_mimic.tasks.tracking.config.env import (
    make_g1_ladder_training_robot_cfg,
    resolve_g1_training_xml,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the G1 ladder policy with RL-only PPO (mjlab)."
    )
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help=(
            "Learning iterations for this invocation. When resuming, this is "
            "the number of additional iterations."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--logger",
        choices=["tensorboard", "wandb", "swanlab"],
        default="tensorboard",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--robot_xml",
        type=str,
        default=None,
        help=(
            "Canonical G1 MuJoCo XML to augment with the ladder scene "
            "(default: assets/robots/unitree_g1/g1_29dof.xml)."
        ),
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help="Single-node multi-GPU helper, for example: --gpu_ids 0 1 2 3.",
    )
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument("--video_length", type=int, default=300)
    return parser.parse_args(argv)


def _run_worker(args: argparse.Namespace) -> None:
    (
        torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    env: Any | None = None
    rank = os.environ.get("RANK", "0")

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        print(f"[INFO] Rank {rank} received signal {signum}, shutting down...")
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        shared_train._destroy_process_group(torch)
        raise KeyboardInterrupt

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    configure_torch_backends()
    _task_name, env_cfg, agent_cfg, runner_cls = load_task_components(
        LADDER_RL_TASK,
        load_env_cfg=load_env_cfg,
        load_rl_cfg=load_rl_cfg,
        load_runner_cls=load_runner_cls,
    )

    env_cfg.seed = shared_train._resolve_worker_seed(args.seed)
    robot_xml = resolve_g1_training_xml(args.robot_xml)
    if not robot_xml.is_file():
        raise FileNotFoundError(f"G1 training MuJoCo XML not found: {robot_xml}")
    env_cfg.scene.entities["robot"] = make_g1_ladder_training_robot_cfg(robot_xml)
    env_cfg.robot_xml = str(robot_xml)
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError(f"--num_envs must be positive, got {args.num_envs}")
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        if args.max_iterations <= 0:
            raise ValueError(
                f"--max_iterations must be positive, got {args.max_iterations}"
            )
        agent_cfg.max_iterations = args.max_iterations
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.resume is not None:
        validate_checkpoint_path(args.resume)

    device = shared_train._resolve_device(args, torch)
    log_root = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    ladder_cmd = env_cfg.commands["ladder"]
    swanlab_active = shared_train._configure_experiment_logger(
        logger_name=args.logger,
        agent_cfg=agent_cfg,
        env_cfg=env_cfg,
        log_dir=log_dir,
        run_config={
            "task": LADDER_RL_TASK,
            "experiment_name": agent_cfg.experiment_name,
            "robot_xml": str(robot_xml),
            "num_envs": env_cfg.scene.num_envs,
            "max_iterations": agent_cfg.max_iterations,
            "start_rung": ladder_cmd.start_rung,
            "num_rungs": len(ladder_cmd.rung_site_names),
            "curriculum_success_threshold": ladder_cmd.curriculum_success_threshold,
            "curriculum_window_size": ladder_cmd.curriculum_window_size,
            "curriculum_min_phase_steps": list(ladder_cmd.curriculum_min_phase_steps),
            "boundary_state_reset_prob": ladder_cmd.boundary_state_reset_prob,
            "boundary_state_bank_size": ladder_cmd.boundary_state_bank_size,
            "stabilization_dwell_steps": ladder_cmd.stabilization_dwell_steps,
            "stabilization_dwell_max_steps": ladder_cmd.stabilization_dwell_max_steps,
            "max_stabilization_support_offset_error": (
                ladder_cmd.max_stabilization_support_offset_error
            ),
            "release_preload_dwell_steps": ladder_cmd.release_preload_dwell_steps,
            "release_ramp_steps": ladder_cmd.release_ramp_steps,
            "release_final_dwell_steps": ladder_cmd.release_final_dwell_steps,
            "release_recovery_steps": ladder_cmd.release_recovery_steps,
            "pre_release_timeout_steps": ladder_cmd.pre_release_timeout_steps,
            "max_release_torso_speed": ladder_cmd.max_release_torso_speed,
            "max_release_torso_orientation_error": (
                ladder_cmd.max_release_torso_orientation_error
            ),
            "max_release_support_offset_error": (
                ladder_cmd.max_release_support_offset_error
            ),
            "release_soft_timeconst": ladder_cmd.release_soft_timeconst,
            "release_soft_impedance": ladder_cmd.release_soft_impedance,
            "hand_target_dwell_steps": ladder_cmd.hand_target_dwell_steps,
            "foot_target_dwell_steps": ladder_cmd.foot_target_dwell_steps,
        },
    )

    render_mode = "rgb_array" if args.video else None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
        if args.video:
            from mjlab.utils.wrappers import VideoRecorder

            env = VideoRecorder(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step
                % (args.video_interval * env_cfg.decimation)
                == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        RunnerCls = runner_cls if runner_cls is not None else MjlabOnPolicyRunner
        runner = RunnerCls(
            env,
            build_runner_cfg_dict(agent_cfg),
            log_dir=log_dir,
            device=device,
        )
        if args.resume is not None:
            print(f"[INFO] Resuming ladder training from: {args.resume}")
            runner.load(args.resume)
        print(f"[INFO] Running {agent_cfg.max_iterations} ladder RL iterations")
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=False,
        )
    except KeyboardInterrupt:
        print(f"[INFO] Rank {rank} interrupted; exiting gracefully.")
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        if swanlab_active:
            with contextlib.suppress(Exception):
                import swanlab

                swanlab.finish()
        shared_train._destroy_process_group(torch)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main(argv: Sequence[str] | None = None) -> None:
    cli_argv = list(sys.argv if argv is None else argv)
    args = parse_args(cli_argv[1:])
    if shared_train._should_launch_multi_gpu(args):
        shared_train._launch_multi_gpu(args, cli_argv)
        return
    _run_worker(args)


if __name__ == "__main__":
    main()
