#!/usr/bin/env python3
"""Play back a trained tracking or ladder policy in simulation.

Viewer options:
  native  -- MuJoCo native window (default, requires display)
  viser   -- browser-based 3D viewer at http://localhost:8012

Usage:
    # Native window
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets_precomputed

    # Browser viewer (no display required)
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets_precomputed \
        --viewer viser

    # Record video instead of interactive viewer
    python train_mimic/scripts/play.py \
        --checkpoint logs/rsl_rl/g1_tracking/2026-.../model_30000.pt \
        --motion_file data/datasets_precomputed \
        --video

    # Play only the stabilize/first-hand/second-hand ladder prefix
    python train_mimic/scripts/play.py \
        --task G1-Ladder-Climb-RL \
        --checkpoint logs/rsl_rl/g1_ladder_rl/2026-.../model_60000.pt \
        --ladder_phase second_hand
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from train_mimic.app import (
    DEFAULT_TASK,
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    resolve_device,
    validate_checkpoint_path,
    validate_motion_file,
)
from train_mimic.ladder_playback import (
    LADDER_PLAY_PHASES,
    configure_ladder_play_phase,
)
from train_mimic.tasks.tracking.config.constants import (
    LADDER_RL_TASK,
    SUPPORTED_TASKS,
    TRACKING_TASKS,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained G1 policy.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help=(
            "Precomputed motion dataset root; required for tracking tasks and "
            "not used by the ladder task"
        ),
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--viewer",
        type=str,
        default="native",
        choices=["native", "viser"],
        help="native: MuJoCo window (requires display); viser: browser at localhost:8012",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record video instead of interactive viewer",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=SUPPORTED_TASKS,
        help="Task id to play (default: %(default)s)",
    )
    parser.add_argument(
        "--ladder_phase",
        type=str,
        choices=tuple(LADDER_PLAY_PHASES),
        default=None,
        help=(
            "Deepest ladder phase to play. Each episode starts with stabilize "
            "and ends after this phase; default: second_foot (full climb)."
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args.num_envs}")
    if args.task in TRACKING_TASKS:
        if args.motion_file is None:
            raise ValueError("--motion_file is required for tracking tasks")
        if args.ladder_phase is not None:
            raise ValueError("--ladder_phase is only valid for G1-Ladder-Climb-RL")
        return
    if args.task == LADDER_RL_TASK and args.motion_file is not None:
        raise ValueError("--motion_file is not used by G1-Ladder-Climb-RL")


def main() -> None:
    args = parse_args()

    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)

    (
        torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        _load_env_cfg,
        _load_rl_cfg,
        _load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    try:
        validate_checkpoint_path(args.checkpoint)
        if args.task in TRACKING_TASKS:
            validate_motion_file(args.motion_file)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    configure_torch_backends()

    # Load configs (play=True disables corruption, push_robot, etc.)
    task_name, env_cfg, agent_cfg, runner_cls = load_task_components(
        args.task,
        play=True,
        load_env_cfg=_load_env_cfg,
        load_rl_cfg=_load_rl_cfg,
        load_runner_cls=_load_runner_cls,
    )

    # Override for playback
    env_cfg.scene.num_envs = args.num_envs
    if args.task in TRACKING_TASKS:
        env_cfg.commands["motion"].motion_file = args.motion_file
    else:
        selected_phase = configure_ladder_play_phase(env_cfg, args.ladder_phase)
        print(f"[INFO] Ladder playback prefix: stabilize through {selected_phase}.")

    device = resolve_device(args.device, torch)

    # render_mode only needed for video recording
    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if args.video:
        from mjlab.utils.wrappers import VideoRecorder

        log_dir = os.path.dirname(args.checkpoint)
        env = VideoRecorder(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=500,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy (force tensorboard to avoid wandb init during playback).
    log_dir = os.path.dirname(args.checkpoint)
    agent_dict = build_runner_cfg_dict(agent_cfg, force_tensorboard=True)
    RunnerCls = runner_cls or MjlabOnPolicyRunner
    runner = RunnerCls(env, agent_dict, log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Apply the configured reset state before the first policy observation.
    obs, _ = env.reset()

    if args.video:
        # Run a fixed number of steps then close
        for _ in range(500):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    elif args.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
