#!/usr/bin/env python3
"""Record one RL-only G1 ladder-policy rollout directly to MP4.

Unlike ``benchmark_ladder.py``, this entry point does not aggregate rewards,
classify terminations, calculate metrics, or write text/JSON reports.  It uses
one environment and stops at the first episode termination or the requested
frame limit.

Example:
    python train_mimic/scripts/record_ladder_video.py \
        --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
        --output ladder.mp4 --frames 1000
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record one trained G1 ladder-policy episode to MP4 without "
            "benchmark metrics."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=("Output MP4 path (default: <checkpoint_dir>/videos/ladder-<model>.mp4)."),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1000,
        help="Maximum number of video frames (default: 1000).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Unrecorded policy steps before resetting and recording (default: 0).",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=4.5,
        help="Distance from the fixed overview camera to its target (default: 4.5).",
    )
    parser.add_argument(
        "--camera_azimuth",
        type=float,
        default=30.0,
        help="Fixed outside-ladder camera azimuth in degrees (default: 30).",
    )
    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=-5.0,
        help="Fixed overview camera elevation in degrees (default: -5).",
    )
    parser.add_argument(
        "--camera_lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-0.45, 0.0, 1.45),
        help=("World-space camera target as X Y Z (default: -0.45 0.0 1.45)."),
    )
    parser.add_argument(
        "--camera_fovy",
        type=float,
        default=48.0,
        help="Vertical field of view in degrees (default: 48).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output FPS (default: policy frequency, normally 50).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--robot_xml",
        type=str,
        default=None,
        help=(
            "Canonical G1 MuJoCo XML to augment with the ladder scene "
            "(default: assets/robots/unitree_g1/g1_29dof.xml)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output MP4.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.frames <= 0:
        raise ValueError(f"--frames must be positive, got {args.frames}")
    if args.warmup_steps < 0:
        raise ValueError(f"--warmup_steps must be >= 0, got {args.warmup_steps}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError(
            f"--width and --height must be positive, got {args.width}x{args.height}"
        )
    if args.fps is not None and args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.camera_distance <= 0.0:
        raise ValueError(
            f"--camera_distance must be positive, got {args.camera_distance}"
        )
    if not 1.0 <= args.camera_fovy < 180.0:
        raise ValueError(f"--camera_fovy must be in [1, 180), got {args.camera_fovy}")


def _configure_recording_camera(env_cfg: Any, args: argparse.Namespace) -> None:
    """Use a fixed outside-ladder overview instead of torso tracking."""

    env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.WORLD
    env_cfg.viewer.entity_name = None
    env_cfg.viewer.body_name = None
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    env_cfg.viewer.distance = args.camera_distance
    env_cfg.viewer.azimuth = args.camera_azimuth
    env_cfg.viewer.elevation = args.camera_elevation
    env_cfg.viewer.fovy = args.camera_fovy


def _configure_video_backend() -> None:
    # These variables must be set before importing MuJoCo/GL-dependent modules.
    if "MUJOCO_GL" not in os.environ:
        backend = "egl" if platform.system() == "Linux" else "glfw"
        os.environ["MUJOCO_GL"] = backend
        print(f"[INFO] Defaulting MUJOCO_GL={backend}")
    if (
        os.environ.get("MUJOCO_GL", "").lower() == "egl"
        and "PYOPENGL_PLATFORM" not in os.environ
    ):
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("[INFO] Defaulting PYOPENGL_PLATFORM=egl")


def _output_path(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.output is None:
        path = checkpoint.parent / "videos" / f"ladder-{checkpoint.stem}.mp4"
    else:
        path = Path(args.output).expanduser().resolve()
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"--output must end with .mp4, got {path}")
    return path


def _render_frame(unwrapped: Any) -> Any:
    frame = unwrapped.render()
    if frame is None:
        raise RuntimeError("render() returned None; expected RGB video frame")
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    try:
        output_path = _output_path(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if output_path.exists() and not args.force:
        print(
            f"Error: output already exists: {output_path}. "
            "Pass --force to overwrite it."
        )
        return 1

    _configure_video_backend()
    # On Windows, load Torch DLLs before MuJoCo/GLFW enters the process.
    import torch

    # Import the training stack only after selecting the platform GL backend.
    from train_mimic.app import (
        build_runner_cfg_dict,
        import_training_stack,
        load_task_components,
        resolve_device,
    )
    from train_mimic.tasks.tracking.config.constants import LADDER_RL_TASK

    (
        imported_torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    if imported_torch is not torch:
        raise RuntimeError("Training stack imported a different torch module")
    from train_mimic.tasks.tracking.config.env import (
        make_g1_ladder_training_robot_cfg,
        resolve_g1_training_xml,
    )

    configure_torch_backends()
    _task_name, env_cfg, agent_cfg, runner_cls = load_task_components(
        LADDER_RL_TASK,
        play=True,
        load_env_cfg=load_env_cfg,
        load_rl_cfg=load_rl_cfg,
        load_runner_cls=load_runner_cls,
    )

    robot_xml = resolve_g1_training_xml(args.robot_xml)
    if not robot_xml.is_file():
        print(f"Error: G1 training MuJoCo XML not found: {robot_xml}")
        return 1

    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.scene.entities["robot"] = make_g1_ladder_training_robot_cfg(robot_xml)
    env_cfg.robot_xml = str(robot_xml)
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    _configure_recording_camera(env_cfg, args)

    device = resolve_device(args.device, torch)
    env: Any | None = None
    video_writer: Any | None = None
    frames_written = 0

    try:
        try:
            env = ManagerBasedRlEnv(
                cfg=env_cfg,
                device=device,
                render_mode="rgb_array",
            )
        except Exception as exc:
            raise RuntimeError(
                "Ladder video environment initialization failed. Check the "
                "selected MUJOCO_GL backend and that the Warp kernel cache is "
                f"writable. Original error: {exc}"
            ) from exc

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cfg = build_runner_cfg_dict(agent_cfg, force_tensorboard=True)
        RunnerCls = runner_cls or MjlabOnPolicyRunner
        runner = RunnerCls(
            env,
            runner_cfg,
            log_dir=str(Path(args.checkpoint).resolve().parent),
            device=device,
        )
        runner.load(args.checkpoint, map_location=device)
        policy = runner.get_inference_policy(device=device)

        # ManagerBasedRlEnv construction leaves the raw XML qpos in place.
        # Explicit reset is required to apply the configured climbing keyframe
        # and initialize the hand/foot ladder FSM before the first video frame.
        obs, _ = env.reset()
        for _ in range(args.warmup_steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, _rewards, _dones, _extras = env.step(actions)
        if args.warmup_steps:
            obs, _ = env.reset()

        unwrapped = env.unwrapped
        video_fps = args.fps or max(1, int(round(1.0 / unwrapped.step_dt)))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import imageio.v2 as imageio

        video_writer = imageio.get_writer(
            str(output_path),
            fps=video_fps,
            quality=8,
        )
        print(
            f"[INFO] Recording up to {args.frames} frames at {video_fps} FPS "
            f"to: {output_path}"
        )

        video_writer.append_data(_render_frame(unwrapped))
        frames_written = 1
        for _ in range(1, args.frames):
            with torch.no_grad():
                actions = policy(obs)
            obs, _rewards, dones, _extras = env.step(actions)
            if bool(torch.any(dones).item()):
                print("[INFO] Episode terminated; stopping before automatic reset")
                break
            video_writer.append_data(_render_frame(unwrapped))
            frames_written += 1
    finally:
        if video_writer is not None:
            video_writer.close()
        if env is not None:
            env.close()

    print(f"Saved {frames_written} frames: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
