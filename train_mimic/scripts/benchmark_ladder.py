#!/usr/bin/env python3
"""Evaluate an RL-only G1 ladder checkpoint and optionally record an MP4.

This entry point is intentionally independent from the motion-tracking
benchmark: it does not load a motion dataset and reports ladder-specific
success, failure, timeout, progress, grip, and reach metrics.

Examples:
    python train_mimic/scripts/benchmark_ladder.py \
        --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
        --num_envs 64 --num_eval_steps 5000

    python train_mimic/scripts/benchmark_ladder.py \
        --checkpoint logs/rsl_rl/g1_ladder_rl/<run>/model_60000.pt \
        --num_envs 1 --video --video_length 1000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from train_mimic.app import (
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    resolve_device,
    validate_checkpoint_path,
)
from train_mimic.tasks.tracking.config.constants import LADDER_RL_TASK


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an RL-only G1 ladder-climbing PPO checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--robot_xml",
        type=str,
        default=None,
        help=(
            "Canonical G1 MuJoCo XML to augment with the ladder scene "
            "(default: assets/robots/unitree_g1/g1_29dof.xml)."
        ),
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--num_eval_steps",
        type=int,
        default=5000,
        help="Policy steps measured after warmup (default: 5000).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Unmeasured policy steps before a fresh evaluation reset (default: 100).",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=20.0,
        help=(
            "Evaluation episode timeout in seconds (default: 20). "
            "Timeouts count as completed unsuccessful episodes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Frames to record (default: one evaluation episode).",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="MP4 output directory (default: benchmark_results/videos).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Text and JSON report directory (default: benchmark_results).",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args.num_envs}")
    if args.num_eval_steps <= 0:
        raise ValueError(
            f"--num_eval_steps must be positive, got {args.num_eval_steps}"
        )
    if args.warmup_steps < 0:
        raise ValueError(f"--warmup_steps must be >= 0, got {args.warmup_steps}")
    if args.episode_length_s <= 0.0:
        raise ValueError(
            f"--episode_length_s must be positive, got {args.episode_length_s}"
        )
    if args.video and args.num_envs != 1:
        raise ValueError("--video requires --num_envs 1")
    if args.video_length is not None and args.video_length <= 0:
        raise ValueError(f"--video_length must be positive, got {args.video_length}")


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _extend_tensor_values(
    values: list[float],
    tensor: object,
    torch_module: Any,
) -> None:
    if not isinstance(tensor, torch_module.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    values.extend(tensor.detach().float().cpu().reshape(-1).tolist())


def _render_frame(unwrapped: object) -> np.ndarray:
    frame = unwrapped.render()
    if frame is None:
        raise RuntimeError("render() returned None; expected render_mode='rgb_array'")
    return frame


def _configure_video_backend() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("[INFO] --video enabled, defaulting MUJOCO_GL=egl")
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print("[INFO] --video enabled, defaulting PYOPENGL_PLATFORM=egl")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)

    try:
        validate_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if args.video:
        # Must be set before importing MuJoCo/GL-dependent training modules.
        _configure_video_backend()

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
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.scene.entities["robot"] = make_g1_ladder_training_robot_cfg(robot_xml)
    env_cfg.robot_xml = str(robot_xml)

    device = resolve_device(args.device, torch)
    render_mode = "rgb_array" if args.video else None
    env: Any | None = None
    video_writer: Any | None = None
    video_path: Path | None = None

    try:
        try:
            env = ManagerBasedRlEnv(
                cfg=env_cfg,
                device=device,
                render_mode=render_mode,
            )
        except Exception as exc:
            if not args.video:
                raise
            raise RuntimeError(
                "Video renderer initialization failed. Set MUJOCO_GL to a "
                "backend available on this machine (usually egl or osmesa)."
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

        # Apply the configured ladder climbing keyframe before warmup/evaluation.
        obs, _ = env.reset()
        if args.warmup_steps:
            print(f"[INFO] Running {args.warmup_steps} warmup steps")
            for _ in range(args.warmup_steps):
                with torch.no_grad():
                    actions = policy(obs)
                obs, _rewards, _dones, _extras = env.step(actions)
            # Start all measured episodes from a clean, synchronized state.
            obs, _ = env.reset()

        unwrapped = env.unwrapped
        command = unwrapped.command_manager.get_term("ladder")
        termination_manager = unwrapped.termination_manager
        metric_keys = sorted(command.metrics.keys())
        metric_series: dict[str, list[float]] = {key: [] for key in metric_keys}
        reward_series: list[float] = []
        completed_episode_rewards: list[float] = []
        completed_episode_lengths: list[float] = []
        completed_episode_progress: list[float] = []
        reset_log_series: dict[str, list[float]] = {}

        successes = 0
        failures = 0
        timeouts = 0
        termination_counts = {name: 0 for name in termination_manager.active_terms}
        episode_reward = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
        episode_length = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        episode_peak_progress = torch.zeros(
            args.num_envs, dtype=torch.float32, device=device
        )

        if args.video:
            import imageio.v2 as imageio

            if args.video_length is None:
                args.video_length = min(
                    args.num_eval_steps,
                    int(unwrapped.max_episode_length),
                )
            if args.num_eval_steps < args.video_length:
                print(
                    "[INFO] Increasing --num_eval_steps from "
                    f"{args.num_eval_steps} to {args.video_length} to cover video"
                )
                args.num_eval_steps = args.video_length
            video_folder = Path(args.video_folder or "benchmark_results/videos")
            video_folder.mkdir(parents=True, exist_ok=True)
            video_path = video_folder / f"ladder-{Path(args.checkpoint).stem}.mp4"
            video_fps = max(1, int(round(1.0 / unwrapped.step_dt)))
            video_writer = imageio.get_writer(
                str(video_path),
                fps=video_fps,
                quality=8,
            )
            print(f"[INFO] Recording {args.video_length} frames to: {video_path}")

        print(
            f"[INFO] Evaluating {args.num_envs} environment(s) for "
            f"{args.num_eval_steps} policy steps"
        )
        for step in range(args.num_eval_steps):
            current_progress = command.rung_progress.detach()
            episode_peak_progress = torch.maximum(
                episode_peak_progress,
                current_progress,
            )
            for key in metric_keys:
                _extend_tensor_values(
                    metric_series[key],
                    command.metrics[key],
                    torch,
                )

            if video_writer is not None and step < args.video_length:
                video_writer.append_data(_render_frame(unwrapped))

            with torch.no_grad():
                actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            _extend_tensor_values(reward_series, rewards, torch)
            episode_reward += rewards
            episode_length += 1

            done_mask = dones > 0
            success_mask = termination_manager.get_term("success").bool()
            timeout_mask = termination_manager.time_outs.bool()
            explicit_failure_mask = torch.zeros_like(done_mask)
            for name in termination_manager.active_terms:
                term_cfg = termination_manager.get_term_cfg(name)
                if name != "success" and not term_cfg.time_out:
                    explicit_failure_mask |= termination_manager.get_term(name).bool()

            classified_success = done_mask & success_mask
            classified_failure = done_mask & ~classified_success & explicit_failure_mask
            classified_timeout = (
                done_mask & ~classified_success & ~classified_failure & timeout_mask
            )
            # Future non-timeout termination terms are failures even if they were
            # not part of the task when this benchmark was written.
            classified_failure |= done_mask & ~classified_success & ~classified_timeout

            successes += int(classified_success.sum().item())
            timeouts += int(classified_timeout.sum().item())
            failures += int(classified_failure.sum().item())
            for name in termination_counts:
                termination_counts[name] += int(
                    termination_manager.get_term(name).sum().item()
                )

            if torch.any(done_mask):
                # A successful episode has, by definition, reached the final rung.
                episode_peak_progress[classified_success] = 1.0
                completed_episode_rewards.extend(
                    episode_reward[done_mask].detach().cpu().tolist()
                )
                completed_episode_lengths.extend(
                    episode_length[done_mask].detach().float().cpu().tolist()
                )
                completed_episode_progress.extend(
                    episode_peak_progress[done_mask].detach().cpu().tolist()
                )
                episode_reward[done_mask] = 0.0
                episode_length[done_mask] = 0
                episode_peak_progress[done_mask] = 0.0

                extras_log = extras.get("log", {}) if isinstance(extras, dict) else {}
                if isinstance(extras_log, dict):
                    for key, value in extras_log.items():
                        if not key.startswith(
                            (
                                "Episode_Reward/",
                                "Episode_Termination/",
                                "Metrics/ladder/",
                            )
                        ):
                            continue
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.detach().float().mean().item())
                        elif isinstance(value, (float, int)):
                            scalar = float(value)
                        else:
                            continue
                        reset_log_series.setdefault(key, []).append(scalar)
    finally:
        if video_writer is not None:
            video_writer.close()
        if env is not None:
            env.close()

    completed_episodes = successes + failures + timeouts
    success_rate = _rate(successes, completed_episodes)
    failure_rate = _rate(failures, completed_episodes)
    timeout_rate = _rate(timeouts, completed_episodes)
    reward_stats = _stats(reward_series)
    episode_reward_stats = _stats(completed_episode_rewards)
    episode_length_stats = _stats(completed_episode_lengths)
    episode_progress_stats = _stats(completed_episode_progress)
    metric_stats = {key: _stats(values) for key, values in metric_series.items()}
    reset_log_stats = {key: _stats(values) for key, values in reset_log_series.items()}

    print("\nLadder Benchmark Results:")
    print(f"  completed_episodes: {completed_episodes}")
    print(f"  successes: {successes}")
    print(f"  failures: {failures}")
    print(f"  timeouts: {timeouts}")
    print(f"  success_rate: {_format_number(success_rate)}")
    print(f"  failure_rate: {_format_number(failure_rate)}")
    print(f"  timeout_rate: {_format_number(timeout_rate)}")
    print(
        "  mean_terminal_rung_progress: "
        f"{_format_number(episode_progress_stats['mean'])}"
    )
    print(f"  mean_episode_reward: {_format_number(episode_reward_stats['mean'])}")
    print(f"  mean_episode_length: {_format_number(episode_length_stats['mean'], 2)}")
    print(f"  mean_step_reward: {_format_number(reward_stats['mean'])}")
    print("\nStep metric distributions (mean / p50 / p95):")
    for key in metric_keys:
        stats = metric_stats[key]
        print(
            f"  {key}: {_format_number(stats['mean'])} / "
            f"{_format_number(stats['p50'])} / "
            f"{_format_number(stats['p95'])}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_stem = f"{LADDER_RL_TASK}-{Path(args.checkpoint).stem}"
    text_path = output_dir / f"{report_stem}.txt"
    json_path = output_dir / f"{report_stem}.json"

    report = {
        "task": LADDER_RL_TASK,
        "checkpoint": args.checkpoint,
        "robot_xml": str(robot_xml),
        "device": str(device),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_eval_steps": args.num_eval_steps,
        "warmup_steps": args.warmup_steps,
        "episode_length_s": args.episode_length_s,
        "evaluated_transitions": args.num_envs * args.num_eval_steps,
        "completed_episodes": completed_episodes,
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "timeout_rate": timeout_rate,
        "step_reward_stats": reward_stats,
        "episode_reward_stats": episode_reward_stats,
        "episode_length_stats": episode_length_stats,
        "episode_peak_rung_progress_stats": episode_progress_stats,
        "metric_stats": metric_stats,
        "termination_counts": termination_counts,
        "reset_log_stats": reset_log_stats,
        "video": str(video_path) if video_path is not None else None,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"task: {LADDER_RL_TASK}",
        f"checkpoint: {args.checkpoint}",
        f"robot_xml: {robot_xml}",
        f"num_envs: {args.num_envs}",
        f"num_eval_steps: {args.num_eval_steps}",
        f"warmup_steps: {args.warmup_steps}",
        f"episode_length_s: {args.episode_length_s}",
        f"completed_episodes: {completed_episodes}",
        f"successes: {successes}",
        f"failures: {failures}",
        f"timeouts: {timeouts}",
        f"success_rate: {_format_number(success_rate, 6)}",
        f"failure_rate: {_format_number(failure_rate, 6)}",
        f"timeout_rate: {_format_number(timeout_rate, 6)}",
        "mean_terminal_rung_progress: "
        f"{_format_number(episode_progress_stats['mean'], 6)}",
        f"mean_episode_reward: {_format_number(episode_reward_stats['mean'], 6)}",
        f"mean_episode_length: {_format_number(episode_length_stats['mean'], 6)}",
        f"mean_step_reward: {_format_number(reward_stats['mean'], 6)}",
        "",
        "termination_counts:",
    ]
    lines.extend(
        f"{name}: {count}" for name, count in sorted(termination_counts.items())
    )
    lines.append("")
    lines.append("metric_stats(mean,std,p50,p95,min,max):")
    for key in sorted(metric_stats):
        stats = metric_stats[key]
        lines.append(
            f"{key}: "
            + ", ".join(
                _format_number(stats[field], 6)
                for field in ("mean", "std", "p50", "p95", "min", "max")
            )
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved summary to: {text_path}")
    print(f"Saved detailed JSON to: {json_path}")
    if video_path is not None:
        print(f"Saved video: {video_path}")
    if completed_episodes == 0:
        print(
            "[WARN] No episodes completed. Increase --num_eval_steps or reduce "
            "--episode_length_s before interpreting success rate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
