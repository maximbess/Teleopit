"""Run with conda run -n teleopit python -m teleopit.sim.o6_grasp."""

import argparse
import json

from .config import load_config
from .evaluate import run_suite, run_trial
from .presets import ladder_rung_config
from .replay import with_replay


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config overrides")
    parser.add_argument(
        "--preset", choices=("nominal", "ladder-rung"), default="nominal"
    )
    parser.add_argument(
        "--wrist-load",
        help="o6_wrist_load_v1 JSON trace; source and frame must be explicit",
    )
    parser.add_argument(
        "--replay-scale", type=float, help="override the trace amplitude multiplier"
    )
    parser.add_argument(
        "--replay-fade",
        type=float,
        help="seconds of endpoint fading (0 for exact samples)",
    )
    parser.add_argument("--output", default="outputs/o6_grasp")
    parser.add_argument("--seed", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--suite", action="store_true")
    mode.add_argument("--viewer", action="store_true")
    mode.add_argument("--render", action="store_true")
    return parser


def resolve_config(args, parser):
    base = ladder_rung_config() if args.preset == "ladder-rung" else None
    config = load_config(args.config, base=base)
    config = resolve_replay(config, args, parser)
    if args.suite and config.load_profile == "replay":
        parser.error(
            "--suite cannot be combined with wrist replay; run a single replay trial"
        )
    return config


def resolve_replay(config, args, parser):
    replay_path = args.wrist_load or config.wrist_load_path
    if replay_path:
        config = with_replay(
            config,
            replay_path,
            fade_duration=config.replay_fade_duration
            if args.replay_fade is None
            else args.replay_fade,
            scale=config.replay_scale
            if args.replay_scale is None
            else args.replay_scale,
        )
    elif args.replay_fade is not None or args.replay_scale is not None:
        parser.error("replay options require --wrist-load or a replay config")
    return config


def main():
    parser = argument_parser()
    args = parser.parse_args()
    config = resolve_config(args, parser)
    if args.viewer:
        from .viewer import show

        show(config)
    elif args.render:
        from .viewer import render_frames

        render_frames(config, args.output)
    elif args.suite:
        results = run_suite(config, args.output, seed=args.seed)
        print(
            json.dumps(
                [
                    {
                        k: row[k]
                        for k in (
                            "scenario",
                            "unsupported_success",
                            "verified_seconds",
                            "failure_observed",
                            "release_clear",
                        )
                    }
                    for row in results
                ],
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                run_trial(
                    "baseline" if args.preset == "nominal" else "ladder_rung",
                    config,
                    args.output,
                    seed=args.seed,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
