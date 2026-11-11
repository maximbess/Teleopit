"""Explicit, gated support-validation experiments; default G1/O6 remains hold-only."""

import argparse
import json

from .long_hold import run_hold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    hold = commands.add_parser("long-hold")
    hold.add_argument("--seconds", type=float, choices=(30, 60, 120), default=60)
    hold.add_argument(
        "--physics-dt", type=float, choices=(0.001, 0.0005), default=0.001
    )
    hold.add_argument("--output", required=True)
    hold.add_argument("--grasp-config")
    capacity = commands.add_parser("capacity-case")
    capacity.add_argument("--source-run", required=True)
    capacity.add_argument("--state", choices=("initial", "settled"), default="initial")
    capacity.add_argument("--side", choices=("left", "right"), required=True)
    capacity.add_argument(
        "--physics-dt", type=float, choices=(0.001, 0.0005), default=0.001
    )
    capacity.add_argument("--output", required=True)
    matrix = commands.add_parser("capacity-matrix")
    matrix.add_argument("--source-run", required=True)
    matrix.add_argument("--state", choices=("initial", "settled"), default="settled")
    matrix.add_argument(
        "--physics-dt", type=float, choices=(0.001, 0.0005), default=0.001
    )
    matrix.add_argument("--prior-trials", type=int, choices=range(9), default=0)
    matrix.add_argument("--output", required=True)
    transfer = commands.add_parser("transfer")
    transfer.add_argument("--evidence", required=True)
    transfer.add_argument(
        "--mode", choices=("load", "unload", "release", "reacquire"), default="load"
    )
    transfer.add_argument("--side", choices=("left", "right"), default="left")
    transfer.add_argument(
        "--target-share", type=float, choices=(0.05, 0.1, 0.2), default=0.05
    )
    transfer.add_argument("--cycles", type=int, choices=(1, 3), default=1)
    transfer.add_argument("--alternating", action="store_true")
    transfer.add_argument("--three-seconds", type=int, choices=(1, 3, 10), default=1)
    transfer.add_argument("--plateau-seconds", type=int, choices=(10, 30), default=10)
    transfer.add_argument(
        "--physics-dt", type=float, choices=(0.001, 0.0005), default=0.001
    )
    transfer.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "long-hold":
        from teleopit.sim.g1_o6.settings import load_settings

        _, pose = load_settings(grasp_path=args.grasp_config)
        result = run_hold(
            args.output, seconds=args.seconds, physics_dt=args.physics_dt, pose=pose
        )
    elif args.command == "capacity-matrix":
        from .capacity_matrix import run_matrix

        result = run_matrix(
            args.source_run,
            args.output,
            state=args.state,
            dt=args.physics_dt,
            prior_trials=args.prior_trials,
        )
    elif args.command == "transfer":
        from .transfer import run_transfer

        result = run_transfer(
            args.evidence,
            args.output,
            mode=args.mode,
            side=args.side,
            target_share=args.target_share,
            cycles=args.cycles,
            alternating=args.alternating,
            three_seconds=args.three_seconds,
            plateau_seconds=args.plateau_seconds,
            physics_dt=args.physics_dt,
        )
    else:
        from .capacity import qualified_source, run_capacity

        source = qualified_source(args.source_run, args.state)
        result = run_capacity(
            source, args.side, args.output, physics_dt=args.physics_dt
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
