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
    args = parser.parse_args()
    result = run_hold(args.output, seconds=args.seconds, physics_dt=args.physics_dt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
