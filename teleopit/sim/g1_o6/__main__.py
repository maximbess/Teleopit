"""CPU physical G1/O6 ladder feasibility experiment."""

import argparse
import json
from pathlib import Path

from .run import run_trial, scenarios
from .settings import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", default="outputs/g1_o6/current")
    parser.add_argument("--grasp-config", help="current GraspPose JSON overrides")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--viewer", action="store_true")
    modes.add_argument("--video", action="store_true")
    modes.add_argument("--suite", action="store_true")
    args = parser.parse_args()
    try:
        cfg, pose = load_settings(args.config, args.grasp_config)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.viewer:
        from .viewer import show

        result = show(cfg, args.output, pose=pose)
    elif args.video:
        from .viewer import record

        result = record(cfg, args.output, pose=pose)
    elif args.suite:
        result = {
            name: run_trial(c, Path(args.output) / name, pose=pose)
            for name, c in scenarios(cfg).items()
        }
        (Path(args.output) / "suite.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    else:
        result = run_trial(cfg, args.output, pose=pose)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
