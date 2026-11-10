"""CPU physical G1/O6 ladder feasibility experiment."""

import argparse
import json
from pathlib import Path

from .config import Config
from .run import run_trial, scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", default="outputs/g1_o6/nominal")
    parser.add_argument(
        "--posture", choices=["baseline", "thumb-below"], default="baseline"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--viewer", action="store_true")
    modes.add_argument("--video", action="store_true")
    modes.add_argument("--suite", action="store_true")
    args = parser.parse_args()
    cfg = Config(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    cfg.validate()
    if args.viewer:
        from .viewer import show

        result = show(cfg, args.output, args.posture)
    elif args.video:
        from .viewer import record

        result = record(cfg, args.output, args.posture)
    elif args.suite:
        result = {
            name: run_trial(c, Path(args.output) / name, posture=args.posture)
            for name, c in scenarios(cfg).items()
        }
        (Path(args.output) / "suite.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    else:
        result = run_trial(cfg, args.output, posture=args.posture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
