import argparse
import json
from pathlib import Path

from .config import DiagnosticConfig


def main():
    parser = argparse.ArgumentParser(
        description="CPU G1/O6 diagnostic experiments; failures remain failures"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    nominal = commands.add_parser("nominal")
    nominal.add_argument("--config", default="docs/g1_o6/nominal.json")
    nominal.add_argument("--continue-after-failure", type=float, default=0)
    nominal.add_argument(
        "--posture", choices=["baseline", "thumb-below"], default="baseline"
    )
    nominal.add_argument("--output", required=True)
    for name in ("free-close", "load-response"):
        sub = commands.add_parser(name)
        sub.add_argument("--source-run", default="outputs/g1_o6/diagnostics/baseline")
        sub.add_argument("--side", choices=["both", "left", "right"], default="both")
        sub.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "nominal":
        from teleopit.sim.g1_o6.config import Config

        from .nominal import run_nominal

        result = run_nominal(
            Config(**json.loads(Path(args.config).read_text())),
            args.output,
            DiagnosticConfig(continuation_s=args.continue_after_failure),
            args.posture,
        )
    elif args.command == "free-close":
        from .experiments import free_close

        result = free_close(args.source_run, args.output, args.side)
    else:
        from .experiments import load_response

        result = load_response(args.source_run, args.output, args.side)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
