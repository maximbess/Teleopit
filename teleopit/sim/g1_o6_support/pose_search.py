"""Run the frozen long-HOLD pose campaign without enabling support transfers."""

import argparse
from pathlib import Path

from .pose_campaign import Campaign, campaign_lock, initial_screen, mandatory
from .pose_selection import (
    adaptive_geometry,
    promote_long,
    promote_sixty,
    qualify,
    select_controllers,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/g1_o6/pose_search")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("mandatory")
    commands.add_parser("screen")
    for name in ("adaptive", "controllers", "promote-60", "promote-120", "qualify"):
        commands.add_parser(name)
    trial = commands.add_parser("trial")
    trial.add_argument("candidate")
    trial.add_argument("stage", choices=("screen", "60s", "120s", "repeat", "fine"))
    trial.add_argument("--retry-reason")
    args = parser.parse_args()
    with campaign_lock(args.output):
        campaign = Campaign(args.output)
        if args.command == "mandatory":
            mandatory(campaign)
        elif args.command == "screen":
            initial_screen(campaign)
        elif args.command == "trial":
            campaign.run(args.candidate, args.stage, args.retry_reason)
        elif args.command != "init":
            {
                "adaptive": adaptive_geometry,
                "controllers": select_controllers,
                "promote-60": promote_sixty,
                "promote-120": promote_long,
                "qualify": qualify,
            }[args.command](campaign)


if __name__ == "__main__":
    main()
