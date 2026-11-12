"""Run the frozen long-HOLD pose campaign without enabling support transfers."""

import argparse
from pathlib import Path

from .pose_campaign import Campaign, campaign_lock, initial_screen, mandatory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/g1_o6/pose_search")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("mandatory")
    commands.add_parser("screen")
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


if __name__ == "__main__":
    main()
