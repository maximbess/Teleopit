"""Run bounded G6 batches and prerequisite-gated loading/hand-transfer stages."""

import argparse
import json
from pathlib import Path

from .g6_campaign import G6Campaign, STAGES, best_parent, promote, run_batch
from .g6_pipeline import Pipeline
from .g6_stages import milestone_report
from .pose_campaign import campaign_lock


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output", type=Path, default=Path("outputs/g1_o6/g6_stability")
    )
    result.add_argument("--workers", type=int, default=3)
    sub = result.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    control = sub.add_parser("import-control")
    control.add_argument("path", type=Path)
    batch = sub.add_parser("batch")
    batch.add_argument("batch", choices=list("ABCD"))
    batch.add_argument("--parent", required=True)
    trial = sub.add_parser("trial")
    trial.add_argument("candidate")
    trial.add_argument("stage", choices=list(STAGES))
    retry = sub.add_parser("retry")
    retry.add_argument("candidate")
    retry.add_argument("stage", choices=list(STAGES))
    retry.add_argument("--reason", required=True)
    for command in ("promote", "downstream", "status", "search"):
        sub.add_parser(command)
    return result


def dispatch(campaign, args):
    if args.action == "prepare":
        campaign.prepare()
    elif args.action == "import-control":
        campaign.import_control(args.path)
    elif args.action == "batch":
        print(
            json.dumps(
                {"selected_parent": run_batch(campaign, args.batch, args.parent)}
            )
        )
    elif args.action == "trial":
        campaign.run_many([(args.candidate, args.stage)])
    elif args.action == "retry":
        campaign.retry(args.candidate, args.stage, args.reason)
    else:
        run_stage(campaign, args.action)


def run_stage(campaign, action):
    if action == "search":
        parent = "control"
        for batch in "ABCD":
            parent = run_batch(campaign, batch, parent)
        campaign.note(
            "best_screened_candidate",
            best_parent(campaign, campaign.state["candidates"]),
        )
    if action in ("search", "promote"):
        promote(campaign)
    if action in ("search", "downstream"):
        print(json.dumps(Pipeline(campaign).run(), indent=2))
    if action == "status":
        print(json.dumps(milestone_report(campaign), indent=2))


def main():
    args = parser().parse_args()
    with campaign_lock(args.output):
        dispatch(G6Campaign(args.output, args.workers), args)


if __name__ == "__main__":
    main()
