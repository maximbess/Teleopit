"""Resume the frozen ladder/material experiment without changing the default."""

import argparse
import json
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import write_json

from .ladder_campaign import (
    LadderCampaign,
    code_provenance,
    conditions,
    file_hash,
    long_qualified,
    ranked_promotions,
    run_checksums,
    run_identity,
    specification,
)
from .pose_campaign import campaign_lock


def import_reference(campaign):
    from .ladder_identity import verify_reviewed_model

    verification_model = verify_reviewed_model(
        campaign.root / "reference" / "reviewed_model.npz"
    )
    campaign.note("reviewed_model_preservation", verification_model)
    provenance = code_provenance()
    verification = json.loads(
        (campaign.root / "reference_verification.json").read_text()
    )
    for seconds, stage in ((30, "screen"), (120, "120s")):
        path, result = verified_reference(
            campaign.root, verification[str(seconds)], seconds
        )
        if campaign.previous("G0_reference", stage):
            continue
        condition = campaign.state["candidates"]["G0_reference"]
        identity = run_identity(
            condition=condition,
            reset=campaign.reset_for(condition),
            seconds=seconds,
            dt=0.001,
            provenance=provenance,
        )
        campaign.state["runs"].append(
            {
                "id": "G0_reference_" + stage,
                "candidate": "G0_reference",
                "stage": stage,
                "status": "COMPLETE",
                "output": path.relative_to(campaign.root).as_posix(),
                "result": result,
                "reviewed_reference_import": True,
                "identity": identity,
                "target_provenance": provenance,
                "files_sha256": run_checksums(path),
                "evidence": "pre-refactor fresh reproduction, compared to reviewed reference; consumes one original long slot",
            }
        )
        campaign.save()


def verified_reference(root, verification, seconds):
    if not verification["identical"]:
        raise ValueError(
            "reference reproduction differs; review numerical evidence before import"
        )
    path = root / "reference" / f"{seconds}s"
    if file_hash(path / "steps.jsonl") != verification["new_trace_sha256"]:
        raise ValueError("reference trace changed after verified comparison")
    result = json.loads((path / "summary.json").read_text())
    if result != verification["summary"]:
        raise ValueError("reference summary changed after verified comparison")
    return path, result


def screens(campaign, group):
    previous_groups = {"A": [], "B": ["A"], "C": ["A", "B"]}[group]
    required = [r for r in conditions() if r["stage_group"] in previous_groups]
    if not completed(campaign, [r["id"] for r in required], "screen"):
        raise ValueError("complete preceding screen stage before continuing")
    campaign.run_many(
        [(r["id"], "screen") for r in conditions() if r["stage_group"] == group]
    )


def promote(campaign, stage):
    if not completed(campaign, [r["id"] for r in conditions()], "screen"):
        raise ValueError("complete all feasible screens before ranking")
    if stage == "120s" and any(
        r["stage"] == "60s" and r["status"] != "COMPLETE"
        for r in campaign.state["runs"]
    ):
        raise ValueError("complete all 60-second attempts before ranking")
    preceding = "screen" if stage == "60s" else "60s"
    limit = campaign.state["specification"]["budgets"][stage]
    decision = campaign.decision(
        "promotions_" + stage,
        lambda: promotion_decision(campaign, preceding, stage, limit),
    )
    write_json(campaign.root / ("ranked_" + stage + ".json"), decision)
    campaign.run_many([(name, stage) for name in decision["selected"]])


def completed(campaign, names, stage):
    return all(
        (campaign.previous(name, stage) or {}).get("status") == "COMPLETE"
        for name in names
    )


def promotion_decision(campaign, preceding, stage, limit):
    decision = ranked_promotions(campaign, preceding, 25)
    existing = {r["candidate"] for r in campaign.state["runs"] if r["stage"] == stage}
    decision["selected"] = [
        name for name in decision["selected"] if name not in existing
    ][: limit - len(existing)]
    return decision


def confirm(campaign):
    selected = campaign.decision(
        "finalists",
        lambda: [
            r["candidate"]
            for r in campaign.state["runs"]
            if r["stage"] == "120s" and long_qualified(r.get("result", {}), 120)
        ][:2],
    )
    campaign.run_many([(name, "repeat") for name in selected])
    campaign.run_many(
        [
            (name, "fine")
            for name in selected
            if long_qualified(campaign.previous(name, "repeat")["result"], 120)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/g1_o6/ladder_material")
    )
    parser.add_argument("--workers", type=int, default=3)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "freeze",
        "prepare",
        "import-reference",
        "A",
        "B",
        "C",
        "promote-60",
        "promote-120",
        "confirm",
        "all",
    ):
        commands.add_parser(name)
    trial = commands.add_parser("trial")
    trial.add_argument("condition")
    trial.add_argument("stage", choices=("screen", "60s", "120s", "repeat", "fine"))
    retry = commands.add_parser("retry")
    retry.add_argument("condition")
    retry.add_argument("stage", choices=("screen", "60s", "120s", "repeat", "fine"))
    retry.add_argument("--reason", required=True)
    args = parser.parse_args()
    with campaign_lock(args.output):
        campaign = LadderCampaign(args.output, args.workers)
        dispatch(campaign, args)


def dispatch(campaign, args):
    actions = {
        "freeze": lambda: write_json(
            args.output / "specification.json", specification()
        ),
        "prepare": campaign.prepare,
        "import-reference": lambda: import_reference(campaign),
        "confirm": lambda: confirm(campaign),
        "all": lambda: run_all(campaign),
        "trial": lambda: campaign.run_many([(args.condition, args.stage)]),
        "retry": lambda: campaign.retry(args.condition, args.stage, args.reason),
    }
    if args.command in ("A", "B", "C"):
        screens(campaign, args.command)
    elif args.command.startswith("promote-"):
        promote(campaign, args.command.split("-")[1] + "s")
    else:
        actions[args.command]()


def run_all(campaign):
    campaign.prepare()
    import_reference(campaign)
    for group in ("A", "B", "C"):
        screens(campaign, group)
    for stage in ("60s", "120s"):
        promote(campaign, stage)
    confirm(campaign)


if __name__ == "__main__":
    main()
