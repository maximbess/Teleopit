"""Shared qualification identity for nearby-condition validation and reports."""

STAGES = ("120s", "repeat", "fine")


def stage_qualification(records, name, digest):
    matched = [
        r
        for r in records
        if r["candidate"] == name
        and r["status"] == "COMPLETE"
        and r["configuration_sha256"] == digest
    ]
    return {
        stage: any(
            r["stage"] == stage and r["result"].get("progression_justified")
            for r in matched
        )
        for stage in STAGES
    }


def qualified_candidate(campaign, name):
    return all(
        stage_qualification(
            campaign.state["runs"], name, campaign.candidate(name).digest()
        ).values()
    )


def validate_robustness(campaign, candidate):
    if candidate.parent not in campaign.state["candidates"] or not qualified_candidate(
        campaign, candidate.parent
    ):
        raise ValueError(
            "robustness requires a qualified parent with matching identity"
        )
    parent = campaign.candidate(candidate.parent)
    if (candidate.pose, candidate.controller, candidate.ik_seed) != (
        parent.pose,
        parent.controller,
        parent.ik_seed,
    ):
        raise ValueError("robustness must retain the frozen parent pose and controller")


def robustness(campaign):
    from dataclasses import replace

    from .pose_results import campaign_results

    selected = campaign.decision(
        "robustness_parent",
        lambda: next(
            (
                r["candidate"]
                for r in campaign_results(campaign)
                if qualified_candidate(campaign, r["candidate"])
            ),
            None,
        ),
    )
    if selected is None:
        campaign.note(
            "robustness_status", "BLOCKED: no nominal/repeat/fine qualified parent"
        )
        return
    parent = campaign.candidate(selected)
    candidates = [
        replace(
            parent,
            id=f"{selected}_robustness_{i + 1}",
            parent=selected,
            kind="robustness",
            family="robustness",
            reset_offsets=offset,
            changes={"reset_offsets": offset},
        )
        for i, offset in enumerate(
            campaign.state["specification"]["robustness"]["trials"]
        )
    ]
    campaign.register(candidates)
    campaign.run_many([(c.id, "robustness") for c in candidates])
