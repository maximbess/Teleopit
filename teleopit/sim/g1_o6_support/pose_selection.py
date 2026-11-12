"""Freeze adaptive choices from measured screens within the declared families."""

from itertools import combinations

from .pose_candidates import (
    Candidate,
    controller_candidates,
    offset_and_branch_candidates,
    variant,
)
from .pose_results import (
    campaign_results,
    rank,
    read,
    series,
    stationary_target_motion,
    strict_contact_pass,
)


def geometry_parents(results):
    eligible = [
        r
        for r in results
        if r["stage"] == "screen"
        and r["configuration"]["kind"] == "geometry"
        and strict_contact_pass(r)
    ]
    return [r["candidate"] for r in sorted(eligible, key=rank)[:2]]


def combine_improvements(parent, children):
    result = []
    for first, second in combinations(children, 2):
        if set(first.changes) & set(second.changes):
            continue
        name = f"{parent.id}_combination_{len(result) + 1}"
        result.append(
            variant(parent, name, "combination", {**first.changes, **second.changes})
        )
    return result


def supported_combinations(campaign, results):
    screens = {r["candidate"]: r for r in results if r["stage"] == "screen"}
    output = []
    for parent_id in ("reference", "rung10"):
        parent = screens[parent_id]
        children = [
            campaign.candidate(r["candidate"])
            for r in screens.values()
            if supported_child(r, parent)
        ]
        output.extend(combine_improvements(campaign.candidate(parent_id), children))
    return output[:4]


def supported_child(result, parent):
    return (
        result["configuration"]["parent"] == parent["candidate"]
        and len(result["configuration"]["changes"]) == 1
        and strict_contact_pass(result)
        and rank(result) < rank(parent)
        and not any((result.get("materially_reduced_hand_load") or {}).values())
    )


def adaptive_geometry(campaign):
    results = campaign_results(campaign)
    selected = campaign.decision(
        "offset_and_ik_parents", lambda: geometry_parents(results)
    )
    candidates = offset_and_branch_candidates([campaign.candidate(n) for n in selected])
    records = campaign.decision(
        "supported_combinations",
        lambda: [c.record() for c in supported_combinations(campaign, results)],
    )
    combinations = [Candidate.from_dict(r) for r in records]
    campaign.register([*candidates, *combinations])
    campaign.run_many([(c.id, "screen") for c in [*candidates, *combinations]])


def select_controllers(campaign):
    results = campaign_results(campaign)
    selected = campaign.decision(
        "controller_parents", lambda: geometry_parents(results)
    )
    mapping = read(campaign.root / "runs/reference_screen/resolved.json")[
        "command_joint_mapping"
    ]
    candidates = controller_candidates(
        [campaign.candidate(n) for n in selected], mapping
    )
    candidates = conditional_arm(campaign, candidates, selected)
    campaign.register(candidates)
    campaign.run_many([(c.id, "screen") for c in candidates])


def conditional_arm(campaign, candidates, parents):
    if not parents:
        return candidates
    evidence = campaign.decision(
        "arm_tracking_evidence", lambda: arm_tracking_evidence(campaign)
    )
    if evidence is None:
        campaign.note(
            "conditional_arm_feedback",
            "NOT RUN: no stationary-target wrist-tracking-error evidence",
        )
        return candidates
    parent = campaign.candidate(parents[0])
    replacement = variant(
        parent,
        parent.id + "_arm_feedback",
        "arm_feedback",
        {},
        kind="controller",
        controller="arm_feedback",
    )
    campaign.note(
        "conditional_arm_feedback",
        {
            "replaces": parent.id + "_preload_0p14",
            "candidate": replacement.record(),
            "evidence": evidence,
            "hypothesis": "Persistent rung-relative wrist tracking error after thumb target settles; this timing alone does not establish an arm-first cause",
        },
    )
    return [
        replacement if c.id == parent.id + "_preload_0p14" else c for c in candidates
    ]


def arm_tracking_evidence(campaign):
    record = campaign.previous("reference", "120s")
    if record is None or record["status"] != "COMPLETE":
        return None
    evidence = stationary_target_motion(series(campaign.root / record["output"]))
    if evidence and any(
        h["thumb_target_range_rad"] <= 1e-4 and h["mount_excursion_m"] > 0.002
        for h in evidence["hands"].values()
    ):
        return {"source": record["output"], **evidence}
    return None


def promote_sixty(campaign):
    requests = campaign.decision(
        "sixty_second_promotions", lambda: sixty_requests(campaign)
    )
    campaign.run_many(requests)


def sixty_requests(campaign):
    results = campaign_results(campaign)
    requests = []
    for kind, maximum in (("geometry", 6), ("controller", 6)):
        used = sum(
            r["stage"] == "60s" and campaign.candidate(r["candidate"]).kind == kind
            for r in campaign.state["runs"]
            if not r.get("retry_of")
        )
        eligible = sixty_eligible(campaign, results, kind)
        requests.extend(
            (r["candidate"], "60s")
            for r in sorted(eligible, key=rank)[: max(0, maximum - used)]
        )
    return requests


def sixty_eligible(campaign, results, kind):
    return [
        r
        for r in results
        if r["stage"] == "screen"
        and r["configuration"]["kind"] == kind
        and strict_contact_pass(r)
        and campaign.previous(r["candidate"], "60s") is None
    ]


def promote_long(campaign):
    selected = campaign.decision(
        "first_120_promotions", lambda: long_requests(campaign)
    )
    campaign.run_many([(n, "120s") for n in selected])


def long_requests(campaign):
    results = campaign_results(campaign)
    used = sum(
        r["stage"] == "120s" and not r.get("retry_of") for r in campaign.state["runs"]
    )
    eligible = [
        r
        for r in results
        if r["stage"] == "60s"
        and strict_contact_pass(r)
        and campaign.previous(r["candidate"], "120s") is None
    ]
    selected = [r["candidate"] for r in sorted(eligible, key=rank)[: max(0, 6 - used)]]
    return selected


def finalists(results):
    return [
        r["candidate"]
        for r in results
        if r["stage"] == "120s"
        and r["result"]["progression_justified"]
        and strict_contact_pass(r)
    ][:2]


def qualify(campaign):
    frozen = campaign.decision(
        "frozen_finalists",
        lambda: [
            {"id": n, "configuration_sha256": campaign.candidate(n).digest()}
            for n in finalists(campaign_results(campaign))
        ],
    )
    selected = [r["id"] for r in frozen]
    campaign.run_many([(n, "repeat") for n in selected])
    eligible = [
        n
        for n in selected
        if campaign.previous(n, "repeat")["result"]["progression_justified"]
    ]
    campaign.run_many([(n, "fine") for n in eligible])
