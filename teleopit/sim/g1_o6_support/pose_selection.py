"""Freeze adaptive choices from measured screens within the declared families."""

from itertools import combinations

from .pose_candidates import (
    controller_candidates,
    offset_and_branch_candidates,
    variant,
)
from .pose_results import campaign_results, rank, read, strict_contact_pass


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
            if r["configuration"]["parent"] == parent_id
            and len(r["configuration"]["changes"]) == 1
            and strict_contact_pass(r)
            and rank(r) < rank(parent)
            and not any((r.get("materially_reduced_hand_load") or {}).values())
        ]
        output.extend(combine_improvements(campaign.candidate(parent_id), children))
    return output[:4]


def adaptive_geometry(campaign):
    results = campaign_results(campaign)
    selected = geometry_parents(results)
    campaign.note("offset_and_ik_parents", selected)
    candidates = offset_and_branch_candidates([campaign.candidate(n) for n in selected])
    combinations = supported_combinations(campaign, results)
    campaign.note("supported_combinations", [c.record() for c in combinations])
    campaign.register([*candidates, *combinations])
    campaign.run_many([(c.id, "screen") for c in [*candidates, *combinations]])


def select_controllers(campaign):
    results = campaign_results(campaign)
    selected = geometry_parents(results)
    campaign.note("controller_parents", selected)
    mapping = read(campaign.root / "runs/reference_screen/resolved.json")[
        "command_joint_mapping"
    ]
    candidates = controller_candidates(
        [campaign.candidate(n) for n in selected], mapping
    )
    campaign.note(
        "conditional_arm_feedback",
        "NOT RUN: use the six isolated finger-law interventions; no separately established arm-first mechanism",
    )
    campaign.register(candidates)
    campaign.run_many([(c.id, "screen") for c in candidates])


def promote_sixty(campaign):
    results = campaign_results(campaign)
    requests = []
    for kind, maximum in (("geometry", 6), ("controller", 6)):
        used = sum(
            r["stage"] == "60s" and campaign.candidate(r["candidate"]).kind == kind
            for r in campaign.state["runs"]
            if not r.get("retry_of")
        )
        eligible = [
            r
            for r in results
            if r["stage"] == "screen"
            and r["configuration"]["kind"] == kind
            and strict_contact_pass(r)
            and campaign.previous(r["candidate"], "60s") is None
        ]
        requests.extend(
            (r["candidate"], "60s")
            for r in sorted(eligible, key=rank)[: max(0, maximum - used)]
        )
    campaign.note("sixty_second_promotions", requests)
    campaign.run_many(requests)


def promote_long(campaign):
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
    campaign.note("first_120_promotions", selected)
    campaign.run_many([(n, "120s") for n in selected])


def finalists(results):
    return [
        r["candidate"]
        for r in results
        if r["stage"] == "120s" and r["result"]["progression_justified"]
    ][:2]


def qualify(campaign):
    selected = finalists(campaign_results(campaign))
    campaign.note(
        "frozen_finalists",
        [
            {"id": n, "configuration_sha256": campaign.candidate(n).digest()}
            for n in selected
        ],
    )
    campaign.run_many([(n, "repeat") for n in selected])
    eligible = [
        n
        for n in selected
        if campaign.previous(n, "repeat")["result"]["progression_justified"]
    ]
    campaign.run_many([(n, "fine") for n in eligible])
