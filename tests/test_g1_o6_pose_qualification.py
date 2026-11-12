"""Qualification must remain a conjunction of the unchanged physical gates."""

from dataclasses import replace

import pytest


def qualified_records(candidate):
    return [
        {
            "candidate": candidate.id,
            "stage": stage,
            "status": "COMPLETE",
            "configuration_sha256": candidate.digest(),
            "result": {"progression_justified": True},
        }
        for stage in ("120s", "repeat", "fine")
    ]


def test_robustness_is_blocked_until_same_identity_passes_all_three_runs(tmp_path):
    from teleopit.sim.g1_o6_support.pose_campaign import Campaign
    from teleopit.sim.g1_o6_support.pose_candidates import Candidate

    campaign = Campaign(tmp_path)
    parent = campaign.candidate("reference")
    candidate = Candidate(
        "nearby",
        parent=parent.id,
        kind="robustness",
        reset_offsets={"root_offset_x": 0.002},
    )
    campaign.register([candidate])
    with pytest.raises(ValueError, match="qualified parent"):
        campaign.reserve_many([(candidate.id, "robustness")])
    records = qualified_records(parent)
    campaign.state["runs"] = records[:2]
    with pytest.raises(ValueError, match="qualified parent"):
        campaign.reserve_many([(candidate.id, "robustness")])
    campaign.state["runs"] = records
    records[-1]["configuration_sha256"] = "different-controller"
    with pytest.raises(ValueError, match="qualified parent"):
        campaign.reserve_many([(candidate.id, "robustness")])
    records[-1]["configuration_sha256"] = parent.digest()
    assert len(campaign.reserve_many([(candidate.id, "robustness")])) == 1


def test_robustness_configuration_changes_only_one_declared_reset_offset(tmp_path):
    from teleopit.sim.g1_o6_support.pose_campaign import Campaign
    from teleopit.sim.g1_o6_support.pose_candidates import Candidate

    campaign = Campaign(tmp_path)
    parent = campaign.candidate("reference")
    campaign.state["runs"] = qualified_records(parent)
    changed = Candidate(
        "retuned",
        parent=parent.id,
        kind="robustness",
        controller="freeze_all",
        reset_offsets={"wrist_offset_x": -0.002},
    )
    campaign.register([changed])
    with pytest.raises(ValueError, match="frozen parent"):
        campaign.reserve_many([(changed.id, "robustness")])
    for offset in (0.003, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="2 mm"):
            replace(changed, reset_offsets={"wrist_offset_x": offset})
    with pytest.raises(ValueError, match="one declared"):
        replace(
            changed, reset_offsets={"wrist_offset_x": 0.002, "root_offset_x": 0.002}
        )
    with pytest.raises(ValueError, match="robustness"):
        replace(changed, kind="geometry")


def test_finalist_cannot_disagree_with_its_contact_verdict():
    from teleopit.sim.g1_o6_support.pose_selection import finalists

    result = {
        "candidate": "contradictory",
        "stage": "120s",
        "strict": "PASS",
        "result": {
            "progression_justified": True,
            "contact_quality": {"status": "FAIL"},
        },
    }
    assert finalists([result]) == []


def test_transfer_long_hold_gate_preserves_new_contact_failure():
    from copy import deepcopy

    from teleopit.sim.g1_o6_support.qualification import long_hold_gate

    result = {
        "strict_status": "PASS",
        "drift_assessment": {"status": "PASS"},
        "contact_quality": {"status": "PASS"},
        "physics_audit": {
            "no_external_support": True,
            "unintended_support_steps": 0,
            "housing_contact_steps": 0,
        },
    }
    runs = [
        {
            "path": str(i),
            "metadata": {
                "experiment": {"requested_hold_s": 120},
                "config": {"physics_dt": dt},
            },
            "summary": deepcopy(result),
        }
        for i, dt in enumerate((0.001, 0.001, 0.0005))
    ]
    assert long_hold_gate(runs)
    runs[-1]["summary"]["contact_quality"]["status"] = "FAIL"
    assert not long_hold_gate(runs)
