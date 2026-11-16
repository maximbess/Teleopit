import json

import pytest

from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate, batch_candidates


def test_g6_campaign_freezes_candidates_and_refuses_failed_promotion(tmp_path):
    from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign

    campaign = G6Campaign(tmp_path)
    campaign.register(batch_candidates("A", G6Candidate("control")))
    assert len(campaign.state["candidates"]) == 13
    campaign.freeze_batch("A", "control")
    reopened = G6Campaign(tmp_path)
    assert reopened.state == campaign.state
    with pytest.raises(ValueError, match="preceding"):
        campaign.validate_stage("a_00", "screen")
    with pytest.raises(ValueError, match="immutable"):
        campaign.register([G6Candidate("a_00")])


def test_campaign_resume_checks_every_evidence_file(tmp_path):
    from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign

    campaign = G6Campaign(tmp_path)
    record = {
        "status": "COMPLETE",
        "identity": "abc",
        "output": "run",
        "files_sha256": {"summary.json": "wrong"},
    }
    (tmp_path / "run").mkdir()
    (tmp_path / "run/summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        campaign.verify_resume(record, "abc")
    record["status"] = "ERROR"
    with pytest.raises(ValueError, match="technical recovery"):
        campaign.verify_resume(record, "abc")


def test_downstream_is_blocked_without_stable_pose(tmp_path):
    from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign
    from teleopit.sim.g1_o6_support.g6_stages import milestone_report

    campaign = G6Campaign(tmp_path)
    report = milestone_report(campaign)
    assert report["initial_pose"]["status"] == "NOT RUN"
    assert all(
        value["status"] == "BLOCKED"
        for key, value in report.items()
        if key != "initial_pose"
    )
    json.dumps(report, allow_nan=False)


def test_capacity_budget_exhaustion_is_not_a_pass():
    from teleopit.sim.g1_o6_support.capacity_matrix import summarize_matrix

    result = {
        "trials": [],
        "out_of_budget_cases": [{"name": "required"}],
        "unrun_boundary_cases": 0,
    }
    summarize_matrix(result, [{}])
    assert result["loading_status"] == "BLOCKED"
    assert not result["reserve_qualified"]
