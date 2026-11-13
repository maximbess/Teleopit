"""Immutable single-writer ledger behavior independent of expensive physics."""

import pytest


def prepared_campaign(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support import ladder_campaign as module

    monkeypatch.setattr(module, "code_provenance", lambda: {"code": "test"})
    campaign = module.LadderCampaign(tmp_path)
    campaign.note("reference_physics", {"test": True})
    campaign.note(
        "exact_reference_qpos_inspections",
        {g: {"status": "NOT RUN"} for g in ("G0", "G1", "G2", "G3", "G4", "G5", "G6")},
    )
    for geometry in ("G0", "G1", "G2", "G3", "G4", "G5", "G6"):
        campaign.note("reset_" + geometry, {"status": "INFEASIBLE", "pose": None})
    return campaign


def test_invalid_batch_does_not_leave_reserved_runs(tmp_path, monkeypatch):
    campaign = prepared_campaign(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="preceding"):
        campaign.run_many([("G0_reference", "screen"), ("G1_reference", "60s")])
    assert campaign.state["runs"] == []
    with pytest.raises(ValueError, match="duplicate"):
        campaign.run_many([("G0_reference", "screen")] * 2)
    assert campaign.state["runs"] == []


def test_completed_runs_resume_only_exact_identity_and_checksums(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support.ladder_campaign import run_checksums, run_condition

    campaign = prepared_campaign(tmp_path, monkeypatch)

    def execute(records, workers):
        assert workers == 2
        for record in records:
            path = tmp_path / record["output"]
            result = run_condition(record["payload"], path, 30, 0.001)
            record["files_sha256"] = run_checksums(path)
            campaign.complete(record, result)

    monkeypatch.setattr(campaign, "execute", execute)
    campaign.run_many([("G0_reference", "screen")], workers=2)
    campaign.run_many([("G0_reference", "screen")], workers=2)
    assert len(campaign.state["runs"]) == 1
    campaign.state["candidates"]["G0_reference"]["geometry"] = "G1"
    with pytest.raises(ValueError, match="identity"):
        campaign.run_many([("G0_reference", "screen")])
    campaign.state["candidates"]["G0_reference"]["geometry"] = "G0"
    (tmp_path / "runs/G0_reference_screen/summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        campaign.run_many([("G0_reference", "screen")])


def test_unfinished_runs_and_budget_exhaustion_never_resume_as_pass(
    tmp_path, monkeypatch
):
    campaign = prepared_campaign(tmp_path, monkeypatch)
    campaign.state["runs"] = [
        {"candidate": "G0_reference", "stage": "screen", "status": "RUNNING"}
    ]
    with pytest.raises(ValueError, match="interrupted"):
        campaign.run_many([("G0_reference", "screen")])
    campaign.state["runs"] = [{"candidate": str(i), "stage": "60s"} for i in range(8)]
    with pytest.raises(ValueError, match="budget"):
        campaign.validate_stage("G1_reference", "60s")


def test_reservations_keep_commercial_and_lower_friction_opportunities(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support.ladder_campaign import ranked_promotions

    campaign = prepared_campaign(tmp_path, monkeypatch)
    for condition in campaign.state["candidates"].values():
        result = {
            "strict_status": "PASS",
            "contact_quality": {"status": "PASS"},
            "physics_audit": {
                "no_external_support": True,
                "unintended_support_steps": 0,
                "housing_contact_steps": 0,
            },
            "drift_assessment": {"status": "FAIL", "windows": []},
            "strict": {"peak_effort_fraction": 0.2},
        }
        campaign.state["runs"].append(
            {
                "candidate": condition["id"],
                "stage": "screen",
                "status": "COMPLETE",
                "result": result,
            }
        )
    selection = ranked_promotions(campaign, "screen", 8)
    selected = [campaign.state["candidates"][name] for name in selection["selected"]]
    assert len(selected) == 8
    assert any(c["geometry"] == "G5" for c in selected)
    assert any(c["geometry"] == "G6" for c in selected)
    assert any(c["material"] and c["material"]["scope"] == "H" for c in selected)
    assert any(
        c["geometry"] == "G6" and c["material"] and c["material"]["scope"] == "M"
        for c in selected
    )


def test_worker_executor_writes_complete_immutable_rejections(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from teleopit.sim.g1_o6_support import ladder_campaign as module

    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    campaign = prepared_campaign(tmp_path, monkeypatch)
    campaign.run_many([("G0_reference", "screen"), ("G1_reference", "screen")])
    assert all(r["status"] == "COMPLETE" for r in campaign.state["runs"])
    assert all(
        r["result"]["strict_status"] == "NOT RUN" for r in campaign.state["runs"]
    )
    assert all(r["files_sha256"]["summary.json"] for r in campaign.state["runs"])


def test_technical_retry_preserves_original_and_refuses_physical_failures(
    tmp_path, monkeypatch
):
    campaign = prepared_campaign(tmp_path, monkeypatch)
    name, stage, identity, payload = campaign.plan_requests(
        [("G0_reference", "screen")]
    )[0]
    old = campaign.reserve(name, stage, None, None)
    old.update(identity=identity, payload=payload)
    executed = []
    monkeypatch.setattr(
        campaign, "execute", lambda records, workers: executed.extend(records)
    )
    campaign.retry(name, stage, "process interrupted before completion")
    assert len(campaign.state["runs"]) == 2
    assert executed[0]["retry_of"] == old["id"]
    assert executed[0]["output"] != old["output"]
    executed[0]["status"] = "COMPLETE"
    with pytest.raises(ValueError, match="technical retry"):
        campaign.retry(name, stage, "try again for a pass")
