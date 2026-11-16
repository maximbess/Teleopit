"""Synthetic scheduler tests never stand in for physical campaign evidence."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate, batch_candidates


def test_frozen_campaign_retains_negative_runs_and_rejects_changed_source(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import g6_campaign as module
    from teleopit.sim.g1_o6_support import ladder_campaign

    monkeypatch.setattr(module, "code_provenance", lambda: {"source": "one"})
    monkeypatch.setattr(ladder_campaign, "ProcessPoolExecutor", ThreadPoolExecutor)
    campaign = module.G6Campaign(tmp_path, workers=1)
    campaign.note("reference_physics", {"unchanged": True})

    def rejected(payload, output, seconds, dt):
        output.mkdir(parents=True)
        result = {"strict_status": "FAIL", "progression_justified": False}
        (output / "summary.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(campaign, "trial_function", rejected)
    names = campaign.freeze_batch("A", "control")
    campaign.run_many([(names[0], "probe")])
    original = json.loads(campaign.path.read_text())
    campaign.run_many([(names[0], "probe")])
    assert len(campaign.state["runs"]) == 1
    assert campaign.state["runs"] == original["runs"]
    with pytest.raises(ValueError, match="preceding"):
        campaign.run_many([(names[0], "screen")])
    with pytest.raises(ValueError, match="technical retry"):
        campaign.retry(names[0], "probe", "physical failure cannot be retried")
    module.promote(campaign)
    assert not campaign.decision("promote_60s", lambda: ["should not run"])
    assert module.run_batch(campaign, "A", "control") == "a_00"
    assert module.best_parent(campaign, []) == "control"
    assert len(campaign.state["runs"]) == 12
    monkeypatch.setattr(module, "code_provenance", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="changed"):
        campaign.run_many([(names[1], "probe")])


def test_technical_recovery_preserves_error_and_allocates_new_attempt(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import g6_campaign as module
    from teleopit.sim.g1_o6_support import ladder_campaign

    monkeypatch.setattr(module, "code_provenance", lambda: {})
    monkeypatch.setattr(ladder_campaign, "ProcessPoolExecutor", ThreadPoolExecutor)
    campaign = module.G6Campaign(tmp_path, workers=1)
    campaign.note("reference_physics", {})
    campaign.register(batch_candidates("A", G6Candidate("control")))

    def error(*args):
        raise OSError("synthetic disk interruption")

    monkeypatch.setattr(campaign, "trial_function", error)
    with pytest.raises(RuntimeError, match="technical failures"):
        campaign.run_many([("a_00", "probe")])
    assert campaign.state["runs"][0]["status"] == "ERROR"

    def recovered(payload, output, seconds, dt):
        output.mkdir(parents=True)
        (output / "summary.json").write_text("{}")
        return {"strict_status": "FAIL", "progression_justified": False}

    monkeypatch.setattr(campaign, "trial_function", recovered)
    campaign.retry("a_00", "probe", "synthetic storage recovered")
    assert [r["status"] for r in campaign.state["runs"]] == ["ERROR", "COMPLETE"]
    assert campaign.state["runs"][1]["retry_of"] == "a_00_probe"


def test_downstream_run_stops_without_a_qualified_pose(tmp_path):
    from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign
    from teleopit.sim.g1_o6_support.g6_pipeline import Pipeline

    result = Pipeline(G6Campaign(tmp_path)).run()
    assert result["load_bearing_grasp"]["status"] == "BLOCKED"
    assert not (tmp_path / "stages").exists()
    assert json.loads((tmp_path / "milestones.json").read_text()) == result


def test_report_requires_both_orders_and_all_fine_confirmations():
    from teleopit.sim.g1_o6_support.g6_stages import (
        downstream_milestones,
        transfer_schedule,
    )

    requests = transfer_schedule()
    runs = [
        {
            "path": str(i),
            "metadata": {
                "experiment": {"physics_dt": 0.001, **r},
                "config": {"physics_dt": r.get("physics_dt", 0.001)},
            },
            "summary": {
                "physical_status": "PASS",
                "capacity_status": "PASS",
                "plateau_convergence": {"status": "PASS"},
                "load_measurements": {"continuous_in_band_s": 60},
            },
        }
        for i, r in enumerate(requests)
    ]
    proof = SimpleNamespace(qualified_transfers=lambda: runs)
    report = downstream_milestones({}, proof)
    assert all(r["status"] == "PASS" for r in report.values())
    runs.pop()
    assert (
        downstream_milestones({}, proof)["adjacent_tread_transfer"]["status"] != "PASS"
    )


def test_state_capacity_declares_measured_and_planned_bounds_before_running(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import g6_capacity as module
    from test_g1_o6_g6_state_evidence import sample

    targets = {s + "_hand": "left_ladder_rung_07" for s in ("left", "right")}
    source = SimpleNamespace(
        path=tmp_path,
        metadata={"support_targets": targets},
        model=SimpleNamespace(
            body=lambda _: SimpleNamespace(id=1),
            body_subtreemass=[0, 34.08],
            opt=SimpleNamespace(gravity=[0, 0, -9.81]),
        ),
    )
    monkeypatch.setattr(module, "qualified_source", lambda *a, **k: source)
    monkeypatch.setattr(module, "read_rows", lambda _: [sample()])
    plan = module.capacity_plan(tmp_path)
    assert plan["bounds"]["force_N"] > 100
    assert {s["side"] for s in plan["states"]} == {"left", "right"}

    def failed(source, side, path, refs, dt, budget):
        assert (path.parent / "declared_matrix.json").exists()
        return {
            "trials": [{}],
            "loading_status": "BLOCKED",
            "physical_status": "FAIL",
            "reserve_qualified": False,
        }

    monkeypatch.setattr(module, "evaluate_side", failed)
    result = module.run_g6_capacity(tmp_path, tmp_path / "capacity")
    assert result["physical_status"] == "FAIL"
    assert len(result["states"]) == 1


def test_short_native_g6_trial_exports_owned_scene_and_diagnostics(tmp_path):
    from teleopit.sim.g1_o6_support.g6_trial import make_environment, run_candidate
    from teleopit.sim.g1_o6_support.ladder_identity import physical_identity

    candidate = G6Candidate("software_fixture")
    env = make_environment(candidate, 0.02, 0.001)
    payload = {
        "candidate": candidate.record(),
        "reference_physics": physical_identity(env.model),
    }
    result = run_candidate(payload, tmp_path / "trial", 0.02, 0.001)
    assert not result["progression_justified"]
    assert result["physics_audit"]["no_external_support"]
    assert result["physics_audit"]["native_hand_friction"]["status"] == "PASS"
    assert (tmp_path / "trial/physics_contacts.jsonl.gz").stat().st_size > 0
    assert (tmp_path / "trial/body_contact_events.json").exists()
