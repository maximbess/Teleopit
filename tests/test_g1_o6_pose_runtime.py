"""Exercise run boundaries and real scored diagnostics without long qualification."""

import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_scored_observer_records_actual_contacts_without_applied_support(tmp_path):
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.pose_observer import PoseObserver

    env = Environment(Config(hold_seconds=0.1))
    while env.phase not in ("HOLD", "ABORT"):
        env.step()
    assert env.phase == "HOLD"
    observer = PoseObserver(env, tmp_path)
    env.step(observer)
    row = observer.control_sample(env)
    observer.close()
    assert observer.report()["no_external_support"]
    assert observer.contact_report()["status"] == "PASS"
    for hand in row["hands"].values():
        assert hand["quality"]["thumb_opposition"]
        assert hand["quality"]["ordered"]
        assert hand["contacts"] and all(c["target"] for c in hand["contacts"])
    triggers = json.loads((tmp_path / "physics_window_triggers.json").read_text())
    assert "hold_entry" in triggers[0]["signals"]
    assert (tmp_path / "physics_windows.jsonl.gz").stat().st_size > 0


def test_error_retry_preserves_original_attempt_and_physical_failure_is_final(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import pose_campaign

    campaign = pose_campaign.Campaign(tmp_path)
    calls = []

    def runner(candidate, path, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("demonstrated software fixture error")
        return {"strict_status": "FAIL", "progression_justified": False}

    monkeypatch.setattr(pose_campaign, "run_candidate", runner)
    with pytest.raises(OSError):
        campaign.run("reference", "screen")
    with pytest.raises(ValueError, match="documented"):
        campaign.run("reference", "screen")
    campaign.run("reference", "screen", "software fixture error corrected")
    assert len(campaign.state["runs"]) == 2 and calls[0] != calls[1]
    assert campaign.state["runs"][0]["status"] == "ERROR"
    assert campaign.state["runs"][1]["retry_of"] == "reference_screen"
    assert campaign.run("reference", "screen")["strict_status"] == "FAIL"
    assert len(calls) == 2


def test_single_writer_lock_and_mandatory_duration_stop(tmp_path):
    from teleopit.sim.g1_o6_support.pose_campaign import (
        campaign_lock,
        initial_screen,
        mandatory,
    )

    with campaign_lock(tmp_path), pytest.raises(OSError), campaign_lock(tmp_path):
        pass
    with campaign_lock(tmp_path):
        pass
    calls = []

    def run(name, stage):
        calls.append((name, stage))
        return {
            "strict_status": "FAIL" if name == "rung10" or stage == "120s" else "PASS"
        }

    campaign = SimpleNamespace(
        run=run,
        previous=lambda *a: {"result": {"strict_status": "PASS"}},
        run_many=lambda jobs: calls.extend(jobs),
    )
    mandatory(campaign)
    assert ("rung10", "60s") not in calls and ("reference", "120s") in calls
    calls.clear()
    initial_screen(campaign)
    assert len(calls) == 17 and calls[0] == ("reference", "120s")


def test_prior_reset_references_preserve_the_exact_resolved_state(tmp_path):
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.pose_candidates import Candidate
    from teleopit.sim.g1_o6_support.pose_trial import SearchEnvironment, prior_resets

    env = SearchEnvironment(Candidate("fixture"), 0.1, 0.001)
    path = tmp_path / "fixture"
    path.mkdir()
    (path / "resolved.json").write_text(
        json.dumps(
            {
                "candidate": env.candidate.record(),
                "controller": env.hold_controller.metadata(env),
                "initial": snapshot(env),
            }
        )
    )
    records = [
        {
            "candidate": "fixture",
            "stage": "screen",
            "status": "COMPLETE",
            "output": "fixture",
            "result": {},
        }
    ]
    references = prior_resets(tmp_path, records)
    np.testing.assert_array_equal(references[0]["qpos"], env.data.qpos)
    records[0]["result"]["reset_status"] = "INFEASIBLE"
    assert prior_resets(tmp_path, records) == []


def test_cli_registration_and_robustness_block_before_physics(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support import pose_search
    from teleopit.sim.g1_o6_support.pose_candidates import Candidate

    configuration = tmp_path / "candidate.json"
    configuration.write_text(json.dumps(Candidate("standalone").record()))
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pose_search",
            "--output",
            str(output),
            "register",
            "--candidate-config",
            str(configuration),
        ],
    )
    pose_search.main()
    monkeypatch.setattr(
        "sys.argv", ["pose_search", "--output", str(output), "robustness"]
    )
    pose_search.main()
    state = json.loads((output / "campaign_state.json").read_text())
    assert "standalone" in state["candidates"] and state["runs"] == []
    assert any("BLOCKED" in str(r["value"]) for r in state["decisions"])
