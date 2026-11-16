"""A bounded feedback search cannot skip its premise audit or physical gates."""

import json
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_seek_requires_two_real_failures_and_an_immutable_complete_audit(tmp_path):
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import FeedbackCampaign

    campaign = FeedbackCampaign(tmp_path)
    assert len(campaign.state["candidates"]) == 5
    with pytest.raises(ValueError, match="premise"):
        campaign.validate_stage("seek_slow", "probe")
    audit = {"failed_run_ids": ["retain_slow_probe", "retain_fast_probe"],
             "premise": "An acquired but invalid wrist reference can preserve opposition.",
             "removed_assignment": "Do not retain an invalid grasp reference; seek the measured thumb gap during acquisition.",
             "observed_actors": {"hands": 2, "unloaded_thumbs": 2, "feet": 2}}
    with pytest.raises(ValueError, match="both completed"):
        campaign.record_audit(audit)
    campaign.state["runs"] = [
        {"id": name + "_probe", "candidate": name, "stage": "probe", "status": "COMPLETE",
         "result": {"strict_status": "FAIL", "contact_quality": {"status": "FAIL"}}}
        for name in ("retain_slow", "retain_fast")]
    with pytest.raises(ValueError, match="observed_actors"):
        campaign.record_audit({**audit, "observed_actors": {}})
    campaign.record_audit(audit)
    campaign.validate_stage("seek_slow", "probe")
    assert FeedbackCampaign(tmp_path).state == campaign.state
    with pytest.raises(ValueError, match="immutable"):
        campaign.record_audit({**audit, "premise": "changed later"})
    with pytest.raises(ValueError, match="preceding"):
        campaign.validate_stage("seek_slow", "screen")
    with pytest.raises(ValueError, match="budget"):
        campaign.register([replace(campaign.candidate("control"), id="extra")])


def test_feedback_identity_and_resume_reject_controller_changes(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support import g6_campaign, ladder_campaign
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import FeedbackCampaign
    from teleopit.sim.g1_o6_support.g6_arm import ArmFeedback

    monkeypatch.setattr(g6_campaign, "code_provenance", lambda: {"source": "frozen"})
    monkeypatch.setattr(ladder_campaign, "ProcessPoolExecutor", ThreadPoolExecutor)
    campaign = FeedbackCampaign(tmp_path, workers=1)
    campaign.note("reference_physics", {})
    def failed(payload, output, seconds, dt):
        output.mkdir(parents=True)
        result = {"strict_status": "FAIL", "contact_quality": {"status": "FAIL"}, "progression_justified": False}
        (output / "summary.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(campaign, "trial_function", failed)
    campaign.run_many([("retain_slow", "probe")])
    original = json.loads(campaign.path.read_text())
    campaign.run_many([("retain_slow", "probe")])
    assert campaign.state["runs"] == original["runs"]
    payload = campaign.state["runs"][0]["payload"]
    assert "g6_feedback_experiment" in payload["command"]
    assert payload["candidate"]["controller"]["rate_rad_s"] == .005
    with pytest.raises(ValueError, match="immutable"):
        campaign.register([replace(campaign.candidate("retain_slow"), controller=ArmFeedback("retain", .02))])
    with pytest.raises(ValueError, match="technical retry"):
        campaign.retry("retain_slow", "probe", "physical failure")


def test_imported_lower_control_keeps_its_failed_probe_without_a_new_attempt(tmp_path):
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import FeedbackCampaign

    path = tmp_path / "reference"
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"strict_status": "FAIL"}))
    campaign = FeedbackCampaign(tmp_path)
    campaign.import_control(path, stage="probe")
    campaign.import_control(path, stage="probe")
    assert len(campaign.state["runs"]) == 1
    assert campaign.state["runs"][0]["id"] == "control_probe"


def test_short_feedback_trial_preserves_native_scene_and_reports_actual_activation(tmp_path):
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment, run_candidate
    from teleopit.sim.g1_o6_support.ladder_identity import physical_identity

    candidate = candidates()[1]
    env = make_environment(candidate, .02, .001)
    payload = {"candidate": candidate.record(), "reference_physics": physical_identity(env.model)}
    result = run_candidate(payload, tmp_path / "trial", .02, .001)
    assert not result["progression_justified"]
    assert result["physics_audit"]["no_external_support"]
    assert result["physics_audit"]["native_hand_friction"]["status"] == "PASS"
    meta = json.loads((tmp_path / "trial/resolved.json").read_text())
    assert meta["controller"]["schema"] == "g6_acquisition_arm_feedback_v1"
    assert meta["material"] is None and meta["config"]["friction_scale"] == 1
    rows = [json.loads(line) for line in (tmp_path / "trial/steps.jsonl").read_text().splitlines()]
    state = rows[-1]["pose_search"]["controller_state"]
    assert 3 <= state["activation_time_s"] < 3.1
    assert max(state["command_displacement_m"].values()) <= .005 + 1e-9
    from scripts.g1_o6_g6_arm_report import verify_run

    checked = verify_run(tmp_path / "trial", tmp_path / "commands.json")
    assert checked["status"] == "PASS" and checked["samples"] > 250
