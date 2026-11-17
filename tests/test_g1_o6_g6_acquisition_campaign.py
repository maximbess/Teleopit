"""The finite acquisition search cannot skip audits or reuse an absent grasp."""

from dataclasses import replace

import pytest


def failed(name, *, acquired=False):
    return {"id": name + "_probe", "candidate": name, "stage": "probe", "status": "COMPLETE",
            "result": {"strict_status": "FAIL", "contact_quality": {"status": "FAIL"},
                "physics_audit": {"grasp_acquisition": {"ready_for_hold": acquired},
                    "no_external_support": True, "unintended_support_steps": 0,
                    "housing_contact_steps": 0, "native_hand_friction": {"status": "PASS"}}}}


def test_two_failures_and_measured_immutable_audit_precede_each_new_pair(tmp_path):
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import AcquisitionCampaign

    campaign = AcquisitionCampaign(tmp_path)
    assert len(campaign.state["candidates"]) == 7
    assert campaign.state["specification"]["candidate_limit"] == 9
    with pytest.raises(ValueError, match="audit"):
        campaign.validate_stage("first_025", "probe")
    audit = {"failed_run_ids": ["pitch_025_probe", "pitch_035_probe"],
             "premise": "Thumb pitch alone establishes opposition with synchronous closing.",
             "observed_actors": {"example_fixture": True, "hands": 2},
             "removed_assignment": "Separate thumb seating from finger closure timing."}
    with pytest.raises(ValueError, match="two completed"):
        campaign.record_audit("pitch", audit)
    campaign.state["runs"] = [failed("pitch_025"), failed("pitch_035")]
    with pytest.raises(ValueError, match="observed_actors"):
        campaign.record_audit("pitch", {**audit, "observed_actors": {}})
    campaign.record_audit("pitch", audit)
    campaign.validate_stage("first_025", "probe")
    with pytest.raises(ValueError, match="immutable"):
        campaign.record_audit("pitch", {**audit, "premise": "changed"})
    with pytest.raises(ValueError, match="audit"):
        campaign.validate_stage("yaw_020", "probe")
    with pytest.raises(ValueError, match="preceding"):
        campaign.validate_stage("pitch_025", "screen")


def test_stabilization_requires_measured_acquisition_and_is_derived_only_once(tmp_path):
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import AcquisitionCampaign

    campaign = AcquisitionCampaign(tmp_path)
    campaign.state["runs"] = [failed("pitch_025"), failed("pitch_035")]
    with pytest.raises(ValueError, match="acquired"):
        campaign.select_stabilization()
    campaign.state["runs"][0] = failed("pitch_025", acquired=True)
    names = campaign.select_stabilization()
    assert names == ["stabilize_slow", "stabilize_fast"]
    assert campaign.select_stabilization() == names
    assert len(campaign.state["candidates"]) == 9
    for name in names:
        child = campaign.candidate(name)
        assert child.parent == "pitch_025"
        assert child.pose == campaign.candidate("pitch_025").pose
        assert child.reset == {"root_offset_x": -.04}
        assert child.controller.arm_rate_rad_s in (.005, .02)
    with pytest.raises(ValueError, match="budget"):
        campaign.register([replace(campaign.candidate("control"), id="tenth")])
    assert AcquisitionCampaign(tmp_path).state == campaign.state


def test_short_acquisition_trial_reports_contact_phases_without_qualifying_a_hold(tmp_path):
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import candidates, run_acquisition_candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.ladder_identity import physical_identity

    candidate = candidates()[1]
    env = make_environment(candidate, .02, .001)
    payload = {"candidate": candidate.record(), "reference_physics": physical_identity(env.model)}
    result = run_acquisition_candidate(payload, tmp_path / "trial", .02, .001)
    audit = result["physics_audit"]
    assert not result["progression_justified"]
    assert audit["no_external_support"]
    assert audit["native_hand_friction"]["status"] == "PASS"
    assert set(audit["grasp_acquisition"]["phases"]) == {"INITIALIZE", "ACQUIRE", "REMOVE_ASSISTANCE", "HOLD"}
    assert audit["grasp_acquisition"]["phases"]["HOLD"]["samples"] == 20
    from scripts.g1_o6_g6_acquisition_report import verify_run

    checked = verify_run(tmp_path / "trial", tmp_path / "commands.json")
    assert checked["anatomy"]["status"] == "PASS"
    assert checked["commands"]["maximum_target_rate_rad_s"] <= 1 + 1e-8


def test_stabilization_can_use_an_actually_valid_unassisted_grasp_acquired_during_removal():
    from teleopit.sim.g1_o6_support.g6_acquisition_experiment import acquired

    record = failed("pitch_025")
    record["result"].update(contact_quality={"status": "PASS"}, strict={"unsupported_hold_seconds": 5.})
    assert acquired(record)
    record["result"]["strict"]["unsupported_hold_seconds"] = .02
    assert not acquired(record)
