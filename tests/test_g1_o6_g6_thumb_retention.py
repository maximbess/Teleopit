from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_single_corrections_round_trip_and_change_only_the_declared_target():
    from teleopit.sim.g1_o6.grasp_pose import mount_target
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_thumb_retention_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    entries = {c.id: c for c in candidates()}
    targets = {}
    for name, candidate in entries.items():
        assert G6Candidate.from_dict(candidate.record()) == candidate
        env = make_environment(candidate, 5, .001)
        targets[name] = {s: mount_target(env, candidate.pose, s, 7) for s in ("left","right")}
        assert candidate.pose.support_targets() == entries["control"].pose.support_targets()
        assert env.hold_controller.parent.sides == ("right",)
    for name, by_side in targets.items():
        for changed, reference in zip(by_side["left"], targets["control"]["left"]):
            np.testing.assert_array_equal(changed, reference)
        p, r = by_side["right"]
        p0, r0 = targets["control"]["right"]
        displacement = {"inward_2mm": .002, "inward_4mm": .004}.get(name, 0.)
        np.testing.assert_allclose(p - p0, [displacement, 0, 0], atol=1e-12)
        angle = Rotation.from_matrix(r @ r0.T).magnitude()
        assert angle == pytest.approx(.01 if name == "preorient" else 0., abs=1e-10)
        if name == "preorient":
            actual_axis = Rotation.from_matrix(r @ r0.T).as_rotvec() / angle
            np.testing.assert_allclose(actual_axis, [.631418, .545911, .550720], atol=1e-6)
    with pytest.raises(ValueError):
        replace(entries["preorient"].controller, right_refinement="more_travel")


def test_right_proximal_motion_releases_wrist_travel_and_tracks_the_pad():
    from teleopit.sim.g1_o6_support.g6_thumb_retention import proximal_rate

    j = np.column_stack((np.eye(3), np.zeros(3), np.eye(3)))
    offset = np.array([0., 0., 0., 0., .02, 0., 0.])
    requested = np.array([.001, 0., 0.])
    rate = proximal_rate(j, requested, offset)
    assert 0 < rate[0] <= .005
    assert rate[4] < 0
    np.testing.assert_allclose(j @ rate, requested, atol=1e-6)


def test_downstream_cannot_advance_a_run_that_failed_its_controller_specific_gate(tmp_path):
    from types import SimpleNamespace
    from teleopit.sim.g1_o6_support.g6_stages import qualified_pose

    records = [dict(candidate="x", status="COMPLETE", stage=s, output=s, result={"passed":False})
               for s in ("probe","screen","60s","120s","repeat","fine")]
    campaign = SimpleNamespace(root=tmp_path, state={"candidates":["x"], "runs":records},
                               passes=lambda result, stage: result["passed"])
    assert qualified_pose(campaign) == (None, [])


def test_four_mm_depends_on_measured_clearance_and_acquisition(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support.g6_thumb_retention_experiment import ThumbCampaign

    campaign = ThumbCampaign(tmp_path)
    with pytest.raises(ValueError, match="4 mm"):
        campaign.validate_stage("inward_4mm", "probe")
    assert campaign.state["decisions"] == []
    acquired = {"ready_for_hold": True}
    result = {"physics_audit":{"grasp_acquisition":acquired}, "contact_quality":{"status":"PASS"}}
    campaign.state["runs"].append(dict(id="inward_2mm_probe", candidate="inward_2mm", stage="probe",
                                       status="COMPLETE", result=result))
    signals = dict(time=np.array([5.]), centroid_margin=np.array([.035]))
    monkeypatch.setattr(campaign, "signals", lambda record: signals)
    assert campaign.interior_evidence()["eligible"]
    signals["centroid_margin"][0] = .031
    assert not campaign.interior_evidence()["eligible"]
    acquired["ready_for_hold"] = False
    assert not campaign.interior_evidence()["eligible"]


def test_passing_repeat_with_no_explanation_does_not_block_next_hypothesis(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support.g6_thumb_retention_experiment import ThumbCampaign

    campaign = ThumbCampaign(tmp_path)
    monkeypatch.setattr(campaign, "passes", lambda result, stage: True)
    monkeypatch.setattr(campaign, "explained", lambda name: False)
    for name in campaign.first_pair:
        campaign.state["runs"].append(dict(id=name+"_repeat60", status="COMPLETE",
                                           candidate=name, stage="repeat60", result={}))
    assert len(campaign.measured_failures()["actors"]) == 2


def test_repeated_hold_and_explanation_are_both_required_before_120(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support.g6_thumb_retention_experiment import ThumbCampaign

    campaign = ThumbCampaign(tmp_path)
    monkeypatch.setattr(campaign, "passes", lambda result, stage: result["pass"])
    monkeypatch.setattr(campaign, "explained", lambda name: False)
    with pytest.raises(ValueError, match="60 s repeat"):
        campaign.validate_confirmation("preorient", "repeat60")
    for stage in ("60s", "repeat60"):
        campaign.state["runs"].append(dict(candidate="preorient",stage=stage,status="COMPLETE",result={"pass":True}))
    campaign.validate_confirmation("preorient","repeat60")
    with pytest.raises(ValueError, match="explained"):
        campaign.validate_confirmation("preorient","120s")
    monkeypatch.setattr(campaign, "explained", lambda name: True)
    campaign.validate_confirmation("preorient","120s")


def test_matched_mechanism_distinguishes_delayed_saturation_from_drift():
    from teleopit.sim.g1_o6_support.g6_thumb_retention_experiment import matched_sample, mechanism

    times = np.array([5., 6., 40., 52.028, 60.])
    signals = {"time":times, "control_time":times, "fixed":np.zeros((5,3)),
               "centroid_margin":np.full(5,.03), "applied":np.zeros((5,3))}
    signals["applied"][1,0] = .025
    assert mechanism("preorient", signals)["supported"]
    assert mechanism("recenter", signals)["supported"]
    assert mechanism("inward_2mm", signals)["supported"]
    signals["fixed"][3,0] = .008
    assert not mechanism("preorient", signals)["supported"]
    with pytest.raises(ValueError, match="timestamp"):
        matched_sample(signals, "fixed", 30.)
