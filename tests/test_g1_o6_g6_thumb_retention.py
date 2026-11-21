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
