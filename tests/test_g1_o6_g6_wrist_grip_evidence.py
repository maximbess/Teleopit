import numpy as np
import pytest


def test_reconstruction_rejects_a_false_thumb_force_direction_and_a_wrong_joint_rate():
    from scripts.verify_g1_o6_g6_wrist_grip import check_request

    j = np.column_stack((np.zeros((3,4)), np.eye(3)))
    error = np.array([0., 0., -.01])
    velocity = np.array([0., 0., .002])
    rates = np.r_[np.zeros(4), velocity/1.0001]
    saved = {"error_rung_m":error, "velocity_rung_m_s":velocity, "joint_rate_rad_s":rates}
    check_request(saved, j, error, np.zeros(7), 1., False, "right")
    with pytest.raises(ValueError):
        check_request(saved, j, error, np.zeros(7), 1., True, "right")
    with pytest.raises(ValueError):
        check_request({**saved, "joint_rate_rad_s":-rates}, j, error, np.zeros(7), 1., False, "right")


def test_capture_evidence_uses_only_loaded_thumb_contacts_on_the_correct_tread():
    from scripts.verify_g1_o6_g6_wrist_grip import contact_centroid

    def contact(x, force, digit="thumb", rung="rung7"):
        return {"regions":["right_hand","environment"],"geoms":[digit,rung],"bodies":["right_o6_rh_"+digit+"_distal",rung],
                "normal_force":force,"position":[x,0.,0.]}
    contacts = [contact(1.,.05),contact(3.,.15),contact(8.,.049),contact(20.,2.,rung="rung3"),contact(30.,2.,"index")]
    np.testing.assert_allclose(contact_centroid(contacts,"right","rung7"), [2.5,0.,0.])
    with pytest.raises(ValueError):
        contact_centroid(contacts,"left","rung7")


def test_replayed_servo_respects_slew_travel_effort_joint_range_and_cartesian_boundary():
    from scripts.verify_g1_o6_g6_wrist_grip import joint_step, project

    baseline = np.ones(4)
    old = np.array([.0299, 0., -.01, .01])
    limits = np.tile([-.5, 1.1],(4,1))
    limits[2,1] = .9901
    effort = np.array([0., .98, 0., .97])
    step = joint_step(old, np.full(4,1.), baseline, limits, effort, .02)
    np.testing.assert_allclose(step, [.03,0.,-.0099,.0104])
    assert project(0., .002, lambda v: v <= .005) == .002
    assert .004999 < project(.004, .009, lambda v: v <= .005) <= .005
