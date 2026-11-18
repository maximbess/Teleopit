import numpy as np
import pytest


def test_raw_opposition_needs_opposite_loaded_digit_normals_on_the_intended_tread():
    from scripts.verify_g1_o6_g6_grip_feedback import grasp_sample

    def contact(digit, normal, force=1., rung="rung7"):
        return {"regions": ["environment", "right_hand"], "geoms": [rung, digit],
                "bodies": [rung, "right_o6_rh_" + digit + "_link"],
                "normal_on_second_world": [0., 0., normal], "position": [0., 0., 0.], "normal_force": force}

    row = {"support_targets": {"right_hand": "rung7"}, "contacts": [contact("thumb", -1.), contact("index", 1.)]}
    loads, opposed = grasp_sample(row, "right")
    assert opposed and loads.tolist() == [1., 1., 0., 0., 0.]
    for ineffective in (contact("index", 1., .049), contact("index", -1.), contact("index", 1., rung="rung3")):
        row["contacts"][1] = ineffective
        assert not grasp_sample(row, "right")[1]


def test_independent_cap_check_rejects_false_yaw_motion_and_wrong_force_direction():
    from scripts.verify_g1_o6_g6_grip_feedback import check_cap

    anchor = np.full((2, 6), .4)
    loads = np.tile([.7, .5, 1.4, 1.4, 1.4], (2, 1))
    spec = {"law": "pressure", "pressure_gain_rad_per_Ns": .12}
    expected = anchor.copy()
    expected[:, 2] += .0008
    args = (anchor, anchor, anchor + .02, loads, [True, True], np.zeros(2), .02, spec, False)
    check_cap(expected, *args)
    for axis in (1, 2):
        altered = expected.copy()
        altered[:, axis] -= .0008
        with pytest.raises(ValueError, match="feedback cap"):
            check_cap(altered, *args)
    check_cap(anchor, anchor, anchor, anchor, loads, [False, True], np.zeros(2), .02, spec, True)


def test_reconstructed_contacts_reject_a_missing_solver_loss_or_an_invented_sample():
    from scripts.verify_g1_o6_g6_grip_feedback import ContactEvidence

    record = ContactEvidence()
    record.observe({"support_targets": {"left_hand": "rung7", "right_hand": "rung7"}, "contacts": []}, .001)
    saved = {"samples": 1, "loss_steps": dict(record.losses), "first_loss_s": dict(record.first)}
    assert record.verify(saved)["minimum_N"]["right"]["thumb"] == 0
    for key, bad in (("samples", 2), ("loss_steps", {**record.losses, "right_thumb": 0}),
                     ("first_loss_s", {**record.first, "left_thumb": .002})):
        with pytest.raises(AssertionError):
            record.verify({**saved, key: bad})
