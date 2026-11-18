"""Verify the actual command and contact evidence at its acquisition frequency."""

import numpy as np
import pytest


def test_balance_uses_signed_environment_contacts_and_gravity():
    from scripts.verify_g1_o6_g6_whole_body import static_residual

    contacts = [
        {"regions":["environment","right_foot"],"force_on_second_world":[1,0,170]},
        {"regions":["left_foot","environment"],"force_on_second_world":[1,0,-148]},
        {"regions":["left_hand","other_robot"],"force_on_second_world":[4,0,200]},
    ]
    np.testing.assert_array_equal(static_residual(contacts, [0,0,-345]), [0,0,-27])


def test_native_body_bounds_reject_early_fast_and_large_applied_targets():
    from scripts.verify_g1_o6_g6_whole_body import body_sample

    baseline = np.zeros(15)
    previous = baseline.copy()
    previous[2] = .07
    current = previous.copy()
    current[2] += .0012
    assert body_sample(current, previous, baseline, .02, True)[0] == pytest.approx(.0712)
    for angle, last, on in ((.0712, .07, False), (.0713, .07, True), (.1202, .12, True)):
        current[2], previous[2] = angle, last
        with pytest.raises(AssertionError):
            body_sample(current, previous, baseline, .02, on)


def test_left_finger_verifier_counts_single_unloaded_step_on_intended_tread():
    from scripts.verify_g1_o6_g6_whole_body import left_loading

    digits = ("index", "middle", "ring", "pinky")
    def contact(digit, force, rung):
        return {"geoms": ["left_o6_geom", rung], "regions": ["left_hand", "environment"],
                "bodies": ["left_o6_lh_" + digit + "_distal", "ladder"],
                "normal_force": force, "normal_on_second_world": [0,0,1], "position": [0,0,1]}
    target = "left_ladder_rung_07"
    loaded = [contact(d, .8, target) for d in digits]
    missing = [*loaded[:-1], contact("pinky", 5., "left_ladder_rung_06"), contact("pinky", .049, target)]
    def row(phase, contacts):
        return {"phase": phase, "contacts": contacts, "support_targets": {"left_hand": target}}
    values = left_loading([row("ACQUIRE", []), row("HOLD", loaded),
                           row("HOLD", missing), row("HOLD", loaded)], .001)
    assert values["hold_steps"] == 3
    assert values["unloaded_steps"] == {"index":0, "middle":0, "ring":0, "pinky":1}
    assert values["minimum_N"]["pinky"] == 0
    assert values["first_unloaded_hold_s"]["pinky"] == .002
