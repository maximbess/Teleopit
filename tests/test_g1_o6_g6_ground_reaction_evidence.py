"""Independent evidence enforces native sample alignment and peak force."""

import numpy as np
import pytest


def test_interval_includes_midstep_impact_and_intended_finger_dropout():
    from scripts.verify_g1_o6_g6_ground_reaction import native_interval

    target = "left_ladder_rung_07"
    def row(time, foot_force, pinky):
        contacts = [{"regions":["environment","left_foot"],"geoms":["rung","foot"],
                     "force_on_second_world":[0.,0.,foot_force]}]
        for group, force in (("index",1.),("middle",1.),("ring",1.),("pinky",pinky)):
            # A horizontal finger normal isolates this fixture's vertical balance.
            contacts.append({"regions":["environment","left_hand"],"geoms":[target,group],
                             "normal_force":force,"bodies":["world",f"left_o6_{group}_distal"],
                             "normal_on_second_world":[1.,0,0],"force_on_second_world":[force,0,0],
                             "position":[0.,0.,1.]})
        return {"phase":"HOLD","time_s":time,"contacts":contacts, "support_targets":{"left_hand":target}}
    rows = [row(1.001,325.,1.),row(1.002,300.,.04),row(1.003,325.,1.)]
    result = native_interval(iter(rows), 3, 1.003, .001, np.array([0.,0.,-325.]))
    assert result["peak_N"] == pytest.approx(np.hypot(25.,3.04))
    assert result["minimum_left_N"] == 0. and result["violations"] == 1
    with pytest.raises(ValueError, match="interval"):
        native_interval(iter([rows[0],rows[0],rows[2]]), 3, 1.003, .001, np.array([0,0,-325.]))
    with pytest.raises(ValueError, match="interval"):
        native_interval(iter(rows[:2]), 3, 1.003, .001, np.array([0,0,-325.]))


def test_verifier_rejects_a_false_forecast_even_when_the_control_endpoint_passes():
    from scripts.verify_g1_o6_g6_ground_reaction import compare_prediction

    actual = {"peak_N":26., "minimum_left_N":.4}
    saved = {"choice":"nominal","evaluations":1,"forecast":{"peak_N":26., "finger_N":.4}}
    with pytest.raises(AssertionError):
        compare_prediction(saved, actual, (1., .3), 9)
    saved["forecast"]["finger_N"] = .3
    assert compare_prediction(saved, actual, (1., .3), 9) == 0.
    saved["forecast"]["peak_N"] = 1.
    with pytest.raises(AssertionError):
        compare_prediction(saved, actual, (1., .3), 9)
    saved["forecast"]["peak_N"] = 26.
    saved["evaluations"] = 10
    with pytest.raises(AssertionError):
        compare_prediction(saved, actual, (1., .3), 9)
