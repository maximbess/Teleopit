import numpy as np
import pytest


def test_preview_selection_receipt_rejects_a_wrong_winner_and_a_false_executed_thumb():
    from scripts.verify_g1_o6_g6_grip_preview import check_selection, check_thumb_forecast

    def force(peak, finger, left, right):
        return {"peak_N": peak,"finger_N": finger,"left_thumb_N": left,"right_thumb_N": right}
    baseline, unsafe, safe = force(12., 1., 0., .7), force(25., 1., .8, .8), force(15., 1., .51, .52)
    state = {"choice": "raise","extra_evaluations": 2, "nominal_forecast": baseline, "tested_forecasts": [unsafe,safe]}
    check_selection(state, safe, True)
    with pytest.raises(ValueError):
        check_selection({**state, "choice": "thumb"}, safe, True)
    with pytest.raises(ValueError):
        check_selection({**state, "extra_evaluations": 3}, safe, True)
    with pytest.raises(ValueError):
        check_selection(state, safe, False)
    assert check_thumb_forecast(safe, [.51,.52]) < 1e-12
    with pytest.raises(ValueError):
        check_thumb_forecast(safe, [.49,.52])


def test_alternate_applied_ceiling_protects_native_preload_slew_and_independent_thumb_yaw():
    from scripts.verify_g1_o6_g6_grip_preview import check_alternate

    previous = np.full((2,6), .4)
    nominal = previous.copy()
    anchor = previous.copy()
    native = previous + .1
    expected = nominal + .0008
    expected[:,1] = .4
    check_alternate(expected, "raise", nominal, previous, anchor,native,.02)
    altered = expected.copy()
    altered[1,1] += .0008
    with pytest.raises(ValueError):
        check_alternate(altered, "raise", nominal, previous, anchor,native,.02)
    with pytest.raises(ValueError):
        check_alternate(expected,"lower",nominal,previous,anchor,native,.02)
