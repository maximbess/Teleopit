"""Support accounting and stage gates cannot pass from commands alone."""

import numpy as np


def test_signed_hand_share_uses_weight_and_excludes_internal_wrist_loads():
    from teleopit.sim.g1_o6_support.measurements import load_shares

    supports = {
        name: {"force_world": force}
        for name, force in {
            "left_hand": [100, 0, 20],
            "right_hand": [100, 0, -5],
            "left_foot": [0, 0, 90],
            "right_foot": [0, 0, 95],
        }.items()
    }
    row = {"supports": supports, "wrist_world_on_hand": {"left": [999] * 6}}
    result = load_shares(row, 20, 10)
    np.testing.assert_allclose(result["total_hand_share"], 0.075, rtol=0, atol=1e-15)
    assert result["per_hand_share"] == {"left": 0.1, "right": -0.025}


def test_late_drift_uses_vector_motion_not_only_distance_from_reset():
    from teleopit.sim.g1_o6_support.measurements import window_motion

    time = np.linspace(0, 30, 301)
    # Constant radial distance conceals tangential movement.
    points = np.c_[0.02 * np.cos(time / 60), 0.02 * np.sin(time / 60), time * 0]
    result = window_motion(time, points)
    assert result["slope_per_s"] > 0.0003
    assert result["excursion"] > 0.009


def test_short_or_missing_windows_cannot_qualify_convergence():
    from teleopit.sim.g1_o6_support.measurements import late_drift

    assert late_drift([], 120)["status"] == "NOT RUN"


def test_opening_reduces_measured_start_without_a_closing_preload():
    from teleopit.sim.g1_o6.holding import opening_targets

    start = np.array([0.04, 0.45, 1.0, 1.1, 1.2, 1.3])
    np.testing.assert_array_equal(opening_targets(start, 0), start)
    np.testing.assert_array_equal(opening_targets(start, 0.5), 0.5 * start)
    np.testing.assert_array_equal(opening_targets(start, 1), np.zeros(6))
