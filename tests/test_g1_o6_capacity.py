"""Capacity protocols must finish settling, loading and recovery without a fixture."""

import numpy as np


def test_capacity_schedule_has_scored_ramp_down_and_recovery():
    from teleopit.sim.g1_o6_support.capacity import CapacityConfig, capacity_schedule

    cfg = CapacityConfig()
    assert capacity_schedule(6, cfg) == ("SETTLE", 1, 0, 0)
    assert capacity_schedule(8, cfg) == ("LOAD_RAMP", 1, 0, 0)
    assert capacity_schedule(11, cfg) == ("PLATEAU", 1, 0, 1)
    assert capacity_schedule(21, cfg) == ("RAMP_DOWN", 1, 0, 1)
    assert capacity_schedule(24, cfg) == ("RECOVERY", 1, 0, 0)


def test_measured_references_are_simultaneous_and_settled():
    from teleopit.sim.g1_o6_support.capacity import measured_references

    rows = [
        {
            "time": t,
            "phase": "HOLD",
            "assistance_gain": 0,
            "wrist_world_on_hand": {"left": w},
        }
        for t, w in [
            (5, [9, 0, 0, 0, 1, 0]),
            (10, [2, 0, 0, 0, 4, 0]),
            (30, [1, 2, 3, 4, 5, 6]),
        ]
    ]
    refs = measured_references(rows, "left")
    for ref in refs:
        original = next(row for row in rows if row["time"] == ref["time_s"])
        np.testing.assert_array_equal(
            ref["wrench_world_at_mount"], original["wrist_world_on_hand"]["left"]
        )
    assert any("settled" in ref["selection"] for ref in refs)
