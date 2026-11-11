"""Measured, persistent transition guards cannot be satisfied by commands."""

import numpy as np


def row(left=10, right=10, feet=90, wrist=3):
    return {
        "supports": {
            name: {"force_world": [0, 0, force]}
            for name, force in zip(
                ("left_hand", "right_hand", "left_foot", "right_foot"),
                (left, right, feet, feet),
            )
        },
        "wrist_world_on_hand": {"left": [0, 0, wrist, 0, 0, 0]},
    }


def test_unload_needs_meaningful_reference_and_compensating_reactions():
    from teleopit.sim.g1_o6_support.guards import unloading

    assert unloading(row(), row(1, 19), "left")["passed"]
    assert not unloading(row(1), row(0, 11), "left")["passed"]
    assert not unloading(row(), row(1), "left")["passed"]
    assert not unloading(row(), row(1, 19, wrist=9), "left")["passed"]
    assert not unloading(row(), row(-3, 23), "left")["passed"]


def test_guard_requires_continuous_elapsed_time_and_resets_on_bad_sample():
    from teleopit.sim.g1_o6_support.guards import Persistence

    guard = Persistence(0.5)
    assert not guard.update(1, True)
    assert not guard.update(1.48, True)
    assert not guard.update(1.49, False)
    assert not guard.update(2, True)
    assert guard.update(2.5, True)


def test_release_requires_actual_open_motion_contacts_and_clearance():
    from teleopit.sim.g1_o6_support.guards import released

    start, opened = np.ones(6), np.zeros(6)
    assert released(start, opened, 0, 0.006)
    assert not released(start, start, 0, 0.006)
    assert not released(start, opened, 1, 0.006)
    assert not released(start, opened, 0, 0.004)
    assert not released(opened, opened, 0, 0.006)


def test_share_target_uses_measured_per_hand_values():
    from teleopit.sim.g1_o6_support.guards import share_settled

    assert share_settled(row(), 0.1, 200)
    assert not share_settled(row(0, 20), 0.1, 200)
    assert not share_settled(row(2, 2), 0.1, 200)


def test_capacity_coverage_preserves_wrench_direction_and_tested_reserve():
    from teleopit.sim.g1_o6_support.guards import wrench_covered

    demand = np.array([1, -2, 3, 0.1, 0, -0.2])
    assert wrench_covered(demand, [1.2 * demand])
    assert not wrench_covered(demand, [demand])
    assert not wrench_covered(demand, [-1.5 * demand])
    assert not wrench_covered(demand, [np.abs(2 * demand)])
