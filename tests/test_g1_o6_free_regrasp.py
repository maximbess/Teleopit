import numpy as np


def test_stance_servos_preserve_captured_load_and_share_targets_at_different_measured_poses():
    from types import SimpleNamespace

    from teleopit.sim.g1_o6_support.regrasp_balance import StanceServos

    ids = np.array([0, 2])
    model = SimpleNamespace(
        actuator_gainprm=np.full((3, 3), 20.0),
        actuator_biasprm=np.tile([0.0, -20.0, -2.0], (3, 1)),
    )
    original = np.array([0.1, -0.3])
    command = np.array([0.2, -0.1])
    servos = StanceServos(model, ids, original, command, np.array([2.0, 3.0]))
    np.testing.assert_allclose(servos.target(0), command)
    for t in (0.4, 1.0, 2.0):
        target = servos.target(t)
        np.testing.assert_allclose(
            model.actuator_gainprm[ids, 0] * (target - original),
            20.0 * (command - original),
        )
    assert model.actuator_gainprm[1, 0] == 20.0
    assert (model.actuator_biasprm[1] == [0.0, -20.0, -2.0]).all()
    np.testing.assert_allclose(model.actuator_gainprm[ids, 0], [40.0, 60.0])
    np.testing.assert_array_equal(servos.target(2.0), servos.target(20.0))


def test_supporting_hand_closes_from_its_own_encoders_with_finite_preload_and_travel():
    from teleopit.sim.g1_o6_support.regrasp_actions import DigitCommand
    from teleopit.sim.g1_o6_support.regrasp_balance import prepare_support_grip

    anchor = np.array([0.23, 0.5, 0.8, 0.9, 0.9, 0.95])
    closed = np.array([0.35, 0.5, 1.35, 1.35, 1.35, 1.35])
    measured = np.array([0.21, 0.49, 0.75, 0.87, 0.91, 1.3])
    gains = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    digit = DigitCommand(closed, gains, 0.12)
    digit.follow(anchor, 1.0)
    prepare_support_grip(digit, anchor)
    target = digit.advance(1.0, measured, 0.02)
    assert np.all(target <= closed) and np.all(target <= anchor + 0.25)
    assert 0.12 < gains[0] * (target[0] - measured[0]) < 0.35
    assert target[-1] < measured[-1]


def test_returned_digits_retain_flexor_reserve_and_restore_travel_on_the_next_open():
    from teleopit.sim.g1_o6_support.regrasp_actions import DigitCommand
    from teleopit.sim.g1_o6_support.regrasp_balance import HoldingDigits

    closed = np.array([0.35, 0.5, 1.35, 1.35, 1.35, 1.35])
    actual = np.array([0.18, 0.5, 0.7, 0.8, 0.9, 0.9])
    digit = DigitCommand(closed, np.full(6, 3.0), 0.12)
    digit.follow(actual, 1.0)
    hold = HoldingDigits(digit)
    hold.select(True, actual)
    target = digit.advance(1.0, actual, 0.02)
    assert np.all(target[2:] > actual[2:] + 0.04)
    hold.select(True, actual + 0.2)
    assert np.all(digit.closed[2:] < actual[2:] + 0.1)
    hold.select(False, target)
    np.testing.assert_array_equal(digit.closed, closed)
    assert digit.preload == 0.12


def test_sparse_contacts_reconstruct_signed_world_load_on_each_of_four_supports():
    from scripts.g1_o6_free_regrasp_audit import native_forces

    # Geoms 0,1 are the rungs, 2..5 the four supports, 6 another surface.
    contacts = np.array(
        [
            [0, 0, 2, 0.01, 0.008],
            [0, 3, 0, 3.0, -2.0],
            [0, 1, 4, 50.0, 45.0],
            [1, 5, 1, 60.0, -58.0],
            [1, 2, 6, 7.0, -5.0],
            [1, 1, 2, 1.2, 1.0],
        ]
    )
    forces = native_forces(
        contacts, np.array([-1, -1, 0, 1, 2, 3, -1]), [0, 0, 1, 1], 2
    )
    np.testing.assert_array_equal(
        forces, [[0.008, 2.0, 45.0, 0.0], [0.0, 0.0, 0.0, 58.0]]
    )


def test_completed_campaign_reports_only_measured_modes_and_marks_final_regrasp():
    from scripts.g1_o6_regrasp_report import event_marks, summary
    from teleopit.sim.g1_o6_support.regrasp_results import measured_windows

    row = {
        "pelvis": "free",
        "side": "right",
        "offset_mm": 0,
        "moving_completed": True,
        "supported_cycle": True,
        "opposite_grasp_and_feet_retained": True,
        "missing": None,
    }
    assert [(r["pelvis"], r["hand"]) for r in summary({"trial": row})] == [
        ("free", "right")
    ]
    monitored = summary({"trial": {**row, "supported_cycle": False}})[0]
    assert monitored["opposite_grasp_and_feet_retained"] == 1
    assert monitored["continuously_supported_cycles"] == 0
    marks = event_marks(
        {"open_request2": 9.0, "grasp_verified2": 14.0, "recovered2": 16.0}, {}, 2.0
    )
    assert [t for t, _ in marks] == [7.0, 12.0, 14.0]
    windows = {"opening2": [9.0, 10.0], "retreat2": [10.0, 11.0], "hold2": [14.0, 16.0]}
    measured, _ = measured_windows({"windows": windows, "missing": None}, np.inf)
    assert {w for w, _, _ in measured} == set(windows)


def test_replay_includes_open_and_both_holds_and_rejects_an_interrupted_cycle():
    import pytest

    from scripts.view.view_rung7_regrasp import cycle_frames

    clock = np.arange(8.0, 16.1, 0.25)
    events = {"open_request1": 10.0, "recovered2": 15.0}
    frames = cycle_frames(clock, {"events": events, "missing": None})
    np.testing.assert_array_equal(clock[frames], np.arange(10.0, 15.1, 0.25))
    with pytest.raises(ValueError, match="continuous"):
        cycle_frames(clock, {"events": events, "missing": "first hold lost"})


def test_held_hand_keeps_its_wrist_equilibrium_when_the_control_repeats_a_rung_goal():
    from types import SimpleNamespace

    from teleopit.sim.g1_o6_support.regrasp_balance import FloatingTrial, HoldingWrist

    trial = object.__new__(FloatingTrial)
    wrist = object.__new__(HoldingWrist)
    wrist.arm = np.array([0.1, -0.4])
    goal = np.array([0.22, -0.3])
    wrist.path = SimpleNamespace(target=lambda distance: goal)
    trial.holding_wrist = wrist
    before = trial.selected_arm(None, holding=True)
    np.testing.assert_array_equal(trial.selected_arm(goal, holding=True), before)
    assert np.all((before > wrist.arm) & (before < goal))
    np.testing.assert_array_equal(trial.selected_arm(goal, holding=False), goal)
    assert trial.holding_wrist is None
