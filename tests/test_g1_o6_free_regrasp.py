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
