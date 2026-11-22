import numpy as np
import pytest

from teleopit.sim.g1_o6_support.regrasp_actions import DigitCommand, fixed_model


def test_independent_signed_commands_open_from_encoders_and_hold_target():
    measured = np.array([0.26, 0.48, 1.1, 0.8, 0.75, 0.7])
    closed = np.array([0.35, 0.5, 1.35, 1.35, 1.35, 1.35])
    hands = [DigitCommand(closed, np.full(6, 4.0), 0.12) for _ in range(2)]
    for hand in hands:
        hand.follow(measured + 0.02, 1.0)

    def tick(hand, request, angles=measured, dt=0.02):
        return hand.advance(request, angles, dt).copy()

    start = tick(hands[1], 0)
    left = tick(hands[0], -3)
    np.testing.assert_allclose(left, measured * 0.98)
    np.testing.assert_array_equal(tick(hands[0], 0, measured / 2), left)
    np.testing.assert_array_equal(tick(hands[1], 0), start)
    for _ in range(49):
        tick(hands[0], -1)
    np.testing.assert_allclose(hands[0].target, 0, atol=1e-12)
    np.testing.assert_allclose(
        tick(hands[0], 1, measured * 0, 0.5), np.minimum(0.5 * closed, 0.12 / 4)
    )
    with pytest.raises(ValueError):
        tick(hands[0], np.nan)


def test_pelvis_fixture_holds_six_axes_and_leaves_fingers_free():
    import mujoco

    xml = """<mujoco><worldbody><body name="pelvis"><freejoint/>
      <geom size=".1"/><body pos="0 0 .2"><joint name="digit"/>
      <geom size=".03"/></body></body></worldbody></mujoco>"""
    pose = np.array([0.1, 0.2, 0.8, 1, 0, 0, 0])
    model, _ = fixed_model(xml, pose)
    data = mujoco.MjData(model)
    data.qpos[:7] = pose
    data.qfrc_applied[[0, 4, 6]] = [0.3, 0.2, 0.001]
    for _ in range(200):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[:7], pose, atol=2e-5)
    assert abs(data.qpos[7]) > 0.001
    assert model.neq == 1


def test_contact_classes_require_the_correct_loaded_opposing_surfaces():
    from teleopit.sim.g1_o6_support.regrasp_measure import hand_state

    def pad(digit, z, normal, target=True, load=1.0):
        return {
            "group": digit,
            "target": target,
            "normal_force_N": load,
            "normal_on_hand_world": [0, 0, normal],
            "position_world_m": [0, 0, z],
            "rung_position_m": [0, 0, z],
        }

    contacts = [
        pad("thumb", -0.018, -1),
        *[pad(d, 0.018, 1) for d in ("index", "middle", "ring", "pinky")],
    ]
    assert hand_state([], 0.0) == "air"
    assert hand_state([dict(c, target=False) for c in contacts], 0.0) == "wrong_surface"
    assert hand_state(contacts, 0.0) == "grasp"
    for change in (
        {"normal_force_N": 0.01},
        {"normal_on_hand_world": [0, 0, 1]},
        {"position_world_m": [0, 0, 0.02]},
    ):
        assert hand_state([dict(contacts[0], **change), *contacts[1:]], 0.0) == "touch"


def test_confirmation_accounts_for_a_single_lost_native_sample():
    from teleopit.sim.g1_o6_support.regrasp_trial import Confirmation

    clock = Confirmation(0.5, 0.001)
    assert not clock.update([True] * 499)
    assert not clock.update([False, True])
    assert not clock.update([True] * 498)
    assert clock.update([True])


def test_physical_step_drives_only_requested_servos_and_reset_clears_both():
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.g6_wrist_grip_experiment import candidates
    from teleopit.sim.g1_o6_support.regrasp_actions import PhysicalHands

    env = make_environment(candidates()[1], 5.0, 0.001)
    policy = PhysicalHands.install(env)
    left, right = (env.hand_act[s] for s in ("left", "right"))
    for _ in range(8):
        row = policy.step(grip=(2.0, 0.0))
    assert np.linalg.norm(env.data.ctrl[left]) > 0.05
    np.testing.assert_array_equal(env.data.ctrl[right], np.zeros(6))
    assert max(row["physical_hands"]["left"]["measured_rad"]) > 0.01
    assert row["physical_hands"]["left"]["previous_request"] == 1.0
    policy.reset()
    policy.step(grip=(0.0, 1.0))
    assert env.hold_controller is policy
    np.testing.assert_array_equal(env.data.ctrl[left], np.zeros(6))
    assert max(env.data.ctrl[right]) > 0.001
    assert env.data.time == pytest.approx(0.020)
    env.reset()
    policy.step(grip=(0.0, 1.0))
    assert env.hold_controller is policy
    np.testing.assert_array_equal(env.data.ctrl[left], np.zeros(6))
    assert max(env.data.ctrl[right]) > 0.001


def test_paired_effect_removes_drift_in_each_support_and_uses_percentage_points():
    from teleopit.sim.g1_o6_support.regrasp_results import paired_window

    baseline = np.array([5.0, 4.0, 160.0, 165.0])
    drift = np.array([0.5, -0.5, 2.0, -2.0])
    effect = np.array([-5.0, 1.0, 2.0, 2.0])
    ctrl_base = baseline + [0.0, 0.0, 1.0, -1.0]

    def series(a, b):
        return {"time": np.arange(4.0), "fz": np.vstack([a, a, b, b])}

    rows = paired_window(
        series(baseline, baseline + drift + effect),
        series(ctrl_base, ctrl_base + drift),
        (0, 2),
        (2, 4),
        334.0,
    )
    np.testing.assert_allclose([r["effect_N"] for r in rows], effect)
    np.testing.assert_allclose([r["effect_pp"] for r in rows], 100 * effect / 334.0)
    np.testing.assert_allclose(
        [r["release_pct_BW"] for r in rows], 100 * (baseline + drift + effect) / 334.0
    )


def test_signed_forces_include_tiny_contacts_and_account_for_other_surfaces():
    from types import SimpleNamespace

    from teleopit.sim.g1_o6_support.regrasp_measure import NativeTrace

    env = SimpleNamespace(support_targets={"left_hand": "rung", "left_foot": "step"})
    trace = SimpleNamespace(
        geom_ids={"finger": 0, "sole": 1, "rung": 2, "rail": 3, "step": 4}, detail=[]
    )

    def contact(geoms, regions, fz, normal):
        return {
            "geoms": geoms,
            "regions": regions,
            "force_on_second_world": [0, 0, fz],
            "normal_force": normal,
            "normal_on_second_world": [0, 0, 1],
            "position": [0, 0, 0],
            "slip_m_s": 0.006,
            "distance": 0.0,
        }

    rows = [
        contact(["finger", "rung"], ["left_hand", "environment"], 2.0, 3.0),
        contact(["rung", "finger"], ["environment", "left_hand"], 0.01, 0.01),
        contact(["step", "sole"], ["environment", "left_foot"], 170.0, 170.0),
        contact(["finger", "rail"], ["left_hand", "environment"], -2.0, 2.0),
    ]
    fz, slip, loaded, other, wrong = NativeTrace.record_contacts(trace, env, rows)
    np.testing.assert_allclose(fz, [-1.99, 0, 170, 0])
    np.testing.assert_array_equal(loaded, [True, False, True, False])
    assert other == 2 and wrong == 1 and slip[0] == 0.006


def test_release_needs_encoder_motion_no_contact_and_five_mm_on_each_sample():
    from teleopit.sim.g1_o6_support.regrasp_trial import Trial

    trial = Trial.__new__(Trial)
    trial.i = 0
    trial.start_angles = np.full(6, 0.5)
    angles = np.zeros((5, 2, 6))
    angles[3, 0, 4] = 0.14
    values = {
        "digits": angles,
        "gap": np.array([0.005, 0.00499, 0.05, 0.05, np.nan]),
        "external_count": np.array([0, 0, 1, 0, 0]),
    }
    trial.last = values.__getitem__
    np.testing.assert_array_equal(trial.clear(), [True, False, False, False, False])
    trial.start_angles[:] = 0.0
    assert not trial.clear().any()


def test_loss_invalidates_settled_comparison_but_preserves_transient():
    from teleopit.sim.g1_o6_support.regrasp_results import measured_windows

    result = {
        "windows": {"baseline": [0, 2], "clear1": [3, 5], "hold1_transient": [7, 9]},
        "missing": "opposition lost",
    }
    windows, unavailable = measured_windows(result, 2.3)
    assert ("clear1", [3, 5], "transient, settled interval unavailable") in windows
    assert ("hold1_transient", [7, 9], "incomplete") in windows
    assert unavailable["clear1"] == "required support lost before window ended"
    assert unavailable["hold1"] == "opposition lost"


def test_matched_control_freezes_the_same_tread_offset_in_each_opening():
    from types import SimpleNamespace

    from teleopit.sim.g1_o6_support.regrasp_trial import Trial

    trial = SimpleNamespace(time=0.0, tick_s=0.02, windows={}, event=lambda name: None)
    trial.path = SimpleNamespace(target=lambda distance, lateral=1.0: lateral)
    commands = []

    def tick(phase, arm):
        commands.append(arm)
        trial.time = round(trial.time + 0.02, 8)

    trial.tick = tick
    matched = {
        "events": {
            "open_request1": 0.0,
            "opened1": 0.04,
            "open_request2": 0.06,
            "opened2": 0.08,
        },
        "end_s": 0.10,
        "windows": {},
    }
    Trial.control(trial, matched)
    assert commands == [0.0, 0.0, None, 1.0, None]


def test_pelvis_angle_is_zero_at_the_rounded_source_quaternion():
    from teleopit.sim.g1_o6_support.regrasp_measure import pelvis_angle

    unit = np.array([0.99915, 0.0, 0.041219, 0.0])
    normalized = unit / np.linalg.norm(unit)
    np.testing.assert_allclose(
        pelvis_angle(np.array([normalized, -normalized]), unit), 0.0, atol=1e-14
    )
    assert pelvis_angle(
        [np.cos(0.03), 0, np.sin(0.03), 0], [1, 0, 0, 0]
    ) == pytest.approx(0.06)
