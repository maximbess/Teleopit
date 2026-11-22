import numpy as np
import pytest


def test_unassisted_model_accepts_internal_finger_mimics_and_rejects_a_world_pin():
    import mujoco

    from teleopit.sim.g1_o6_support.regrasp_unassisted import free_body_setup

    robot = """<mujoco><worldbody><body name="pelvis"><freejoint/><geom size=".1"/>
    <body name="left_hand"><joint name="finger"/><geom size=".01"/>
    <body name="tip"><joint name="distal"/><geom size=".01"/></body></body></body>
    </worldbody><equality><joint joint1="distal" joint2="finger"/>{}</equality></mujoco>"""
    hands = {"left": {"passive": [("distal", "finger", 1.0)]}}
    setup = free_body_setup(mujoco.MjModel.from_xml_string(robot.format("")), hands)
    assert setup["free_root_dofs"] == 6
    assert setup["internal_finger_couplings"] == [["distal", "finger"]]
    for pin in (
        '<weld body1="pelvis"/>',
        '<joint joint1="finger"/>',
        '<connect body1="left_hand" anchor="0 0 0"/>',
    ):
        with pytest.raises(ValueError, match="coupling|fixture"):
            free_body_setup(mujoco.MjModel.from_xml_string(robot.format(pin)), hands)
    with pytest.raises(ValueError, match="couplings"):
        free_body_setup(mujoco.MjModel.from_xml_string(robot.format("")), {})
    for source, error in (
        (robot.replace("<freejoint/>", ""), "free root"),
        (robot.replace('name="pelvis"', 'name="pelvis" gravcomp="1"'), "compensation"),
        (
            robot.replace("<worldbody>", '<option gravity="0 0 0"/><worldbody>'),
            "gravity",
        ),
    ):
        with pytest.raises(ValueError, match=error):
            free_body_setup(mujoco.MjModel.from_xml_string(source.format("")), hands)


def test_hold_start_restores_measured_velocities_and_rejects_an_assisted_frame():
    from teleopit.sim.g1_o6_support.regrasp_unassisted import launch_state

    frame = {
        "phase": "HOLD",
        "assistance_gain": 0.0,
        "assistance_world": [0.0] * 6,
        "disturbance_world": [0.0] * 3,
        "base_pose": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "base_velocity": [0.01] * 6,
        "joint_positions": [0.4, 0.5],
        "applied_position_targets_rad": [0.41],
        "diagnostic_v1": {
            "joints": {"b": {"velocity_rad_s": -0.2}, "a": {"velocity_rad_s": 0.1}}
        },
        "supports": {
            name: {"loaded": True}
            for name in ("left_hand", "right_hand", "left_foot", "right_foot")
        },
    }
    qpos, qvel, ctrl = launch_state(frame, ["a", "b"])
    np.testing.assert_array_equal(qvel, [0.01] * 6 + [0.1, -0.2])
    np.testing.assert_array_equal(qpos[-2:], [0.4, 0.5])
    np.testing.assert_array_equal(ctrl, [0.41])
    with pytest.raises(ValueError, match="unassisted"):
        launch_state({**frame, "assistance_world": [1.0] * 6}, ["a", "b"])


def test_releasing_relaxes_only_the_arm_contact_preload_at_the_same_pose():
    from types import SimpleNamespace

    from teleopit.sim.g1_o6_support.regrasp_unassisted import UnassistedTrial

    trial = object.__new__(UnassistedTrial)
    trial.i, trial.balance = 0, None
    trial.path = SimpleNamespace(preload=np.array([0.12, -0.04]))
    joint_goal = np.array([0.4, -0.3])
    opened, _ = trial.actions(joint_goal, -1.0)
    closed, _ = trial.actions(joint_goal, 1.0)
    np.testing.assert_allclose(opened[0], [0.34, -0.28])
    np.testing.assert_array_equal(closed[0], joint_goal)
    assert opened[1] is None


def test_episode_check_includes_the_preparation_before_the_scored_baseline():
    from scripts.g1_o6_free_regrasp_audit import launch_checks

    trace = {
        "time": np.array([0.0, 0.001, 0.002]),
        "state": np.full((3, 2), 3),
        "slip": np.zeros((3, 4)),
        "applied": np.zeros(3),
        "assist_gain": np.zeros(3),
        "fixture": np.zeros((3, 6)),
        "other_fz": np.zeros(3),
    }
    limits = {
        "max_slip_m_s": 0.02,
        "assistance_force_N": 0.0,
        "assistance_moment_Nm": 0.0,
        "disturbance_N": 0.0,
    }
    assert launch_checks(trace, limits, 0.002)["passed"]
    trace["state"][0, 0] = 2
    result = launch_checks(trace, limits, 0.002)
    assert not result["passed"] and result["preparation_hand_losses"] == [1, 0]
    trace["state"][0, 0] = 3
    trace["applied"][0] = 1.0
    result = launch_checks(trace, limits, 0.002)
    assert not result["passed"] and result["maximum_applied"] == 1.0
    trace["applied"][0] = 0.0
    assert not launch_checks(trace, {**limits, "assistance_force_N": 1.0}, 0.002)[
        "passed"
    ]


def test_preloading_grips_slews_from_applied_angles_without_slowing_the_wrists(
    monkeypatch,
):
    from types import SimpleNamespace

    from teleopit.sim.g1_o6_support.g6_grip_preview import PreviewReference
    from teleopit.sim.g1_o6_support.regrasp_actions import PhysicalHands

    policy = object.__new__(PhysicalHands)
    policy.fingers = np.array([1, 3])
    policy.requests = {"left": 1.0, "right": 1.0}
    policy.digits = {
        "left": SimpleNamespace(target=np.array([0.8])),
        "right": SimpleNamespace(target=np.array([0.3])),
    }
    env = SimpleNamespace(
        hand_act={"left": [1], "right": [3]},
        config=SimpleNamespace(control_dt=0.02),
        target=np.array([0.1, 0.5, -0.1, 0.6]),
    )
    monkeypatch.setattr(
        PreviewReference, "desired", lambda *_: np.array([0.3, 0.0, -0.4, 0.0])
    )
    policy.grip_slew_rad_s = 0.03
    np.testing.assert_allclose(policy.desired(env, 1.0), [0.3, 0.5006, -0.4, 0.5994])
    policy.grip_slew_rad_s = None
    np.testing.assert_allclose(policy.desired(env, 1.0), [0.3, 0.8, -0.4, 0.3])


def test_a_regrasp_with_loaded_fingers_still_fails_if_the_returned_hand_slips():
    from scripts.g1_o6_free_regrasp_audit import cycle_passes

    cfg = {
        "config": {"physics_dt": 0.001, "max_slip_m_s": 0.02},
        "criteria": {"persistence_s": 0.5},
        "position_tolerance_m": 0.003,
        "rotation_tolerance_rad": 0.035,
    }
    cycle = {
        "position_error_mm": 1.0,
        "angle_error_rad": 0.01,
        "detached_steps": 500,
        "clear_steps": 500,
        "confirmation_steps": 500,
        "hold_steps": 2000,
        "invalid_confirm_or_hold_steps": 0,
        "max_regrasp_slip_m_s": 0.01,
    }
    assert cycle_passes(cycle, cfg)
    assert not cycle_passes({**cycle, "max_regrasp_slip_m_s": 0.021}, cfg)


def test_still_closed_control_keeps_its_applied_servo_equilibrium_at_the_new_rung_goal():
    from types import SimpleNamespace

    from teleopit.sim.g1_o6_support.regrasp_balance import HoldingWrist

    wrist = object.__new__(HoldingWrist)
    wrist.arm = np.array([-0.1, 0.6])
    current = np.array([0.3, -0.4])
    path = SimpleNamespace(
        env=SimpleNamespace(target=np.array([0.3, 0.0, -0.4])),
        ids=[0, 2],
        target=lambda _: current,
    )
    wrist.recapture(path)
    np.testing.assert_array_equal(wrist.target(), current)
    np.testing.assert_array_equal(wrist.target(current), current)
