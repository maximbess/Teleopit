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
