"""Strict hold gates on a controlled state, independent of the physics model."""

from types import SimpleNamespace

import numpy as np
import pytest

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.env import Environment, initial_errors


def controller(phase):
    env = Environment.__new__(Environment)
    env.config = Config()
    env.hands = {"left": {}, "right": {}}
    env.data = SimpleNamespace(time=0.0)
    env.phase = phase
    env.phase_start = env.stable = env.bad = 0.0
    env.support_targets = {
        s + "_" + p: "test_rung" for s in env.hands for p in ("hand", "foot")
    }
    env.events = []
    env.abort_reasons = []
    env.hold_passed = False
    return env


def stable_row():
    return {
        "base_drift_m": 0,
        "base_angle_rad": 0,
        "penetration_m": 0,
        "joint_limit_error_rad": 0,
        "mimic_error_rad": 0,
        "effort_fraction": [0.5],
        "numerical_warnings": [0],
        "unintended_support": [],
        "base_velocity": [0] * 6,
        "static_balance_residual_N": [0] * 3,
        "wrist_world_on_hand": {s: [0] * 6 for s in ("left", "right")},
        "supports": {
            s + "_" + p: {"loaded": True, "opposing": True, "max_slip": 0}
            for s in ("left", "right")
            for p in ("hand", "foot")
        },
    }


def tick(env, row, seconds):
    for _ in range(round(seconds / env.config.control_dt)):
        env.data.time += env.config.control_dt
        env.advance(row)
        if env.phase in ("ABORT", "HOLD_COMPLETE"):
            break


def test_static_gate_requires_entire_interval():
    env = controller("HOLD")
    row = stable_row()
    tick(env, row, 9.8)
    assert env.phase == "HOLD" and not env.hold_passed
    row["supports"]["left_hand"]["opposing"] = False
    tick(env, row, 0.02)
    row["supports"]["left_hand"]["opposing"] = True
    tick(env, row, 0.3)
    assert env.phase == "ABORT" and not env.hold_passed
    assert env.abort_reasons == ["full_interval_not_stable"]


@pytest.mark.parametrize(
    "support", ["left_hand", "right_hand", "left_foot", "right_foot"]
)
def test_every_support_is_required_throughout_hold(support):
    env = controller("HOLD")
    row = stable_row()
    row["supports"][support]["loaded"] = False
    tick(env, row, 0.48)
    assert env.phase == "HOLD"
    tick(env, row, 0.04)
    assert env.phase == "ABORT" and support in env.abort_reasons
    assert not env.hold_passed


def test_nonopposing_hand_contact_cannot_complete_hold():
    env = controller("HOLD")
    row = stable_row()
    row["supports"]["left_hand"]["opposing"] = False
    tick(env, row, 0.6)
    assert env.phase == "ABORT" and "left_hand" in env.abort_reasons


@pytest.mark.parametrize(
    "change",
    [
        {"physics_dt": 0.003},
        {"delay_steps": -1},
        {"friction_scale": 0},
        {"assistance_force_N": -1},
        {"max_effort_fraction": 2},
        {"root_offset_x": float("nan")},
    ],
)
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        Config(**change).validate()


def test_initial_preflight_rejects_collision_or_unreachable_wrist():
    ik = {"left": {"position_error_m": 0.01, "angle_error_rad": 0}}
    assert initial_errors({"penetration_m": 0.001}, ik) == [
        "initial_penetration",
        "unreachable_grasp",
    ]


def test_internal_robot_contacts_do_not_enter_force_balance():
    import mujoco

    from teleopit.sim.g1_o6.diagnostics import contacts

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="pelvis" pos="0 0 1"><freejoint/><geom size=".01" contype="0" conaffinity="0"/>
        <body name="left_o6_finger" pos="-.08 0 0"><joint/><geom name="a" size=".1"/></body>
        <body name="right_o6_finger" pos=".08 0 0"><joint/><geom name="b" size=".1"/></body>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rows, external = contacts(model, data)
    assert len(rows) == 1 and rows[0]["normal_force"] > 0
    assert rows[0]["regions"] == ["left_hand", "right_hand"]
    np.testing.assert_array_equal(external, [0, 0, 0])


def test_success_finishes_without_authorizing_release():
    env = controller("HOLD")
    tick(env, stable_row(), 10.04)
    assert env.phase == "HOLD_COMPLETE"
    assert env.hold_passed
    assert all(e["next"] != "UNLOAD_SELECTED_HAND" for e in env.events)
