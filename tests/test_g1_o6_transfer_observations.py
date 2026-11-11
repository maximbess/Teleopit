"""Synthetic measured observations exercise orchestration, never physical claims."""

import copy

import numpy as np
import pytest

from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.posture_report import annotate
from teleopit.sim.g1_o6_support.protocol import Step
from teleopit.sim.g1_o6_support.transfer import TransferTrial, support_positions


@pytest.fixture
def trial():
    env = Environment()
    request = {
        "mode": "reacquire",
        "side": "left",
        "target_share": 0.05,
        "cycles": 1,
        "alternating": False,
        "three_seconds": 1,
        "plateau_seconds": 10,
        "physics_dt": 0.001,
    }
    task = TransferTrial(env, request)
    env.phase = "HOLD"
    row = snapshot(env)
    annotate(env, row)
    for name, item in row["supports"].items():
        item.update(
            loaded=True,
            opposing=True,
            max_slip=0,
            force_world=[
                0,
                0,
                10 if name.endswith("hand") else (env.mass * 9.81 - 20) / 2,
            ],
        )
    row["static_balance_residual_N"] = [0, 0, 0]
    row["posture"]["thumb_below_fingers_above"] = {"left": True, "right": True}
    row["wrist_world_on_hand"] = {"left": [0] * 6, "right": [0] * 6}
    task.controller.baseline = copy.deepcopy(row)
    task.controller.reference = copy.deepcopy(row)
    task.controller.latest = row
    return task, row


def phase(task, name):
    task.protocol.steps = [Step(name, 10, 15, side="left")]
    task.protocol.index = 0
    task.protocol.started = -10
    task.controller.previous = None


def test_measured_unload_and_reload_features(trial):
    task, row = trial
    phase(task, "UNLOAD_SETTLE")
    for name, item in row["supports"].items():
        item["force_world"][2] += -9 if name == "left_hand" else 3
    values = task.features(row)
    assert values["unloaded"]
    assert not values["accepted_load"]
    assert "left_hand" not in task.required_supports()
    assert not task.strict_reasons(row)
    row["supports"]["left_hand"]["max_slip"] = 0.03
    assert "contact_slip" in task.strict_reasons(row)
    values = task.features(task.controller.reference)
    assert values["accepted_load"]
    assert values["baseline_share"]


def test_each_controller_phase_is_bounded_and_preserves_live_state(trial):
    task, row = trial
    env, controller = task.env, task.controller
    saved = env.data.qpos.copy()
    for name in (
        "BASELINE",
        "LOAD_RAMP",
        "LOADED_HOLD",
        "UNLOAD_RAMP",
        "UNLOAD_SETTLE",
        "OPEN",
        "RETRACT",
        "THREE_SUPPORT_HOLD",
        "APPROACH",
        "CLOSE",
        "VERIFY_GRASP",
        "RELOAD",
        "POST_HOLD",
        "RETURN",
        "RECOVERY",
    ):
        phase(task, name)
        controller.enter(env, row)
        desired = controller.desired(env, 1)
        assert np.isfinite(desired).all()
        assert np.max(np.abs(controller.offset)) <= 0.08
        np.testing.assert_array_equal(env.data.qpos, saved)
    np.testing.assert_allclose(
        list(controller.target_forces.values()),
        [v["force_world"][2] for v in controller.baseline["supports"].values()],
    )
    assert set(controller.motion) == {
        "goal_world_m",
        "position_error_m",
        "angle_error_rad",
        "planned_clearance",
    }


def test_clearance_and_tracking_are_measured_and_serializable(trial):
    import json

    task, row = trial
    phase(task, "OPEN")
    task.controller.enter(task.env, row)
    assert not task.open_features(row, "left")["opened"]  # No actual opening motion.
    task.controller.open_start["left"] = np.ones(6)
    values = task.open_features(row, "left")
    assert values["opened"]
    assert row["release_clearance"]["target_rung"] == "left_ladder_rung_11"
    json.dumps(values)
    phase(task, "LOADED_HOLD")
    task.positions = {
        k: p + np.array([0.003, 0, 0]) for k, p in support_positions(task.env).items()
    }
    assert "support_location_moved" in task.strict_reasons(row)


def test_phase_entry_captures_reference_and_rejects_an_already_loaded_baseline(trial):
    task, row = trial
    task.rows = [row]
    phase(task, "LOAD_RAMP")
    task.on_transition("BASELINE", row)
    assert task.env.phase == "ABORT"
    assert task.env.abort_reasons == ["target_not_meaningful_above_baseline"]
    task.env.phase = "HOLD"
    phase(task, "UNLOAD_RAMP")
    task.on_transition("LOADED_HOLD", row)
    assert task.controller.reference["supports"] == row["supports"]
    task.protocol.failure = "test_failure"
    task.on_transition("UNLOAD_RAMP", row)
    assert task.env.abort_reasons == ["test_failure"]


def test_controller_or_strict_failure_aborts_and_preserves_reason(trial):
    task, row = trial
    phase(task, "LOAD_SETTLE")
    task.controller.failure = "arm_trajectory_infeasible_or_colliding"
    task.update(row)
    assert task.env.phase == "ABORT"
    assert task.env.abort_reasons == [task.controller.failure]


def test_static_failure_uses_existing_persistence_then_blocks_phase_completion(trial):
    task, row = trial
    task.protocol.steps = [
        Step("BASELINE", 0.6, 0.62, hold=True),
        Step("LOAD_RAMP", 10, 10),
    ]
    bad = copy.deepcopy(row)
    bad["base_drift_m"] = 0.04
    task.update(bad)
    assert task.env.phase == "HOLD"  # A single crossing is not a persistent abort.
    for i in range(1, 32):
        if task.env.phase == "ABORT":
            break
        recovered = copy.deepcopy(row)
        recovered["time"] = i * 0.02
        task.update(recovered)
    assert task.env.phase == "ABORT"
    assert "full_interval_not_stable" in task.env.abort_reasons
    assert "drift" in task.failures


def test_load_integrator_freezes_at_physical_command_limits(trial):
    task, _row = trial
    phase(task, "LOADED_HOLD")
    env = task.env
    joints = env.model.actuator_trnid[task.controller.indices, 0]
    env.target[task.controller.indices] = env.model.jnt_range[joints, 1]
    task.controller.desired(env, 1)
    np.testing.assert_array_equal(
        task.controller.offset, np.zeros_like(task.controller.offset)
    )


def test_physics_contact_after_detachment_latches_first_failure(trial):
    from types import SimpleNamespace

    task, _ = trial
    phase(task, "THREE_SUPPORT_HOLD")
    geoms = list(task.controller.clearance["left"].hand)
    task.check_detachment([SimpleNamespace(geom1=geoms[0], geom2=geoms[1])], 1)
    assert not task.failures
    other = task.env.model.geom("left_ladder_rung_12").id
    task.check_detachment([SimpleNamespace(geom1=geoms[0], geom2=other)], 2)
    task.check_detachment([SimpleNamespace(geom1=geoms[0], geom2=other)], 3)
    assert task.failures == {"released_hand_physics_contact": 2}
