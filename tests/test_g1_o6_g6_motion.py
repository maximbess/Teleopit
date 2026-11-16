"""Measured trajectory/load bookkeeping cannot qualify commanded values."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np


def stationary_rows(phase, seconds=60):
    return [
        {
            "time": t + 5,
            "phase": "HOLD",
            "experiment_phase": phase,
            "experiment_phase_time_s": t,
            "transfer": {"side": "left", "measured": {"share": True}},
            "base_pose": [0, 0, 1, 1, 0, 0, 0],
            "com": [0, 0, 1],
            "supports": {
                s + "_hand": {"force_world": [0, 0, 15]} for s in ("left", "right")
            },
            "hand_in_target_rung": {
                s: {"position": [0, 0, 0], "rotation": np.eye(3).ravel().tolist()}
                for s in ("left", "right")
            },
        }
        for t in np.arange(0, seconds + 0.01, 0.02)
    ]


def test_loaded_metrics_require_continuous_band_and_stationary_final_window():
    from teleopit.sim.g1_o6_support.g6_transfer_results import (
        load_measurements,
        transfer_motion_gate,
    )

    rows = stationary_rows("LOADED_HOLD")
    result = load_measurements(rows, 100, 0.02)
    assert result["status"] == "PASS" and result["continuous_in_band_s"] >= 60
    assert result["per_hand_min_max"]["left"] == [0.15, 0.15]
    request = dict(mode="load", plateau_seconds=60)
    assert transfer_motion_gate(request, rows)["passed"]
    rows[-1]["transfer"]["measured"]["share"] = False
    assert load_measurements(rows, 100, 0.02)["continuous_in_band_s"] == 0
    rows[-1]["base_pose"][0] = 0.01
    assert not transfer_motion_gate(request, rows)["passed"]


def test_adjacent_motion_result_keeps_first_second_and_complete_success_separate():
    from teleopit.sim.g1_o6_support.g6_transfer_results import (
        adjacent_progress,
        transfer_motion_gate,
    )

    request = dict(mode="adjacent", side="left", plateau_seconds=10)
    events = [dict(previous="POST_HOLD", side="left")]
    partial = adjacent_progress(request, events, False, False)
    assert partial["first_hand"] == "PASS" and partial["second_hand"] == "FAIL"
    assert partial["complete_sequence"] == "BLOCKED"
    rows = stationary_rows("POST_HOLD", 30)
    assert not transfer_motion_gate(request, rows)["passed"]
    right = [{**row, "transfer": {**row["transfer"], "side": "right"}} for row in rows]
    assert transfer_motion_gate(request, rows + right)["passed"]
    events.append(dict(previous="POST_HOLD", side="right"))
    assert adjacent_progress(request, events, True, True)["complete_sequence"] == "PASS"
    assert (
        adjacent_progress(
            {**request, "mode": "adjacent-first"}, events[:1], True, True
        )["second_hand"]
        == "NOT RUN"
    )


def test_adjacent_actuator_targets_are_bounded_without_live_pose_writes(monkeypatch):
    from teleopit.sim.g1_o6_support import g6_transfer as module
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.protocol import Protocol, Step

    candidate = G6Candidate("fixture")
    candidate = replace(
        candidate, pose=replace(candidate.pose, hand_rungs=(7, 7), foot_rungs=(2, 2))
    )
    env = make_environment(candidate, 30, 0.001)
    protocol = Protocol(
        [Step("OPEN", 1, 2), Step("REACH", 15, 30), Step("CLOSE", 1, 2)]
    )
    controller = module.AdjacentController(env, env.hold_controller, protocol, 0.3)
    controller.enter(env, {})
    site = env.data.site(env.hands["left"]["mount"])
    plan = {
        "status": "FEASIBLE",
        "waypoints_world_m": [site.xpos.tolist()] * 2,
        "rotation_world": site.xmat.reshape(3, 3).tolist(),
    }
    monkeypatch.setattr(module, "plan_reach", lambda *a: plan)
    protocol.index = 1
    env.phase = "HOLD"
    before = env.data.qpos.copy()
    controller.enter(env, {})
    # This tests the command boundary with a deterministic scratch solution.
    joints = list(range(1, 8))
    indices = [
        int(np.flatnonzero(env.model.actuator_trnid[:, 0] == j)[0]) for j in joints
    ]
    monkeypatch.setattr(
        module,
        "arm_target",
        lambda *a: dict(
            joints=joints,
            qpos=env.target[indices] + 0.1,
            scratch=env.data,
            position_error_m=0,
            angle_error_rad=0,
        ),
    )
    controller.clearance["left"] = SimpleNamespace(
        measure=lambda _: {"clearance_m": 0.01}
    )
    desired = controller.desired(env, 1)
    np.testing.assert_array_equal(env.data.qpos, before)
    np.testing.assert_allclose(desired[indices] - env.target[indices], 0.05 * 0.02)
    assert env.support_targets["left_hand"] == "left_ladder_rung_08"
    controller.clearance["left"] = SimpleNamespace(
        measure=lambda _: {"clearance_m": -0.0001}
    )
    controller.desired(env, 1)
    assert controller.failure == "adjacent_command_infeasible_or_colliding"
    protocol.index = 2
    controller.enter(env, {})
    np.testing.assert_array_equal(controller.mounts["left"][0], site.xpos)
