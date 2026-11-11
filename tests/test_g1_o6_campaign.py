"""Synthetic campaign decisions; these checks never qualify a physical maneuver."""

import json
from types import SimpleNamespace

import numpy as np
import pytest


def request(mode="load", **kwargs):
    return {
        "mode": mode,
        "side": "left",
        "target_share": 0.05,
        "cycles": 1,
        "alternating": False,
        "three_seconds": 1,
        "plateau_seconds": 10,
        "physics_dt": 0.001,
        **kwargs,
    }


def qualified(mode, **kwargs):
    return {"metadata": {"experiment": request(mode, **kwargs)}}


def test_progression_requires_each_duration_side_and_cycle():
    from teleopit.sim.g1_o6_support.qualification import transition_prerequisites

    assert transition_prerequisites(request(), []) == []
    assert transition_prerequisites(request(target_share=0.1), []) == [
        "previous_load_level"
    ]
    assert not transition_prerequisites(request(target_share=0.1), [qualified("load")])
    assert transition_prerequisites(request(plateau_seconds=30), []) == [
        "ten_second_load_plateau"
    ]
    runs = [qualified("load")]
    assert not transition_prerequisites(request("unload"), runs)
    assert "both_sides_closed_unload_reload" in transition_prerequisites(
        request("release"), runs
    )
    for side in ("left", "right"):
        runs.append(qualified("unload", side=side))
    assert not transition_prerequisites(request("release"), runs)
    assert transition_prerequisites(request("release", three_seconds=3), runs)
    runs.append(qualified("release", side="left", three_seconds=1))
    assert not transition_prerequisites(request("release", three_seconds=3), runs)
    for side in ("left", "right"):
        runs.append(qualified("release", side=side, three_seconds=10))
    assert not transition_prerequisites(request("reacquire"), runs)
    assert transition_prerequisites(request("reacquire", cycles=3), runs)
    for side in ("left", "right"):
        runs.append(qualified("reacquire", side=side, cycles=1))
    assert not transition_prerequisites(request("reacquire", cycles=3), runs)
    assert transition_prerequisites(
        request("reacquire", cycles=3, alternating=True), runs
    )
    for side in ("left", "right"):
        runs.append(qualified("reacquire", side=side, cycles=3))
    assert not transition_prerequisites(
        request("reacquire", cycles=3, alternating=True), runs
    )


def test_capacity_orchestrator_respects_budget_and_fine_repeats(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support import capacity_matrix as matrix

    calls = []
    reference = {
        "time_s": 30,
        "selection": ["settled"],
        "wrench_world_at_mount": [1, 2, 3, 0.1, 0.2, 0.3],
    }

    def trial(*args, **kwargs):
        calls.append(kwargs)
        return {
            "strict_pass": True,
            "plateau_complete": True,
            "physical_retention": True,
        }

    monkeypatch.setattr(matrix, "run_capacity", trial)
    outcome = matrix.evaluate_side(object(), "right", tmp_path, [reference], 0.001, 48)
    assert outcome["reserve_qualified"]
    assert outcome["physical_status"] == "PASS"
    assert len(calls) <= 48
    assert calls[-1]["physics_dt"] == 0.0005
    with pytest.raises(ValueError, match="trial budget"):
        matrix.evaluate_side(object(), "right", tmp_path, [reference], 0.001, 1)


def test_pose_convergence_checks_all_axes_and_orientation():
    from teleopit.sim.g1_o6_support.measurements import late_drift

    rows = [
        {
            "hold_time_s": t,
            "base_pose": [0, 0, 0, 1, 0, 0, 0],
            "com": [0, 0, 0],
            "hand_in_target_rung": {
                s: {"position": [0, 0, 0], "rotation": np.eye(3).ravel().tolist()}
                for s in ("left", "right")
            },
        }
        for t in range(121)
    ]
    assert late_drift(rows, 120)["status"] == "PASS"
    for r in rows:
        r["hand_in_target_rung"]["right"]["position"][1] = r["hold_time_s"] * 0.0001
    assert late_drift(rows, 120)["status"] == "FAIL"


def test_reference_window_is_measured_not_commanded():
    from teleopit.sim.g1_o6_support.transfer import mean_reference, validate_request

    rows = [
        {
            "time": i,
            "supports": {"left_hand": {"force_world": [0, 0, force]}},
            "command": 99,
        }
        for i, force in enumerate((100, 9, 11))
    ]
    assert mean_reference(rows)["supports"]["left_hand"]["force_world"] == [0, 0, 10]
    for invalid in (
        request(mode="fly"),
        request("release", cycles=3),
        request("unload", alternating=True),
    ):
        with pytest.raises(ValueError):
            validate_request(invalid)


def test_replay_rejects_short_recording_and_keeps_world_axes(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support import capacity_matrix as matrix

    rows = [
        {"time": t, "wrist_world_on_hand": {"left": [t, -1, 2, 0.1, 0.2, 0.3]}}
        for t in (0, 5, 10)
    ]
    source = SimpleNamespace()
    with pytest.raises(ValueError, match="10 s"):
        matrix.run_replay(source, "left", tmp_path, rows[:2], 0.001)
    monkeypatch.setattr(matrix, "run_hand", lambda *a, **kw: kw)
    kwargs = matrix.run_replay(source, "left", tmp_path, rows, 0.0005)
    assert kwargs["load_reference"]["follower"] is False
    assert kwargs["load_reference"]["simultaneous_wrenches_N_Nm"] == [
        r["wrist_world_on_hand"]["left"] for r in rows
    ]


def test_evidence_rejects_a_changed_raw_log(tmp_path):
    from teleopit.sim.g1_o6_support.qualification import read_run

    (tmp_path / "manifest.json").write_text(
        json.dumps({"files_sha256": {"summary.json": "wrong"}})
    )
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity mismatch"):
        read_run(tmp_path)
