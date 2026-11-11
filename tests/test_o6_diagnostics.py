"""Behavioral contracts for CPU diagnostic evidence; no training dependencies."""

import mujoco
import numpy as np
import pytest


def test_moment_shift_and_rotated_mount_force():
    from teleopit.sim.o6_diagnostics.mechanics import shift_wrench

    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    wrench = np.r_[rotation @ [2, 0, 0], rotation @ [0, 0, 3]]
    np.testing.assert_allclose(
        shift_wrench(wrench, [1, 0, 0], [0, 0, 0]), [0, 2, 0, 0, 0, 5]
    )


def test_mimic_offset_and_constraint_classification():
    from teleopit.sim.o6_diagnostics.mechanics import joint_details

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
    <joint name="a"/><geom size=".1"/><body><joint name="b"/><geom size=".05"/></body>
    </body></worldbody><equality><joint joint1="b" joint2="a" polycoef=".2 .5 0 0 0"/></equality></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[:] = [0.4, 0.4]
    mujoco.mj_forward(model, data)
    result = joint_details(model, data)
    assert result["mimic"][0]["offset_rad"] == 0.2
    assert abs(result["mimic"][0]["residual_rad"]) < 1e-12
    assert result["joints"]["b"]["constraint_Nm"]["contact"] == 0


def test_all_link_momentum_matches_known_external_load():
    from teleopit.sim.o6_diagnostics.mechanics import momentum

    m = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".001" gravity="0 0 0"/>
    <worldbody><body name="base"><freejoint/><inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    start = momentum(m, d, [1])
    d.xfrc_applied[1] = [4, 0, 0, 0, 0, 2]
    for _ in range(20):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    end = momentum(m, d, [1])
    np.testing.assert_allclose(
        (
            np.r_[end["linear_kg_m_s"], end["angular_kg_m2_s"]]
            - np.r_[start["linear_kg_m_s"], start["angular_kg_m2_s"]]
        )
        / 0.02,
        [4, 0, 0, 0, 0, 2],
        atol=1e-8,
    )


def test_articulated_link_motion_contributes_to_whole_robot_momentum():
    from teleopit.sim.o6_diagnostics.mechanics import momentum

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="base"><freejoint/>
    <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
    <body name="moving"><joint axis="0 0 1"/><inertial pos="1 0 0" mass="1" diaginertia=".2 .2 .2"/>
    </body></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[-1] = 1
    mujoco.mj_forward(m, d)
    result = momentum(m, d, [1, 2])
    np.testing.assert_allclose(result["linear_kg_m_s"], [0, 1, 0])
    np.testing.assert_allclose(result["angular_kg_m2_s"], [0, 0, 0.7])


def test_off_center_load_at_rotated_mount_changes_momentum_with_correct_sign():
    from teleopit.sim.o6_diagnostics.mechanics import momentum

    m = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".001" gravity="0 0 0"/>
    <worldbody><body quat=".70710678 0 0 .70710678"><freejoint/>
    <inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/><site name="mount" pos="1 0 0"/>
    </body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mount = d.site("mount")
    rot = mount.xmat.reshape(3, 3)
    mujoco.mj_applyFT(
        m, d, rot @ [0, 2, 0], rot @ [0, 0, 3], mount.xpos, 1, d.qfrc_applied
    )
    mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    result = momentum(m, d, [1])
    np.testing.assert_allclose(
        np.r_[result["linear_kg_m_s"], result["angular_kg_m2_s"]] / 0.001,
        [-2, 0, 0, 0, 0, 5],
        atol=1e-8,
    )


def test_reference_load_uses_simultaneous_zero_assistance_hold_samples():
    from teleopit.sim.o6_diagnostics.loads import select_references

    def row(t, f, m, phase="HOLD", gain=0):
        return {
            "time": t,
            "phase": phase,
            "assistance_gain": gain,
            "wrist_world_on_hand": {"left": [f, 0, 0, 0, m, 0]},
        }

    rows = [
        row(0, 100, 100, "ACQUIRE", 1),
        row(1, -10, 1),
        row(2, -2, -4),
        row(3, -3, 2),
    ]
    selected = select_references(rows, "left")
    assert selected[0]["selection"] == ["max_moment"]
    assert selected[0]["time_s"] == 2
    assert selected[0]["wrench_world_at_mount"] == [-2, 0, 0, 0, -4, 0]
    assert len(select_references([row(1, -1, -2)], "left")) == 1
    with pytest.raises(ValueError):
        select_references(rows[:1], "left")
    continued = row(4, 200, 200)
    continued["diagnostic_unscored"] = True
    assert select_references(rows + [continued], "left") == selected


def test_event_persistence_tracks_each_gate_independently():
    from teleopit.sim.o6_diagnostics.events import Events

    events = Events(0.1, 0.3)
    for t, values in [
        (0.1, {"a": 2}),
        (0.2, {"a": 2}),
        (0.3, {"b": 2}),
        (0.4, {"b": 2}),
        (0.5, {"b": 2}),
    ]:
        events.update(t, "HOLD", values, {"a": 1, "b": 1})
    assert "a" not in events.persistent
    assert events.persistent["b"]["first_crossing_s"] == 0.3
    assert events.persistent["b"]["persistence_onset_s"] == 0.5
    events.update(0.6, "HOLD", {"a": 2}, {"a": 1, "b": 1})
    assert "a" not in events.persistent


def test_pose_is_rung_relative_without_relabelling_world_pose():
    from teleopit.sim.o6_diagnostics.mechanics import relative_pose

    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pose = relative_pose([1, 2, 0], np.eye(3), [1, 0, 0], rotation)
    np.testing.assert_allclose(pose["position_m"], [2, 0, 0])
    np.testing.assert_allclose(np.array(pose["rotation"]).reshape(3, 3), rotation.T)


def test_continuation_keeps_original_failure_and_zero_assistance(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.run import run_trial
    from teleopit.sim.o6_diagnostics import nominal, plots
    from teleopit.sim.o6_diagnostics.config import DiagnosticConfig
    from teleopit.sim.o6_diagnostics.io import read_rows

    monkeypatch.setattr(plots, "nominal_plot", lambda *args: None)
    cfg = Config(acquire_seconds=0.1, remove_seconds=0.1, stability_seconds=0.04)
    expected = run_trial(cfg, tmp_path / "original")
    result = nominal.run_nominal(
        cfg, tmp_path / "diagnostic", DiagnosticConfig(continuation_s=0.06)
    )
    original = read_rows(tmp_path / "original/steps.jsonl")
    observed = read_rows(tmp_path / "diagnostic/steps.jsonl")
    assert expected["final_phase"] == "ABORT"
    assert all(result["original_strict_outcome"][k] == v for k, v in expected.items())
    assert all(
        new[k] == v for old, new in zip(original, observed) for k, v in old.items()
    )
    continued = [r for r in observed if r.get("diagnostic_unscored")]
    assert continued and all(
        r["phase"] == "HOLD" and r["next_phase"] == "HOLD" for r in continued
    )
    assert all(
        r["assistance_gain"] == 0 and not any(r["assistance_world"]) for r in continued
    )
    assert result["diagnostic_end_time_s"] > result["would_abort_at"]
    assert result["diagnostic_end_reason"] == "continuation_time_limit"
    assert result["original_strict_outcome"]["static_hold_passed"] is False
    assert observed[-1]["base_pose"] != original[-1]["base_pose"]


def test_successful_diagnostic_matches_ordinary_and_never_continues(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.run import run_trial
    from teleopit.sim.o6_diagnostics import nominal, plots
    from teleopit.sim.o6_diagnostics.config import DiagnosticConfig
    from teleopit.sim.o6_diagnostics.io import read_rows

    monkeypatch.setattr(plots, "nominal_plot", lambda *args: None)
    cfg = Config(hold_seconds=0.04)
    ordinary = run_trial(cfg, tmp_path / "ordinary")
    result = nominal.run_nominal(
        cfg, tmp_path / "diagnostic", DiagnosticConfig(continuation_s=0.06)
    )
    strict = result["original_strict_outcome"]
    assert strict is not None and strict["final_phase"] == "HOLD_COMPLETE"
    assert strict["static_hold_passed"] and not strict["release_reacquire_passed"]
    assert strict["abort_time_s"] is None and result["would_abort_at"] is None
    assert not result["continuation_unscored"]
    assert all(strict[k] == v for k, v in ordinary.items())
    before = read_rows(tmp_path / "ordinary/steps.jsonl")
    after = read_rows(tmp_path / "diagnostic/steps.jsonl")
    assert len(before) == len(after)
    assert all(new[k] == v for old, new in zip(before, after) for k, v in old.items())


def test_historical_reader_preserves_schema_axes_targets_and_failure(tmp_path):
    import json

    from teleopit.sim.o6_diagnostics.io import read_rows

    # A v1 record is data, never input for reconstructing the current controller.
    old = {
        "time": 5.54,
        "phase": "HOLD",
        "next_phase": "ABORT",
        "support_targets": {"left_hand": "left_ladder_rung_08"},
        "wrist_world_on_hand": {"left": [-1.0, 2.0, -30.0, -0.3, 0.2, 0.1]},
        "diagnostic_v1": {"from_hold": {"translation_m": 0.0947}},
    }
    path = tmp_path / "historical.jsonl"
    path.write_text(json.dumps(old) + "\n")
    original = path.read_bytes()
    assert read_rows(path) == [old]
    assert path.read_bytes() == original
