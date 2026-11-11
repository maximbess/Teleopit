"""Physical contracts of the independent left/right diagnostic harness."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.metadata import resolved
from teleopit.sim.o6_diagnostics.config import DiagnosticConfig
from teleopit.sim.o6_diagnostics.harness import HandHarness


@pytest.fixture(scope="module")
def source():
    env = Environment(Config())
    return SimpleNamespace(
        model=env.model,
        data=env.data,
        hands=env.hands,
        targets=env.support_targets,
        metadata=resolved(env, snapshot(env)),
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_authentic_hand_preserves_compiled_parameters_and_has_no_world_contacts(
    source, side
):
    h = HandHarness(source, side, fixed=True)
    assert h.model.nu == 6 and h.model.neq == 5 and h.model.nq == 11
    assert len(h.hand["active"]) == 6
    for name in h.hand["active"]:
        assert name.startswith(side + "_o6_" + ("lh_" if side == "left" else "rh_"))
    np.testing.assert_allclose(
        h.data.site(h.hand["mount"]).xmat,
        source.data.site(source.hands[side]["mount"]).xmat,
    )
    for b in h.bodies:
        original = source.model.body(h.model.body(b).name).id
        np.testing.assert_array_equal(
            h.model.body_inertia[b], source.model.body_inertia[original]
        )
        assert h.model.body_mass[b] == source.model.body_mass[original]
    assert all(int(b) in h.bodies for b in h.model.geom_bodyid)
    h.command(1)
    for _ in range(100):
        mujoco.mj_step(h.model, h.data)
    assert all(
        all(h.model.geom_bodyid[g] in h.bodies for g in (c.geom1, c.geom2))
        for c in h.data.contact
    )


def test_zero_fixture_and_load_at_current_rotated_mount(source):
    h = HandHarness(source, "left")
    assert h.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert all(t == mujoco.mjtEq.mjEQ_JOINT for t in h.model.eq_type)
    before = h.data.qpos.copy()
    rotation = h.data.site(h.hand["mount"]).xmat.reshape(3, 3)
    applied = np.r_[rotation @ [2.0, 0, 0], rotation @ [0, 0, 0.3]]
    h.apply(0, applied)
    np.testing.assert_array_equal(h.data.qpos, before)
    np.testing.assert_array_equal(h.fixture, np.zeros(6))
    expected = np.zeros(h.model.nv)
    mujoco.mj_applyFT(
        h.model,
        h.data,
        applied[:3],
        applied[3:],
        h.data.site(h.hand["mount"]).xpos,
        h.base,
        expected,
    )
    np.testing.assert_allclose(h.data.qfrc_applied, expected)
    row = h.sample()
    assert not any(row["fixture_stiffness"]) and not any(row["fixture_damping"])


def test_independent_reset_and_hold_reference_and_missing_contact(source):
    h = HandHarness(source, "right")
    h.data.qpos[2] += 0.01
    mujoco.mj_forward(h.model, h.data)
    h.telemetry.start_hold(h.data)
    row = h.sample()
    assert row["from_reset"]["translation_m"] == pytest.approx(0.01)
    assert row["from_hold"]["translation_m"] == 0
    support = row["supports"]["right_hand"]
    assert support["contact_loss"] and support["slip_m_s"] is None


def test_fixed_base_momentum_balance_is_explicitly_not_evaluated(source):
    h = HandHarness(source, "left", fixed=True)
    row = h.sample()
    assert (
        row["momentum_rate"]["status"] == "not_evaluated_fixed_base_reaction_unmeasured"
    )
    assert row["momentum_rate"]["balance_residual_N_Nm"] is None


def test_guard_stops_on_fall_and_numerical_warning(source):
    from teleopit.sim.o6_diagnostics.hand_run import physics_status

    h = HandHarness(source, "right")
    cfg = DiagnosticConfig()
    reference = (h.base, h.data.body(h.base).xpos.copy())
    warnings = [w.number for w in h.data.warning]
    h.data.qpos[2] -= 0.26
    assert physics_status(h, cfg, reference, warnings) == "fall_drop"
    h.data.warning[0].number += 1
    assert physics_status(h, cfg, reference, warnings) == "new_numerical_warning"


@pytest.mark.parametrize(
    "field,value",
    [("continuation_s", 4), ("max_rotation_rad", np.nan), ("plateau_s", -1)],
)
def test_invalid_diagnostic_schedule_rejected(field, value):
    with pytest.raises(ValueError):
        replace(DiagnosticConfig(), **{field: value}).validate()


def test_terminal_failure_is_preserved_and_missing_drift_is_unknown():
    from teleopit.sim.o6_diagnostics.hand_results import (
        drift_peak,
        outcome,
        terminal_failure,
    )

    failure = {}
    terminal = {"warnings": [0, 1], "reason": "new_numerical_warning", "time_s": 1.25}
    assert terminal_failure(failure, terminal)
    assert outcome(False, failure, True) == "numerical_termination"
    assert failure["numerical_warning"] == 1.25
    assert outcome(False, {"grasp_loss": 1}, False) == "grasp_loss"
    assert drift_peak([{"from_hold": {"translation_m": None}}], "translation_m") is None


def test_schedule_has_two_seconds_unsupported_before_ramp():
    from teleopit.sim.o6_diagnostics.hand_run import load_schedule

    cfg = DiagnosticConfig()
    for time in [6.0, 7.0, 7.999]:
        phase, amount, gain, load = load_schedule(time, cfg, "primary")
        assert (phase, amount, gain, load) == ("SETTLE", 1, 0, 0)
    assert load_schedule(8.0, cfg, "primary")[0] == "LOAD_RAMP"
    assert load_schedule(10.0, cfg, "primary") == ("PLATEAU", 1, 0, 1)


def test_contact_couples_and_moment_arms_exclude_internal_contacts():
    from teleopit.sim.o6_diagnostics.mechanics import contact_wrench

    def row(ids):
        return {
            "body_ids": ids,
            "force_on_second_world_N": [0, 2, 0],
            "couple_on_second_world_Nm": [0, 0, 3],
            "position_world_m": [1, 0, 0],
        }

    np.testing.assert_allclose(
        contact_wrench([row([0, 1]), row([1, 2])], [1, 2], np.zeros(3)),
        [0, 2, 0, 0, 0, 5],
    )
    np.testing.assert_allclose(
        contact_wrench([row([1, 0])], [1], np.zeros(3)), [0, -2, 0, 0, 0, -5]
    )


def test_incomplete_plateau_and_failed_grasp_cannot_pass(source, tmp_path):
    from teleopit.sim.o6_diagnostics.hand_run import integrate_hand

    h = HandHarness(source, "left")
    # Starting far from the rung is an intentional acquisition-failure control.
    h.data.qpos[0] -= 0.2
    mujoco.mj_forward(h.model, h.data)
    cfg = replace(
        DiagnosticConfig(),
        acquire_open_s=0.02,
        acquire_close_s=0.02,
        fixture_remove_s=0.02,
        zero_fixture_settle_s=0.04,
        load_ramp_s=0.04,
        plateau_s=0.1,
        persistence_s=0.04,
    )
    result, rows = integrate_hand(h, cfg, np.zeros(6), "primary", None, tmp_path)
    assert not result["strict_pass"] and not result["plateau_complete"]
    assert not result["physical_retention"]
    assert result["plateau_achieved_s"] < cfg.plateau_s
    assert all(
        r["fixture_gain"] == 0
        for r in rows
        if r["phase"] in ("SETTLE", "LOAD_RAMP", "PLATEAU")
    )


def test_free_close_command_writes_both_hand_evidence(source, tmp_path, monkeypatch):
    from teleopit.sim.o6_diagnostics import experiments as e
    from teleopit.sim.o6_diagnostics import plots
    from teleopit.sim.o6_diagnostics.hand_run import run_hand

    diagnostic_source = SimpleNamespace(**vars(source))
    diagnostic_source.path = tmp_path / "source"
    diagnostic_source.metadata = {
        **source.metadata,
        "assets": {},
        "code_revision": "test",
    }
    cfg = replace(DiagnosticConfig(), free_schedule_s=(0.02, 0.06, 0.04, 0.06, 0.02))
    monkeypatch.setattr(e, "experiment", lambda *_: (diagnostic_source, tmp_path, cfg))
    monkeypatch.setattr(plots, "hand_plot", lambda *_: None)
    result = e.free_close("unused", tmp_path)
    full = [c for c in result["cases"] if c["only_active_joint"] is None]
    assert {(c["side"], c["gravity"]) for c in full} == {
        ("left", True),
        ("left", False),
        ("right", True),
        ("right", False),
    }
    assert all(
        c["fixed_base"] and c["diagnostic_end"]["reason"] == "schedule_complete"
        for c in full
    )
    assert (tmp_path / "free_close_summary.csv").is_file()
    assert all((tmp_path / c["case"] / "scene.xml").is_file() for c in result["cases"])
    import gzip
    import json

    first = tmp_path / full[0]["case"]
    metadata = json.loads((first / "resolved.json").read_text())
    with gzip.open(first / "physics.jsonl.gz", "rt") as stream:
        physics = json.loads(next(stream))
    assert len(metadata["generalized_position_layout"]) == 11
    assert len(physics["generalized_position_qpos"]) == 11
    with pytest.raises(ValueError):
        run_hand(
            diagnostic_source, "left", tmp_path / "bad_wrench", wrench=[np.nan] * 6
        )


def test_harness_uses_source_preload_targets_and_delay(source):
    from teleopit.sim.g1_o6.grasp_pose import GraspPose
    from teleopit.sim.g1_o6.holding import finger_targets
    from teleopit.sim.g1_o6.metadata import controller_settings

    altered = SimpleNamespace(**vars(source))
    pose = GraspPose(
        closed_pose=(0.04, 0.4, 1.2, 1.1, 1.0, 0.9), finger_preload_Nm=0.09
    )
    cfg = Config(delay_steps=2)
    altered.metadata = {
        **source.metadata,
        "controller": controller_settings(pose, cfg),
        "config": asdict(cfg),
        "grasp_pose": asdict(pose),
    }
    h = HandHarness(altered, "left", fixed=True)
    assert h.closed.tolist() == list(pose.closed_pose)
    before = h.data.qpos.copy()
    h.command(1)
    expected = finger_targets(
        1,
        np.array(pose.closed_pose),
        h.data.qpos[h.qpos],
        h.model.actuator_gainprm[:, 0],
        pose.finger_preload_Nm,
    )
    np.testing.assert_allclose(h.desired, expected, rtol=0, atol=0)
    assert max(h.target) <= 0.02
    np.testing.assert_array_equal(h.data.ctrl, np.zeros(6))
    first = h.target.copy()
    h.command(1)
    np.testing.assert_array_equal(h.data.ctrl, np.zeros(6))
    h.command(1)
    np.testing.assert_array_equal(h.data.ctrl, first)
    np.testing.assert_array_equal(h.data.qpos, before)


@pytest.mark.parametrize(
    "change",
    [
        "historical",
        "missing_controller",
        "mapping",
        "rung",
        "preload",
        "initial",
        "mount",
        "nonfinite",
    ],
)
def test_source_rejects_missing_or_inconsistent_controller_contract(
    source, tmp_path, change
):
    import copy
    import json

    from teleopit.sim.o6_diagnostics.harness import Source

    metadata = copy.deepcopy(source.metadata)
    if change == "historical":
        metadata["schema"] = "o6_diagnostic_v1"
    elif change == "missing_controller":
        del metadata["controller"]
    elif change == "mapping":
        metadata["command_joint_mapping"]["left"] = ["wrong"] * 6
    elif change == "rung":
        metadata["support_targets"]["left_hand"] = "left_ladder_rung_08"
    elif change == "preload":
        metadata["controller"]["finger_preload_Nm"] = 0.2
    elif change == "initial":
        del metadata["initial"]["base_pose"]
    elif change == "mount":
        del metadata["model"]["hands"]["left"]["mount"]
    else:
        metadata["initial"]["joint_positions"][0] = float("nan")
    (tmp_path / "resolved.json").write_text(json.dumps(metadata))
    # Contract validation precedes even loading scene.xml.
    with pytest.raises(ValueError, match="source-run"):
        Source(tmp_path)
