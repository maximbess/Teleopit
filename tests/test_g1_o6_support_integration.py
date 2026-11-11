"""Short simulator/software checks, separate from duration/load qualification."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


@pytest.fixture(scope="module")
def short_hold(tmp_path_factory):
    from teleopit.sim.g1_o6_support.long_hold import run_hold

    path = tmp_path_factory.mktemp("short_hold")
    result = run_hold(path, seconds=0.08)
    assert result["strict_status"] == "PASS"
    assert not result["progression_justified"]
    assert result["physics_audit"]["no_external_support"]
    return path


def test_settled_source_preserves_authentic_joint_reset_and_load_audit(short_hold):
    from teleopit.sim.g1_o6_support.capacity import (
        CapacityConfig,
        CapacityTrial,
        qualified_source,
        run_capacity,
    )
    from teleopit.sim.o6_diagnostics.harness import HandHarness

    source = qualified_source(short_hold, "settled")
    assert source.sample_reference["time_s"] > 5
    for side in ("left", "right"):
        h = HandHarness(source, side)
        for name in h.hand["active"]:
            np.testing.assert_allclose(
                h.data.joint(name).qpos, source.data.joint(name).qpos
            )
        trial = CapacityTrial(
            h, CapacityConfig(), np.zeros(6), "support_capacity_v1", None
        )
        assert trial.capacity_schedule(0, trial.cfg)[1] == 1
        h.apply(0, np.zeros(6))
        trial.audit_applied_load()
        assert not trial.failure
        h.fixture[:] = 1
        h.data.qfrc_applied[0] = 1
        trial.audit_applied_load()
        assert set(trial.failure) == {"fixture_support", "undeclared_applied_force"}
    with pytest.raises(ValueError, match="source state"):
        qualified_source(short_hold, "unknown")
    for invalid in (
        [0] * 5,
        [float("nan")] * 6,
        [51, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
    ):
        with pytest.raises(ValueError):
            run_capacity(source, "left", "unused", wrench=invalid)


def test_capacity_and_world_replay_step_use_the_existing_servo_harness(
    short_hold, tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import capacity
    from teleopit.sim.g1_o6_support.capacity_matrix import ReplayTrial
    from teleopit.sim.o6_diagnostics.harness import HandHarness

    source = capacity.qualified_source(short_hold)
    # Short software smoke: explicitly not a load qualification protocol.
    cfg = capacity.CapacityConfig(
        acquire_open_s=0.02,
        acquire_close_s=0.02,
        fixture_remove_s=0.02,
        zero_fixture_settle_s=0.02,
        load_ramp_s=0.02,
        plateau_s=0.02,
        ramp_down_s=0.02,
        recovery_s=0.02,
    )
    monkeypatch.setattr(capacity, "CapacityConfig", lambda: cfg)
    result = capacity.run_capacity(source, "right", tmp_path / "short_capacity")
    assert result["protocol"] == "support_capacity_v1"
    assert not result["strict_pass"]  # A transient/no acquired grasp cannot pass.
    h = HandHarness(source, "left")
    replay = ReplayTrial(
        h,
        cfg,
        np.zeros(6),
        "support_replay_v1",
        None,
        time=[20, 30],
        wrenches=[[1, -2, 3, 0.1, 0.2, 0.3], [2, -1, 4, 0.2, 0.3, 0.4]],
    )
    replay.clock = 0.11  # Inside the plateau (starts at 0.10).
    replay.physics_step(io.StringIO())
    np.testing.assert_allclose(
        h.external, [1.001, -1.999, 3.001, 0.1001, 0.2001, 0.3001], atol=1e-12
    )
    assert np.max(np.abs(h.fixture)) == 0


def test_transfer_runner_short_hold_only_integration_retains_capacity_block(
    short_hold, tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import transfer
    from teleopit.sim.g1_o6_support.protocol import Step
    from teleopit.sim.g1_o6_support.qualification import identity, read_run

    source = read_run(short_hold)
    proof = SimpleNamespace(
        source=source,
        identity=identity(source["metadata"]),
        passed_wrenches={"left": [], "right": []},
    )
    monkeypatch.setattr(
        transfer,
        "transfer_steps",
        lambda *args: [Step("BASELINE", 0.52, 0.54, hold=True)],
    )
    request = {
        "mode": "load",
        "side": "left",
        "target_share": 0.05,
        "cycles": 1,
        "alternating": False,
        "three_seconds": 1,
        "plateau_seconds": 10,
        "physics_dt": 0.001,
    }
    path = tmp_path / "runner_smoke"
    path.mkdir()
    result = transfer.execute_transfer(proof, path, request)
    assert result["physical_status"] == "PASS"
    assert result["capacity_status"] == "BLOCKED"
    assert not result["progression_justified"]
    assert result["physics_audit"]["nonzero_applied_steps"] == 0
    rows = [
        json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()
    ]
    assert {r["experiment_phase"] for r in rows if "experiment_phase" in r} == {
        "BASELINE"
    }
    # This test deliberately stops before load transfer; no qualification claim.
    with pytest.raises(FileExistsError):
        transfer.run_transfer(Path("missing"), path)


def test_recorded_source_parameters_differ_from_fixture_only_when_declared(short_hold):
    from teleopit.sim.g1_o6_support.qualification import Evidence
    from teleopit.sim.o6_diagnostics.io import write_json

    book = short_hold.parent / "evidence.json"
    write_json(book, {"source_run": short_hold.name, "long_holds": [short_hold.name]})
    evidence = Evidence(book)
    request = {
        "mode": "load",
        "side": "left",
        "target_share": 0.05,
        "plateau_seconds": 10,
    }
    assert evidence.reasons(request) == [
        "stage_1_requires_120s_nominal_repeat_fine_and_convergence",
        "stage_2_requires_both_hand_tested_wrench_reserve",
    ]
    model = mujoco.MjModel.from_xml_path(str(short_hold / "scene.xml"))
    assert model.opt.timestep == 0.001


def test_capacity_campaign_blocks_replay_after_real_unacquired_short_smoke(
    short_hold, tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import capacity
    from teleopit.sim.g1_o6_support.capacity_matrix import run_matrix

    cfg = capacity.CapacityConfig(
        acquire_open_s=0.02,
        acquire_close_s=0.02,
        fixture_remove_s=0.02,
        zero_fixture_settle_s=0.02,
        load_ramp_s=0.02,
        plateau_s=0.02,
        ramp_down_s=0.02,
        recovery_s=0.02,
    )
    monkeypatch.setattr(capacity, "CapacityConfig", lambda: cfg)
    result = run_matrix(short_hold, tmp_path / "matrix", state="initial")
    assert all(
        s["loading_status"] == "BLOCKED" and s["replay"]["physical_status"] == "BLOCKED"
        for s in result["sides"].values()
    )


def test_capacity_accounting_rejects_fixture_and_changed_physics(short_hold):
    import copy

    from teleopit.sim.g1_o6_support.qualification import capacity_wrenches, read_run

    source = read_run(short_hold)["metadata"]
    meta = {
        "side": "left",
        "source_grasp_pose": source["grasp_pose"],
        "source_config": source["config"],
        "source_support_targets": source["support_targets"],
        "assets": source["assets"],
        "controller": source["controller"],
        "fixed_base": False,
        "gravity_world_m_s2": [0, 0, -9.81],
    }
    # Pure acceptance predicate on synthetic records; never written as evidence.
    result = {
        "strict_pass": True,
        "physical_retention": True,
        "plateau_complete": True,
        "requested_wrench_world_at_mount": [1, -2, 3, 0, 0, 0],
    }
    run = {"metadata": meta, "summary": result}
    assert capacity_wrenches([run], source)["left"] == [
        result["requested_wrench_world_at_mount"]
    ]
    for change in (
        {"fixed_base": True},
        {"source_config": {**source["config"], "friction_scale": 2}},
        {"gravity_world_m_s2": [0, 0, 0]},
    ):
        changed = copy.deepcopy(run)
        changed["metadata"].update(change)
        assert not capacity_wrenches([changed], source)["left"]
