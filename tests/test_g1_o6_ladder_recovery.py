"""Recover a wrong scene and reproduce acquisition using actual coupled physics."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import mujoco


def test_scene_recovery_and_acquisition_prefix_preserve_recorded_states(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from scripts.g1_o6_ladder_acquisition import supplement
    from scripts.g1_o6_ladder_reconstruct import reconstruct
    from scripts.verify_g1_o6_ladder_campaign import verify_transition_coverage
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6_support.ladder_campaign import (
        code_provenance,
        run_checksums,
    )
    from teleopit.sim.ladder_config import geometry_matrix

    # Acquisition is shortened only in this software fixture, never in the campaign.
    cfg = replace(Config(), acquire_seconds=0.02, remove_seconds=0.02)
    ladder = geometry_matrix()["G0"]
    env = Environment(cfg, ladder=ladder)
    source = tmp_path / "runs" / "software_fixture"
    source.mkdir(parents=True)
    metadata, rows = recorded_prefix(env)
    (source / "resolved.json").write_text(json.dumps(metadata))
    (source / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "scene.xml").write_text(
        '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    )
    record = {
        "id": "software_fixture",
        "identity": "software fixture; no qualification",
        "output": "runs/software_fixture",
        "files_sha256": run_checksums(source),
        "payload": {
            "provenance": code_provenance(),
            "condition": {"geometry": "G0", "material": None},
            "ladder": asdict(ladder),
        },
    }
    record = json.loads(json.dumps(record))  # Match the ledger's JSON boundary.
    proof = reconstruct(tmp_path, record)
    assert proof["reset_qpos_equal"] and not proof["original_scene_has_coupled_model"]
    recovered = mujoco.MjModel.from_binary_path(
        str(tmp_path / "reconstructions/software_fixture/scene.mjb")
    )
    assert recovered.nq == env.model.nq
    assert reconstruct(tmp_path, record) == proof
    prefix = supplement(tmp_path, record)
    assert prefix["compared_control_rows"] == len(rows)
    assert prefix["end_s"] < 1
    assert supplement(tmp_path, record) == prefix
    assert verify_transition_coverage(
        tmp_path / "acquisition_windows/software_fixture", 0.001
    ) == ["INITIALIZE", "ACQUIRE", "REMOVE_ASSISTANCE", "HOLD"]


def recorded_prefix(env):
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.metadata import resolved
    from teleopit.sim.g1_o6_support.ladder_identity import physical_identity
    from teleopit.sim.ladder_config import compiled_ladder

    rows = [snapshot(env)]
    metadata = resolved(env, rows[0])
    metadata.update(
        ladder=compiled_ladder(env.model, env.ladder),
        material=None,
        physical_identity=physical_identity(env.model),
    )
    hold_entry = None
    while env.data.time < 1:
        rows.append(env.step())
        if env.phase == "HOLD" and hold_entry is None:
            hold_entry = env.data.time
        if hold_entry is not None and env.data.time >= hold_entry + 0.259:
            return metadata, rows
    raise AssertionError("software fixture never reached HOLD")
