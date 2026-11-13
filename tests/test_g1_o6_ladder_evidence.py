"""Independent evidence checks reject coefficient, force and artifact tampering."""

import gzip
import json

import pytest

from scripts.g1_o6_ladder_reconstruct import verify_existing
from scripts.verify_g1_o6_ladder_campaign import check_requested_material, verify_forces
from teleopit.sim.g1_o6_support.ladder_campaign import file_hash


def test_requested_material_cannot_be_satisfied_by_a_different_pair_coefficient():
    requested = {"scope": "H", "mu": 0.4}
    audit = {
        **requested,
        "requested_tangential_mu": 0.4,
        "pairs": [
            {"robot": "left_o6_thumb", "friction": [0.4, 0.4, 0.02, 0.002, 0.002]}
        ],
    }
    check_requested_material(requested, audit)
    audit["pairs"][0]["friction"][0] = 0.8
    with pytest.raises(AssertionError):
        check_requested_material(requested, audit)
    audit["pairs"][0].update(
        robot="left_foot_collision", friction=[0.4, 0.4, 0.02, 0.002, 0.002]
    )
    with pytest.raises(AssertionError):
        check_requested_material(requested, audit)


def test_force_stream_detects_hidden_assistance_and_missing_steps(tmp_path):
    result = {"physics_audit": {"scored_steps": 2, "nonzero_applied_steps": 0}}
    rows = [{"qfrc_applied": [0], "xfrc_applied": [[0] for _ in range(2)]}] * 2
    path = tmp_path / "applied_forces.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    assert verify_forces(tmp_path, result)["status"] == "PASS"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"qfrc_applied": [1], "xfrc_applied": [[0]]}) + "\n")
        stream.write(json.dumps(rows[1]) + "\n")
    with pytest.raises(AssertionError):
        verify_forces(tmp_path, result)
    result["physics_audit"]["nonzero_applied_steps"] = 1
    assert verify_forces(tmp_path, result)["status"] == "FAIL"
    result["physics_audit"]["scored_steps"] = 3
    with pytest.raises(AssertionError):
        verify_forces(tmp_path, result)


def test_reconstructed_scene_resume_checks_both_original_and_recovered_bytes(tmp_path):
    original, recovered = tmp_path / "original", tmp_path / "recovered"
    original.mkdir()
    recovered.mkdir()
    (original / "scene.xml").write_text("original")
    (recovered / "scene.xml").write_text("recovered")
    proof = {
        "original_scene_sha256": file_hash(original / "scene.xml"),
        "files_sha256": {"scene.xml": file_hash(recovered / "scene.xml")},
    }
    (recovered / "proof.json").write_text(json.dumps(proof))
    assert verify_existing(original, recovered) == proof
    (original / "scene.xml").write_text("altered")
    with pytest.raises(AssertionError):
        verify_existing(original, recovered)
    (original / "scene.xml").write_text("original")
    (recovered / "scene.xml").write_text("altered")
    with pytest.raises(AssertionError):
        verify_existing(original, recovered)


def test_transition_window_rejects_a_missing_physics_interval():
    from scripts.verify_g1_o6_ladder_campaign import assert_transition_interval

    times = [i / 1000 for i in range(750, 1251)]
    assert_transition_interval(times, 1, 0.001)
    with pytest.raises(AssertionError):
        assert_transition_interval(times[:100] + times[120:], 1, 0.001)


def test_archive_preserves_original_bytes_and_adds_a_portable_scene(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from zipfile import ZipFile

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from scripts import package_g1_o6_ladder as packager

    monkeypatch.chdir(tmp_path)
    original = Path("evidence/run/scene.xml")
    original.parent.mkdir(parents=True)
    raw = b'<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    original.write_bytes(raw)
    docs = Path("docs")
    docs.mkdir()
    monkeypatch.setattr(packager, "ROOT", Path("evidence"))
    monkeypatch.setattr(packager, "DOCS", docs)
    monkeypatch.setattr(packager, "ASSETS", ())
    monkeypatch.setattr(packager, "collect_files", lambda *args: [original])
    monkeypatch.setattr(packager, "implementation_revisions", lambda files: [])
    monkeypatch.setattr(
        packager.subprocess, "check_output", lambda *args, **kw: "software-fixture"
    )
    output = Path("evidence.zip")
    packager.package(output, Path("portable"))
    with ZipFile(output) as archive:
        assert archive.read(original.as_posix()) == raw
        assert "evidence/run/scene_portable.xml" in archive.namelist()
        manifest = json.loads(archive.read("evidence_manifest.json"))
        assert manifest["original_bytes_preserved"]
        assert manifest["portable_scenes"][0]["identical_public_numeric_fields"] > 400
    assert original.read_bytes() == raw
    with pytest.raises(FileExistsError):
        packager.package(output, Path("portable"))
