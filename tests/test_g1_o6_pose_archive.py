"""Portable evidence keeps originals and recompiles the same physical model."""

import json
import subprocess
import zipfile
from pathlib import Path

import mujoco


def test_campaign_archive_relocates_meshes_and_retains_source_and_originals(
    tmp_path, monkeypatch
):
    from scripts import package_g1_o6_support as package

    relative = tmp_path.relative_to(Path.cwd())
    assets, root, docs = (relative / n for n in ("assets", "runs", "docs"))
    for p in (assets, root / "control", docs):
        p.mkdir(parents=True)
    mesh = assets / "tetra.obj"
    mesh.write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    scene = root / "control/scene.xml"
    scene.write_text(
        f'<mujoco><asset><mesh name="shape" file="{mesh.resolve().as_posix()}"/></asset>'
        '<worldbody><body><freejoint/><geom type="mesh" mesh="shape"/></body></worldbody></mujoco>'
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (scene.parent / "resolved.json").write_text(json.dumps({"code_revision": revision}))
    (scene.parent / "manifest.json").write_text(json.dumps({"files_sha256": {}}))
    monkeypatch.setattr(package, "ASSETS", (assets.as_posix(),))
    archive, staging = tmp_path / "test.zip", tmp_path / "portable"
    package.package(archive, staging, root, docs, ("control",))
    receipt = json.loads((docs / "artifact.json").read_text())
    assert receipt["portable_scene_count"] == 1
    with zipfile.ZipFile(archive) as saved:
        original = saved.read((scene.parent / "scene_original.xml").as_posix())
        portable = saved.read(scene.as_posix())
        assert original == scene.read_bytes() and original != portable
        assert any(
            n.startswith("provenance/" + revision + "/teleopit/")
            for n in saved.namelist()
        )
    relocated = mujoco.MjModel.from_xml_path(str(staging / scene))
    baseline = mujoco.MjModel.from_xml_path(str(scene))
    assert (
        package.compare_numeric_fields(baseline, relocated, ("npaths", "nbuffer")) > 100
    )
    assert package.verify(archive)["source_revisions_included"] == [revision]
