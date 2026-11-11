"""Preserve raw evidence and add portable scenes with identical compiled physics."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import numpy as np

RUNS = (
    "control_before",
    "control_after",
    "control_final",
    "hold_60s",
    "hold_120s",
    "correction_30s",
    "correction_60s",
    "correction_120s",
    "capacity_initial",
    "capacity_settled",
    "capacity_repeat",
    "capacity_fine",
    "free_close",
    "free_close_final",
    "gates",
    "gates_verified",
    "videos",
)
ASSETS = ("assets/robots/unitree_g1", "teleopit/sim/o6_grasp/assets")
DOCS = Path("docs/g1_o6/support_transfer")
ROOT = Path("outputs/g1_o6/support_transfer")


def sha_stream(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def sha_file(path):
    with path.open("rb") as stream:
        return sha_stream(stream)


def directory_files(directories):
    return {
        p
        for directory in directories
        for p in directory.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
    }


def check_artifacts():
    return {
        p
        for p in ROOT.iterdir()
        if p.is_file() and p.suffix in (".json", ".txt", ".py", ".npz")
    }


def collect_files():
    directories = [ROOT / name for name in RUNS] + [Path(p) for p in ASSETS] + [DOCS]
    files = directory_files(directories) | check_artifacts()
    files.update(Path("scripts").glob("*g1_o6*support*.py"))
    files.add(Path("scripts/g1_o6_state_video.py"))
    return sorted(
        p for p in files if p.name not in ("artifact.json", "evidence_manifest.json")
    )


def portable_scene(path):
    tree = ET.parse(path)
    for element in tree.iter():
        name = element.get("file")
        if name:
            asset = Path(name).resolve()
            asset.relative_to(Path.cwd())  # Do not collect unrelated external paths.
            if not asset.is_file():
                raise FileNotFoundError(asset)
            element.set(
                "file", os.path.relpath(asset, path.parent.resolve()).replace("\\", "/")
            )
    return ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)


def compare_models(original, portable):
    before = mujoco.MjModel.from_xml_path(str(original))
    after = mujoco.MjModel.from_xml_path(str(portable))
    checked = []
    for name in dir(before):
        value = getattr(before, name)
        if isinstance(value, np.ndarray) and not name.endswith("pathadr"):
            np.testing.assert_array_equal(value, getattr(after, name), err_msg=name)
            checked.append(name)
    for name in ("timestep", "tolerance", "integrator", "cone", "solver", "iterations"):
        assert getattr(before.opt, name) == getattr(after.opt, name), name
    return len(checked)


class Archive:
    def __init__(self, output):
        self.zip = zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6)
        self.hashes = {}

    def file(self, path, name=None):
        name = name or path.as_posix()
        self.zip.write(path, name)
        self.hashes[name] = sha_file(path)

    def data(self, name, payload):
        if not isinstance(payload, bytes):
            payload = (json.dumps(payload, indent=2) + "\n").encode()
        self.zip.writestr(name, payload)
        self.hashes[name] = hashlib.sha256(payload).hexdigest()


def add_files(archive, files, staging):
    scenes, manifests = [], []
    for path in files:
        if path.name == "scene.xml":
            payload = portable_scene(path)
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            fields = compare_models(path, target)
            archive.file(path, (path.parent / "scene_original.xml").as_posix())
            archive.data(path.as_posix(), payload)
            scenes.append(
                {
                    "scene": path.as_posix(),
                    "identical_numeric_arrays": fields,
                    "original_scene": "scene_original.xml",
                }
            )
        elif path.name == "manifest.json" and path.is_relative_to(ROOT):
            manifests.append(path)
            archive.file(path, (path.parent / "manifest_original.json").as_posix())
        else:
            archive.file(path)
    return scenes, manifests


def add_manifests(archive, paths):
    # Deepest runs first so parent indexes can reference the new child manifests.
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        value = json.loads(path.read_text())
        parent = path.parent.as_posix() + "/"
        value["files_sha256"] = {
            name[len(parent) :]: sha
            for name, sha in archive.hashes.items()
            if name.startswith(parent) and "/" not in name[len(parent) :]
        }
        value["portable_archive"] = {
            "original_manifest": "manifest_original.json",
            "scene_rewrite": "file references only; all compiled numeric model arrays identical",
        }
        for child in value.get("child_runs", {}).values():
            child["sha256"] = archive.hashes[parent + child["manifest"]]
        archive.data(path.as_posix(), value)


def verify(output):
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("evidence_manifest.json"))
        for name, expected in manifest["files_sha256"].items():
            with archive.open(name) as stream:
                assert sha_stream(stream) == expected, name
    return manifest


def package(output, staging):
    if output.exists() or staging.exists():
        raise FileExistsError("choose new archive and staging paths")
    for name in ASSETS:
        shutil.copytree(name, staging / name)
    archive = Archive(output)
    try:
        scenes, manifests = add_files(archive, collect_files(), staging)
        add_manifests(archive, manifests)
        record = {
            "schema": "g1_o6_support_archive_v1",
            "implementation_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "physical_assets_included": True,
            "portable_scenes": scenes,
            "files_sha256": dict(archive.hashes),
        }
        archive.data("evidence_manifest.json", record)
    finally:
        archive.zip.close()
    verified = verify(output)
    checksum = sha_file(output)
    output.with_suffix(".zip.sha256").write_text(checksum + "  " + output.name + "\n")
    receipt = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": checksum,
        "implementation_revision": verified["implementation_revision"],
        "verified_files": len(verified["files_sha256"]),
        "physical_assets_included": True,
        "portable_scene_count": len(verified["portable_scenes"]),
        "publication_status": "not uploaded",
    }
    (DOCS / "artifact.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    package(args.output, args.staging)
