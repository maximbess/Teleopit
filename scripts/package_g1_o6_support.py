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


def check_artifacts(root):
    return {
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix in (".json", ".txt", ".py", ".npz")
    }


def collect_files(root=ROOT, docs=DOCS, runs=RUNS):
    directories = [root / name for name in runs] + [Path(p) for p in ASSETS] + [docs]
    files = directory_files(directories) | check_artifacts(root)
    files.update(Path("scripts").glob("*g1_o6*.py"))
    files.add(Path("scripts/g1_o6_state_video.py"))
    return sorted(
        p for p in files if p.name not in ("artifact.json", "evidence_manifest.json")
    )


def portable_scene(path):
    tree = ET.parse(path)
    for element in tree.iter():
        name = element.get("file")
        if name:
            asset = Path(name)
            asset = (asset if asset.is_absolute() else path.parent / asset).resolve()
            asset.relative_to(Path.cwd())  # Do not collect unrelated external paths.
            if not asset.is_file():
                raise FileNotFoundError(asset)
            element.set(
                "file", os.path.relpath(asset, path.parent.resolve()).replace("\\", "/")
            )
    return ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)


def compare_numeric_fields(before, after, ignored=()):
    checked = []
    for name in dir(before):
        if name.startswith("_") or name.endswith("pathadr") or name in ignored:
            continue
        value = getattr(before, name)
        if isinstance(value, (np.ndarray, int, float)):
            np.testing.assert_array_equal(value, getattr(after, name), err_msg=name)
            checked.append(name)
    return len(checked)


def compare_models(original, portable):
    before = mujoco.MjModel.from_xml_path(str(original))
    after = mujoco.MjModel.from_xml_path(str(portable))
    # Rewriting filenames changes the path string table and its buffer allocation.
    fields = compare_numeric_fields(before, after, ("npaths", "nbuffer"))
    return fields + compare_numeric_fields(before.opt, after.opt)


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


def add_files(archive, files, staging, root=ROOT):
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
                    "identical_numeric_fields": fields,
                    "excluded_bookkeeping": "private fields, filename offsets, npaths, nbuffer",
                    "original_scene": "scene_original.xml",
                }
            )
        elif path.name == "manifest.json" and path.is_relative_to(root):
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
            "scene_rewrite": "file references only; compiled public numeric physics fields identical",
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


def implementation_revisions(files):
    revisions = {
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    }
    for path in files:
        if path.name == "resolved.json":
            record = json.loads(path.read_text())
            revision = record.get("code_revision")
            if revision:
                revisions.add(revision)
    return sorted(revisions)


def add_sources(archive, revisions):
    for revision in revisions:
        paths = subprocess.check_output(
            [
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                revision,
                "teleopit",
                "scripts",
                "pyproject.toml",
                "requirements.txt",
            ],
            text=True,
        ).splitlines()
        for name in paths:
            if Path(name).suffix in (".py", ".toml", ".yaml", ".yml", ".json"):
                payload = subprocess.check_output(
                    ["git", "show", revision + ":" + name]
                )
                archive.data("provenance/" + revision + "/" + name, payload)


def package(output, staging, root=ROOT, docs=DOCS, runs=RUNS):
    if output.exists() or staging.exists():
        raise FileExistsError("choose new archive and staging paths")
    for name in ASSETS:
        shutil.copytree(name, staging / name)
    archive = Archive(output)
    try:
        files = collect_files(root, docs, runs)
        scenes, manifests = add_files(archive, files, staging, root)
        add_manifests(archive, manifests)
        revisions = implementation_revisions(files)
        add_sources(archive, revisions)
        record = {
            "schema": "g1_o6_support_archive_v1",
            "implementation_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "physical_assets_included": True,
            "source_revisions_included": revisions,
            "source_usage": "Committed source snapshots under provenance/<revision>; per-run resolved.json also retains any dirty diff and source hashes. Assets remain at archive root.",
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
    (docs / "artifact.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT)
    parser.add_argument("--docs-root", type=Path, default=DOCS)
    parser.add_argument("--runs", nargs="+", default=RUNS)
    args = parser.parse_args()
    package(args.output, args.staging, args.runs_root, args.docs_root, args.runs)
