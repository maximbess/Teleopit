"""Recover owned model XML from frozen inputs, preserving every original artifact.

Run at the recorded campaign revision before changing simulation source. This
does no dynamics and cannot turn a physical failure into a passing experiment.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.g1_o6.material import Material
from teleopit.sim.g1_o6_support.ladder_campaign import code_provenance, file_hash
from teleopit.sim.g1_o6_support.ladder_identity import (
    physical_identity,
    verify_reviewed_model,
)
from teleopit.sim.ladder_config import LadderConfig, compiled_ladder


def read(path):
    return json.loads(path.read_text())


def assert_same_arrays(first, second):
    count = 0
    for name in dir(first):
        value = getattr(first, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(second, name), err_msg=name)
            count += 1
    assert first.opt.timestep == second.opt.timestep
    return count


def check_source(record):
    provenance = record.get("payload", {}).get(
        "provenance", record.get("target_provenance")
    )
    for name, expected in provenance["source_sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(
                "checkout recorded simulation source before reconstruction: " + name
            )
    actual = code_provenance()
    assert actual["assets"] == provenance["assets"]
    assert actual["runtime"] == provenance["runtime"]
    return provenance


def reconstruct(root, record):
    source = root / record["output"]
    destination = root / "reconstructions" / record["id"]
    if (destination / "proof.json").exists():
        return verify_existing(source, destination)
    provenance = check_source(record)
    metadata = read(source / "resolved.json")
    original = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
    env = rebuild_environment(record, metadata)
    reference_check = verify_environment(root, record, metadata, env)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "scene.xml").write_text(env.xml)
    mujoco.mj_saveModel(env.model, str(destination / "scene.mjb"), None)
    restored = mujoco.MjModel.from_xml_path(str(destination / "scene.xml"))
    arrays = assert_same_arrays(env.model, restored)
    proof = reconstruction_proof(
        record, provenance, source, destination, original, env, reference_check, arrays
    )
    (destination / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def verify_existing(source, destination):
    proof = read(destination / "proof.json")
    for name, expected in proof["files_sha256"].items():
        assert file_hash(destination / name) == expected
    assert file_hash(source / "scene.xml") == proof["original_scene_sha256"]
    return proof


def rebuild_environment(record, metadata):
    condition = record.get("payload", {}).get("condition")
    ladder = LadderConfig(**record["payload"]["ladder"]) if condition else None
    material = (
        Material(**condition["material"])
        if condition and condition["material"]
        else None
    )
    return Environment(
        Config(**metadata["config"]),
        pose=GraspPose(**metadata["grasp_pose"]),
        ladder=ladder,
        material=material,
    )


def verify_environment(root, record, metadata, env):
    first = metadata["initial"]
    np.testing.assert_array_equal(
        env.data.qpos, [*first["base_pose"], *first["joint_positions"]]
    )
    if record.get("payload", {}).get("condition"):
        assert physical_identity(env.model) == metadata["physical_identity"]
        assert compiled_ladder(env.model, env.ladder) == metadata["ladder"]
        assert env.audit.get("material") == metadata["material"]
        return None
    return verify_reviewed_model(root / "reference" / "reviewed_model.npz")


def reconstruction_proof(
    record, provenance, source, destination, original, env, reference_check, arrays
):
    return {
        "status": "PASS",
        "method": "deterministic reconstruction at frozen source; no integration",
        "source_run": record["output"],
        "run_identity": record["identity"],
        "source_revision": provenance["code_revision"],
        "source_sha256": provenance["source_sha256"],
        "assets": provenance["assets"],
        "runtime": provenance["runtime"],
        "original_scene_sha256": file_hash(source / "scene.xml"),
        "original_scene_has_coupled_model": original.nq == env.model.nq
        and original.ngeom == env.model.ngeom,
        "original_scene_dimensions": {
            k: int(getattr(original, k)) for k in ("nbody", "nq", "ngeom")
        },
        "reconstructed_scene_dimensions": {
            k: int(getattr(env.model, k)) for k in ("nbody", "nq", "ngeom")
        },
        "physical_identity": physical_identity(env.model),
        "reset_qpos_equal": True,
        "compiled_geometry_equal": env.ladder is not None,
        "material_audit_equal": env.ladder is not None,
        "reviewed_reference_comparison": reference_check,
        "xml_roundtrip_identical_arrays": arrays,
        "files_sha256": {
            name: file_hash(destination / name) for name in ("scene.xml", "scene.mjb")
        },
    }


def main(root):
    records = [
        r
        for r in read(root / "campaign_state.json")["runs"]
        if r["status"] == "COMPLETE"
    ]
    for record in records:
        proof = reconstruct(root, record)
        print(
            record["id"],
            "PASS",
            "original coupled model:",
            proof["original_scene_has_coupled_model"],
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/g1_o6/ladder_material")
    )
    main(parser.parse_args().root)
