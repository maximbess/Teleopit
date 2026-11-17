"""Independent artifact, source, model, material and applied-force verification."""

import argparse
import gzip
import json
import zipfile
from pathlib import Path

import mujoco

from scripts.package_g1_o6_support import compare_numeric_fields
from scripts.verify_g1_o6_ladder_campaign import verify_forces, verify_material
from teleopit.sim.g1_o6_support.ladder_campaign import digest, file_hash
from teleopit.sim.g1_o6_support.scene_contract import canonical
from teleopit.sim.g1_o6_support.ladder_identity import (
    assert_preserved,
    physical_identity,
    verify_reviewed_model,
)
from teleopit.sim.ladder_config import compiled_ladder, geometry_matrix


def read(path):
    return json.loads(path.read_text())


def validate_retry(record, previous):
    assert previous["status"] in ("ERROR", "RUNNING")
    assert record.get("retry_of") == previous["id"]
    assert record.get("technical_retry_reason", "").strip()
    assert previous["identity"] == digest(previous["payload"])
    assert record["identity"] == previous["identity"] == digest(record["payload"])
    assert record["output"] != previous["output"]


def verified_history(records):
    assert len({r["id"] for r in records}) == len(records), "duplicate attempt ID"
    previous, completed, recoveries = {}, [], []
    for record in records:
        assert record["status"] in ("COMPLETE", "ERROR", "RUNNING")
        key = record["candidate"], record["stage"]
        prior = previous.get(key)
        if prior is None:
            assert not record.get("retry_of")
        else:
            validate_retry(record, prior)
            recoveries.append({"failed_attempt": prior["id"], "recovered_by": record["id"], "identity": record["identity"]})
        previous[key] = record
        if record["status"] == "COMPLETE":
            completed.append(record)
    assert all(r["status"] == "COMPLETE" for r in previous.values()), "unfinished attempt lacks completed recovery"
    return completed, recoveries


def check_source(root, state):
    import hashlib

    snapshot = read(root / "source/runtime_source.json")
    frozen = next(d["value"] for d in state["decisions"] if d["name"] == "source")
    assert frozen["dirty_diff"] == ""
    assert snapshot["revision"] == frozen["code_revision"]
    assert snapshot["files_sha256"] == frozen["source_sha256"]
    with zipfile.ZipFile(root / "source/runtime_source.zip") as archive:
        for name, expected in snapshot["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    return {
        "status": "PASS",
        "revision": snapshot["revision"],
        "exact_source_files": len(snapshot["files_sha256"]),
    }


def hand_tread_contact(contact):
    return any("_o6_" in n for n in contact["geoms"]) and any(
        "_ladder_rung_" in n for n in contact["geoms"]
    )


def contacts(path, scored_steps):
    count, held = 0, 0
    phases, values = set(), set()
    with gzip.open(path / "physics_contacts.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            phases.add(row["phase"])
            held += row["phase"] == "HOLD"
            for contact in row["contacts"]:
                if hand_tread_contact(contact):
                    values.add(tuple(contact["friction"]))
    assert held == scored_steps
    assert values <= {(1.8, 1.8, 0.02, 0.002, 0.002)}
    return {
        "status": "PASS",
        "physics_rows": count,
        "scored_rows": held,
        "phases": sorted(phases),
        "observed_hand_vectors": [list(v) for v in values],
    }


def verify_record(root, record, reference):
    assert record["status"] == "COMPLETE", record["id"]
    path = root / record["output"]
    for name, expected in record["files_sha256"].items():
        assert file_hash(path / name) == expected, str(path / name)
    metadata = read(path / "resolved.json")
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    xml = path / "scene_portable.xml"
    if not xml.exists():
        xml = path / "scene.xml"
    xml_model = mujoco.MjModel.from_xml_path(str(xml))
    # MuJoCo clears its process-local compiler signature on MJB loading.
    fields = compare_numeric_fields(model, xml_model, ("npaths", "nbuffer", "signature"))
    fields += compare_numeric_fields(model.opt, xml_model.opt)
    assert_preserved(physical_identity(model), reference)
    assert canonical(compiled_ladder(model, geometry_matrix()["G6"])) == metadata["ladder"], record["id"]
    assert metadata["material"] is None and metadata["config"]["friction_scale"] == 1
    result = {
        "id": record["id"],
        "status": "PASS",
        "identical_model_fields": fields,
        "files_checked": len(record["files_sha256"]),
    }
    summary = read(path / "summary.json")
    if "physics_audit" in summary:
        result["forces"] = verify_forces(path, summary)
        result["material"] = verify_material(path, {"material": None}, model)
        if not record.get("imported_control"):
            result["contacts"] = contacts(
                path, summary["physics_audit"]["scored_steps"]
            )
    return result


def verify(root):
    state = read(root / "campaign_state.json")
    completed, recoveries = verified_history(state["runs"])
    reference = next(
        d["value"] for d in state["decisions"] if d["name"] == "reference_physics"
    )
    counts = {
        stage: len({r["candidate"] for r in state["runs"] if r["stage"] == stage})
        for stage in state["specification"]["budgets"]
    }
    assert all(
        counts[s] <= limit for s, limit in state["specification"]["budgets"].items()
    )
    assert len(state["candidates"]) <= 40
    result = {
        "software_status": "PASS",
        "source": check_source(root, state),
        "counts": counts,
        "technical_recoveries": recoveries,
        "default_preservation": verify_reviewed_model(root / "source/reviewed_model.npz"),
        "runs": [verify_record(root, r, reference) for r in completed],
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": "PASS", "counts": counts, "runs": len(result["runs"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_stability"))
    verify(parser.parse_args().root)
