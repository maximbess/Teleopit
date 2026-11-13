"""Independent checks of immutable run bytes and requested/observed materials."""

import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np

from teleopit.sim.g1_o6.material import mixed_parameters
from teleopit.sim.g1_o6.model import provenance as asset_provenance
from teleopit.sim.g1_o6_support.ladder_campaign import condition_status, file_hash
from teleopit.sim.g1_o6_support.ladder_experiment import verified_reference
from teleopit.sim.g1_o6_support.ladder_identity import (
    assert_preserved,
    physical_identity,
)


def read(path):
    return json.loads(path.read_text())


def baseline_contacts(path):
    observed = {}
    with (path / "steps.jsonl").open() as stream:
        for line in stream:
            for contact in json.loads(line)["contacts"]:
                if any("_ladder_rung_" in name for name in contact["geoms"]):
                    key = "|".join(sorted(contact["geoms"]))
                    record = observed.setdefault(
                        key, {"coefficients": [], "samples": 0}
                    )
                    if contact["friction"] not in record["coefficients"]:
                        record["coefficients"].append(contact["friction"])
                    record["samples"] += 1
    return observed


def verify_material(path, condition, model):
    metadata = read(path / "resolved.json")
    check_requested_material(condition["material"], metadata.get("material"))
    observed = (
        read(path / "observed_friction.json")
        if (path / "observed_friction.json").exists()
        else baseline_contacts(path)
    )
    selected = {
        "|".join(sorted((p["robot"], p["rung"]))): p["friction"]
        for p in (metadata.get("material") or {}).get("pairs", [])
    }
    groups = check_contact_coefficients(model, selected, observed)
    return {
        "status": "PASS",
        "requested": condition["material"] or {"scope": "native reference", "mu": 1.8},
        "observed_pairs": len(observed),
        "observed_regions": sorted(groups),
        "observed_tangential_by_region": groups,
        "observation_resolution": "physics step"
        if (path / "observed_friction.json").exists()
        else "original reference control rows; unchanged native model",
    }


def check_contact_coefficients(model, selected, observed):
    groups = {}
    for key, record in observed.items():
        geoms = [model.geom(name).id for name in key.split("|")]
        expected = selected.get(key, mixed_parameters(model, *geoms)["friction"])
        for coefficients in record["coefficients"]:
            np.testing.assert_array_equal(coefficients, expected, err_msg=key)
        region = "hand" if "_o6_" in key else "other_robot"
        groups.setdefault(region, set()).update(c[0] for c in record["coefficients"])
    return {region: sorted(values) for region, values in groups.items()}


def check_requested_material(requested, audit):
    if requested is None:
        assert audit is None
        return
    assert requested == {key: audit[key] for key in ("scope", "mu")}
    assert audit["requested_tangential_mu"] == requested["mu"]
    assert audit["pairs"]
    for pair in audit["pairs"]:
        check_selected_pair(requested, pair)


def check_selected_pair(requested, pair):
    assert pair["friction"][:2] == [requested["mu"]] * 2
    if requested["scope"] == "H":
        assert "_o6_" in pair["robot"]


def verify_forces(path, result):
    audit = result["physics_audit"]
    if not (path / "applied_forces.jsonl.gz").exists():
        return {
            "status": "PASS" if audit["no_external_support"] else "FAIL",
            "method": "reviewed reference every-physics-step boolean audit; detailed arrays retained in event windows",
            "scored_steps": audit["scored_steps"],
            "nonzero_steps": audit["nonzero_applied_steps"],
        }
    count, nonzero = 0, 0
    with gzip.open(path / "applied_forces.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            nonzero += int(
                bool(np.any(row["qfrc_applied"]) or np.any(row["xfrc_applied"]))
            )
    assert count == audit["scored_steps"]
    assert nonzero == audit["nonzero_applied_steps"]
    return {
        "status": "PASS" if nonzero == 0 else "FAIL",
        "scored_steps": count,
        "nonzero_steps": nonzero,
        "method": "all actual applied arrays at every scored physics step",
    }


def verify(root, completed_only=False):
    state = read(root / "campaign_state.json")
    reference = next(
        d["value"] for d in state["decisions"] if d["name"] == "reference_physics"
    )
    reports = []
    for record in state["runs"]:
        if record["status"] != "COMPLETE":
            if completed_only:
                continue
            raise ValueError("unfinished run: " + record["id"])
        reports.append(
            verify_record(
                root,
                record,
                state["candidates"][record["candidate"]],
                reference,
                not completed_only,
            )
        )
    counts = verify_budgets(state)
    status = qualification_statuses(state)
    output = {
        "software_status": "PASS",
        "reference_comparisons": verify_reference_history(root),
        "partial": completed_only,
        "distinct_conditions_by_stage": counts,
        "runs": reports,
        "physical_qualification": status,
    }
    name = "verification_partial.json" if completed_only else "verification.json"
    (root / name).write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "software_status": "PASS",
                "counts": counts,
                "qualified": [k for k, v in status.items() if v == "QUALIFIED"],
            }
        )
    )


def verify_reference_history(root):
    verification = read(root / "reference_verification.json")
    for seconds in (30, 120):
        verified_reference(root, verification[str(seconds)], seconds)
    return {"status": "PASS", "trace_hashes_and_summaries_match_comparison": True}


def verify_record(root, record, condition, reference, require_acquisition=False):
    path = root / record["output"]
    for name, checksum in record["files_sha256"].items():
        if file_hash(path / name) != checksum:
            raise ValueError("changed immutable artifact: " + str(path / name))
    report = {
        "id": record["id"],
        "integrity": "PASS",
        "files": len(record["files_sha256"]),
    }
    if (path / "scene.xml").exists():
        model, proof = verified_model(root, record, reference)
        report.update(
            material=verify_material(path, condition, model),
            external_forces=verify_forces(path, read(path / "summary.json")),
            robot_solver_controller="PASS",
            reconstructed_scene="PASS",
            original_scene_has_coupled_model=proof["original_scene_has_coupled_model"],
        )
    if record["stage"] == "screen":
        report["acquisition_windows"] = verify_acquisition(
            root, record, require_acquisition
        )
    return report


def verify_acquisition(root, record, required):
    path = root / "acquisition_windows" / record["id"]
    if not required and not (path / "verification.json").exists():
        return {"status": "PENDING"}
    proof = read(path / "verification.json")
    assert proof["status"] == "PASS"
    assert proof["source_run_identity"] == record["identity"]
    assert proof["source_trace_sha256"] == record["files_sha256"]["steps.jsonl"]
    assert proof["prefix_program_sha256"] == file_hash(
        root / "source/acquisition_windows_program.py"
    )
    for name, expected in proof["files_sha256"].items():
        assert file_hash(path / name) == expected
    phases = verify_transition_coverage(path, proof["physics_dt"])
    return {
        "status": "PASS",
        "identical_control_rows": proof["compared_control_rows"],
        "phases": phases,
        "purpose": "separate deterministic acquisition prefix; no qualification result",
    }


def verify_transition_coverage(path, dt):
    triggers = read(path / "physics_window_triggers.json")
    with gzip.open(path / "physics_windows.jsonl.gz", "rt") as stream:
        times = [json.loads(line)["time_s"] for line in stream]
    phases = []
    for trigger in triggers:
        for reason in trigger["signals"]:
            if reason.startswith("phase_transition:"):
                phases.append(reason.split(":")[1])
                assert_transition_interval(times, trigger["time_s"], dt)
    assert set(phases) == {"INITIALIZE", "ACQUIRE", "REMOVE_ASSISTANCE", "HOLD"}
    return phases


def assert_transition_interval(times, center, dt):
    selected = [
        time
        for time in times
        if max(dt, center - 0.25) - 1e-8 <= time <= center + 0.25 + 1e-8
    ]
    assert selected[0] <= max(dt, center - 0.25) + dt + 1e-8
    assert selected[-1] >= center + 0.25 - dt - 1e-8
    assert max(np.diff(selected)) <= dt + 1e-8


def verified_model(root, record, reference):
    reconstruction = root / "reconstructions" / record["id"]
    proof = read(reconstruction / "proof.json")
    assert asset_provenance() == proof["assets"]
    assert proof["original_scene_sha256"] == file_hash(
        root / record["output"] / "scene.xml"
    )
    assert proof["run_identity"] == record["identity"]
    for name, checksum in proof["files_sha256"].items():
        assert file_hash(reconstruction / name) == checksum
    model = mujoco.MjModel.from_binary_path(str(reconstruction / "scene.mjb"))
    assert_preserved(physical_identity(model), reference)
    assert physical_identity(model) == proof["physical_identity"]
    return model, proof


def verify_budgets(state):
    counts = {
        stage: len({r["candidate"] for r in state["runs"] if r["stage"] == stage})
        for stage in state["specification"]["budgets"]
    }
    assert all(
        counts[stage] <= maximum
        for stage, maximum in state["specification"]["budgets"].items()
    )
    return counts


def qualification_statuses(state):
    return {
        name: condition_status(
            [
                r
                for r in state["runs"]
                if r["candidate"] == name and r["status"] == "COMPLETE"
            ]
        )
        for name in state["candidates"]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/g1_o6/ladder_material")
    )
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    verify(args.root, args.completed_only)
