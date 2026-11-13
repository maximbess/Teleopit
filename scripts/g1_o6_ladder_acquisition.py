"""Supplement acquisition-transition windows with verified deterministic prefixes.

These stop 0.26 s after HOLD entry, never qualify a condition, and never replace
an original failure. Every control state must equal the immutable primary run.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from g1_o6_ladder_reconstruct import (
    check_source,
    rebuild_environment,
    verify_environment,
)

from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6_support.ladder_campaign import file_hash, run_checksums
from teleopit.sim.g1_o6_support.ladder_observer import LadderObserver

FIELDS = (
    "time",
    "phase",
    "base_pose",
    "joint_positions",
    "base_drift_m",
    "contacts",
    "supports",
    "wrist_world_on_hand",
    "failure_reasons",
)


def compare_row(actual, expected):
    changed = [field for field in FIELDS if actual.get(field) != expected.get(field)]
    if changed:
        raise ValueError(f"acquisition differs at {actual['time']}: {changed}")


def record_prefix(env, source, observer):
    count, hold_entry = 1, None
    with (source / "steps.jsonl").open() as stream:
        compare_row(snapshot(env), json.loads(next(stream)))
        while env.data.time < 10:
            row = env.step(observer)
            compare_row(row, json.loads(next(stream)))
            observer.control_sample(env, row)
            count += 1
            if env.phase == "HOLD" and hold_entry is None:
                hold_entry = env.data.time
            if hold_entry is not None and env.data.time >= hold_entry + 0.259:
                return count, hold_entry
            if env.phase in ("ABORT", "HOLD_COMPLETE"):
                raise ValueError(
                    "primary trajectory ended before acquisition prefix completed"
                )
    raise ValueError("HOLD entry missing from bounded acquisition prefix")


def supplement(root, record):
    root = Path(root)
    destination = root / "acquisition_windows" / record["id"]
    receipt = destination / "verification.json"
    if receipt.exists():
        proof = json.loads(receipt.read_text())
        for name, expected in proof["files_sha256"].items():
            assert file_hash(destination / name) == expected
        return proof
    if destination.exists():
        raise FileExistsError(
            "retain interrupted prefix and choose a new evidence directory"
        )
    provenance = check_source(record)
    source = root / record["output"]
    assert file_hash(source / "steps.jsonl") == record["files_sha256"]["steps.jsonl"]
    metadata = json.loads((source / "resolved.json").read_text())
    env = rebuild_environment(record, metadata)
    verify_environment(root, record, metadata, env)
    destination.mkdir(parents=True)
    observer = LadderObserver(env, destination)
    try:
        count, hold_entry = record_prefix(env, source, observer)
    finally:
        observer.close()
    proof = {
        "status": "PASS",
        "purpose": "acquisition-transition evidence only; no qualification or promotion",
        "source_run": record["output"],
        "source_run_identity": record["identity"],
        "source_trace_sha256": record["files_sha256"]["steps.jsonl"],
        "source_revision": provenance["code_revision"],
        "prefix_program_sha256": file_hash(Path(__file__)),
        "compared_control_rows": count,
        "identical_fields": list(FIELDS),
        "hold_entry_s": hold_entry,
        "end_s": env.data.time,
        "physics_dt": env.config.physics_dt,
        "no_state_resets_after_initialization": True,
        "files_sha256": run_checksums(destination),
    }
    receipt.write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main(root, workers):
    state = json.loads((root / "campaign_state.json").read_text())
    records = [
        r for r in state["runs"] if r["stage"] == "screen" and r["status"] == "COMPLETE"
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(supplement, root, record): record["id"] for record in records
        }
        for future in as_completed(jobs):
            proof = future.result()
            print(
                jobs[future],
                proof["status"],
                proof["compared_control_rows"],
                "identical rows",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/g1_o6/ladder_material")
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    main(args.root, args.workers)
