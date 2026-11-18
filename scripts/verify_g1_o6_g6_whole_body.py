"""Reconstruct per-finger loading and body commands for the bounded G6 search."""

import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.g1_o6_g6_acquisition_report import selected_contacts, verify_targets
from scripts.g1_o6_g6_arm_report import verify_run as verify_arm
from scripts.verify_g1_o6_g6 import read, verified_history
from teleopit.sim.g1_o6_support.pose_contacts import FINGERS


def body_sample(current, previous, baseline, dt, active):
    offset = float(np.max(abs(current - baseline)))
    rate = float(np.max(abs(current - previous)) / dt)
    assert offset <= (.12 if active else 0) + 1e-9, ("body offset", offset, active)
    assert rate <= .06 + 1e-8, ("body rate", rate)
    return offset, rate


def body_indices(model):
    # Resolve independently of the controller metadata or saved offset array.
    parts = ("_hip_", "_knee_", "_ankle_")
    names = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
    indices = [i for i, n in enumerate(names) if n.startswith("waist_") or any(p in n for p in parts)]
    assert len(indices) == 15
    return indices


def static_residual(contacts, weight_world):
    result = np.array(weight_world, dtype=float)
    for contact in contacts:
        first, second = contact["regions"]
        sign = int(first == "environment") - int(second == "environment")
        result += sign * np.asarray(contact["force_on_second_world"])
    return result


def observe_balance(record, row, weight):
    residual = static_residual(row["contacts"], weight)
    np.testing.assert_allclose(residual, row["static_balance_residual_N"], atol=1e-9)
    magnitude = float(np.linalg.norm(residual))
    record["peak_N"] = max(record["peak_N"], magnitude)
    exceeded = magnitude >= 25.
    assert exceeded == ("force_balance" in row["failure_reasons"])
    if exceeded:
        record["violations"].append({"hold_s": row["hold_time_s"], "residual_N": magnitude,
            "residual_world_N": residual.tolist(),
            "support_z_N": {name: data["force_world"][2] for name, data in row["supports"].items()}})


def verify_body(path, model, active):
    indices = body_indices(model)
    peak = np.zeros(2)
    weight = model.opt.gravity * model.body_subtreemass[model.body("pelvis").id]
    balance = {"peak_N": 0., "violations": []}
    count = 0
    with (path / "steps.jsonl").open() as stream:
        rows = map(json.loads, stream)
        first = next(rows)
        baseline = np.asarray(first["applied_position_targets_rad"])[indices]
        previous, time = baseline, first["time"]
        for row in rows:
            current = np.asarray(row["applied_position_targets_rad"])[indices]
            sample = body_sample(current, previous, baseline, row["time"] - time, active and row["phase"] == "HOLD")
            peak = np.maximum(peak, sample)
            if row["phase"] == "HOLD":
                observe_balance(balance, row, weight)
            previous, time = current, row["time"]
            count += 1
    assert count > 0
    return {"control_updates": count, "peak_offset_rad": peak[0], "peak_rate_rad_s": peak[1],
            "hold_force_balance": balance}


def left_forces(row):
    forces = dict.fromkeys(FINGERS[1:], 0.)
    for contact in selected_contacts(row, "left"):
        group = contact["group"]
        if group in forces:
            forces[group] += contact["force"]
    return forces


def left_loading(rows, dt):
    minimum = dict.fromkeys(FINGERS[1:], float("inf"))
    unloaded = dict.fromkeys(minimum, 0)
    first = dict.fromkeys(minimum)
    steps = 0
    for row in rows:
        if row["phase"] != "HOLD":
            continue
        steps += 1
        for finger, force in left_forces(row).items():
            minimum[finger] = min(minimum[finger], force)
            if force < .05:
                unloaded[finger] += 1
                if first[finger] is None:
                    first[finger] = steps * dt
    assert steps > 0
    return {"hold_steps": steps, "minimum_N": minimum, "unloaded_steps": unloaded,
            "first_unloaded_hold_s": first}


def verify_run(path, report):
    meta, summary = read(path / "resolved.json"), read(path / "summary.json")
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    with gzip.open(path / "physics_contacts.jsonl.gz", "rt") as stream:
        loading = left_loading(map(json.loads, stream), meta["config"]["physics_dt"])
    assert loading["hold_steps"] == summary["physics_audit"]["scored_steps"]
    np.testing.assert_allclose(min(loading["minimum_N"].values()),
                               summary["contact_quality"]["minimum_group_normal_N"]["left"]["finger"], atol=1e-10)
    body = verify_body(path, model, meta["controller"]["mode"] == "g6_whole_body")
    return {"status": "PASS", "left_fingers": loading, "body_commands": body,
            "all_servo_limits": verify_targets(path, meta, model),
            "arm_commands": verify_arm(path, report / (path.name + "_arm.json"))}


def verify(root, run_id=None):
    state = read(root / "campaign_state.json")
    report = root / "report"
    report.mkdir(exist_ok=True)
    records = [r for r in state["runs"] if r["id"] == run_id] if run_id else verified_history(state["runs"])[0]
    if not records or any(r["status"] != "COMPLETE" for r in records):
        raise ValueError("verification requires completed runs")
    for record in records:
        result = verify_run(root / record["output"], report)
        (report / (record["id"] + "_verification.json")).write_text(json.dumps(result, indent=2) + "\n")
        print(record["id"], result["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_whole_body"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    verify(args.root, args.run_id)
