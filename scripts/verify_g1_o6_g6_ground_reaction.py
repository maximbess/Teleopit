"""Compare scratch forecasts to independent native contact and command records."""

import argparse
import gzip
import json
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

from scripts.verify_g1_o6_g6_whole_body import left_forces, read, static_residual, verify, verify_run as verify_body


def measurements(row, weight):
    return float(np.linalg.norm(static_residual(row["contacts"], weight))), min(left_forces(row).values())


def native_interval(rows, count, end, dt, weight):
    values = []
    for i, row in enumerate(islice(rows, count), 1):
        if abs(row["time_s"] - (end - (count-i) * dt)) > 1e-8:
            raise ValueError("native interval timestamps do not align with the applied control")
        values.append(measurements(row, weight))
    if len(values) != count:
        raise ValueError("incomplete native interval")
    forces, fingers = zip(*values)
    return {"peak_N": max(forces), "minimum_left_N": min(fingers), "violations": sum(f >= 25. for f in forces)}


def compare_prediction(saved, actual, endpoint, limit):
    expected = np.array([max(actual["peak_N"], endpoint[0]), min(actual["minimum_left_N"], endpoint[1])])
    observed = [saved["forecast"]["peak_N"], saved["forecast"]["finger_N"]]
    assert 1 <= saved["evaluations"] <= limit
    np.testing.assert_allclose(observed, expected, rtol=0., atol=1e-8)
    return float(np.max(abs(observed - expected)))


def verify_intervals(steps, native, meta):
    dt = meta["config"]["physics_dt"]
    count = round(meta["config"]["control_dt"] / dt)
    weight = np.array([0., 0., -9.81 * meta["robot_mass_kg"]])
    profile = meta["controller"].get("ground_reaction_governor")
    choices = Counter()
    controls, violations, maximum, error, evaluations = 0, 0, 0., 0., 0
    minimum = float("inf")
    for row in steps:
        if row["phase"] != "HOLD":
            continue
        actual = native_interval(native, count, row["time"], dt, weight)
        controls += 1
        violations += actual["violations"]
        maximum = max(maximum, actual["peak_N"])
        minimum = min(minimum, actual["minimum_left_N"])
        if profile:
            saved = row["pose_search"]["controller_state"]["ground_reaction_preview"]
            endpoint = measurements({"contacts": row["contacts"], "support_targets": meta["support_targets"]}, weight)
            bound = 9 if profile["governor"] == "collective" else 21
            error = max(error, compare_prediction(saved, actual, endpoint, bound))
            choices[saved["choice"]] += 1
            evaluations = max(evaluations, saved["evaluations"])
    assert controls > 0 and next(native, None) is None
    return {"control_intervals": controls, "native_steps": controls * count, "peak_native_N": maximum,
            "native_violations": violations, "minimum_left_N": minimum, "choices": dict(choices),
            "most_previews_per_interval": evaluations, "maximum_forecast_error_N": error if profile else None}


def verify_run(path, report):
    body = verify_body(path, report)
    meta = read(path / "resolved.json")
    with (path / "steps.jsonl").open() as steps, gzip.open(path / "physics_contacts.jsonl.gz", "rt") as contacts:
        native = (r for r in map(json.loads, contacts) if r["phase"] == "HOLD")
        checked = verify_intervals(map(json.loads, steps), native, meta)
    audit = read(path / "summary.json")["physics_audit"]
    physics, control = (audit["ground_reaction"][s] for s in ("physics", "control"))
    assert checked["native_steps"] == physics["samples"] == audit["scored_steps"]
    assert checked["native_violations"] == physics["violations"]
    assert checked["control_intervals"] == control["samples"]
    np.testing.assert_allclose(checked["peak_native_N"], [physics["peak_N"], audit["extrema"]["force_balance"]], atol=1e-9)
    observed = body["body_commands"]["hold_force_balance"]
    assert len(observed["violations"]) == control["violations"]
    np.testing.assert_allclose(observed["peak_N"], control["peak_N"], atol=1e-9)
    return {**body, "ground_reaction": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_ground_reaction"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    verify(args.root, args.run_id, verify_function=verify_run)
