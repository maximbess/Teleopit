"""G6 capacity at each critical measured state, with a frozen finite budget."""

import json
from dataclasses import asdict
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import (
    manifest,
    output_directory,
    read_rows,
    write_json,
)

from .capacity import CapacityBounds, measured_references, qualified_source
from .capacity_matrix import evaluate_side, load_cases
from .qualification import critical_groups


def capacity_plan(source_path, planned_share=0.16, budget=192):
    source = qualified_source(source_path, "settled")
    rows = read_rows(source.path / "steps.jsonl")
    groups = critical_groups(rows, source.metadata["support_targets"])
    weight = source.model.body_subtreemass[source.model.body("pelvis").id] * abs(
        source.model.opt.gravity[2]
    )
    entries = {}
    for (side, target, phase), samples in groups.items():
        for reference in measured_references(samples, side):
            key = (side, reference["time_s"])
            entries.setdefault(
                key,
                {
                    "side": side,
                    "target": target,
                    "phase": phase,
                    "reference": reference,
                },
            )
    references = [e["reference"] for e in entries.values()]
    # Two times the largest simultaneous demand covers the declared diagnostic
    # scales as well as the mandatory 1.2 reserve; these are not actuator limits.
    bounds = CapacityBounds.for_demands(references, weight * planned_share, reserve=2)
    result = {
        "source_run": str(source.path),
        "bounds": asdict(bounds),
        "per_side_trial_budget": budget,
        "planned_share_per_hand": planned_share,
        "robot_weight_N": float(weight),
        "states": list(entries.values()),
    }
    return result


def state_references(entry, weight, planned_share):
    reference = entry["reference"]
    result = [reference]
    if planned_share:
        # Explicit planned direction, separate from the simultaneous measured sample.
        result.append(
            {
                "time_s": reference["time_s"],
                "side": entry["side"],
                "selection": ["planned_symmetric_vertical_support"],
                "source_kind": "declared_search_demand_not_measured_capacity",
                "wrench_world_at_mount": [0, 0, -weight * planned_share, 0, 0, 0],
            }
        )
    return result


def run_g6_capacity(source_path, output, *, planned_share=0.16, budget=192, dt=0.001):
    path = output_directory(output)
    controller = json.loads((Path(source_path) / "resolved.json").read_bytes())["controller"]
    if controller.get("isolated_runner_support") is False:
        result = {"schema": "g1_o6_g6_state_capacity_v1", "physical_status": "BLOCKED",
                  "capacity_status": "BLOCKED", "states": [], "controller_schema": controller["schema"],
                  "reason": "The free-wrist bench cannot execute the source controller's coupled feedback."}
        write_json(path / "summary.json", result)
        manifest(path, "g6_state_capacity", {"source_run": str(source_path)})
        return result
    plan = capacity_plan(source_path, planned_share, budget)
    write_json(path / "declared_matrix.json", plan)
    result = {
        "schema": "g1_o6_g6_state_capacity_v1",
        "physical_status": "BLOCKED",
        "capacity_status": "BLOCKED",
        "states": [],
        "remaining_trials": {s: budget for s in ("left", "right")},
    }
    for index, entry in enumerate(plan["states"]):
        side = entry["side"]
        source = qualified_source(
            source_path, "settled", sample_time=entry["reference"]["time_s"]
        )
        source.capacity_bounds = CapacityBounds(**plan["bounds"])
        references = state_references(entry, plan["robot_weight_N"], planned_share)
        required = len(load_cases(references, source.capacity_bounds)) + 3
        if required > result["remaining_trials"][side]:
            result["reason"] = (
                "declared state matrix exceeds remaining budget; untested states uncovered"
            )
            break
        report = evaluate_side(
            source,
            side,
            path / f"state_{index:03d}_{side}",
            references,
            dt,
            result["remaining_trials"][side],
        )
        result["remaining_trials"][side] -= len(report["trials"])
        result["states"].append({**entry, "result": report})
        write_json(path / "summary.json", result)
        if not report["reserve_qualified"]:
            result.update(
                physical_status=report["physical_status"],
                reason="zero retention, reserve or declared budget failed",
            )
            break
    else:
        result["capacity_status"] = "PASS" if result["states"] else "BLOCKED"
        result["physical_status"] = (
            "PASS"
            if all(s["result"]["physical_status"] == "PASS" for s in result["states"])
            else "FAIL"
        )
    write_json(path / "summary.json", result)
    manifest(path, "g6_state_capacity", {})
    return result


def capacity_run_paths(path):
    return sorted(str(p.parent) for p in Path(path).rglob("resolved.json"))
