"""Bounded experiments, fresh resets and explicit unexecuted outcomes."""

import itertools
from dataclasses import asdict

import numpy as np

from .config import DiagnosticConfig
from .hand_run import run_hand
from .harness import Source
from .io import code_state, manifest, output_directory, read_rows, write_csv, write_json
from .loads import select_references


def sides(side):
    return ("left", "right") if side == "both" else (side,)


def experiment(source_run, output):
    source = Source(source_run)
    path = output_directory(output)
    cfg = DiagnosticConfig()
    write_json(
        path / "resolved.json",
        {
            "schema": "o6_experiment_v1",
            **code_state(),
            "config": asdict(cfg),
            "source_run": str(source.path),
            "source_assets": source.metadata["assets"],
        },
    )
    return source, path, cfg


def save_experiment(path, name, cases, extra):
    result = {
        "schema": "o6_" + name + "_v1",
        "units": {
            "time": "s",
            "length": "m",
            "angle": "rad",
            "force": "N",
            "moment": "Nm",
        },
        "frame": "world",
        "wrench_origin": "current wrist_mount",
        "cases": cases,
        **extra,
    }
    write_json(path / (name + ".json"), result)
    compact = [
        {
            k: v
            for k, v in c.items()
            if k
            not in (
                "saturation",
                "physics_extrema",
                "first_threshold_violation",
                "first_persistent_violation_per_gate",
            )
        }
        for c in cases
    ]
    write_csv(path / (name + ".csv"), compact)
    manifest(
        path,
        name,
        {"side": "both" if len({c["side"] for c in cases}) == 2 else cases[0]["side"]},
    )
    return result


def implicated(result, source, side, cfg):
    active = source.hands[side]["active"]
    names = set()
    for name, error in result["closed_endpoint_errors_rad"].items():
        if error >= cfg.settling_position_rad:
            names.add(name.removesuffix("_servo"))
    names.update(saturated_commands(result, cfg))
    for event in result["first_threshold_violation"]:
        joint = event.split(":")[-1]
        if joint in active and event.startswith(("lower_limit:", "upper_limit:")):
            names.add(joint)
    # A passive range/mimic failure belongs to the associated active command.
    names.update(passive_commands(result, source.hands[side]["passive"]))
    return [i for i, name in enumerate(active) if name in names]


def saturated_commands(result, cfg):
    return {
        n.removesuffix("_servo")
        for n, item in result["saturation"].items()
        if item["longest_s"] >= cfg.persistence_s
    }


def passive_commands(result, relations):
    affected = {event.split(":")[-1] for event in result["first_threshold_violation"]}
    return {active for passive, active, _ in relations if passive in affected}


def free_close(source_run, output, side="both"):
    source, path, cfg = experiment(source_run, output)
    cases = []
    for current in sides(side):
        for gravity in (True, False):
            case = current + ("_gravity" if gravity else "_no_gravity")
            result = run_hand(
                source, current, path / case, fixed=True, gravity=gravity, config=cfg
            )
            cases.append(dict(case=case, executed=True, **result))
            for index in implicated(result, source, current, cfg):
                label = case + "_joint_" + str(index)
                single = run_hand(
                    source,
                    current,
                    path / label,
                    fixed=True,
                    gravity=gravity,
                    only=index,
                    config=cfg,
                )
                cases.append(dict(case=label, executed=True, **single))
    result = save_experiment(
        path,
        "free_close_results",
        cases,
        {
            "additional_settling_criteria": {
                "position_rad": cfg.settling_position_rad,
                "velocity_rad_s": cfg.settling_velocity_rad_s,
            }
        },
    )
    write_csv(
        path / "free_close_summary.csv",
        [
            {
                k: c[k]
                for k in (
                    "case",
                    "side",
                    "gravity",
                    "only_active_joint",
                    "strict_pass",
                    "outcome",
                    "max_joint_error_rad",
                    "max_mimic_error_rad",
                    "max_penetration_m",
                    "reopened_settled",
                    "endpoint_reached_without_self_contact",
                    "closed_endpoint_errors_rad",
                    "self_contact_bodies",
                )
            }
            for c in cases
        ],
    )
    manifest(path, "free_close", {"side": side})
    return result


def planned_cases(side, reference, cfg):
    cases = []
    measured = np.array(reference["wrench_world_at_mount"])
    masks = {
        "force_only": [1, 1, 1, 0, 0, 0],
        "moment_only": [0, 0, 0, 1, 1, 1],
        "combined": [1] * 6,
    }
    for family, mask in masks.items():
        for scale in cfg.scales:
            cases.append(
                {
                    "case": f"{side}_{family}_{scale:.4f}",
                    "side": side,
                    "family": family,
                    "scale": scale,
                    "reference_time_s": reference["time_s"],
                    "reference_kind": "measured"
                    if family == "combined"
                    else "synthetic_decomposition",
                    "requested_wrench_world_at_mount": (
                        measured * mask * scale
                    ).tolist(),
                }
            )
    return cases


def qualification(zero):
    if not zero["zero_load_qualified"]:
        return "not_qualified_at_zero_load"
    return "qualified" if zero["strict_pass"] else "reserve_failed_diagnostic_only"


def brackets(cases):
    ordered = sorted((c for c in cases if c["executed"]), key=lambda c: c["scale"])
    return [
        (a["scale"], b["scale"])
        for a, b in itertools.pairwise(ordered)
        if a["strict_pass"] != b["strict_pass"]
    ]


def execute(source, path, cfg, case, *, dt=None, protocol="primary"):
    result = run_hand(
        source,
        case["side"],
        path / case["case"],
        wrench=case["requested_wrench_world_at_mount"],
        dt=dt,
        protocol=protocol,
        config=cfg,
    )
    return {**case, **result, "executed": True}


def skipped(case, why):
    return {
        **case,
        "executed": False,
        "outcome": "unexecuted",
        "reason": why,
        "strict_pass": None,
        "physical_retention": None,
        "plateau_achieved_s": None,
        "achieved_force_norm_N": None,
        "achieved_moment_norm_Nm": None,
    }


def confirm(source, path, cfg, case):
    results = []
    for label, dt in [
        ("repeat", source.model.opt.timestep),
        ("half_step", source.model.opt.timestep / 2),
    ]:
        new = {
            **case,
            "case": case["case"] + "_" + label,
            "confirmation_of": case["case"],
        }
        results.append(
            execute(
                source, path, cfg, new, dt=dt, protocol=case.get("protocol", "primary")
            )
        )
    return results


def refine_direction(source, path, cfg, cases):
    extra = []
    for low, high in brackets(cases)[:2]:
        original = next(c for c in cases if c["scale"] == high)
        scale = (low + high) / 2
        case = {
            **original,
            "case": original["side"] + "_" + original["family"] + f"_{scale:.4f}",
            "scale": scale,
            "requested_wrench_world_at_mount": (
                np.array(original["requested_wrench_world_at_mount"]) * scale / high
            ).tolist(),
        }
        extra.append(execute(source, path, cfg, case))
    all_cases = cases + extra
    extra.extend(confirm_boundaries(source, path, cfg, all_cases))
    return extra


def confirm_boundaries(source, path, cfg, all_cases):
    extra = []
    passing = [c for c in all_cases if c["strict_pass"]]
    if passing:
        largest = max(passing, key=lambda c: c["scale"])
        extra.extend(confirm(source, path, cfg, largest))
        failing = [c for c in all_cases if not c["strict_pass"]]
        if failing:
            extra.extend(
                confirm(
                    source,
                    path,
                    cfg,
                    min(failing, key=lambda c: abs(c["scale"] - largest["scale"])),
                )
            )
    return extra


def side_matrix(source, path, cfg, side, references):
    zero_case = {
        "case": side + "_zero",
        "side": side,
        "family": "zero",
        "scale": 0.0,
        "reference_time_s": None,
        "requested_wrench_world_at_mount": [0.0] * 6,
    }
    zero = execute(source, path, cfg, zero_case)
    cases = [zero]
    status = qualification(zero)
    primary = planned_cases(side, references[0], cfg)
    if status == "not_qualified_at_zero_load":
        cases.extend(skipped(c, status) for c in primary)
        seated = {
            **primary[-1],
            "case": side + "_combined_seated_diagnostic",
            "protocol": "seated_diagnostic",
        }
        seated = execute(source, path, cfg, seated, protocol="seated_diagnostic")
        cases.append(seated)
        cases.extend(confirm(source, path, cfg, zero))
        cases.extend(confirm(source, path, cfg, seated))
    else:
        results = [execute(source, path, cfg, c) for c in primary]
        cases.extend(results)
        for family in ("force_only", "moment_only", "combined"):
            cases.extend(
                refine_direction(
                    source, path, cfg, [c for c in results if c["family"] == family]
                )
            )
        if not any(c["strict_pass"] for c in results):
            cases.extend(confirm(source, path, cfg, zero))
            cases.extend(confirm(source, path, cfg, results[-1]))
        cases.extend(
            confirm_other_references(source, path, cfg, side, references, results)
        )
    return status, cases


def confirm_other_references(source, path, cfg, side, references, results):
    retained = [
        c for c in results if c["family"] == "combined" and c["physical_retention"]
    ]
    if not retained:
        return []
    scale = max(c["scale"] for c in retained)
    cases = []
    for index, reference in enumerate(references[1:], 1):
        case = {
            "case": f"{side}_reference_{index}",
            "side": side,
            "family": "combined_confirmation",
            "scale": scale,
            "reference_time_s": reference["time_s"],
            "requested_wrench_world_at_mount": (
                np.array(reference["wrench_world_at_mount"]) * scale
            ).tolist(),
        }
        cases.append(execute(source, path, cfg, case))
    return cases


def load_response(source_run, output, side="both"):
    source, path, cfg = experiment(source_run, output)
    rows = read_rows(source.path / "steps.jsonl")
    references = {s: select_references(rows, s) for s in sides(side)}
    status = {}
    cases = []
    write_json(path / "signed_reference_samples.json", references)
    for current in sides(side):
        status[current], trials = side_matrix(
            source, path, cfg, current, references[current]
        )
        cases.extend(trials)
    from .plots import load_plot

    load_plot(cases, path)
    return save_experiment(
        path,
        "load_response",
        cases,
        {
            "qualification": status,
            "references": references,
            "capacity_note": "Only tested directions and protocols; not a manufacturer payload rating. Unexecuted loads are unknown.",
        },
    )
