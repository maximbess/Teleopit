"""Check floating release and hold trials against the native contacts and measured wrist poses."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.diagnostics import region
from teleopit.sim.g1_o6_support.guards import detached, measured_opening
from teleopit.sim.g1_o6_support.regrasp import write_json
from teleopit.sim.g1_o6_support.regrasp_measure import STATES, SUPPORTS
from teleopit.sim.g1_o6_support.regrasp_results import interval, read_run


def native_forces(contacts, owners, targets, steps):
    sample, first, second = contacts[:, :3].astype(int).T
    result = np.zeros((steps, len(targets)))
    for i, target in enumerate(targets):
        on_second = (first == target) & (owners[second] == i)
        on_first = (second == target) & (owners[first] == i)
        weights = contacts[:, 4] * (on_second.astype(int) - on_first)
        result[:, i] = np.bincount(sample, weights=weights, minlength=steps)
    return result


def force_error(directory, cfg, trace):
    model = mujoco.MjModel.from_binary_path(str(Path(cfg["source"]) / "scene.mjb"))
    assert cfg["geom_names"] == [model.geom(g).name for g in range(model.ngeom)]
    ids = {name: i for i, name in enumerate(SUPPORTS)}
    owners = np.array([ids.get(region(model, b), -1) for b in model.geom_bodyid])
    targets = [model.geom(cfg["support_targets"][name]).id for name in SUPPORTS]
    with np.load(directory / "contacts.npz") as data:
        fz = native_forces(data["contacts"], owners, targets, len(trace["time"]))
    return float(np.max(abs(fz - trace["fz"])))


def required_supports(trace, active, hands):
    states = trace["state"][active][:, hands]
    return {
        "native_steps": int(active.sum()),
        "hand_grasp_losses": (states != STATES.index("grasp")).sum(axis=0).tolist(),
        "thumb_opposition_losses": (~trace["thumb_opposed"][active][:, hands])
        .sum(axis=0)
        .tolist(),
        "minimum_digit_normal_N": trace["finger_load"][active][:, hands]
        .min(axis=0)
        .tolist(),
        "foot_contact_losses": (~trace["loaded"][active, 2:]).sum(axis=0).tolist(),
        "foot_minimum_Fz_N": trace["fz"][active, 2:].min(axis=0).tolist(),
        "max_required_slip_m_s": float(trace["slip"][active][:, [*hands, 2, 3]].max()),
        "max_applied": float(trace["applied"][active].max()),
        "max_assist_gain": float(trace["assist_gain"][active].max()),
        "max_fixture_wrench": float(abs(trace["fixture"][active]).max()),
        "max_other_Fz_N": float(abs(trace["other_fz"][active]).max()),
        "unintended_loaded_steps": int(np.count_nonzero(trace["wrong"][active])),
    }


def release_cycle(events, commands, trace, cfg, number):
    side = int(cfg["side"] == "right")

    def tick(key):
        return int(np.argmin(abs(commands["time"] - events[key])))

    def winding(end, length):
        return interval(trace["time"], (events[end] - length, events[end]))

    confirmed = cfg["criteria"]["persistence_s"]
    clear = winding("clear_verified" + number, confirmed)
    open_tick = tick("open_request" + number)
    # The native encoder sample at the end of the preceding tick is aligned to its interval start.
    start = np.flatnonzero(trace["time"] < commands["time"][open_tick] - 1e-8)[-1]
    opened = measured_opening(
        trace["digits"][start, side], trace["digits"][clear, side]
    )
    detached_samples = detached(
        opened, trace["external_count"][clear], trace["gap"][clear]
    )
    close = tick("close_request" + number)
    error = commands["position_rung"][close, side] - commands["goal_rung"][close]
    rotation = (
        commands["rotation_rung"][tick("settled"), side]
        @ commands["rotation_rung"][close, side].T
    )
    grasp = winding("grasp_verified" + number, confirmed)
    hold = interval(
        trace["time"],
        (events["grasp_verified" + number], events["grasp_verified" + number] + 2),
    )
    return {
        "position_error_mm": float(np.linalg.norm(error) * 1000),
        "angle_error_rad": float(Rotation.from_matrix(rotation).magnitude()),
        "clear_steps": int(clear.sum()),
        "detached_steps": int(detached_samples.sum()),
        "minimum_gap_mm": float(trace["gap"][clear].min() * 1000),
        "confirmation_steps": int(grasp.sum()),
        "hold_steps": int(hold.sum()),
        "invalid_confirm_or_hold_steps": int(
            np.sum(trace["state"][grasp | hold, side] != STATES.index("grasp"))
        ),
        "post_grasp_mean_Fz_N": trace["fz"][hold].mean(axis=0).tolist(),
    }


def audit_branch(directory):
    outcome, cfg, trace = read_run(directory)
    active = trace["time"] >= outcome["windows"]["baseline"][0] - 1e-8
    moving = int(cfg["side"] == "right")
    hands = [1 - moving] if cfg["branch"] == "release" else [0, 1]
    report = required_supports(trace, active, hands)
    report["hand_order"] = [SUPPORTS[i] for i in hands]
    report["contact_force_max_error_N"] = force_error(directory, cfg, trace)
    report["first_missing"] = outcome["missing"]
    checks = [
        outcome["missing"] is None,
        not any(report["hand_grasp_losses"] + report["foot_contact_losses"]),
        report["max_required_slip_m_s"] < cfg["config"]["max_slip_m_s"],
        report["contact_force_max_error_N"] < 1e-10,
        max(
            report[k]
            for k in (
                "max_applied",
                "max_assist_gain",
                "max_fixture_wrench",
                "unintended_loaded_steps",
            )
        )
        == 0,
    ]
    if cfg["branch"] == "release":
        cycles = {}
        events = outcome["events"]
        with np.load(directory / "commands.npz") as saved:
            commands = {
                k: saved[k]
                for k in ("time", "position_rung", "goal_rung", "rotation_rung")
            }
        for n in ("1", "2"):
            if "grasp_verified" + n in events:
                cycle = release_cycle(events, commands, trace, cfg, n)
                cycles[n] = cycle
                checks.extend(
                    [
                        cycle["position_error_mm"]
                        <= cfg["position_tolerance_m"] * 1000,
                        cycle["angle_error_rad"] <= cfg["rotation_tolerance_rad"],
                        cycle["detached_steps"]
                        == cycle["clear_steps"]
                        == cycle["confirmation_steps"]
                        == round(
                            cfg["criteria"]["persistence_s"]
                            / cfg["config"]["physics_dt"]
                        ),
                        cycle["hold_steps"] == round(2 / cfg["config"]["physics_dt"]),
                        cycle["invalid_confirm_or_hold_steps"] == 0,
                    ]
                )
        checks.append(len(cycles) == 2)
        report["cycles"] = cycles
        dwell = trace["phase"] == "clear1"
        report["clear_dwell_steps"] = int(dwell.sum())
        checks.append(
            report["clear_dwell_steps"] == round(2 / cfg["config"]["physics_dt"])
            and "clear1" in outcome["windows"]
        )
    report["passed"] = all(checks)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    branches = sorted(args.root.glob("*/*/outcome.json"))
    reports = {
        p.parent.relative_to(args.root).as_posix(): audit_branch(p.parent)
        for p in branches
    }
    write_json(args.output, reports)
    raise SystemExit(0 if reports and all(r["passed"] for r in reports.values()) else 1)
