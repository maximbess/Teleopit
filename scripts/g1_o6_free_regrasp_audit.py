"""Check floating release and hold trials against the native contacts and measured wrist poses."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.diagnostics import region
from teleopit.sim.g1_o6_support.guards import detached, measured_opening
from teleopit.sim.g1_o6_support.regrasp import write_json
from teleopit.sim.g1_o6_support.regrasp_measure import STATES, SUPPORTS
from teleopit.sim.g1_o6_support.regrasp_results import failures, interval, read_run
from teleopit.sim.g1_o6_support.regrasp_unassisted import free_body_setup


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
    if cfg["geom_names"] != [model.geom(g).name for g in range(model.ngeom)]:
        raise ValueError("Contact geometry order differs from the recorded scene.")
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


def launch_checks(trace, limits, opening):
    before = trace["time"] < opening - 1e-8
    losses = (trace["state"][before] != STATES.index("grasp")).sum(axis=0).tolist()
    slip = float(trace["slip"][before].max(initial=0))
    external = {
        "maximum_applied": float(np.max(abs(trace["applied"]))),
        "maximum_assist_gain": float(np.max(abs(trace["assist_gain"]))),
        "maximum_fixture_wrench": float(np.max(abs(trace["fixture"]))),
        "maximum_other_vertical_reaction_N": float(np.max(abs(trace["other_fz"]))),
    }
    caps = [
        limits[k]
        for k in ("assistance_force_N", "assistance_moment_Nm", "disturbance_N")
    ]
    return {
        **external,
        "all_native_steps": len(trace["time"]),
        "preparation_hand_losses": losses,
        "preparation_max_slip_m_s": slip,
        "passed": all(
            (
                abs(trace["time"][0]) < 1e-8,
                not any(caps),
                not any(external.values()),
                not any(losses),
                slip < limits["max_slip_m_s"],
            )
        ),
    }


def unassisted_receipt(cfg, trace, opening):
    source = Path(cfg["source"])
    model = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
    hands = json.loads((source / "resolved.json").read_bytes())["model"]["hands"]
    body = free_body_setup(model, hands)
    if any(cfg["unassisted_launch"][k] != v for k, v in body.items()):
        raise ValueError("The recorded free-body setup differs from the source model.")
    report = launch_checks(trace, cfg["config"], opening)
    monitors = failures(trace, cfg, 0.0)
    net = (
        trace["external_force"]
        + trace["fixture"][:, :3]
        + [0.0, 0.0, cfg["mass_kg"] * cfg["gravity_z"]]
    )
    report.update(
        body=body,
        first_monitor_events=monitors,
        peak_net_external_force_N=float(np.linalg.norm(net, axis=1).max()),
    )
    report["passed"] = report["passed"] and all(t is None for t in monitors.values())
    return report


def detached_steps(trace, side, start_s, mask):
    start = np.argmin(abs(trace["time"] - start_s))
    opened = measured_opening(trace["digits"][start, side], trace["digits"][mask, side])
    return int(
        detached(opened, trace["external_count"][mask], trace["gap"][mask]).sum()
    )


def release_cycle(events, commands, trace, cfg, number):
    side = int(cfg["side"] == "right")

    def tick(key):
        return int(np.argmin(abs(commands["time"] - events[key])))

    def winding(end, length):
        return interval(trace["time"], (events[end] - length, events[end]))

    confirmed = cfg["criteria"]["persistence_s"]
    clear = winding("clear_verified" + number, confirmed)
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
        "detached_steps": detached_steps(
            trace, side, events["open_request" + number], clear
        ),
        "minimum_gap_mm": float(trace["gap"][clear].min() * 1000),
        "confirmation_steps": int(grasp.sum()),
        "hold_steps": int(hold.sum()),
        "max_regrasp_slip_m_s": float(trace["slip"][grasp | hold, side].max()),
        "invalid_confirm_or_hold_steps": int(
            np.sum(trace["state"][grasp | hold, side] != STATES.index("grasp"))
        ),
        "post_grasp_mean_Fz_N": trace["fz"][hold].mean(axis=0).tolist(),
    }


def cycle_passes(cycle, cfg):
    dt = cfg["config"]["physics_dt"]
    samples = round(cfg["criteria"]["persistence_s"] / dt)
    return all(
        (
            cycle["position_error_mm"] <= cfg["position_tolerance_m"] * 1000,
            cycle["angle_error_rad"] <= cfg["rotation_tolerance_rad"],
            cycle["detached_steps"]
            == cycle["clear_steps"]
            == cycle["confirmation_steps"]
            == samples,
            cycle["hold_steps"] == round(2 / dt),
            cycle["invalid_confirm_or_hold_steps"] == 0,
            cycle["max_regrasp_slip_m_s"] < cfg["config"]["max_slip_m_s"],
        )
    )


def check_release(directory, outcome, cfg, trace, report):
    with np.load(directory / "commands.npz") as saved:
        commands = {
            k: saved[k] for k in ("time", "position_rung", "goal_rung", "rotation_rung")
        }
    events = outcome["events"]
    cycles = {
        n: release_cycle(events, commands, trace, cfg, n)
        for n in ("1", "2")
        if "grasp_verified" + n in events
    }
    report["cycles"] = cycles
    dwell = trace["phase"] == "clear1"
    report["clear_dwell_steps"] = int(dwell.sum())
    report["clear_dwell_detached_steps"] = detached_steps(
        trace, int(cfg["side"] == "right"), events["open_request1"], dwell
    )
    return all(
        (
            len(cycles) == 2,
            all(cycle_passes(c, cfg) for c in cycles.values()),
            report["clear_dwell_detached_steps"]
            == report["clear_dwell_steps"]
            == round(2 / cfg["config"]["physics_dt"]),
            "clear1" in outcome["windows"],
        )
    )


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
        checks.append(check_release(directory, outcome, cfg, trace, report))
    if "unassisted_launch" in cfg:
        report["full_episode"] = unassisted_receipt(
            cfg, trace, outcome["events"]["settled"]
        )
        checks.append(report["full_episode"]["passed"])
    report["passed"] = all(checks)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports = {
        f"{pair.name}/{branch}": audit_branch(pair / branch)
        for pair in sorted(args.root.glob("free_*"))
        if pair.is_dir()
        for branch in ("release", "control")
    }
    write_json(args.output, reports)
    raise SystemExit(0 if reports and all(r["passed"] for r in reports.values()) else 1)
