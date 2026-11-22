"""Score native support episodes and compare identical clock windows on a matched pair."""

import json
from pathlib import Path

import numpy as np

from .regrasp_measure import STATES, SUPPORTS, pelvis_angle


def interval(time, window):
    # The solver clock labels the start of [t,t+dt); no boundary is counted twice.
    return (time >= window[0] - 1e-8) & (time < window[1] - 1e-8)


def paired_window(release, control, baseline, window, weight):
    def forces(t, bounds):
        return t["fz"][interval(t["time"], bounds)]

    samples, matched = forces(release, window), forces(control, window)
    change = samples.mean(axis=0) - forces(release, baseline).mean(axis=0)
    control_change = matched.mean(axis=0) - forces(control, baseline).mean(axis=0)
    effect = change - control_change
    return [
        {
            "support": name,
            "release_N": samples[:, i].mean(),
            "release_pct_BW": samples[:, i].mean() * 100 / weight,
            "control_N": matched[:, i].mean(),
            "control_pct_BW": matched[:, i].mean() * 100 / weight,
            "change_N": change[i],
            "change_pp": change[i] * 100 / weight,
            "control_change_N": control_change[i],
            "control_change_pp": control_change[i] * 100 / weight,
            "effect_N": effect[i],
            "effect_pp": effect[i] * 100 / weight,
            "min_N": samples[:, i].min(),
            "max_N": samples[:, i].max(),
            "min_effect_N": min(samples[:, i] - matched[:, i]),
            "max_effect_N": max(samples[:, i] - matched[:, i]),
        }
        for i, name in enumerate(SUPPORTS)
    ]


def read_run(directory):
    outcome, config = (
        json.loads((directory / name).read_bytes())
        for name in ("outcome.json", "resolved.json")
    )
    with np.load(directory / "physics.npz") as trace:
        samples = dict(trace)
    samples["tilt"] = pelvis_angle(samples["pelvis_quat"], config["initial_qpos"][3:7])
    return outcome, config, samples


def first_at(trace, mask):
    indices = np.flatnonzero(mask)
    return round(float(trace["time"][indices[0]]), 6) if len(indices) else None


def failures(t, cfg, start):
    moving = int(cfg["side"] == "right")
    active = t["time"] >= start - 1e-8
    limit = cfg["config"]
    opposed = t["state"] == STATES.index("grasp")
    required = [1 - moving, 2, 3] if cfg["branch"] == "release" else [0, 1, 2, 3]
    conditions = {
        SUPPORTS[i] + "_loss": ~(opposed[:, i] if i < 2 else t["loaded"][:, i])
        for i in required
    }
    conditions.update(
        {
            SUPPORTS[i] + "_slip": t["slip"][:, i] >= limit["max_slip_m_s"]
            for i in required
        }
    )
    conditions.update(
        drift=t["drift"] >= limit["max_drift_m"],
        tilt=t["tilt"] >= limit["max_angle_rad"],
        effort=t["effort"] >= limit["max_effort_fraction"],
        joint=t["joint_excess"] >= limit["max_joint_error_rad"],
        mimic=t["mimic"] >= limit["max_mimic_error_rad"],
        penetration=t["penetration"] >= limit["max_penetration_m"],
        unintended=t["wrong"] > 0,
        numerical=t["warnings"] > 0,
        speed=t["base_speed"] >= 0.05,
        angular_speed=t["base_angular_speed"] >= 0.2,
        applied=(t["applied"] != 0.0) | (t["assist_gain"] != 0.0),
    )
    net = (
        t["external_force"]
        + t["fixture"][:, :3]
        + [0, 0, cfg["mass_kg"] * cfg["gravity_z"]]
    )
    conditions["static_residual"] = np.linalg.norm(net, axis=1) >= 25.0
    return {name: first_at(t, active & mask) for name, mask in conditions.items()}


def contact_events(trace, outcome):
    events = dict(outcome["events"])
    for cycle in ("1", "2"):
        opening = events.get("open_request" + cycle)
        if opening is None:
            continue
        stop = events.get("approach" + cycle, outcome["end_s"])
        mask = interval(trace["time"], (opening, stop))
        loaded = mask & (trace["external_count"] != 0)
        if np.any(loaded):
            last = np.flatnonzero(loaded)[-1]
            subsequent = mask & (np.arange(len(mask)) > last)
            events["last_contact_end" + cycle] = first_at(trace, subsequent)
        else:
            events["last_contact_end" + cycle] = opening
        events["first_gap_5mm" + cycle] = first_at(
            trace, mask & (trace["gap"] >= 0.005) & (trace["external_count"] == 0)
        )
    return events


def momentum_balance(t, cfg, bounds, end_momentum):
    mask = interval(t["time"], bounds)
    idx = np.flatnonzero(mask)
    p1 = t["momentum"][idx[-1] + 1] if idx[-1] + 1 < len(mask) else end_momentum
    dt = cfg["config"]["physics_dt"]
    net = t["fz"][mask].sum(axis=1) + t["other_fz"][mask] + t["fixture"][mask, 2]
    net += cfg["mass_kg"] * cfg["gravity_z"]
    return {
        "mean_net_Fz_N": net.mean(),
        "delta_vertical_momentum_Ns": p1 - t["momentum"][idx[0]],
        "external_impulse_Ns": net.sum() * dt,
        "mean_dynamic_residual_N": net.mean()
        - (p1 - t["momentum"][idx[0]]) / (len(idx) * dt),
    }


def check_pair(directory, start, stance_ids):
    with (
        np.load(directory / "release" / "commands.npz") as r,
        np.load(directory / "control" / "commands.npz") as c,
    ):
        np.testing.assert_array_equal(r["time"], c["time"])
        mask = r["time"] <= start + 1e-8
        for key in ("qpos", "qvel", "ctrl", "target"):
            np.testing.assert_array_equal(r[key][mask], c[key][mask])
        error = abs(r["ctrl"][:, stance_ids] - c["ctrl"][:, stance_ids]).max()
        if error > 1e-12:
            raise ValueError("matched stance commands diverged")
        return {
            "identical_precommand_ticks": int(mask.sum()),
            "maximum_stance_difference_rad": float(error),
        }


def score_pair(directory):
    directory = Path(directory)
    outcome, cfg, r = read_run(directory / "release")
    other, ccfg, c = read_run(directory / "control")
    baseline = outcome["windows"]["baseline"]
    check = check_pair(directory, baseline[1], cfg["stance_actuators"])
    pre = r["time"] < baseline[1] - 1e-8
    for key in ("fz", "state", "loaded", "momentum"):
        np.testing.assert_array_equal(r[key][pre], c[key][pre])
    loss = failures(r, cfg, baseline[1])
    control_loss = failures(c, ccfg, baseline[1])
    first_loss = min(first_support_loss(loss), first_support_loss(control_loss))
    windows, unavailable = measured_windows(outcome, first_loss)
    weight = cfg["mass_kg"] * abs(cfg["gravity_z"])
    rows, balance = [], {}
    for name, bounds, status in windows:
        rows.extend(
            dict(window=name, status=status, start_s=bounds[0], end_s=bounds[1], **row)
            for row in paired_window(r, c, baseline, bounds, weight)
        )
        balance[name] = {
            branch: momentum_balance(trace, cfg, bounds, result["end_momentum_Ns"])
            for branch, trace, result in (
                ("release", r, outcome),
                ("control", c, other),
            )
        }
    return {
        "matching": check,
        "release_first_events": loss,
        "control_first_events": control_loss,
        "events": contact_events(r, outcome),
        "unavailable": unavailable,
        "balance": balance,
        "forces": rows,
    }


def first_support_loss(events):
    return min(
        (v for k, v in events.items() if k.endswith("_loss") and v is not None),
        default=np.inf,
    )


def measured_windows(outcome, first_loss):
    phases = (
        "baseline",
        "opening1",
        "retreat1",
        "clear1",
        "hold1",
        "opening2",
        "retreat2",
        "hold2",
    )
    windows, unavailable = [], {}
    for name in phases:
        bounds = outcome["windows"].get(name)
        if bounds is None:
            unavailable[name] = outcome["missing"] or "window not reached"
            bounds = outcome["windows"].get(name + "_transient")
            if bounds:
                windows.append((name + "_transient", bounds, "incomplete"))
            continue
        status = "measured"
        if name in ("clear1", "hold1", "hold2") and first_loss < bounds[1] - 1e-8:
            unavailable[name] = "required support lost before window ended"
            status = "transient, settled interval unavailable"
        windows.append((name, bounds, status))
    return windows, unavailable
