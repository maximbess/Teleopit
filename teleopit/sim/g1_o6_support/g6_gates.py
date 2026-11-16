"""Fail-closed G6 phase and evidence gates, separate from legacy evidence."""

from .ladder_campaign import screen_passes
from .long_hold import long_qualified

MOVING_PHASES = {
    "OPEN",
    "RETRACT",
    "THREE_SUPPORT_HOLD",
    "REACH",
    "APPROACH",
    "CLOSE",
    "VERIFY_GRASP",
}


def required_contact_sides(phase, selected):
    return {"left", "right"} - ({selected} if phase in MOVING_PHASES else set())


def hold_passes(run, seconds):
    result = run["summary"]
    return (
        run["metadata"]["experiment"].get("requested_hold_s") == seconds
        and result.get("strict", {}).get("unsupported_hold_seconds", 0)
        >= seconds - 0.021
        and screen_passes(result)
    )


def initial_pose_gate(runs):
    unique = {r["path"]: r for r in runs}.values()
    if not all(any(hold_passes(r, duration) for r in unique) for duration in (30, 60)):
        return False
    long = [r for r in unique if stationary_qualified(r)]
    timesteps = [r["metadata"]["config"]["physics_dt"] for r in long]
    return timesteps.count(0.001) >= 2 and timesteps.count(0.0005) >= 1


def stationary_qualified(run):
    return hold_passes(run, 120) and long_qualified(run["summary"], 120)


def loaded_hold_gate(runs):
    valid = {r["path"]: r for r in runs if loaded_passes(r)}
    timesteps = [r["metadata"]["config"]["physics_dt"] for r in valid.values()]
    return timesteps.count(0.001) >= 2 and timesteps.count(0.0005) >= 1


def loaded_passes(run):
    experiment = run["metadata"]["experiment"]
    result = run["summary"]
    return (
        experiment.get("mode") == "load"
        and experiment.get("target_share") == 0.3
        and experiment.get("plateau_seconds") == 60
        and result.get("physical_status") == "PASS"
        and result.get("capacity_status") == "PASS"
        and result.get("plateau_convergence", {}).get("status") == "PASS"
        and result.get("load_measurements", {}).get("continuous_in_band_s", 0) >= 59.98
    )
