"""Reproducible deterministic scenarios, JSONL trajectories and JSON summaries."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .env import O6GraspEnv


def scenarios(config):
    """One-variable comparisons; finite rung length is evaluated separately."""
    config = replace(config, evaluation_mode="unsupported")
    rung_rotation = Rotation.from_quat(
        np.asarray(config.rung_quat)[[1, 2, 3, 0]]
    ).as_matrix()
    along, across = rung_rotation[:, 0], rung_rotation[:, 1]
    small = resized_rung(config, 5 / 6)
    large = resized_rung(config, 7 / 6)
    return {
        "free_motion": replace(
            config, rung_enabled=False, evaluation_mode="acquisition"
        ),
        "acquisition": replace(
            config, evaluation_mode="acquisition", load_force=(0, 0, 0)
        ),
        "fixture_removal": replace(config, load_force=(0, 0, 0)),
        "pullout_small": config,
        "pullout_large": replace(
            config, load_force=tuple(30 * along), load_frame="world"
        ),
        "twist_along": replace(
            config,
            load_force=(0, 0, 0),
            load_torque=tuple(0.15 * along),
            load_frame="world",
        ),
        "twist_across": replace(
            config,
            load_force=(0, 0, 0),
            load_torque=tuple(0.15 * across),
            load_frame="world",
        ),
        "combined": replace(config, load_torque=(0, 0.1, 0)),
        "impulse": replace(config, load_force=(0, 0, -10), load_profile="impulse"),
        "release": replace(config, load_profile="constant"),
        "friction_low": replace(config, friction=0.25),
        "alignment_offset": replace(
            config, palm_pos=tuple(np.asarray(config.palm_pos) + [0.004, 0, 0])
        ),
        "rung_small": small,
        "rung_large": large,
        "rectangular": replace(config, cross_section="rectangular"),
        "actuator_delay": replace(config, actuator_delay_steps=3),
        "finite_edge": replace(
            config,
            length=0.06,
            rung_pos=tuple(np.asarray(config.rung_pos) + 0.04 * along),
        ),
        "fine_timestep": replace(
            config,
            physics_dt=config.physics_dt / 2,
            solver_iterations=config.solver_iterations * 2,
            solver_tolerance=config.solver_tolerance / 10,
        ),
    }


def resized_rung(config, factor):
    if config.cross_section == "circular":
        return replace(config, radius=config.radius * factor)
    return replace(config, width=config.width * factor, height=config.height * factor)


def summarize(name, config, initial, rows):
    unsupported = [row for row in rows if row["unsupported"]]
    assisted = [
        row for row in rows if row["fixture_gain"] > 0 and row["mode"] != "RELEASE"
    ]
    summary = {
        "scenario": name,
        "evaluation_mode": config.evaluation_mode,
        "initial_penetration": initial["max_penetration"],
        "unsupported_seconds": len(unsupported) * config.control_dt,
        "assisted_contact_seconds": sum(bool(row["contacts"]) for row in assisted)
        * config.control_dt,
        "release_clear": rows[-1]["release_clear"],
        "max_translation": peak(unsupported, "translation_drift"),
        "max_rotation": peak(unsupported, "rotation_drift"),
        "final_verification": rows[-1]["verification"],
    }
    summary.update(numerical_summary(rows))
    summary.update(holding_summary(unsupported, config.control_dt))
    summary["unsupported_success"] = trial_passes(config, summary)
    return summary


def peak(rows, field):
    return max((row[field] for row in rows), default=0.0)


def holding_summary(rows, dt):
    verified = [row for row in rows if row["verification"] == "VERIFIED"]
    failures = [row for row in rows if row["verification"] in ("FAILED", "SLIPPING")]
    longest = peak(verified, "stable_duration")
    reasons = sorted(
        {reason for row in failures for reason in row["verification_reasons"]}
    )
    return {
        "verified_seconds": len(verified) * dt,
        "longest_stable_seconds": longest,
        "failure_observed": bool(failures),
        "failure_reasons": reasons,
        "first_failure": failure_sample(failures),
    }


def failure_sample(rows):
    if not rows:
        return None
    row = rows[0]
    return {
        key: row[key]
        for key in (
            "time",
            "verification",
            "verification_reasons",
            "external_wrench",
            "translation_drift",
            "rotation_drift",
        )
    }


def numerical_summary(rows):
    fields = {
        "max_contact_slip": "contact_slip_speed",
        "max_penetration": "max_penetration",
        "max_coupling_error": "coupling_error",
        "max_effort_fraction": "effort_fraction",
        "max_joint_limit_violation": "joint_limit_violation",
    }
    summary = {name: peak(rows, field) for name, field in fields.items()}
    summary["numerical_warnings"] = any(any(row["numerical_warnings"]) for row in rows)
    summary["peak_external_force"] = max(
        np.linalg.norm(row["external_wrench"][:3]) for row in rows
    )
    return summary


def trial_passes(config, summary):
    checks = (
        summary["longest_stable_seconds"]
        >= config.success_duration + config.verification_window,
        not summary["failure_observed"],
        not summary["numerical_warnings"],
        summary["initial_penetration"] < 1e-6,
        summary["max_penetration"] < config.max_penetration,
        summary["max_coupling_error"] < config.max_coupling_error,
        summary["max_joint_limit_violation"] < config.max_joint_limit_violation,
    )
    return bool(all(checks))


def run_trial(name, config, output, *, seed=0, callback=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = O6GraspEnv(config)
    config = env.config  # includes any HOLD extension required by the full trace
    _, initial = env.reset(seed=seed)
    resolved = asdict(config)
    if env.replay is not None:
        snapshot = f"{name}.replay.json"
        (output / snapshot).write_bytes(env.replay.raw_bytes)
        resolved["wrist_load_path"] = snapshot  # relative to the saved config
    (output / f"{name}.config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (output / f"{name}.xml").write_text(env.scene_xml)
    rows = []
    with (output / f"{name}.jsonl").open("w") as stream:
        stream.write(json.dumps(dict(event="reset", **initial)) + "\n")
        while True:
            obs, _, _, done, info = env.step(np.zeros(6))
            info["actor_observation"] = obs.tolist()
            stream.write(json.dumps(info) + "\n")
            rows.append(info)
            if callback is not None:
                callback(env, info)
            if done:
                break
    summary = summarize(name, config, initial, rows)
    summary.update(seed=seed, mujoco_version=mujoco.__version__, config=asdict(config))
    if env.replay is not None:
        summary.update(
            replay=env.replay.metadata, replay_result=replay_summary(rows, config)
        )
        summary["unsupported_success"] &= (
            summary["replay_result"]["verified_seconds"] >= config.success_duration
        )
    (output / f"{name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_suite(config, output, *, seed=0):
    if config.load_profile == "replay":
        raise ValueError(
            "run a single replay trial; the analytic suite substitutes its own loads"
        )
    results = [
        run_trial(name, cfg, output, seed=seed)
        for name, cfg in scenarios(config).items()
    ]
    path = Path(output)
    (path / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Deterministic O6 evaluation",
        "",
        "| Scenario | Unsupported verified (s) | Success | Failure/sliding | Release clear | Peak penetration (mm) |",
        "|---|---:|---|---|---|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['scenario']} | {row['verified_seconds']:.2f} | {row['unsupported_success']} | "
            f"{row['failure_observed']} | {row['release_clear']} | {1000 * row['max_penetration']:.2f} |"
        )
    (path / "summary.md").write_text("\n".join(lines) + "\n")
    return results


def replay_summary(rows, config):
    active = [row for row in rows if row["wrist_replay"]["active"]]
    verified = [
        row
        for row in active
        if row["verification"] == "VERIFIED" and row["unsupported"]
    ]
    return {
        "active_seconds": len(active) * config.control_dt,
        "verified_seconds": len(verified) * config.control_dt,
        "scale": config.replay_scale,
        "fade_duration_s": config.replay_fade_duration,
        "peak_applied_force_N": max(
            (np.linalg.norm(row["external_wrench"][:3]) for row in active), default=0.0
        ),
        "peak_applied_moment_Nm": max(
            (np.linalg.norm(row["external_wrench"][3:]) for row in active), default=0.0
        ),
    }
