"""Reproducible logs and replay traces, including failed trials."""

import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .diagnostics import snapshot
from .env import Environment
from .metadata import resolved
from .posture_report import annotate, summarize_posture


def run_trial(
    config,
    output,
    callback=None,
    *,
    pose=None,
    physics_observer=None,
):
    env = Environment(config, pose=pose)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    initial = snapshot(env)
    annotate(env, initial)
    metadata = resolved(env, initial)
    (output / "resolved.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Save actual compiled friction overrides, actuator limits, etc.
    mujoco.mj_saveLastXML(str(output / "scene.xml"), env.model)
    rows = [initial]
    with (output / "steps.jsonl").open("w") as stream:
        stream.write(json.dumps(initial) + "\n")
        while env.phase not in ("ABORT", "HOLD_COMPLETE"):
            row = env.step(physics_observer)
            annotate(env, row)
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            if callback is not None:
                callback(env, row)
    summary = summarize(env, rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_traces(rows, metadata, summary, output)
    return summary


def summarize(env, rows):
    holds = [r for r in rows if r["phase"] == "HOLD" and r["assistance_gain"] == 0]
    return {
        "static_hold_passed": env.hold_passed and not env.initial_errors,
        "release_reacquire_passed": False,
        "final_phase": env.phase,
        "failure_reasons": env.abort_reasons,
        "events": env.events,
        "unsupported_hold_seconds": len(holds) * env.config.control_dt,
        "max_penetration_m": max(r["penetration_m"] for r in rows),
        "peak_effort_fraction": max(max(r["effort_fraction"]) for r in rows),
        "peak_wrist_force_N": peak_load(rows, slice(0, 3)),
        "peak_wrist_moment_Nm": peak_load(rows, slice(3, 6)),
        "final_diagnostics": final_diagnostics(env, rows[-1]),
        "initial_penetration_m": rows[0]["penetration_m"],
        "ik": env.ik,
        "numerical_warnings": any(any(r["numerical_warnings"]) for r in rows),
        "ready_for_learning": False,
        "posture": summarize_posture(rows, env.config.control_dt),
    }


def peak_load(rows, component):
    return {
        s: max(np.linalg.norm(r["wrist_world_on_hand"][s][component]) for r in rows)
        for s in ("left", "right")
    }


def final_diagnostics(env, row):
    return {
        "saturated_actuators": {
            env.model.actuator(i).name: row["actuator_effort"][i]
            for i, f in enumerate(row["effort_fraction"])
            if f >= env.config.max_effort_fraction
        },
        "joint_limit_errors": row["joint_limit_errors"],
        "base_drift_m": row["base_drift_m"],
        "supports": row["supports"],
        "static_balance_residual_N": row["static_balance_residual_N"],
        "assistance_world": row["assistance_world"],
    }


def export_traces(rows, metadata, summary, output):
    for side in ("left", "right"):
        trace = {
            "schema": "o6_wrist_load_v1",
            "source": {
                "kind": "simulation",
                "description": f"G1 physical O6 {side} wrist; parent-on-hand wrench; trial {output.name}",
                "commit": metadata["code_revision"],
                "side": side,
                "trial": str(output),
                "resolved_config": metadata["config"],
                "outcome": summary["final_phase"],
            },
            "units": {"time": "s", "force": "N", "torque": "N*m"},
            "frame": "world",
            "application_site": "wrist_mount",
            "point_m": [0, 0, 0],
            "samples": [],
        }
        for row in rows:
            wrench = row["wrist_world_on_hand"][side]
            trace["samples"].append(
                {"time_s": row["time"], "force_N": wrench[:3], "torque_Nm": wrench[3:]}
            )
        (output / f"{side}.wrist.json").write_text(json.dumps(trace, indent=2) + "\n")


def scenarios(cfg):
    return {
        "nominal": cfg,
        "fine_step": replace(cfg, physics_dt=cfg.physics_dt / 2),
        "body_offset": replace(cfg, root_offset_x=0.002),
        "wrist_offset": replace(cfg, wrist_offset_x=0.002),
        "friction_low": replace(cfg, friction_scale=0.25 / 1.8),
        "disturbance": replace(cfg, disturbance_N=10.0),
        "delay": replace(cfg, delay_steps=2),
    }
