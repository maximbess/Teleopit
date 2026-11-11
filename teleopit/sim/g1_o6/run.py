"""Reproducible logs and replay traces, including failed trials."""

import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from .diagnostics import snapshot
from .env import Environment
from .model import provenance

POSTURES = ("baseline", "thumb-below", "thumb-below-balanced")


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def create_environment(config, posture, hold_only=False):
    if posture not in POSTURES:
        raise ValueError(f"unknown posture: {posture}")
    pose = None
    if posture == "thumb-below-balanced":
        from .grasp_pose import GraspPose

        pose = GraspPose()
    env = Environment(config, pose=pose, hold_only=hold_only)
    if posture == "thumb-below":
        from teleopit.sim.o6_diagnostics.posture import install_candidate

        install_candidate(env)
    return env


def run_trial(
    config,
    output,
    callback=None,
    *,
    posture="baseline",
    hold_only=False,
    physics_observer=None,
):
    env = create_environment(config, posture, hold_only)
    pose = env.pose
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    initial = snapshot(env)
    if pose is not None:
        from .posture_report import annotate

        annotate(env, initial)
    metadata = {
        "code_revision": revision(),
        "code_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        ),
        "config": asdict(config),
        "mujoco_version": mujoco.__version__,
        "assets": provenance(),
        "model": env.audit,
        "ik": env.ik,
        "initial": initial,
        "posture": posture,
        "support_targets": getattr(env, "support_targets", None),
    }
    if pose is not None:
        metadata["grasp_pose"] = asdict(pose)
        metadata["hold_only"] = env.hold_only
    metadata["model"]["initial_com_world"] = initial["com"]
    (output / "resolved.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Save actual compiled friction overrides, actuator limits, etc.
    mujoco.mj_saveLastXML(str(output / "scene.xml"), env.model)
    rows = [initial]
    with (output / "steps.jsonl").open("w") as stream:
        stream.write(json.dumps(initial) + "\n")
        while env.phase not in ("DONE", "ABORT", "HOLD_COMPLETE"):
            row = env.step(physics_observer)
            if pose is not None:
                annotate(env, row)
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            if callback is not None:
                callback(env, row)
    summary = summarize(env, rows)
    if pose is not None:
        from .posture_report import summarize_posture

        summary["posture"] = summarize_posture(rows, config.control_dt)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_traces(rows, metadata, summary, output)
    return summary


def summarize(env, rows):
    holds = [r for r in rows if r["phase"] == "HOLD" and r["assistance_gain"] == 0]
    return {
        "static_hold_passed": env.hold_passed and not env.initial_errors,
        "release_reacquire_passed": env.phase == "DONE" and not env.initial_errors,
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
        "right_release": replace(cfg, release_side="right"),
    }
