"""Hold-only qualification; extra telemetry cannot influence the controller."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.metadata import resolved
from teleopit.sim.g1_o6.posture_report import annotate
from teleopit.sim.g1_o6.run import export_traces, summarize
from teleopit.sim.o6_diagnostics.io import (
    manifest,
    output_directory,
    write_csv,
    write_json,
)
from teleopit.sim.o6_diagnostics.nominal import coupled_telemetry, sample

from .audit import PhysicsAudit
from .config import DriftCriteria
from .measurements import late_drift, load_shares


def experiment_metadata(env, initial, experiment):
    metadata = resolved(env, initial)
    metadata.update(
        experiment_schema="g1_o6_support_v1",
        experiment=experiment,
        robot_mass_kg=env.mass,
        seed=0,
        drift_criteria=asdict(DriftCriteria()),
        experiment_source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    if metadata["code_dirty"]:
        metadata["dirty_diff"] = subprocess.check_output(
            ["git", "diff", "HEAD"], text=True
        )
    return metadata


def enrich(env, row, telemetry, hold_start, observer=None):
    annotate(env, row)
    if hold_start is not None:
        telemetry.start_hold(env.data)
    row["diagnostic_v1"] = sample(env, telemetry)
    row["support_measurement"] = load_shares(
        row, env.mass, abs(env.model.opt.gravity[2])
    )
    row["hold_time_s"] = (
        float(env.data.time - hold_start) if hold_start is not None else None
    )
    hook = getattr(observer, "enrich", None)
    if hook is not None:
        hook(env, row)


def run_hold(
    output,
    *,
    seconds=60,
    physics_dt=0.001,
    pose=None,
    environment=None,
    observer_factory=None,
    metadata_extra=None,
):
    env, path, initial, metadata = prepare_hold(
        output, seconds, physics_dt, pose, environment, metadata_extra
    )
    audit = observer_factory(env, path) if observer_factory else PhysicsAudit(env)
    rows = record_hold(env, path, initial, audit)
    result = hold_result(env, rows, seconds, audit)
    write_json(path / "summary.json", result)
    write_csv(path / "physics_events.csv", audit.events.rows)
    write_json(path / "all_phase_events.json", audit.all_events.rows)
    export_traces(rows, metadata, result["strict"], path)
    manifest(path, "support_long_hold", {"strict_status": result["strict_status"]})
    return result


def prepare_hold(output, seconds, physics_dt, pose, environment, extra):
    import mujoco

    cfg = Config(hold_seconds=seconds, physics_dt=physics_dt)
    env = environment if environment is not None else Environment(cfg, pose=pose)
    if env.config.hold_seconds != seconds or env.config.physics_dt != physics_dt:
        raise ValueError(
            "environment duration/timestep differs from requested experiment"
        )
    path = output_directory(output)
    initial = snapshot(env)
    annotate(env, initial)
    metadata = experiment_metadata(
        env, initial, {"mode": "long-hold", "requested_hold_s": seconds}
    )
    metadata.update(extra or {})
    write_json(path / "resolved.json", metadata)
    mujoco.mj_saveLastXML(str(path / "scene.xml"), env.model)
    return env, path, initial, metadata


def summary_row(row):
    # Full detail is already streamed to disk. These three payloads are not
    # inputs to the original evaluator, drift calculation or wrist export.
    return {
        k: v
        for k, v in row.items()
        if k not in ("diagnostic_v1", "pose_search", "support_measurement")
    }


def record_hold(env, path, initial, audit):
    telemetry = coupled_telemetry(env)
    rows, hold_start = [], None
    try:
        with (path / "steps.jsonl").open("w") as stream:
            enrich(env, initial, telemetry, hold_start, audit)
            record_row(stream, rows, initial)
            while env.phase not in ("ABORT", "HOLD_COMPLETE"):
                row = env.step(audit)
                if env.phase == "HOLD" and hold_start is None:
                    hold_start = float(env.data.time)
                enrich(env, row, telemetry, hold_start, audit)
                record_row(stream, rows, row)
    finally:
        close = getattr(audit, "close", None)
        if close is not None:
            close()
    return rows


def record_row(stream, rows, row):
    stream.write(json.dumps(row) + "\n")
    rows.append(summary_row(row))


def hold_result(env, rows, seconds, audit):
    strict = summarize(env, rows)
    held = [r for r in rows if r["phase"] == "HOLD"]
    result = {
        "schema": "g1_o6_long_hold_v1",
        "software_status": "PASS",
        "strict_status": "PASS" if strict["static_hold_passed"] else "FAIL",
        "strict": strict,
        "drift_assessment": late_drift(held, seconds),
        "physics_audit": audit.report(),
        "maximum_reset_drift_m": max(r["base_drift_m"] for r in rows),
        "final_reset_drift_m": rows[-1]["base_drift_m"],
    }
    finish_observer(audit, result)
    result["progression_justified"] = long_qualified(result, seconds)
    return result


def finish_observer(observer, result):
    if hasattr(observer, "contact_report"):
        result["contact_quality"] = observer.contact_report()


def long_qualified(result, seconds):
    audit = result["physics_audit"]
    return (
        seconds >= 120
        and result["strict_status"] == "PASS"
        and result["drift_assessment"]["status"] == "PASS"
        and result.get("contact_quality", {"status": "PASS"})["status"] == "PASS"
        and audit["no_external_support"]
        and not audit["unintended_support_steps"]
        and not audit["housing_contact_steps"]
    )
