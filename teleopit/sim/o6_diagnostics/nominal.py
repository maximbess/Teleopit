"""Strict nominal reproduction with optional failed/unscored HOLD continuation."""

import copy
import gzip
import json
from dataclasses import asdict

import mujoco
import numpy as np

from teleopit.sim.g1_o6.diagnostics import failure_reasons, region, snapshot
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.metadata import resolved
from teleopit.sim.g1_o6.posture_report import annotate
from teleopit.sim.g1_o6.run import export_traces, summarize

from .config import DiagnosticConfig, emergency
from .events import Events, Saturation
from .io import code_state, manifest, output_directory, write_csv, write_json
from .mechanics import subtree
from .signals import joint_signals
from .telemetry import Telemetry


class DiagnosticStop(Exception):
    pass


def gate_measurements(row, cfg):
    values = {
        "drift": row["base_drift_m"],
        "angle": row["base_angle_rad"],
        "joint_limits": row["joint_limit_error_rad"],
        "mimic": row["mimic_error_rad"],
        "effort": max(row["effort_fraction"]),
        "penetration": row["penetration_m"],
        "velocity": float(np.linalg.norm(row["base_velocity"][:3])),
        "angular_velocity": float(np.linalg.norm(row["base_velocity"][3:])),
        "force_balance": float(np.linalg.norm(row["static_balance_residual_N"])),
    }
    thresholds = {
        "drift": cfg.max_drift_m,
        "angle": cfg.max_angle_rad,
        "joint_limits": cfg.max_joint_error_rad,
        "mimic": cfg.max_mimic_error_rad,
        "effort": cfg.max_effort_fraction,
        "penetration": cfg.max_penetration_m,
        "velocity": 0.05,
        "angular_velocity": 0.2,
        "force_balance": 25.0,
    }
    for name, item in row["supports"].items():
        values["contact_loss:" + name] = float(not item["loaded"])
        thresholds["contact_loss:" + name] = 1.0
        if name.endswith("hand"):
            values["opposition_loss:" + name] = float(not item["opposing"])
            thresholds["opposition_loss:" + name] = 1.0
        values["slip:" + name] = item["max_slip"] if item["loaded"] else None
        thresholds["slip:" + name] = cfg.max_slip_m_s
    return values, thresholds


class Observer:
    def __init__(self, env, diagnostic, stream):
        self.config = diagnostic
        self.stream = stream
        self.continuing = False
        self.data = mujoco.MjData(env.model)
        self.events = Events(env.config.physics_dt, env.config.stability_seconds)
        self.hold_events = Events(env.config.physics_dt, env.config.stability_seconds)
        self.saturation = Saturation(
            [env.model.actuator(i).name for i in range(env.model.nu)],
            env.config.physics_dt,
            env.config.max_effort_fraction,
        )
        self.warnings = [w.number for w in env.data.warning]
        self.reference = (
            env.model.body("pelvis").id,
            env.data.body("pelvis").xpos.copy(),
        )
        self.clock = float(env.data.time)

    def __call__(self, env):
        self.clock += env.config.physics_dt
        mujoco.mj_copyData(self.data, env.model, env.data)
        self.check_numerics()
        mujoco.mj_forward(
            env.model, self.data
        )  # scratch only: never touches the simulation
        proxy = copy.copy(env)
        proxy.data = self.data
        row = snapshot(proxy)
        values, thresholds = gate_measurements(row, env.config)
        named, named_thresholds = joint_signals(env.model, self.data, env.config)
        values.update(named)
        thresholds.update(named_thresholds)
        self.wrist_peaks(row)
        phase = "DIAGNOSTIC_HOLD" if self.continuing else env.phase
        self.events.update(self.clock, phase, values, thresholds)
        hold = env.phase == "HOLD" and env.assistance_gain == 0
        if hold:
            self.hold_events.update(self.clock, phase, values, thresholds)
        fractions = {
            env.model.actuator(i).name: f for i, f in enumerate(row["effort_fraction"])
        }
        self.saturation.update(self.clock, fractions, hold)
        self.stream.write(
            json.dumps(
                {
                    "time_s": self.clock,
                    "phase": phase,
                    "resolution_s": env.config.physics_dt,
                    "values": values,
                    "actuator_effort_fraction": row["effort_fraction"],
                    "joint_limit_errors_rad": row["joint_limit_errors"],
                    "numerical_warnings": row["numerical_warnings"],
                }
            )
            + "\n"
        )
        if self.continuing:
            reason = emergency(
                env.model, self.data, self.config, self.reference, self.warnings
            )
            if reason:
                raise DiagnosticStop(reason)

    def check_numerics(self):
        if not np.isfinite(np.r_[self.data.qpos, self.data.qvel, self.data.qacc]).all():
            raise DiagnosticStop("nonfinite_state")
        if any(w.number > old for w, old in zip(self.data.warning, self.warnings)):
            raise DiagnosticStop("new_numerical_warning")

    def wrist_peaks(self, row):
        for side, wrench in row["wrist_world_on_hand"].items():
            for label, components in [
                ("wrist_force_N", wrench[:3]),
                ("wrist_moment_Nm", wrench[3:]),
            ]:
                key = label + ":" + side
                self.events.extrema[key] = max(
                    float(np.linalg.norm(components)), self.events.extrema.get(key, 0.0)
                )


def coupled_telemetry(env):
    targets = env.support_targets
    groups = {
        name: [b for b in range(env.model.nbody) if region(env.model, b) == name]
        for name in targets
    }
    root = env.model.body("pelvis").id
    return Telemetry(
        env.model, env.data, root, subtree(env.model, root), groups, targets, env.hands
    )


def sample(env, telemetry):
    applied = [
        (env.assistance, env.data.qpos[:3]),
        (np.r_[env.disturbance, np.zeros(3)], env.data.qpos[:3]),
    ]
    row = telemetry.sample(
        env.data, getattr(env, "desired_target", env.target), env.target, applied
    )
    row["assistance_wrench_world_at_base"] = env.assistance.tolist()
    row["disturbance_wrench_world_at_base"] = np.r_[
        env.disturbance, np.zeros(3)
    ].tolist()
    row["base_velocity_world_m_s"] = env.data.qvel[:3].tolist()
    row["base_angular_velocity_body_rad_s"] = env.data.qvel[3:6].tolist()
    return row


def run_nominal(cfg, output, diagnostic=None, *, pose=None):
    diagnostic = diagnostic or DiagnosticConfig()
    diagnostic.validate()
    path = output_directory(output)
    env = Environment(cfg, pose=pose)
    telemetry = coupled_telemetry(env)
    initial = snapshot(env)
    annotate(env, initial)
    metadata = {
        **resolved(env, initial),
        **code_state(),
        "diagnostic_schema": "o6_diagnostic_v1",
        "diagnostic_config": asdict(diagnostic),
        "legacy_pose_note": "hand_in_target_rung remains world-axis difference and world orientation",
    }
    write_json(path / "resolved.json", metadata)
    mujoco.mj_saveLastXML(str(path / "scene.xml"), env.model)
    trial = CoupledTrial(env, diagnostic, telemetry, initial)
    with (
        gzip.open(path / "physics.jsonl.gz", "wt") as physics,
        (path / "steps.jsonl").open("w") as stream,
    ):
        observer = Observer(env, diagnostic, physics)
        trial.run(observer, stream)
    rows = trial.rows
    strict = trial.strict
    summary = finish_summary(
        strict, observer, rows, trial.entry, trial.reason, cfg.control_dt
    )
    write_json(path / "summary.json", summary)
    write_csv(
        path / "failure_events.csv",
        [dict(scope="all_phases", **r) for r in observer.events.rows]
        + [dict(scope="zero_assistance_hold", **r) for r in observer.hold_events.rows],
    )
    export_traces(
        rows,
        metadata,
        {"final_phase": "ABORT" if strict is None else strict["final_phase"]},
        path,
    )
    from .plots import nominal_plot

    nominal_plot(rows, path)
    manifest(
        path,
        "coupled_nominal",
        {
            "side": "both",
            "physics_dt_s": cfg.physics_dt,
            "control_dt_s": cfg.control_dt,
            "source_revision": metadata["revision"],
            "strict_outcome": summary["original_strict_outcome"],
        },
    )
    return summary


class CoupledTrial:
    def __init__(self, env, config, telemetry, initial):
        self.env, self.config, self.telemetry = env, config, telemetry
        self.rows = [initial]
        self.strict = None
        self.hold_start = None
        self.entry = []
        self.reason = "strict_end"

    def record(self, row, observer, stream):
        env = self.env
        if env.phase == "HOLD" and self.hold_start is None:
            self.hold_start = float(env.data.time)
            self.telemetry.start_hold(env.data)
            self.entry = failure_reasons(row, env.config, env.required_supports())
        annotate(env, row)
        row["diagnostic_unscored"] = observer.continuing
        row["diagnostic_v1"] = sample(env, self.telemetry)
        row["diagnostic_v1"]["saturation"] = observer.saturation.summary()
        self.rows.append(row)
        stream.write(json.dumps(row) + "\n")

    def strict_end(self, row, observer):
        env = self.env
        if env.phase not in ("ABORT", "HOLD_COMPLETE") or self.strict is not None:
            return False
        self.strict = summarize(env, self.rows)
        self.strict["abort_time_s"] = (
            float(env.data.time) if env.phase == "ABORT" else None
        )
        self.strict["saturation"] = copy.deepcopy(observer.saturation.summary())
        if (
            self.config.continuation_s
            and row["phase"] == "HOLD"
            and env.phase == "ABORT"
        ):
            observer.continuing = True
            env.phase = "HOLD"
            env.phase_start = self.hold_start
            return False
        return True

    def expired(self, observer):
        if not observer.continuing:
            return False
        if (
            self.env.data.time
            < self.strict["abort_time_s"] + self.config.continuation_s - 1e-9
        ):
            return False
        self.reason = "continuation_time_limit"
        return True

    def run(self, observer, stream):
        self.rows[0]["diagnostic_v1"] = sample(self.env, self.telemetry)
        stream.write(json.dumps(self.rows[0]) + "\n")
        while True:
            try:
                row = self.env.step(observer, advance_phase=not observer.continuing)
            except DiagnosticStop as exc:
                self.reason = str(exc)
                break
            self.record(row, observer, stream)
            if self.strict_end(row, observer) or self.expired(observer):
                break


def finish_summary(strict, observer, rows, entry, reason, control_dt):
    return {
        "schema": "o6_nominal_diagnostic_v1",
        "original_strict_outcome": strict,
        "first_threshold_violation": observer.events.first,
        "first_persistent_violation_per_gate": observer.hold_events.persistent,
        "would_abort_at": None if strict is None else strict["abort_time_s"],
        "would_abort_reasons": [] if strict is None else strict["failure_reasons"],
        "gates_violated_on_hold_entry": entry,
        "diagnostic_end_time_s": observer.clock,
        "last_control_sample_s": rows[-1]["time"],
        "diagnostic_end_reason": reason,
        "continuation_unscored": observer.continuing,
        "physics_extrema": observer.events.extrema,
        "event_resolution_s": observer.events.dt,
        "control_resolution_s": control_dt,
        "event_order_note": "Events within one physics sample are unordered; per-gate persistence differs from global bad-time abort.",
    }
