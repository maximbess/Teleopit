"""Fresh-reset hand trials; failed gates are latched independently of retention."""

import gzip
import json
from dataclasses import asdict

import mujoco
import numpy as np

from teleopit.sim.o6_grasp.controller import smooth

from .config import DiagnosticConfig, position_guard, state_guard
from .events import Events, Saturation
from .hand_results import SCORED_PHASES, hand_summary
from .harness import HandHarness
from .io import code_state, manifest, output_directory, write_csv, write_json
from .signals import hand_signals


def free_schedule(t, cfg):
    ends = np.cumsum(cfg.free_schedule_s)
    index = min(int(np.searchsorted(ends, t + 1e-9, side="right")), 4)
    start = ends[index - 1] if index else 0
    amount = [
        0,
        smooth((t - start) / cfg.free_schedule_s[1]),
        1,
        1 - smooth((t - start) / cfg.free_schedule_s[3]),
        0,
    ][index]
    return ["OPEN", "CLOSE", "CLOSED", "REOPEN", "OPEN_SETTLE"][index], amount, 0.0, 0.0


def load_schedule(t, cfg, protocol):
    ends = np.cumsum(
        [
            cfg.acquire_open_s,
            cfg.acquire_close_s,
            cfg.fixture_remove_s,
            cfg.zero_fixture_settle_s,
            cfg.load_ramp_s,
            cfg.plateau_s,
        ]
    )
    index = min(int(np.searchsorted(ends, t + 1e-9, side="right")), 5)
    amount = (
        0.0
        if index == 0
        else smooth((t - ends[0]) / cfg.acquire_close_s)
        if index == 1
        else 1.0
    )
    gain = (
        1.0
        if index < 2
        else 1 - smooth((t - ends[1]) / cfg.fixture_remove_s)
        if index == 2
        else 0.0
    )
    load = smooth((t - ends[3]) / cfg.load_ramp_s) if index == 4 else float(index == 5)
    if protocol == "seated_diagnostic":
        load = (
            smooth((t - ends[1]) / cfg.fixture_remove_s)
            if index == 2
            else float(index >= 3)
        )
    return (
        ["OPEN", "CLOSE", "REMOVE_FIXTURE", "SETTLE", "LOAD_RAMP", "PLATEAU"][index],
        amount,
        gain,
        load,
    )


def gate_values(row, cfg, fixed):
    reasons = []
    for name, value, threshold in [
        ("joint_limits", row["max_joint_error_rad"], cfg.max_joint_error_rad),
        ("mimic", row["max_mimic_error_rad"], cfg.max_mimic_error_rad),
        ("penetration", row["penetration_m"], cfg.max_penetration_m),
        (
            "effort",
            max(a["effort_fraction"] for a in row["actuators"].values()),
            cfg.max_effort_fraction,
        ),
    ]:
        if value >= threshold:
            reasons.append(name)
    if any(row["warnings"]):
        reasons.append("numerical_warning")
    if not fixed:
        reasons.extend(grasp_reasons(row, cfg))
    return reasons


def grasp_reasons(row, cfg):
    support = row["supports"][row["side"] + "_hand"]
    reasons = []
    if not support["loaded"] or not support["opposing"]:
        reasons.append("grasp_loss")
    if support["slip_m_s"] is not None and support["slip_m_s"] >= cfg.max_slip_m_s:
        reasons.append("slip")
    drift = row["from_hold"]
    if (
        drift["translation_m"] is not None
        and drift["translation_m"] >= cfg.max_translation_m
    ):
        reasons.append("translation")
    if drift["angle_rad"] is not None and drift["angle_rad"] >= cfg.max_rotation_rad:
        reasons.append("rotation")
    return reasons


def run_hand(
    source,
    side,
    output,
    *,
    fixed=False,
    gravity=True,
    dt=None,
    wrench=None,
    protocol="primary",
    only=None,
    config=None,
):
    cfg = config or DiagnosticConfig()
    cfg.validate()
    path = output_directory(output)
    h = HandHarness(source, side, fixed=fixed, gravity=gravity, dt=dt)
    wrench = np.zeros(6) if wrench is None else np.asarray(wrench, float)
    if wrench.shape != (6,) or not np.isfinite(wrench).all():
        raise ValueError("wrench must contain six finite SI components")
    metadata = hand_metadata(source, h, cfg, protocol, only, wrench)
    write_json(path / "resolved.json", metadata)
    mujoco.mj_saveLastXML(str(path / "scene.xml"), h.model)
    result, rows = integrate_hand(h, cfg, wrench, protocol, only, path)
    result.update(
        side=side,
        physics_dt_s=h.dt,
        fixed_base=fixed,
        gravity=gravity,
        protocol=protocol,
        requested_wrench_world_at_mount=wrench.tolist(),
        only_active_joint=metadata["only_active_joint"],
    )
    write_json(path / "summary.json", result)
    from .plots import hand_plot

    hand_plot(
        rows,
        path,
        f"{side}: {protocol}; {result['outcome']}; no manufacturer payload claim",
    )
    manifest(
        path,
        "free_close" if fixed else "load_response",
        {
            "side": side,
            "physics_dt_s": h.dt,
            "control_dt_s": h.control_dt,
            "source_revision": metadata["revision"],
        },
    )
    return result


def hand_metadata(source, h, cfg, protocol, only, wrench):
    return {
        "schema": "o6_hand_diagnostic_v2",
        **code_state(),
        "side": h.side,
        "config": asdict(cfg),
        "source_run": str(source.path),
        "source_code_revision": source.metadata.get("code_revision"),
        "source_schema": source.metadata["schema"],
        "source_grasp_pose": source.metadata["grasp_pose"],
        "source_support_targets": source.targets,
        "controller": h.controller,
        "assets": source.metadata["assets"],
        "physics_dt_s": h.dt,
        "control_dt_s": h.control_dt,
        "gravity_world_m_s2": h.model.opt.gravity.tolist(),
        "fixed_base": h.fixed,
        "command_joint_mapping": h.hand["active"],
        "generalized_position_layout": qpos_layout(h.model),
        "mapping": h.hand,
        "protocol": protocol,
        "only_active_joint": None if only is None else h.hand["active"][only],
        "requested_wrench_world_at_mount": wrench.tolist(),
        "differences_from_coupled": [
            "G1 and wrist housing removed",
            "no other ladder rungs/rails",
            "no G1 body compensation; closing schedule is the diagnostic protocol",
            "fixed base, no rung"
            if h.fixed
            else "free wrist; explicit standalone acquisition fixture only",
        ],
        "fixture_parameters": {
            k: v for k, v in vars(h.fixture_config).items() if k.startswith("fixture_")
        },
    }


class HandTrial:
    def __init__(self, h, cfg, wrench, protocol, only):
        self.h, self.cfg, self.wrench, self.only = h, cfg, wrench, only
        self.schedule = (
            free_schedule if h.fixed else lambda t, c: load_schedule(t, c, protocol)
        )
        self.end = (
            sum(cfg.free_schedule_s)
            if h.fixed
            else sum(
                (
                    cfg.acquire_open_s,
                    cfg.acquire_close_s,
                    cfg.fixture_remove_s,
                    cfg.zero_fixture_settle_s,
                    cfg.load_ramp_s,
                    cfg.plateau_s,
                )
            )
        )
        self.saturation = Saturation(
            [h.model.actuator(i).name for i in range(6)], h.dt, cfg.max_effort_fraction
        )
        self.events = Events(h.dt, cfg.persistence_s)
        self.scratch = mujoco.MjData(h.model)
        self.rows = []
        self.failure = {}
        self.acquisition = set()
        self.physical_bad = 0.0
        self.clock = 0.0
        self.reason = None
        self.phase = "OPEN"
        self.reference = (h.base, h.data.body(h.base).xpos.copy())
        self.warnings = [w.number for w in h.data.warning]

    def append(self, row, log):
        row["phase"] = self.phase
        self.rows.append(row)
        log.write(json.dumps(row) + "\n")

    def physics_step(self, physics):
        h, cfg = self.h, self.cfg
        self.phase, _, gain, load = self.schedule(self.clock, cfg)
        if not h.fixed and gain == 0:
            h.telemetry.start_hold(h.data)
        h.apply(gain, load * self.wrench)
        mujoco.mj_step(h.model, h.data)
        self.clock += h.dt
        self.reason = physics_status(h, cfg, self.reference, self.warnings)
        if self.reason:
            return False
        mujoco.mj_copyData(self.scratch, h.model, h.data)
        mujoco.mj_forward(h.model, self.scratch)
        values, thresholds = hand_signals(h, self.scratch, cfg)
        self.events.update(self.clock, self.phase, values, thresholds)
        fractions = {
            h.model.actuator(i).name: abs(h.data.actuator_force[i])
            / max(abs(h.model.actuator_forcerange[i]))
            for i in range(6)
        }
        self.saturation.update(
            self.clock, fractions, self.phase in ("CLOSED", *SCORED_PHASES)
        )
        physics.write(
            json.dumps(
                {
                    "time_s": self.clock,
                    "phase": self.phase,
                    "fixture_gain": gain,
                    "effort_Nm": h.data.actuator_force.tolist(),
                    "generalized_position_qpos": h.data.qpos.tolist(),
                    "signals": values,
                    "terminal_guard": self.reason,
                }
            )
            + "\n"
        )
        return True

    def latch_gates(self, row):
        reasons = gate_values(row, self.cfg, self.h.fixed)
        row["failure_reasons"] = reasons
        if self.phase == "REMOVE_FIXTURE":
            self.acquisition.update(reasons)
        scored = self.h.fixed or self.phase in SCORED_PHASES
        if scored:
            for gate in reasons:
                self.failure.setdefault(gate, self.clock)
        physical = set(reasons) - {"effort"}
        self.physical_bad = (
            self.physical_bad + self.h.control_dt if scored and physical else 0.0
        )

    def control_step(self, log, physics):
        h = self.h
        _, amount, _, _ = self.schedule(self.clock, self.cfg)
        h.command(amount, self.only)
        for _ in range(round(h.control_dt / h.dt)):
            if not self.physics_step(physics):
                return False
        mujoco.mj_forward(h.model, h.data)
        row = h.sample()
        self.latch_gates(row)
        self.append(row, log)
        if not h.fixed and self.physical_bad >= self.cfg.persistence_s - 1e-9:
            self.reason = "persistent_physical_failure"
            return False
        return True

    def terminal(self):
        return {
            "time_s": self.clock,
            "simulator_time_s": float(self.h.data.time),
            "reason": self.reason or "schedule_complete",
            "warnings": [int(w.number) for w in self.h.data.warning],
        }


def integrate_hand(h, cfg, wrench, protocol, only, path):
    trial = HandTrial(h, cfg, wrench, protocol, only)
    with (
        (path / "steps.jsonl").open("w") as log,
        gzip.open(path / "physics.jsonl.gz", "wt") as physics,
    ):
        trial.append(h.sample(), log)
        while trial.clock < trial.end - 1e-9:
            if not trial.control_step(log, physics):
                break
        terminal = trial.terminal()
        log.write(json.dumps({"event": "termination", **terminal}) + "\n")
    result = hand_summary(
        trial.rows, trial.failure, trial.acquisition, trial.saturation, terminal, h, cfg
    )
    result.update(
        physics_extrema=trial.events.extrema,
        first_threshold_violation=trial.events.first,
        first_persistent_violation_per_gate=trial.events.persistent,
        event_resolution_s=h.dt,
    )
    write_csv(path / "failure_events.csv", trial.events.rows)
    return result, trial.rows


def physics_status(h, cfg, reference, warnings):
    # Solver-contact guard is sampled before the scratch forward; at most one dt old.
    reason = state_guard(h.data, cfg, warnings)
    if reason or h.fixed:
        return reason
    return position_guard(h.data.qpos[:3], reference[1], cfg)


def qpos_layout(model):
    rows = []
    for j in range(model.njnt):
        free = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        rows.append(
            {
                "joint": model.joint(j).name,
                "start": int(model.jnt_qposadr[j]),
                "components": ["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"]
                if free
                else ["angle_rad"],
            }
        )
    return rows
