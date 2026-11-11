"""Opt-in, prerequisite-gated coupled transfer through existing actuators."""

import copy
import json
from dataclasses import asdict

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.diagnostics import failure_reasons, snapshot
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.g1_o6.posture_report import housing_contact
from teleopit.sim.o6_diagnostics.io import (
    manifest,
    output_directory,
    write_csv,
    write_json,
)
from teleopit.sim.o6_diagnostics.nominal import coupled_telemetry

from .audit import PhysicsAudit
from .config import TransferCriteria
from .control import UNLOADED_PHASES, TransferController
from .guards import released, share_settled, unloading
from .long_hold import enrich, experiment_metadata
from .protocol import Protocol, transfer_steps
from .qualification import Evidence, demands_covered, identity


def mean_reference(rows, seconds=2):
    selected = [r for r in rows if r["time"] >= rows[-1]["time"] - seconds + 1e-8]
    result = copy.deepcopy(selected[-1])
    for name in result["supports"]:
        result["supports"][name]["force_world"] = np.mean(
            [r["supports"][name]["force_world"] for r in selected], axis=0
        ).tolist()
    result["reference_interval_s"] = [selected[0]["time"], selected[-1]["time"]]
    return result


def support_positions(env):
    return {
        name: (
            env.data.site(env.hands[name.split("_")[0]]["mount"]).xpos.copy()
            if name.endswith("hand")
            else env.data.body(name.split("_")[0] + "_ankle_roll_link").xpos.copy()
        )
        for name in env.support_targets
    }


class TransferTrial:
    def __init__(self, env, request):
        self.env, self.request = env, request
        self.cfg = TransferCriteria()
        self.protocol = Protocol(
            transfer_steps(
                request["mode"],
                request["side"],
                request["cycles"],
                request["alternating"],
                request["three_seconds"],
                request["plateau_seconds"],
            )
        )
        self.controller = TransferController(
            env, env.hold_controller, self.protocol, request["target_share"]
        )
        env.hold_controller = self.controller
        self.audit = PhysicsAudit(env)
        self.rows = []
        self.failures = {}
        self.bad = 0
        self.hold_start = None
        self.positions = None

    def required_supports(self):
        names = list(self.env.support_targets)
        if self.protocol.phase in UNLOADED_PHASES | {"RELOAD", "RELOAD_SETTLE"}:
            names.remove(self.protocol.step.side + "_hand")
        return names

    def observe(self, env):
        self.audit.phase = self.protocol.phase if env.phase == "HOLD" else env.phase
        self.audit.optional_supports = set(env.support_targets) - set(
            self.required_supports()
        )
        self.audit(env)
        if env.phase != "HOLD":
            return
        counts = {
            "external_applied_force": self.audit.applied_steps,
            "housing_contact": self.audit.housing_steps,
            "unintended_support": self.audit.prohibited_steps,
        }
        for name, count in counts.items():
            if count:
                self.failures.setdefault(name, float(env.data.time))
        if self.protocol.phase == "THREE_SUPPORT_HOLD":
            self.check_detachment(env.data.contact, float(env.data.time))

    def check_detachment(self, contacts, time):
        geoms = self.controller.clearance[self.protocol.step.side].hand
        if any((int(c.geom1) in geoms) != (int(c.geom2) in geoms) for c in contacts):
            self.failures.setdefault("released_hand_physics_contact", time)

    def update(self, row):
        self.rows.append(row)
        if row["phase"] != "HOLD":
            return
        self.controller.latest = row
        if self.hold_start is None:
            self.hold_start = row["time"]
        phase = self.protocol.phase
        row["experiment_phase"] = phase
        self.controller.enter(self.env, row)
        reasons = self.strict_reasons(row)
        row["failure_reasons"] = reasons
        self.bad = self.bad + self.env.config.control_dt if reasons else 0
        for reason in reasons:
            self.failures.setdefault(reason, row["time"])
        features = self.features(row)
        features["valid"] = not reasons
        features["tick"] = True
        row["transfer"] = {
            "mode": self.request["mode"],
            "side": self.protocol.step.side,
            "target_total_hand_share": self.request["target_share"],
            "commanded_support_Fz_N": self.controller.target_forces,
            "measured": features,
            "feedback_offset_rad": self.controller.offset.tolist(),
            "motion_tracking": self.controller.motion,
            "required_supports": self.required_supports(),
        }
        if self.bad >= self.env.config.stability_seconds or self.controller.failure:
            self.env.abort(reasons or [self.controller.failure])
            return
        self.protocol.update(row["time"], features)
        self.on_transition(phase, row)

    def on_transition(self, previous, row):
        phase = self.protocol.phase
        if phase == previous or self.finish_phase():
            return
        if phase == "LOAD_RAMP":
            self.capture_baseline()
        if phase == "UNLOAD_RAMP":
            self.controller.reference = mean_reference(self.rows)
        self.controller.enter(self.env, row)

    def finish_phase(self):
        if self.failures:
            self.env.abort(["full_interval_not_stable", *self.failures])
            return True
        if self.protocol.phase == "ABORT":
            self.env.abort([self.protocol.failure])
            return True
        if self.protocol.phase == "COMPLETE":
            self.env.hold_passed = True
            self.env.transition("HOLD_COMPLETE")
            return True
        return False

    def capture_baseline(self):
        self.controller.baseline = mean_reference(self.rows)
        self.positions = support_positions(self.env)
        baseline = sum(
            self.controller.baseline["supports"][s + "_hand"]["force_world"][2]
            for s in ("left", "right")
        ) / (self.env.mass * 9.81)
        if self.request["target_share"] - baseline < 0.02:
            self.env.abort(["target_not_meaningful_above_baseline"])

    def strict_reasons(self, row):
        reasons = failure_reasons(row, self.env.config, self.required_supports())
        if housing_contact(row):
            reasons.append("housing_contact")
        # Optional selected supports still obey slip and collision limits.
        if any(
            v["loaded"] and v["max_slip"] >= self.env.config.max_slip_m_s
            for v in row["supports"].values()
        ):
            reasons.append("contact_slip")
        return reasons + self.geometry_reasons(row)

    def geometry_reasons(self, row):
        reasons = []
        if (
            self.protocol.phase in {"UNLOAD_RAMP", "UNLOAD_SETTLE", "UNLOADED_HOLD"}
            and not row["supports"][self.protocol.step.side + "_hand"]["opposing"]
        ):
            reasons.append("selected_grasp_geometry_lost")
        if self.positions is not None and self.protocol.phase in {
            "LOAD_RAMP",
            "LOAD_SETTLE",
            "LOADED_HOLD",
        }:
            motion = {
                k: float(np.linalg.norm(v - self.positions[k]))
                for k, v in support_positions(self.env).items()
            }
            row["support_location_motion_m"] = motion
            if max(motion.values()) > self.cfg.support_motion_m:
                reasons.append("support_location_moved")
        return reasons

    def features(self, row):
        side = self.protocol.step.side
        support = row["supports"][side + "_hand"]
        weight = self.env.mass * abs(self.env.model.opt.gravity[2])
        features = {
            "share": share_settled(row, self.request["target_share"], weight),
            "grasp": support["loaded"]
            and support["opposing"]
            and row["posture"]["thumb_below_fingers_above"][side],
        }
        reference = self.controller.reference
        if reference is not None:
            features.update(load_acceptance(reference, row, side, features))
        baseline = self.controller.baseline
        if baseline is not None:
            features["baseline_share"] = all(
                abs(row["supports"][n]["force_world"][2] - v["force_world"][2])
                <= weight * self.cfg.share_tolerance / 2
                for n, v in baseline["supports"].items()
            )
        if side in self.controller.open_start:
            features.update(self.open_features(row, side))
        return features

    def open_features(self, row, side):
        controller, env = self.controller, self.env
        actual = env.data.qpos[env.act_qpos[env.hand_act[side]]]
        start = controller.open_start[side]
        clearance = controller.clearance[side].measure(env.data)
        row["release_clearance"] = clearance
        site = env.data.site(env.hands[side]["mount"])
        position, rotation = controller.mounts[side]
        error = np.linalg.norm(site.xpos - position)
        angle = Rotation.from_matrix(rotation @ site.xmat.reshape(3, 3).T).magnitude()
        opened = released(start, actual, 0, self.cfg.clearance_m)
        return {
            "opened": opened,
            "detached": released(
                start,
                actual,
                clearance["external_contact_count"],
                clearance["clearance_m"],
            ),
            "approached": bool(opened and error < 0.001 and angle < 0.03),
        }


def load_acceptance(reference, row, side, features):
    report = unloading(reference, row, side)
    row["unloading_measurement"] = report
    accepted = (
        row["supports"][side + "_hand"]["force_world"][2]
        >= 0.8 * reference["supports"][side + "_hand"]["force_world"][2]
    )
    return {
        "unloaded": report["passed"] and features["grasp"],
        "accepted_load": features["share"] and features["grasp"] and accepted,
    }


def run_transfer(
    evidence,
    output,
    *,
    mode="load",
    side="left",
    target_share=0.05,
    cycles=1,
    alternating=False,
    three_seconds=1,
    plateau_seconds=10,
    physics_dt=0.001,
):
    request = {
        "mode": mode,
        "side": side,
        "target_share": target_share,
        "cycles": cycles,
        "alternating": alternating,
        "three_seconds": three_seconds,
        "plateau_seconds": plateau_seconds,
        "physics_dt": physics_dt,
    }
    validate_request(request)
    path = output_directory(output)
    try:
        proof = Evidence(evidence)
        reasons = proof.reasons(request)
    except (OSError, ValueError, KeyError) as exc:
        reasons = [f"missing_or_invalid_evidence: {exc}"]
    if reasons:
        result = {
            "schema": "g1_o6_transfer_v1",
            "experiment": request,
            "software_status": "PASS",
            "physical_status": "BLOCKED",
            "progression_justified": False,
            "reasons": reasons,
        }
        write_json(path / "summary.json", result)
        manifest(path, "blocked_support_transfer", {})
        return result
    return execute_transfer(proof, path, request)


def validate_request(request):
    allowed = {
        "mode": ("load", "unload", "release", "reacquire"),
        "side": ("left", "right"),
        "target_share": (0.05, 0.1, 0.2),
        "cycles": (1, 3),
        "three_seconds": (1, 3, 10),
        "plateau_seconds": (10, 30),
        "physics_dt": (0.001, 0.0005),
    }
    for key, values in allowed.items():
        if request[key] not in values:
            raise ValueError(f"unsupported {key}")
    if request["mode"] != "reacquire" and (
        request["cycles"] != 1 or request["alternating"]
    ):
        raise ValueError("cycles and alternating sides apply only to reacquisition")


def execute_transfer(proof, path, request):
    source = proof.source["metadata"]
    cfg = Config(**{**source["config"], "physics_dt": request["physics_dt"]})
    env = Environment(cfg, pose=GraspPose(**source["grasp_pose"]))
    initial = snapshot(env)
    metadata = experiment_metadata(env, initial, request)
    metadata.update(
        hold_only=False,
        transfer_criteria=asdict(TransferCriteria()),
        qualification_identity=proof.identity,
    )
    if identity(metadata) != proof.identity:
        raise ValueError(
            "current assets/controller/physics differ from qualified evidence"
        )
    write_json(path / "resolved.json", metadata)
    mujoco.mj_saveLastXML(str(path / "scene.xml"), env.model)
    trial = TransferTrial(env, request)
    integrate_transfer(trial, initial, path)
    result = transfer_result(trial, proof, path)
    write_json(path / "summary.json", result)
    write_csv(path / "physics_events.csv", trial.audit.events.rows)
    manifest(path, "support_transfer", {})
    return result


def integrate_transfer(trial, initial, path):
    env = trial.env
    telemetry = coupled_telemetry(env)
    with (path / "steps.jsonl").open("w") as stream:
        enrich(env, initial, telemetry, None)
        stream.write(json.dumps(initial) + "\n")
        while env.phase not in ("ABORT", "HOLD_COMPLETE"):
            row = env.step(trial.observe, advance_phase=env.phase != "HOLD")
            enrich(env, row, telemetry, trial.hold_start)
            trial.update(row)
            stream.write(json.dumps(row) + "\n")


def transfer_result(trial, proof, path):
    env, request = trial.env, trial.request
    audit = trial.audit.report()
    physical = (
        env.phase == "HOLD_COMPLETE"
        and audit["no_external_support"]
        and not audit["unintended_support_steps"]
        and not audit["housing_contact_steps"]
    )
    covered = demands_covered({"path": str(path)}, proof.passed_wrenches)
    result = {
        "schema": "g1_o6_transfer_v1",
        "software_status": "PASS",
        "physical_status": "PASS" if physical else "FAIL",
        "progression_justified": physical and covered,
        "capacity_status": "PASS" if covered else "BLOCKED",
        "capacity_reason": "new observed simultaneous demands require isolated reserve tests"
        if not covered
        else None,
        "experiment": request,
        "final_phase": env.phase,
        "reasons": env.abort_reasons,
        "first_failures": trial.failures,
        "phase_transitions": trial.protocol.events,
        "physics_audit": audit,
        "maximum_reset_drift_m": max(r["base_drift_m"] for r in trial.rows),
        "final_reset_drift_m": trial.rows[-1]["base_drift_m"],
    }
    return result
