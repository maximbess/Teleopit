"""Opt-in, prerequisite-gated coupled transfer through existing actuators."""

import copy
import json
from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.diagnostics import failure_reasons, snapshot
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
from .long_hold import summary_row
from .protocol import Protocol, transfer_steps
from .qualification import Evidence, demands_covered, identity, targeted_demands_covered
from .scene_contract import restore_environment


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
    def __init__(self, env, request, path=None, proof=None):
        from .g6_observer import G6Observer
        from .g6_transfer import AdjacentController
        from .g6_transfer_results import capacity_share_bounds

        self.env, self.request = env, request
        self.proof = proof
        self.capacity_shares = capacity_share_bounds(
            proof, env.mass * abs(env.model.opt.gravity[2])
        )
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
        factory = (
            AdjacentController
            if request["mode"].startswith("adjacent")
            else TransferController
        )
        self.controller = factory(
            env, env.hold_controller, self.protocol, request["target_share"]
        )
        env.hold_controller = self.controller
        self.audit = (
            G6Observer(env, path)
            if env.ladder is not None and path is not None
            else PhysicsAudit(env)
        )
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
        self.audit.selected_side = self.protocol.step.side
        self.audit.phase_token = str(self.protocol.index) + ":" + self.audit.phase
        self.audit.experiment_phase_start = (
            self.protocol.started
            if env.phase == "HOLD" and self.protocol.started is not None
            else env.phase_start
        )
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
        if row["phase"] != "HOLD":
            self.rows.append(summary_row(row))
            return
        self.controller.latest = row
        if self.hold_start is None:
            self.hold_start = row["time"]
        phase = self.protocol.phase
        row["experiment_phase"] = phase
        row["experiment_phase_time_s"] = self.protocol.elapsed(row["time"])
        row["support_targets"] = dict(self.env.support_targets)
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
            "capacity_per_hand_share_bounds": self.capacity_shares,
            "feedback_paused": getattr(self.controller, "feedback_paused", False),
        }
        self.rows.append(summary_row(row))
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
        if previous == "VERIFY_GRASP" and phase == "RELOAD":
            side = self.protocol.step.side
            self.positions[side + "_hand"] = support_positions(self.env)[side + "_hand"]
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
        ) / (self.env.mass * abs(self.env.model.opt.gravity[2]))
        self.initial_hand_share = baseline

    def strict_reasons(self, row):
        from .g6_gates import required_contact_sides

        reasons = failure_reasons(row, self.env.config, self.required_supports())
        if self.env.ladder is not None:
            quality = row.get("pose_search", {}).get("hands", {})
            for side in required_contact_sides(
                self.protocol.phase, self.protocol.step.side
            ):
                if not quality.get(side, {}).get("quality", {}).get("valid", False):
                    reasons.append("anatomical_grasp:" + side)
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
        if self.positions is not None:
            reasons.extend(self.stationary_motion_reasons())
        return reasons

    def stationary_motion_reasons(self):
        moved = support_positions(self.env)
        other = "right" if self.protocol.step.side == "left" else "left"
        names = {
            "foot_support_moved": ("left_foot", "right_foot"),
            "stationary_hand_moved": (other + "_hand",),
        }
        return [
            reason
            for reason, supports in names.items()
            if any(
                np.linalg.norm(moved[n] - self.positions[n]) > self.cfg.support_motion_m
                for n in supports
            )
        ]

    def features(self, row):
        side = self.protocol.step.side
        support = row["supports"][side + "_hand"]
        weight = self.env.mass * abs(self.env.model.opt.gravity[2])
        features = {
            "share": share_settled(
                row,
                self.request["target_share"],
                weight,
                capacity_shares=self.capacity_shares,
            ),
            "grasp": support["loaded"]
            and support["opposing"]
            and row["posture"]["thumb_below_fingers_above"][side],
        }
        if self.env.ladder is not None:
            features["grasp"] &= (
                row.get("pose_search", {})
                .get("hands", {})
                .get(side, {})
                .get("quality", {})
                .get("valid", False)
            )
        if self.protocol.phase == "DESTINATION_CAPACITY":
            features["destination_capacity"] = self.destination_capacity()
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

    def destination_capacity(self):
        return self.proof is not None and targeted_demands_covered(
            self.rows,
            self.proof.target_wrenches,
            self.proof.source["metadata"]["support_targets"],
            capacity_runs=self.proof.runs["capacity_runs"],
        )

    def open_features(self, row, side):
        controller, env = self.controller, self.env
        actual = env.data.qpos[env.act_qpos[env.hand_act[side]]]
        start = controller.open_start[side]
        clearance = controller.clearance[side].measure(env.data)
        row["release_clearance"] = clearance
        site = env.data.site(env.hands[side]["mount"])
        position, rotation = controller.mounts[side]
        if self.protocol.phase == "REACH":
            position = np.asarray(controller.plans[side]["waypoints_world_m"][-1])
            rotation = np.asarray(controller.plans[side]["rotation_world"])
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
        "mode": (
            "load",
            "unload",
            "release",
            "reacquire",
            "adjacent-first",
            "adjacent",
        ),
        "side": ("left", "right"),
        "target_share": (0.05, 0.1, 0.2, 0.3),
        "cycles": (1, 3),
        "three_seconds": (1, 3, 10),
        "plateau_seconds": (10, 30, 60),
        "physics_dt": (0.001, 0.0005),
    }
    for key, values in allowed.items():
        if request[key] not in values:
            raise ValueError(f"unsupported {key}")
    if request["mode"] != "reacquire" and (
        request["cycles"] != 1 or request["alternating"]
    ):
        raise ValueError("cycles and alternating sides apply only to reacquisition")
    if request["mode"].startswith("adjacent") and request["target_share"] != 0.3:
        raise ValueError(
            "adjacent transfer requires the qualified 30 percent allocation"
        )


def execute_transfer(proof, path, request):
    source = proof.source["metadata"]
    env = restore_environment(source, request["physics_dt"])
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
    env.save_scene(path / "scene.xml")
    trial = TransferTrial(env, request, path, proof)
    try:
        integrate_transfer(trial, initial, path)
    finally:
        close = getattr(trial.audit, "close", None)
        if close:
            close()
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
            enrich(env, row, telemetry, trial.hold_start, trial.audit)
            trial.update(row)
            stream.write(json.dumps(row) + "\n")


def transfer_result(trial, proof, path):
    from .g6_transfer_results import (
        adjacent_progress,
        load_measurements,
        transfer_motion_gate,
    )

    env, request = trial.env, trial.request
    audit = trial.audit.report()
    contact = (
        trial.audit.contact_report()
        if hasattr(trial.audit, "contact_report")
        else {"status": "NOT RUN"}
    )
    motion = transfer_motion_gate(request, trial.rows)
    physical = physical_transfer_passed(env, audit, contact, motion)
    covered = demands_covered(
        {"path": str(path)},
        proof.passed_wrenches,
        proof.target_wrenches if proof.is_g6 else None,
        proof.runs["capacity_runs"] if proof.is_g6 else None,
    )
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
        "contact_quality": contact,
        "load_measurements": load_measurements(
            trial.rows, env.mass * abs(env.model.opt.gravity[2]), env.config.control_dt
        ),
        "initial_hand_share": getattr(trial, "initial_hand_share", None),
        "plateau_convergence": motion["plateau_convergence"],
        "destination_convergence": motion["destination_convergence"],
        "final_support_targets": dict(env.support_targets),
        "reach_plans": getattr(trial.controller, "plans", {}),
        "adjacent_progress": adjacent_progress(
            request, trial.protocol.events, physical, covered
        ),
        "maximum_reset_drift_m": max(r["base_drift_m"] for r in trial.rows),
        "final_reset_drift_m": trial.rows[-1]["base_drift_m"],
    }
    return result


def physical_transfer_passed(env, audit, contact, motion):
    return (
        env.phase == "HOLD_COMPLETE"
        and audit["no_external_support"]
        and not audit["unintended_support_steps"]
        and not audit["housing_contact_steps"]
        and (env.ladder is None or contact["status"] == "PASS")
        and motion["passed"]
        and audit.get("native_hand_friction", {"status": "PASS"})["status"] == "PASS"
    )
