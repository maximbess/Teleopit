"""Bounded deterministic controller with explicit assistance and guarded release."""

from collections import deque

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.o6_grasp.controller import smooth
from teleopit.sim.o6_grasp.presets import ladder_rung_config

from .config import Config
from .diagnostics import failure_reasons, snapshot
from .initialize import initial_state
from .model import build_model


class Environment:
    def __init__(self, config=None):
        self.config = config or Config()
        self.model, self.xml, self.hands, self.audit = build_model(self.config)
        self.data = mujoco.MjData(self.model)
        self.mass = float(self.model.body_subtreemass[self.model.body("pelvis").id])
        self.act_qpos = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.hand_act = {
            s: np.array([self.model.actuator(n + "_servo").id for n in h["active"]])
            for s, h in self.hands.items()
        }
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.ik = initial_state(self.model, self.data, self.hands, self.config)
        self.initial_qpos = self.data.qpos.copy()
        self.body_target = self.data.qpos[self.act_qpos].copy()
        self.target = self.body_target.copy()
        self.reference_position = self.data.qpos[:3].copy()
        self.reference_rotation = Rotation.from_quat(self.data.qpos[3:7][[1, 2, 3, 0]])
        self.queue = deque([self.target.copy() for _ in range(self.config.delay_steps)])
        self.phase = "INITIALIZE"
        self.phase_start = 0.0
        self.stable = 0.0
        self.bad = 0.0
        self.assistance = np.zeros(6)
        self.assistance_gain = 1.0
        self.disturbance = np.zeros(3)
        self.abort_reasons = []
        self.events = []
        self.hold_passed = False
        self.unloaded_time = 0.0
        self.confirm_time = 0.0
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)
        row = snapshot(self)
        self.initial_errors = initial_errors(row, self.ik)
        return row

    def transition(self, phase):
        self.events.append(
            {"time": float(self.data.time), "previous": self.phase, "next": phase}
        )
        self.phase = phase
        self.phase_start = float(self.data.time)
        self.stable = 0.0
        self.bad = 0.0

    def abort(self, reasons):
        self.abort_reasons = list(reasons)
        self.transition("ABORT")

    def required_supports(self):
        names = [s + "_" + p for s in self.hands for p in ("hand", "foot")]
        if self.phase in (
            "UNLOAD_SELECTED_HAND",
            "OPEN",
            "RETRACT",
            "HOLD_REMAINING_SUPPORTS",
            "RETURN",
            "REACQUIRE",
        ):
            names.remove(self.config.release_side + "_hand")
        return names

    def advance(self, row):
        cfg = self.config
        elapsed = self.data.time - self.phase_start
        reasons = failure_reasons(row, cfg, self.required_supports())
        self.stable = self.stable + cfg.control_dt if not reasons else 0.0
        self.bad = self.bad + cfg.control_dt if reasons else 0.0
        row["failure_reasons"] = reasons
        if self.phase in ("INITIALIZE", "ACQUIRE", "REMOVE_ASSISTANCE"):
            self.advance_acquisition(elapsed)
        elif self.phase == "HOLD":
            self.advance_hold(reasons, elapsed)
        elif self.phase not in ("ACQUIRE", "REMOVE_ASSISTANCE", "ABORT", "DONE"):
            self.advance_release(row, elapsed, reasons)

    def advance_acquisition(self, elapsed):
        duration, following = {
            "INITIALIZE": (0, "ACQUIRE"),
            "ACQUIRE": (self.config.acquire_seconds, "REMOVE_ASSISTANCE"),
            "REMOVE_ASSISTANCE": (self.config.remove_seconds, "HOLD"),
        }[self.phase]
        if elapsed >= duration:
            self.transition(following)

    def advance_hold(self, reasons, elapsed):
        if self.bad >= self.config.stability_seconds:
            self.abort(reasons)
        elif elapsed >= self.config.hold_seconds:
            if self.stable < self.config.hold_seconds - self.config.control_dt:
                self.abort(["full_interval_not_stable"])
            else:
                self.hold_passed = True
                self.transition("UNLOAD_SELECTED_HAND")

    def advance_release(self, row, elapsed, reasons):
        if self.bad >= self.config.stability_seconds:
            self.abort(reasons)
            return
        if self.phase == "UNLOAD_SELECTED_HAND":
            load = row["wrist_world_on_hand"][self.config.release_side]
            unloaded = (
                np.linalg.norm(load[:3]) < self.config.unload_force_N
                and np.linalg.norm(load[3:]) < self.config.unload_moment_Nm
            )
            self.unloaded_time = (
                getattr(self, "unloaded_time", 0.0) + self.config.control_dt
                if unloaded and not reasons
                else 0.0
            )
            if self.unloaded_time >= self.config.stability_seconds:
                self.transition("OPEN")
            elif elapsed > 5:
                self.abort(["unload_timeout"])
        else:
            self.advance_motion(row, elapsed)

    def advance_motion(self, row, elapsed):
        schedule = {
            "OPEN": (2.0, "RETRACT"),
            "RETRACT": (3.0, "HOLD_REMAINING_SUPPORTS"),
            "HOLD_REMAINING_SUPPORTS": (3.0, "RETURN"),
            "RETURN": (3.0, "REACQUIRE"),
        }
        if self.phase in schedule and elapsed >= schedule[self.phase][0]:
            if self.phase == "RETRACT" and not self.clearance():
                self.abort(["retraction_clearance"])
                return
            self.transition(schedule[self.phase][1])
        elif self.phase == "REACQUIRE" and elapsed >= 3:
            self.confirm_support(row, elapsed)

    def confirm_support(self, row, elapsed):
        item = row["supports"][self.config.release_side + "_hand"]
        stable = (
            item["loaded"]
            and item["opposing"]
            and item["max_slip"] < self.config.max_slip_m_s
        )
        self.confirm_time = (
            self.confirm_time + self.config.control_dt if stable else 0.0
        )
        if self.confirm_time >= self.config.stability_seconds:
            self.transition("DONE")
        elif elapsed > 6:
            self.abort(["reacquire_timeout"])

    def clearance(self):
        rung = self.model.geom("left_ladder_rung_08").id
        for g in range(self.model.ngeom):
            name = self.model.body(self.model.geom_bodyid[g]).name
            if (
                name.startswith(self.config.release_side + "_o6_")
                and self.model.geom_group[g] == 3
                and mujoco.mj_geomDistance(self.model, self.data, g, rung, 0.01, None)
                < 0.01
            ):
                return False
        return True

    def commands(self):
        cfg = self.config
        elapsed = self.data.time - self.phase_start
        desired = self.body_target.copy()
        amount = (
            smooth(elapsed / cfg.acquire_seconds) if self.phase == "ACQUIRE" else 1.0
        )
        if self.phase == "INITIALIZE":
            amount = 0.0
        for side, indices in self.hand_act.items():
            desired[indices] = amount * np.array(ladder_rung_config().closed_pose)
        side = cfg.release_side
        if self.phase in ("OPEN", "RETRACT", "HOLD_REMAINING_SUPPORTS", "RETURN"):
            desired[self.hand_act[side]] *= (
                1 - smooth(elapsed / 2) if self.phase == "OPEN" else 0
            )
        if self.phase == "REACQUIRE":
            desired[self.hand_act[side]] *= smooth(elapsed / 2)
        self.transfer_targets(desired, elapsed)
        limits = self.model.jnt_range[self.model.actuator_trnid[:, 0]]
        desired = np.clip(desired, limits[:, 0], limits[:, 1])
        self.desired_target = desired.copy()
        self.target += np.clip(desired - self.target, -cfg.control_dt, cfg.control_dt)
        self.queue.append(self.target.copy())
        self.data.ctrl[:] = self.queue.popleft()

    def transfer_targets(self, desired, elapsed):
        if self.phase in (
            "UNLOAD_SELECTED_HAND",
            "OPEN",
            "RETRACT",
            "HOLD_REMAINING_SUPPORTS",
            "RETURN",
            "REACQUIRE",
        ):
            scale = smooth(elapsed / 3) if self.phase == "UNLOAD_SELECTED_HAND" else 1.0
            # Small joint-space weight-shift candidate, never a support-state assignment.
            desired[self.model.actuator("waist_roll_joint").id] += (
                0.04 if self.config.release_side == "left" else -0.04
            ) * scale
        if self.phase in ("RETRACT", "HOLD_REMAINING_SUPPORTS", "RETURN"):
            scale = {
                "RETRACT": smooth(elapsed / 3),
                "RETURN": 1 - smooth(elapsed / 3),
            }.get(self.phase, 1.0)
            desired[
                self.model.actuator(
                    self.config.release_side + "_shoulder_pitch_joint"
                ).id
            ] += 0.15 * scale

    def apply_forces(self):
        cfg = self.config
        d = self.data
        gain = 1.0 if self.phase in ("INITIALIZE", "ACQUIRE") else 0.0
        if self.phase == "REMOVE_ASSISTANCE":
            gain = 1 - smooth((d.time - self.phase_start) / cfg.remove_seconds)
        rotation = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]])
        force = (
            3000 * (self.reference_position - d.qpos[:3])
            - 150 * d.qvel[:3]
            + [0, 0, self.mass * 9.81]
        )
        moment = 200 * (
            self.reference_rotation * rotation.inv()
        ).as_rotvec() - 20 * rotation.apply(d.qvel[3:6])
        self.assistance = (
            gain
            * np.r_[
                bounded(force, cfg.assistance_force_N),
                bounded(moment, cfg.assistance_moment_Nm),
            ]
        )
        self.assistance_gain = float(gain)
        self.disturbance = np.array(
            [
                cfg.disturbance_N
                if self.phase == "HOLD" and d.time - self.phase_start < 0.2
                else 0,
                0,
                0,
            ]
        )
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_applyFT(
            self.model,
            d,
            self.assistance[:3] + self.disturbance,
            self.assistance[3:],
            d.qpos[:3],
            self.model.body("pelvis").id,
            d.qfrc_applied,
        )

    def step(self, physics_observer=None, *, advance_phase=True):
        if self.phase in ("ABORT", "DONE"):
            raise RuntimeError("trial finished")
        self.commands()
        for _ in range(round(self.config.control_dt / self.config.physics_dt)):
            self.apply_forces()
            mujoco.mj_step(self.model, self.data)
            if physics_observer is not None:
                physics_observer(self)
        mujoco.mj_forward(self.model, self.data)
        row = snapshot(self)
        if self.initial_errors:
            self.abort(self.initial_errors)
        elif advance_phase:
            self.advance(row)
        row["next_phase"] = self.phase
        return row


def bounded(vector, limit):
    return vector * min(1.0, limit / max(np.linalg.norm(vector), 1e-12))


def initial_errors(row, ik):
    errors = []
    if row["penetration_m"] > 1e-6:
        errors.append("initial_penetration")
    if any(
        v["position_error_m"] > 0.005 or v["angle_error_rad"] > 0.1 for v in ik.values()
    ):
        errors.append("unreachable_grasp")
    return errors
