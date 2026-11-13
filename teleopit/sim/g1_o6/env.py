"""Bounded deterministic controller with explicit acquisition assistance and strict hold gates."""

from collections import deque

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.o6_grasp.controller import smooth

from .config import Config
from .diagnostics import failure_reasons, snapshot
from .grasp_pose import GraspPose, place_hands
from .holding import COMMAND_RATE_RAD_S, HoldController
from .initialize import initial_stance
from .model import build_model


class Environment:
    def __init__(self, config=None, *, pose=None, ladder=None, material=None, adapt_reset=True):
        self.config = config or Config()
        self.pose = pose if pose is not None else GraspPose()
        self.pose.validate()
        self.ladder, self.material = ladder, material
        self.adapt_reset = ladder is not None and adapt_reset
        self.model, self.xml, self.hands, self.audit = build_model(self.config, ladder=ladder, material=material)
        for name in self.pose.support_targets().values():
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
                raise ValueError("support rung absent from compiled ladder: " + name)
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
        initial_stance(self.model, self.data, self.hands, self.config,
                       foot_rungs=self.pose.foot_rungs, adapt=self.adapt_reset)
        self.ik = place_hands(self, self.pose)
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
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.hold_controller = HoldController(
            self,
            closed_pose=self.pose.closed_pose,
            preload_Nm=self.pose.finger_preload_Nm,
        )
        self.initialize_preload()
        row = snapshot(self)
        self.initial_errors = initial_errors(row, self.ik)
        return row

    def initialize_preload(self):
        # A reset may initialize a static servo preload. All subsequent target
        # updates still obey the existing 1 rad/s limiter and actuator bounds.
        limits = self.model.jnt_range[self.model.actuator_trnid[:, 0]]
        self.target = np.clip(
            self.hold_controller.desired(self, 0.0), limits[:, 0], limits[:, 1]
        )
        self.queue = deque([self.target.copy() for _ in range(self.config.delay_steps)])
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)

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
        return list(self.support_targets)

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
                self.transition("HOLD_COMPLETE")

    def commands(self):
        cfg = self.config
        elapsed = self.data.time - self.phase_start
        amount = (
            smooth(elapsed / cfg.acquire_seconds) if self.phase == "ACQUIRE" else 1.0
        )
        if self.phase == "INITIALIZE":
            amount = 0.0
        desired = self.hold_controller.desired(self, amount)
        limits = self.model.jnt_range[self.model.actuator_trnid[:, 0]]
        desired = np.clip(desired, limits[:, 0], limits[:, 1])
        self.desired_target = desired.copy()
        delta = COMMAND_RATE_RAD_S * cfg.control_dt
        self.target += np.clip(desired - self.target, -delta, delta)
        self.queue.append(self.target.copy())
        self.data.ctrl[:] = self.queue.popleft()

    def apply_forces(self):
        cfg = self.config
        d = self.data
        gain = 1.0 if self.phase in ("INITIALIZE", "ACQUIRE") else 0.0
        if self.phase == "REMOVE_ASSISTANCE":
            gain = 1 - smooth((d.time - self.phase_start) / cfg.remove_seconds)
        rotation = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]])
        force = 3000 * (self.reference_position - d.qpos[:3]) - 150 * d.qvel[:3]
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
        if self.phase in ("ABORT", "HOLD_COMPLETE"):
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
