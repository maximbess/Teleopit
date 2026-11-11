"""Slow measured-load feedback through bounded position servos only."""

import mujoco
import numpy as np

from teleopit.sim.g1_o6.holding import opening_targets
from teleopit.sim.o6_grasp.controller import smooth

from .config import TransferCriteria
from .guards import vertical_reactions
from .kinematics import Clearance, arm_target

CRITERIA = TransferCriteria()
OPEN_PHASES = {"OPEN", "RETRACT", "THREE_SUPPORT_HOLD", "APPROACH"}
UNLOADED_PHASES = {
    "UNLOAD_RAMP",
    "UNLOAD_SETTLE",
    "UNLOADED_HOLD",
    *OPEN_PHASES,
    "CLOSE",
    "VERIFY_GRASP",
}


def loading_fraction(phase, fraction):
    return {"LOAD_RAMP": fraction, "RETURN": 1 - fraction, "RECOVERY": 0}.get(phase, 1)


def redistribute(targets, selected, fraction):
    lost = targets[selected] * 0.9 * fraction
    return {n: f - lost if n == selected else f + lost / 3 for n, f in targets.items()}


def bounded_feedback(offset, jacobian, error, kp, effort, dt, cfg=CRITERIA):
    # Compliance solve regularizes predicted support displacement. There is no
    # force assignment counted as evidence: the error is measured external Fz.
    system = np.diag(kp) + 10 * jacobian.T @ jacobian
    correction = np.linalg.solve(system, -jacobian.T @ error)
    delta = np.clip(
        0.2 * dt * correction, -cfg.offset_rate_rad_s * dt, cfg.offset_rate_rad_s * dt
    )
    delta[np.asarray(effort) >= 0.9] = 0  # freeze integrator near effort limits
    return np.clip(offset + delta, -cfg.offset_bound_rad, cfg.offset_bound_rad)


def support_jacobian(env, indices):
    model, data = env.model, env.data
    dofs = model.jnt_dofadr[model.actuator_trnid[indices, 0]]
    rows = []
    for name in env.support_targets:
        side, part = name.split("_")
        if part == "hand":
            site = model.site(env.hands[side]["mount"])
            body, point = site.bodyid[0], data.site(site.id).xpos
        else:
            body = model.body(side + "_ankle_roll_link").id
            point = data.xpos[body]
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, point, body)
        rows.append(jac[2, dofs])
    return np.array(rows)


class TransferController:
    def __init__(self, env, original, protocol, target):
        self.original, self.protocol, self.target_share = original, protocol, target
        self.indices = np.array(
            [
                i
                for i in range(env.model.nu)
                if all(i not in v for v in env.hand_act.values())
            ]
        )
        self.offset = np.zeros(len(self.indices))
        self.baseline = None
        self.reference = None
        self.latest = None
        self.previous = None
        self.open_start = {}
        self.mounts = {}
        self.clearance = {side: Clearance(env, side) for side in env.hands}
        self.target_forces = {}
        self.motion = {}
        self.failure = None

    def enter(self, env, row):
        key = self.protocol.index
        if key == self.previous or self.protocol.phase in ("COMPLETE", "ABORT"):
            return
        self.previous = key
        phase, side = self.protocol.phase, self.protocol.step.side
        if phase == "OPEN":
            self.open_start[side] = env.data.qpos[
                env.act_qpos[env.hand_act[side]]
            ].copy()
            site = env.data.site(env.hands[side]["mount"])
            self.mounts[side] = (site.xpos.copy(), site.xmat.reshape(3, 3).copy())
        if phase in ("RETRACT", "APPROACH"):
            self.arm_previous = env.target.copy()
        self.latest = row

    def desired(self, env, amount):
        desired = self.original.desired(env, amount)
        if env.phase != "HOLD" or self.latest is None or self.baseline is None:
            return desired
        phase, side = self.protocol.phase, self.protocol.step.side
        self.update_load(env)
        desired[self.indices] += self.offset
        if phase in OPEN_PHASES:
            progress = (
                smooth(self.protocol.elapsed(env.data.time) / CRITERIA.opening_s)
                if phase == "OPEN"
                else 1.0
            )
            desired[env.hand_act[side]] = opening_targets(
                self.open_start[side], progress
            )
        if phase in ("RETRACT", "THREE_SUPPORT_HOLD", "APPROACH"):
            self.move_arm(env, desired, side, phase)
        elif side in self.mounts and phase in (
            "CLOSE",
            "VERIFY_GRASP",
            "RELOAD",
            "RELOAD_SETTLE",
            "POST_HOLD",
        ):
            self.move_arm(env, desired, side, "AT_RUNG")
        return desired

    def update_load(self, env):
        phase = self.protocol.phase
        if phase == "BASELINE":
            return
        self.target_forces = self.force_targets(env)
        actual = vertical_reactions(self.latest)
        names = list(env.support_targets)
        error = np.array([self.target_forces[n] - actual[n] for n in names])
        jacobian = support_jacobian(env, self.indices)
        if phase in OPEN_PHASES | {"CLOSE", "VERIFY_GRASP"}:
            # No force objective on a detached hand; distribute through three supports.
            selected = names.index(self.protocol.step.side + "_hand")
            jacobian[selected] = 0
            error[selected] = 0
        effort = np.asarray(self.latest["effort_fraction"])[self.indices].copy()
        limits = env.model.jnt_range[env.model.actuator_trnid[self.indices, 0]]
        commands = env.target[self.indices]
        at_limit = np.minimum(commands - limits[:, 0], limits[:, 1] - commands) < 0.001
        effort[at_limit] = 1  # Disable integration where physical clipping is active.
        self.offset = bounded_feedback(
            self.offset,
            jacobian,
            error,
            self.original.kp[self.indices],
            effort,
            env.config.control_dt,
        )

    def force_targets(self, env):
        phase, side = self.protocol.phase, self.protocol.step.side
        base = vertical_reactions(self.baseline)
        weight = env.mass * abs(env.model.opt.gravity[2])
        increase = self.target_share * weight - base["left_hand"] - base["right_hand"]
        fraction = smooth(self.protocol.elapsed(env.data.time) / CRITERIA.ramp_s)
        increase *= loading_fraction(phase, fraction)
        targets = {
            n: f + (increase / 2 if n.endswith("hand") else -increase / 2)
            for n, f in base.items()
        }
        if phase in UNLOADED_PHASES | {"RELOAD"}:
            unload = {"UNLOAD_RAMP": fraction, "RELOAD": 1 - fraction}.get(phase, 1)
            targets = redistribute(targets, side + "_hand", unload)
        return targets

    def move_arm(self, env, desired, side, phase):
        position, rotation = self.mounts[side]
        progress = smooth(self.protocol.elapsed(env.data.time) / CRITERIA.trajectory_s)
        distance = CRITERIA.retraction_m * (
            1 - progress
            if phase == "APPROACH"
            else progress
            if phase == "RETRACT"
            else 0
            if phase == "AT_RUNG"
            else 1
        )
        goal = position + np.array([-distance, 0, 0])
        result = arm_target(env, side, goal, rotation)
        clearance = self.clearance[side].measure(result["scratch"])
        self.motion = {
            "goal_world_m": goal.tolist(),
            "position_error_m": result["position_error_m"],
            "angle_error_rad": result["angle_error_rad"],
            "planned_clearance": clearance,
        }
        if (
            result["position_error_m"] > 0.001
            or result["angle_error_rad"] > 0.03
            or clearance["clearance_m"] < -env.config.max_penetration_m
        ):
            self.failure = "arm_trajectory_infeasible_or_colliding"
            return
        joints = result["joints"]
        ids = np.array(
            [np.flatnonzero(env.model.actuator_trnid[:, 0] == j)[0] for j in joints]
        )
        step = 0.05 * env.config.control_dt
        desired[ids] = self.arm_previous[ids] + np.clip(
            result["qpos"] - self.arm_previous[ids], -step, step
        )
        self.arm_previous[ids] = desired[ids]
