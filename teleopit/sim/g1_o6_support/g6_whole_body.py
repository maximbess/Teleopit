"""HOLD-entry pelvis and torso feedback through native G1 position servos."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.env import bounded

from .g6_acquisition import GraspAcquisition, GraspEnvironment
from .g6_arm import MAX_RATE_RAD_S

BODY_OFFSET_RAD = .12
BODY_RATE_RAD_S = .06
GAINS = (.8, 1.6)
LEG_AXES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST_AXES = ("yaw", "roll", "pitch")


@dataclass(frozen=True)
class WholeBody(GraspAcquisition):
    arm_rate_rad_s: float = MAX_RATE_RAD_S
    balance: str = "stance"
    gain: float = GAINS[0]

    def __post_init__(self):
        super().__post_init__()
        if (self.order, self.arm_rate_rad_s) != ("together", MAX_RATE_RAD_S):
            raise ValueError("whole-body stabilization requires the acquired parent")
        if self.balance not in ("stance", "wrench") or self.gain not in GAINS:
            raise ValueError("unknown whole-body configuration")


def bounded_offset(previous, increment, bounds, frozen, dt):
    step = np.clip(increment, -BODY_RATE_RAD_S * dt, BODY_RATE_RAD_S * dt)
    return np.clip(previous + np.where(frozen, 0, step), bounds[:, 0], bounds[:, 1])


class Chain:
    def __init__(self, model, body, names):
        self.body = model.body(body).id
        self.names = names
        joints = [model.joint(name).id for name in names]
        actuators = {joint: i for i, joint in enumerate(model.actuator_trnid[:, 0])}
        self.indices = np.array([actuators[j] for j in joints])
        self.dofs = model.jnt_dofadr[joints]
        self.kp = model.actuator_gainprm[self.indices, 0]

    def jacobian(self, env):
        matrix = np.zeros((6, env.model.nv))
        mujoco.mj_jacBody(env.model, env.data, matrix[:3], matrix[3:], self.body)
        return matrix[:, self.dofs]


def attitude_error(target, actual):
    return Rotation.from_matrix(target @ actual.reshape(3, 3).T).as_rotvec()


def damped_step(jacobian, error):
    regularized = jacobian @ jacobian.T + 1e-4 * np.eye(len(error))
    return jacobian.T @ np.linalg.solve(regularized, error)


def stance_rate(chain, env, lever, translation, attitude):
    # If the foot stays planted, its motion relative to the pelvis must
    # cancel the requested world motion of that point on the pelvis.
    twist = np.r_[-translation - np.cross(attitude, lever), -attitude]
    weights = np.array([10., 10., 10., 1., 1., 1.])
    return damped_step(chain.jacobian(env) * weights[:, None], twist * weights)


def foot_wrenches(force, moment, left_lever, right_lever):
    separation = left_lever - right_lever
    residual = moment - np.cross((left_lever + right_lever) / 2, force)
    couple = np.cross(residual, separation) / np.dot(separation, separation)
    ankle = (residual - np.cross(separation, couple)) / 2
    return {"left": np.r_[force / 2 + couple, ankle],
            "right": np.r_[force / 2 - couple, ankle]}


class WholeBodyReference:
    def __init__(self, env):
        self.parent = env.hold_controller
        self.profile = env.grasp_acquisition
        self.chains = {s: Chain(env.model, s + "_ankle_roll_link",
                                [f"{s}_{axis}_joint" for axis in LEG_AXES]) for s in env.hands}
        self.chains["waist"] = Chain(env.model, "torso_link", [f"waist_{a}_joint" for a in WAIST_AXES])
        self.indices = np.concatenate([c.indices for c in self.chains.values()])
        self.limits = env.model.jnt_range[env.model.actuator_trnid[self.indices, 0]]
        self.effort_limit = np.max(abs(env.model.actuator_forcerange[self.indices]), axis=1)
        self.offset = np.zeros(len(self.indices))
        self.anchor = self.latch_time = None

    def desired(self, env, amount):
        command = self.parent.desired(env, amount)
        if env.phase != "HOLD":
            return command
        pelvis = env.data.body("pelvis")
        torso = env.data.body("torso_link")
        if self.anchor is None:
            self.anchor = (pelvis.xpos.copy(), pelvis.xmat.reshape(3, 3).copy(),
                           torso.xmat.reshape(3, 3).copy())
            self.latch_time = float(env.data.time)
        baseline = np.clip(command[self.indices], self.limits[:, 0], self.limits[:, 1])
        bounds = np.clip(self.limits - baseline[:, None], -BODY_OFFSET_RAD, BODY_OFFSET_RAD)
        frozen = abs(env.data.actuator_force[self.indices]) >= .98 * self.effort_limit
        increment = -self.offset if self.parent.inhibition else self.increment(env, pelvis, torso)
        self.offset = bounded_offset(self.offset, increment, bounds, frozen, env.config.control_dt)
        command[self.indices] = baseline + self.offset
        return command

    def increment(self, env, pelvis, torso):
        position, orientation, torso_orientation = self.anchor
        translation = position - pelvis.xpos
        rotation = attitude_error(orientation, pelvis.xmat)
        levers = {s: env.data.xpos[self.chains[s].body] - pelvis.xpos for s in env.hands}
        scale = self.profile.gain * env.config.control_dt
        if self.profile.balance == "stance":
            legs = [scale * stance_rate(self.chains[s], env, levers[s], translation, rotation)
                    for s in env.hands]
        else:
            goals = self.wrench_targets(env, pelvis.id, translation, rotation, levers)
            legs = [np.concatenate(goals) - self.offset[:-len(WAIST_AXES)]]
        waist = self.chains["waist"].jacobian(env)[3:]
        waist_step = scale * damped_step(waist, attitude_error(torso_orientation, torso.xmat))
        return np.concatenate([*legs, waist_step])

    def wrench_targets(self, env, pelvis, translation, rotation, levers):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 0)
        force = bounded(self.profile.gain * (900 * translation - 40 * velocity[3:]), 30.)
        moment = bounded(self.profile.gain * (90 * rotation - 5 * velocity[:3]), 5.)
        reactions = foot_wrenches(force, moment, levers["left"], levers["right"])
        # Reaction torque is supplied at the contact. Command its inverse at
        # the joint through the existing position servo and its native gain.
        return [-self.chains[s].jacobian(env).T @ reactions[s] / self.chains[s].kp for s in env.hands]

    def state(self):
        return {**self.parent.state(), "mode": "g6_whole_body", "body_latch_time_s": self.latch_time,
                "body_offset_rad": self.offset.tolist()}

    def metadata(self, env):
        return {**self.parent.metadata(env), "schema": "g6_whole_body_v1", "mode": "g6_whole_body",
                "balance": self.profile.balance, "body_gain": self.profile.gain,
                "body_offset_bound_rad": BODY_OFFSET_RAD, "body_rate_bound_rad_s": BODY_RATE_RAD_S,
                "body_joint_order": {name: chain.names for name, chain in self.chains.items()},
                "body_reference": "measured world pelvis pose and torso attitude at first HOLD command"}


class WholeBodyEnvironment(GraspEnvironment):
    def reset(self):
        row = super().reset()
        self.hold_controller = WholeBodyReference(self)
        return row
