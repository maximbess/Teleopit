"""Bounded rung-relative wrist reference feedback through original arm servos."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.initialize import arm_joints
from teleopit.sim.g1_o6.metadata import controller_settings


def bounded_correction(offset, rate, command, bounds, effort, dt):
    delta = np.clip(dt * rate, -0.005 * dt, 0.005 * dt)
    delta = np.clip(
        delta,
        np.minimum(bounds[:, 0] - command, 0),
        np.maximum(bounds[:, 1] - command, 0),
    )
    delta[effort >= 0.98] = 0
    return np.clip(offset + delta, -0.03, 0.03)


def rung_pose(env, data, side):
    rung = data.geom(env.support_targets[side + "_hand"])
    mount = data.site(env.hands[side]["mount"])
    rotation = rung.xmat.reshape(3, 3)
    return rotation.T @ (mount.xpos - rung.xpos), rotation.T @ mount.xmat.reshape(3, 3)


def correction_rate(env, data, side, reference, dofs):
    rung = data.geom(env.support_targets[side + "_hand"])
    site = data.site(env.hands[side]["mount"])
    rotation = rung.xmat.reshape(3, 3)
    position = rung.xpos + rotation @ reference[0]
    attitude = rotation @ reference[1]
    error = np.r_[
        position - site.xpos,
        Rotation.from_matrix(attitude @ site.xmat.reshape(3, 3).T).as_rotvec(),
    ]
    translation, angular = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, data, translation, angular, site.id)
    jacobian = np.vstack((10 * translation[:, dofs], angular[:, dofs]))
    weighted = error * [10, 10, 10, 1, 1, 1]
    rate = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + 1e-4 * np.eye(6), weighted
    )
    return rate, error


class ArmReference:
    def __init__(self, env):
        self.base = env.hold_controller
        self.scratch = mujoco.MjData(env.model)
        self.joints = {s: arm_joints(env.model, s) for s in env.hands}
        self.indices = {
            s: np.array(
                [
                    i
                    for i, transmission in enumerate(env.model.actuator_trnid)
                    if transmission[0] in joints
                ]
            )
            for s, joints in self.joints.items()
        }
        self.reset(env)

    def reset(self, env):
        self.reference, self.errors = {}, {}
        self.offset = {s: np.zeros(len(a)) for s, a in self.indices.items()}
        self.raw = env.data.ctrl.copy()
        self.latch_time = None

    def desired(self, env, amount):
        desired = self.base.desired(env, amount)
        if env.phase == "HOLD":
            mujoco.mj_copyData(self.scratch, env.model, env.data)
            mujoco.mj_forward(env.model, self.scratch)
            if not self.reference:
                self.reference = {s: rung_pose(env, self.scratch, s) for s in env.hands}
                self.latch_time = float(env.data.time)
            for side, indices in self.indices.items():
                self.update(env, desired, side, indices)
        self.raw = desired.copy()
        return desired

    def update(self, env, desired, side, indices):
        model = env.model
        joints = model.actuator_trnid[indices, 0]
        rate, error = correction_rate(
            env, self.scratch, side, self.reference[side], model.jnt_dofadr[joints]
        )
        effort = np.abs(env.data.actuator_force[indices]) / np.max(
            np.abs(model.actuator_forcerange[indices]), axis=1
        )
        self.offset[side] = bounded_correction(
            self.offset[side],
            rate,
            desired[indices] + self.offset[side],
            model.jnt_range[joints],
            effort,
            env.config.control_dt,
        )
        desired[indices] += self.offset[side]
        self.errors[side] = error.tolist()

    def state(self):
        return {
            "mode": "arm_feedback",
            "latch_time_s": self.latch_time,
            "reference_at_hold_entry": {
                s: {"position_rung_m": p.tolist(), "rotation_rung": r.tolist()}
                for s, (p, r) in self.reference.items()
            },
            "arm_correction_rad": {s: v.tolist() for s, v in self.offset.items()},
            "tracking_error_world_m_rad": self.errors.copy(),
        }

    def metadata(self, env):
        return {
            **controller_settings(env.pose, env.config),
            "schema": "g1_o6_bounded_arm_reference_v1",
            "mode": "arm_feedback",
            "offset_bound_rad": 0.03,
            "offset_rate_rad_s": 0.005,
            "feedback_gain_per_s": 1.0,
            "jacobian_position_weight": 10.0,
            "damping_squared": 1e-4,
            "effort_freeze_fraction": 0.98,
            "arm_joint_order": {
                s: [env.model.joint(j).name for j in env.model.actuator_trnid[a, 0]]
                for s, a in self.indices.items()
            },
            "reference": "HOLD-entry wrist mount pose in actual rung axes; scratch-data Jacobian",
            "reset": "clear reference and corrections on every fresh reset",
            "isolated_runner_support": False,
        }
