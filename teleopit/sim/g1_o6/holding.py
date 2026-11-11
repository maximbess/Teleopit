"""Static load feedforward and finite finger preload through existing servos."""

import mujoco
import numpy as np


class HoldController:
    """Compute position targets; never apply an external supporting wrench."""

    def __init__(self, env, *, closed_pose, preload_Nm):
        self.closed = np.asarray(closed_pose, dtype=float)
        self.preload = preload_Nm
        model = env.model
        fingers = np.concatenate(list(env.hand_act.values()))
        limit = np.min(model.actuator_forcerange[fingers, 1])
        if not 0 < preload_Nm < limit:
            raise ValueError(
                "finger preload must be positive and below the effort limit"
            )
        dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.kp = model.actuator_gainprm[:, 0].copy()
        self.gravity = env.data.qfrc_bias[dofs].copy() / self.kp
        self.foot_load = foot_weight_torque(env)[dofs] / self.kp

    def desired(self, env, amount):
        desired = env.body_target + self.gravity - self.foot_load
        for indices in env.hand_act.values():
            measured = env.data.qpos[env.act_qpos[indices]]
            desired[indices] = np.minimum(
                amount * self.closed, measured + self.preload / self.kp[indices]
            )
        return desired


def foot_weight_torque(env):
    """Reference statics: equal foot loads with CoP below the reference COM."""
    model, data = env.model, env.data
    torque = np.zeros(model.nv)
    com = data.subtree_com[model.body("pelvis").id]
    for side in env.hands:
        body = model.body(side + "_ankle_roll_link").id
        point = data.xpos[body].copy()
        point[0] = com[0]
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, point, body)
        torque += jacobian.T @ np.array([0, 0, env.mass * 9.81 / 2])
    return torque
