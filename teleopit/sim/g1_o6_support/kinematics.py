"""Scene-wide hand clearance and arm IK on scratch data, never live state."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.diagnostics import region
from teleopit.sim.g1_o6.initialize import arm_joints
from teleopit.sim.g1_o6.posture import geom_clearance


def arm_target(env, side, position, rotation):
    model = env.model
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, env.data)
    joints = arm_joints(model, side)
    addresses = model.jnt_qposadr[joints]

    def residual(values):
        scratch.qpos[addresses] = values
        mujoco.mj_kinematics(model, scratch)
        site = scratch.site(env.hands[side]["mount"])
        return np.r_[
            10 * (site.xpos - position),
            Rotation.from_matrix(rotation @ site.xmat.reshape(3, 3).T).as_rotvec(),
        ]

    limits = model.jnt_range[joints]
    result = least_squares(
        residual,
        np.clip(scratch.qpos[addresses], limits[:, 0], limits[:, 1]),
        bounds=(limits[:, 0], limits[:, 1]),
        max_nfev=60,
    )
    error = residual(result.x)
    mujoco.mj_forward(model, scratch)
    return {
        "joints": joints,
        "qpos": result.x,
        "position_error_m": float(np.linalg.norm(error[:3]) / 10),
        "angle_error_rad": float(np.linalg.norm(error[3:])),
        "scratch": scratch,
    }


def collision_pair(model, first, second):
    a, b = (int(model.geom_bodyid[g]) for g in (first, second))
    if a == b or model.body_weldid[a] == model.body_weldid[b]:
        return False
    if (min(a, b) << 16) + max(a, b) in model.exclude_signature:
        return False
    if structural_neighbors(model, a, b):
        return False
    return bool(
        (model.geom_contype[first] & model.geom_conaffinity[second])
        or (model.geom_contype[second] & model.geom_conaffinity[first])
    )


def structural_neighbors(model, a, b):
    """Adjacent welded groups form the structural hand/mount interface."""
    wa, wb = int(model.body_weldid[a]), int(model.body_weldid[b])
    return (
        wa
        and wb
        and (
            model.body_weldid[model.body_parentid[wa]] == wb
            or model.body_weldid[model.body_parentid[wb]] == wa
        )
    )


class Clearance:
    def __init__(self, env, side):
        self.model = env.model
        self.target = env.support_targets[side + "_hand"]
        self.hand = {
            g
            for g in range(env.model.ngeom)
            if region(env.model, env.model.geom_bodyid[g]) == side + "_hand"
            and env.model.geom_contype[g]
        }
        self.pairs = [
            (g, o)
            for g in self.hand
            for o in range(env.model.ngeom)
            if o not in self.hand and collision_pair(env.model, g, o)
        ]

    def measure(self, data):
        best = (0.1, "distance_cap", None, None)
        for first, second in self.pairs:
            try:
                distance, method = geom_clearance(self.model, data, first, second)
            except ValueError:
                # Exact-zero unknown shapes cannot certify positive clearance.
                distance, method = 0.0, "uncertified_zero"
            if distance < best[0]:
                best = (distance, method, first, second)
        contacts = sum(
            (int(c.geom1) in self.hand) != (int(c.geom2) in self.hand)
            for c in data.contact
        )
        return {
            "clearance_m": best[0],
            "method": best[1],
            "pair": [
                self.model.geom(g).name if g is not None else None for g in best[2:]
            ],
            "external_contact_count": contacts,
            "target_rung": self.target,
        }
