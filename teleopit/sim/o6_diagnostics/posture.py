"""User-requested thumb-below candidate; never substitutes the published baseline."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.env import initial_errors
from teleopit.sim.g1_o6.initialize import arm_joints
from teleopit.sim.o6_grasp.presets import ladder_rung_config

PRESET = {
    "hand_rung": 10,
    "tilt_y_degrees": 60.0,
    "offset_x_m": -0.025,
    "offset_z_m": -0.025,
    "joint_regularization": 0.003,
    "elbow_regularization": 0.03,
    "seeds": 8,
}


def install_candidate(env):
    for side in env.hands:
        env.ik[side] = fit_candidate(env.model, env.data, env.hands[side], side)
    env.support_targets = {
        s + "_" + p: "left_ladder_rung_" + r
        for s in env.hands
        for p, r in (("hand", "10"), ("foot", "04"))
    }
    env.initial_qpos = env.data.qpos.copy()
    env.body_target = env.data.qpos[env.act_qpos].copy()
    env.target = env.body_target.copy()
    env.data.ctrl[:] = env.target
    mujoco.mj_forward(env.model, env.data)
    env.initial_errors = initial_errors(snapshot(env), env.ik)


def candidate_target(data, side):
    bench = ladder_rung_config()
    turn = Rotation.from_euler("y", PRESET["tilt_y_degrees"], degrees=True).as_matrix()
    pos = data.geom("left_ladder_rung_10").xpos + turn @ (
        np.array(bench.palm_pos) - bench.rung_pos
    )
    pos += [
        PRESET["offset_x_m"],
        0.18 if side == "left" else -0.18,
        PRESET["offset_z_m"],
    ]
    rot = turn @ Rotation.from_quat(np.array(bench.palm_quat)[[1, 2, 3, 0]]).as_matrix()
    if side == "left":
        mirror = np.diag([1, -1, 1])
        rot = mirror @ rot @ mirror
    return pos, rot


def fit_candidate(model, data, hand, side):
    joints = arm_joints(model, side)
    addresses = model.jnt_qposadr[joints]
    target, rotation = candidate_target(data, side)

    def residual(q):
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        site = data.site(hand["mount"])
        return np.r_[
            10 * (site.xpos - target),
            Rotation.from_matrix(rotation @ site.xmat.reshape(3, 3).T).as_rotvec(),
            PRESET["joint_regularization"] * q,
            PRESET["elbow_regularization"] * q[3],
        ]

    limits = model.jnt_range[joints]
    seeds = [data.qpos[addresses].copy()] + list(
        np.random.default_rng(0).uniform(
            limits[:, 0], limits[:, 1], (PRESET["seeds"] - 1, 7)
        )
    )
    solutions = [
        least_squares(
            residual,
            np.clip(q, limits[:, 0], limits[:, 1]),
            bounds=(limits[:, 0], limits[:, 1]),
            max_nfev=200,
        )
        for q in seeds
    ]
    solution = min(solutions, key=lambda s: np.linalg.norm(s.fun))
    error = residual(solution.x)
    return {
        "position_error_m": float(np.linalg.norm(error[:3]) / 10),
        "angle_error_rad": float(np.linalg.norm(error[3:6])),
        "target_mount_position": target.tolist(),
        "joints": {model.joint(j).name: float(v) for j, v in zip(joints, solution.x)},
    }
