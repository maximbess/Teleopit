"""Joint-limit bounded reset IK; never called during scored physics stepping."""

import re

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from teleopit.sim.ladder_init import (
    LADDER_INITIAL_JOINT_POS,
    LADDER_INITIAL_ROOT_POS,
    LADDER_INITIAL_ROOT_ROT,
)
from teleopit.sim.o6_grasp.presets import ladder_rung_config


def initial_state(model, data, hands, cfg):
    # The original pose intersects the next rung at the knee by 0.45 mm.
    # Translate the robot/feet 3 mm back before IK, rather than disabling contact.
    data.qpos[:3] = np.array(LADDER_INITIAL_ROOT_POS) + [-0.003, 0, 0]
    data.qpos[3:7] = LADDER_INITIAL_ROOT_ROT
    for j in range(1, model.njnt):
        for pattern, value in LADDER_INITIAL_JOINT_POS.items():
            if re.fullmatch(pattern, model.joint(j).name):
                data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_forward(model, data)
    feet = {
        s: (
            data.body(s + "_ankle_roll_link").xpos.copy(),
            data.body(s + "_ankle_roll_link").xmat.reshape(3, 3).copy(),
        )
        for s in hands
    }
    data.qpos[:3] += [cfg.root_offset_x, 0, cfg.root_offset_z]
    reports = {}
    for side in hands:
        reports[side + "_foot"] = fit_leg(model, data, side, *feet[side])
    for side in hands:
        reports[side] = fit_arm(model, data, hands[side], side, cfg.wrist_offset_x)
    mujoco.mj_forward(model, data)
    return reports


def fit_leg(model, data, side, target, rotation):
    joints = [
        j
        for j in range(model.njnt)
        if model.joint(j).name.startswith(side + "_")
        and any(p in model.joint(j).name for p in ("hip_", "knee_", "ankle_"))
    ]
    addresses = model.jnt_qposadr[joints]

    def residual(values):
        data.qpos[addresses] = values
        mujoco.mj_kinematics(model, data)
        body = data.body(side + "_ankle_roll_link")
        return np.r_[
            10 * (body.xpos - target),
            Rotation.from_matrix(rotation @ body.xmat.reshape(3, 3).T).as_rotvec(),
        ]

    limits = model.jnt_range[joints]
    start = data.qpos[addresses].copy()
    start[0] = -0.7
    start[3] = 1.4
    result = least_squares(
        residual,
        np.clip(start, limits[:, 0], limits[:, 1]),
        bounds=(limits[:, 0], limits[:, 1]),
        max_nfev=250,
    )
    error = residual(result.x)
    return {
        "position_error_m": float(np.linalg.norm(error[:3]) / 10),
        "angle_error_rad": float(np.linalg.norm(error[3:])),
    }


def fit_arm(model, data, hand, side, offset=0.0):
    joints = arm_joints(model, side)
    addresses = model.jnt_qposadr[joints]
    start = data.qpos[addresses].copy()
    target, rotation = arm_target(data, side, offset)

    def residual(values):
        data.qpos[addresses] = values
        mujoco.mj_kinematics(model, data)
        site = data.site(hand["mount"])
        angle = Rotation.from_matrix(rotation @ site.xmat.reshape(3, 3).T).as_rotvec()
        return np.r_[10 * (site.xpos - target), angle, 0.01 * (values - start)]

    solution = solve_arm(residual, start, model.jnt_range[joints])
    error = residual(solution.x)
    return {
        "position_error_m": float(np.linalg.norm(error[:3]) / 10),
        "angle_error_rad": float(np.linalg.norm(error[3:6])),
        "target_mount_position": target.tolist(),
        "joints": {model.joint(j).name: float(v) for j, v in zip(joints, solution.x)},
    }


def arm_joints(model, side):
    return [
        j
        for j in range(model.njnt)
        if model.joint(j).name.startswith(side + "_")
        and any(part in model.joint(j).name for part in ("shoulder", "elbow", "wrist"))
        and "_o6_" not in model.joint(j).name
    ]


def arm_target(data, side, offset):
    rung = data.geom("left_ladder_rung_08").xpos.copy()
    bench = ladder_rung_config()
    # Overhand approach: forearms descend from the shoulders. Rotate the
    # bench grasp around the rung's transverse world X axis; do not force
    # an anatomically unreachable upward-facing wrist below the shoulders.
    overhand = np.diag([1, -1, -1])
    target = rung + overhand @ (np.array(bench.palm_pos) - bench.rung_pos)
    target += [offset, 0.18 if side == "left" else -0.18, 0]
    rotation = (
        overhand
        @ Rotation.from_quat(np.array(bench.palm_quat)[[1, 2, 3, 0]]).as_matrix()
    )
    if side == "left":
        mirror = np.diag([1, -1, 1])
        rotation = mirror @ rotation @ mirror

    return target, rotation


def solve_arm(residual, start, limits):
    rng = np.random.default_rng(0)
    seeds = [start] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(15)]
    solutions = [
        least_squares(
            residual,
            np.clip(seed, limits[:, 0], limits[:, 1]),
            bounds=(limits[:, 0], limits[:, 1]),
            max_nfev=200,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        for seed in seeds
    ]
    return min(solutions, key=lambda s: np.linalg.norm(s.fun[:6]))
