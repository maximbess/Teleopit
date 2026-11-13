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

# Reset IK seeds measured at b34ba73683986ef4252d953def71569b77a22eac.
# Preserve the qualified nonlinear solve without executing an old posture.
ARM_SEEDS = {
    "left": {
        "left_shoulder_pitch_joint": -0.9577842450616211,
        "left_shoulder_roll_joint": 0.060242686566230254,
        "left_shoulder_yaw_joint": 1.318443642120854,
        "left_elbow_joint": 1.3977883998976601,
        "left_wrist_roll_joint": -1.9250857705434818,
        "left_wrist_pitch_joint": -1.1406222594778044,
        "left_wrist_yaw_joint": -0.2901712125336879,
    },
    "right": {
        "right_shoulder_pitch_joint": -0.9577785931388026,
        "right_shoulder_roll_joint": -0.060266905345820965,
        "right_shoulder_yaw_joint": -1.3184464721178988,
        "right_elbow_joint": 1.3977925126948085,
        "right_wrist_roll_joint": 1.9250362872066868,
        "right_wrist_pitch_joint": -1.1406143454044781,
        "right_wrist_yaw_joint": 0.29011917102641377,
    },
}


def initial_stance(model, data, hands, cfg, *, foot_rungs=(4, 4), adapt=False):
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
    if adapt:
        translate_support_frame(model, data, feet, hands, foot_rungs)
    data.qpos[:3] += [cfg.root_offset_x, 0, cfg.root_offset_z]
    reports = {}
    for side in hands:
        reports[side + "_foot"] = fit_leg(model, data, side, *feet[side])
    for side in hands:
        for name, value in ARM_SEEDS[side].items():
            data.joint(name).qpos[0] = value
    mujoco.mj_forward(model, data)
    return reports


def rung_face_displacement(model, rung):
    """Front/top face movement from the historical 110 by 70 mm box."""
    size = model.geom(f"left_ladder_rung_{rung:02d}").size
    return np.array([.110 / 2 - size[0], 0., size[2] - .070 / 2])


def translate_support_frame(model, data, feet, hands, foot_rungs):
    from teleopit.sim.ladder_config import reference_rung_centers

    reference = np.array(reference_rung_centers()[3])
    shifts = []
    for side, rung in zip(hands, foot_rungs):
        shift = data.geom(f"left_ladder_rung_{rung:02d}").xpos - reference
        shift = shift + rung_face_displacement(model, rung)
        position, rotation = feet[side]
        feet[side] = (position + shift, rotation)
        shifts.append(shift)
    data.qpos[:3] += np.mean(shifts, axis=0)


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


def arm_joints(model, side):
    return [
        j
        for j in range(model.njnt)
        if model.joint(j).name.startswith(side + "_")
        and any(part in model.joint(j).name for part in ("shoulder", "elbow", "wrist"))
        and "_o6_" not in model.joint(j).name
    ]
