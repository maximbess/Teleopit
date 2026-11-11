"""A thumb-below rung grasp with bounded wrist flexion and motor preload."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from teleopit.sim.o6_grasp.presets import ladder_rung_config

from .initialize import arm_joints, fit_leg


@dataclass(frozen=True)
class GraspPose:
    hand_rungs: tuple = (11, 11)
    waist_lean_rad: float = 0.3
    hand_tilt_deg: float = 50.0
    grip_offset_m: tuple = (-0.023, 0.0, -0.020)
    hand_spacing_m: float = 0.18
    closed_pose: tuple = (0.05, 0.5, 1.35, 1.35, 1.35, 1.35)
    finger_preload_Nm: float = 0.12

    def validate(self):
        if len(self.hand_rungs) != 2 or any(
            not isinstance(r, int) or not 1 <= r <= 14 for r in self.hand_rungs
        ):
            raise ValueError("hand_rungs must contain two rung numbers in 1..14")
        if len(self.grip_offset_m) != 3 or len(self.closed_pose) != 6:
            raise ValueError("expected a three-axis offset and six finger targets")
        values = [
            self.waist_lean_rad,
            self.hand_tilt_deg,
            self.hand_spacing_m,
            self.finger_preload_Nm,
            *self.grip_offset_m,
            *self.closed_pose,
        ]
        if not np.all(np.isfinite(values)) or self.hand_spacing_m <= 0:
            raise ValueError("pose values must be finite and spacing positive")


def install_balanced(env, pose=None):
    """Select a repeatable reset posture, including subsequent env.reset calls."""
    env.pose = pose or GraspPose()
    env.hold_only = True
    return env.reset()


def place_hands(env, pose):
    pose.validate()
    model, data = env.model, env.data
    waist = model.joint("waist_pitch_joint")
    target = data.qpos[waist.qposadr[0]] + pose.waist_lean_rad
    if not waist.range[0] <= target <= waist.range[1]:
        raise ValueError("waist lean exceeds the original joint range")
    feet = {
        side: (
            data.body(side + "_ankle_roll_link").xpos.copy(),
            data.body(side + "_ankle_roll_link").xmat.reshape(3, 3).copy(),
        )
        for side in env.hands
    }
    data.qpos[waist.qposadr[0]] = target
    reports = {s + "_foot": fit_leg(model, data, s, *feet[s]) for s in env.hands}
    for side, rung in zip(env.hands, pose.hand_rungs):
        reports[side] = fit_arm(env, side, *mount_target(env, pose, side, rung))
    env.support_targets = {
        side + "_" + part: f"left_ladder_rung_{r:02d}"
        for side, rung in zip(env.hands, pose.hand_rungs)
        for part, r in (("hand", rung), ("foot", 4))
    }
    return reports


def mount_target(env, pose, side, rung):
    bench = ladder_rung_config()
    turn = Rotation.from_euler("y", pose.hand_tilt_deg, degrees=True).as_matrix()
    position = (
        env.data.geom(f"left_ladder_rung_{rung:02d}").xpos
        + turn @ (np.array(bench.palm_pos) - bench.rung_pos)
        + pose.grip_offset_m
    )
    position[1] += (1 if side == "left" else -1) * pose.hand_spacing_m
    rotation = (
        turn @ Rotation.from_quat(np.array(bench.palm_quat)[[1, 2, 3, 0]]).as_matrix()
    )
    if side == "left":
        mirror = np.diag([1, -1, 1])
        rotation = mirror @ rotation @ mirror
    return position, rotation


def fit_arm(env, side, target, rotation):
    model, data = env.model, env.data
    joints = arm_joints(model, side)
    addresses = model.jnt_qposadr[joints]
    bounds = model.jnt_range[joints].copy()
    # Search bounds only. The compiled physical joint ranges remain unchanged.
    bounds[-2:, 0] = np.maximum(bounds[-2:, 0], -0.7)
    bounds[-2:, 1] = np.minimum(bounds[-2:, 1], 0.7)

    def residual(q):
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        mount = data.site(env.hands[side]["mount"])
        return np.r_[
            10 * (mount.xpos - target),
            Rotation.from_matrix(rotation @ mount.xmat.reshape(3, 3).T).as_rotvec(),
            0.005 * q,
            0.015 * q[-2:],
        ]

    starts = [
        np.clip(data.qpos[addresses], bounds[:, 0], bounds[:, 1]),
        np.random.default_rng(0).uniform(bounds[:, 0], bounds[:, 1]),
    ]
    solutions = [
        least_squares(residual, q, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=120)
        for q in starts
    ]
    best = min(solutions, key=lambda s: np.linalg.norm(s.fun[:6]))
    error = residual(best.x)
    return {
        "position_error_m": float(np.linalg.norm(error[:3]) / 10),
        "angle_error_rad": float(np.linalg.norm(error[3:6])),
    }
