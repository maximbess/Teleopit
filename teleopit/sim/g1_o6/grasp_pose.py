"""A thumb-below rung grasp with bounded wrist flexion and motor preload."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from teleopit.sim.o6_grasp.config import GraspConfig
from teleopit.sim.o6_grasp.presets import ladder_rung_config

from .initialize import arm_joints, fit_leg


@dataclass(frozen=True)
class GraspPose:
    hand_rungs: tuple = (11, 11)
    foot_rungs: tuple = (4, 4)
    waist_lean_rad: float = 0.3
    hand_tilt_deg: float = 50.0
    grip_offset_m: tuple = (-0.023, 0.0, -0.020)
    hand_spacing_m: float = 0.18
    closed_pose: tuple = (0.05, 0.5, 1.35, 1.35, 1.35, 1.35)
    finger_preload_Nm: float = 0.12

    def __post_init__(self):
        for name in ("hand_rungs", "foot_rungs", "grip_offset_m", "closed_pose"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise TypeError(f"{name} must be an array")
            object.__setattr__(self, name, tuple(value))
        self.validate()

    def validate(self):
        self.validate_vectors()
        values = [
            self.waist_lean_rad,
            self.hand_tilt_deg,
            self.hand_spacing_m,
            self.finger_preload_Nm,
            *self.grip_offset_m,
            *self.closed_pose,
        ]
        if any(type(v) not in (int, float) for v in values):
            raise ValueError("pose values must be numbers")
        if not np.all(np.isfinite(values)) or self.hand_spacing_m <= 0:
            raise ValueError("pose values must be finite and spacing positive")
        if not 0 < self.finger_preload_Nm < GraspConfig().effort_limit:
            raise ValueError(
                "finger preload must be positive and below the effort limit"
            )

    def validate_vectors(self):
        for name in ("hand_rungs", "foot_rungs"):
            if len(getattr(self, name)) != 2 or any(
                type(r) is not int or r < 1 for r in getattr(self, name)
            ):
                raise ValueError(name + " must contain two positive rung numbers")
        if len(self.grip_offset_m) != 3 or len(self.closed_pose) != 6:
            raise ValueError("expected a three-axis offset and six finger targets")

    def support_targets(self):
        return {
            side + "_" + part: f"left_ladder_rung_{r:02d}"
            for side, rung, foot in zip(
                ("left", "right"), self.hand_rungs, self.foot_rungs
            )
            for part, r in (("hand", rung), ("foot", foot))
        }


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
    env.support_targets = pose.support_targets()
    return reports


def mount_target(env, pose, side, rung):
    bench = ladder_rung_config()
    turn = Rotation.from_euler("y", pose.hand_tilt_deg, degrees=True).as_matrix()
    position = (
        env.data.geom(f"left_ladder_rung_{rung:02d}").xpos
        + turn @ (np.array(bench.palm_pos) - bench.rung_pos)
        + pose.grip_offset_m
    )
    position[0] += env.config.wrist_offset_x
    if getattr(env, "adapt_reset", False):
        from .initialize import rung_face_displacement

        position += rung_face_displacement(env.model, rung)
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

    starts = arm_starts(
        data.qpos[addresses], bounds, getattr(env, "reset_arm_seed", None)
    )
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


def arm_starts(current, bounds, seed=None):
    if seed is not None:
        return [np.random.default_rng(seed).uniform(bounds[:, 0], bounds[:, 1])]
    return [
        np.clip(current, bounds[:, 0], bounds[:, 1]),
        np.random.default_rng(0).uniform(bounds[:, 0], bounds[:, 1]),
    ]
