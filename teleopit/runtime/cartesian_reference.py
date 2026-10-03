"""Multi-body Cartesian targets and CPU MuJoCo IK for kinematic previews.

No physics stepping, training, or implicit gripping: the resulting qpos clip is
a reference candidate. Target poses are retained alongside the solved poses.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np


def normalize_quat(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-10:
        raise ValueError("quaternion_wxyz must contain four finite values with nonzero norm")
    return value / np.linalg.norm(value)


def slerp(a, b, u):
    dot = np.dot(a, b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        return normalize_quat(a + u*(b-a))
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1-u)*angle)*a + np.sin(u*angle)*b) / np.sin(angle)


def frame_pose(data, kind, frame_id):
    if kind == "site":
        position, matrix = data.site_xpos[frame_id], data.site_xmat[frame_id]
    else:
        position, matrix = data.xpos[frame_id], data.xmat[frame_id]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, matrix)
    return position.copy(), quat


def rotation_error(target, current):
    conjugate = current * np.array([1, -1, -1, -1])
    delta = np.zeros(4)
    mujoco.mju_mulQuat(delta, target, conjugate)
    if delta[0] < 0:
        delta *= -1
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, delta, 1.0)
    return out


@dataclass
class Track:
    name: str
    kind: str
    target: str
    frame_id: int
    color: np.ndarray
    position_weight: float
    orientation_weight: float
    times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray

    def sample(self, time):
        if time <= self.times[0]:
            return self.positions[0].copy(), self.quaternions[0].copy()
        if time >= self.times[-1]:
            return self.positions[-1].copy(), self.quaternions[-1].copy()
        i = int(np.searchsorted(self.times, time, side="right") - 1)
        u = (time-self.times[i])/(self.times[i+1]-self.times[i])
        blend = u**3*(10 + u*(-15 + 6*u))
        return (self.positions[i] + blend*(self.positions[i+1]-self.positions[i]),
                slerp(self.quaternions[i], self.quaternions[i+1], blend))


def load_reference(path, model, data):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("version") != 1:
        raise ValueError("Unsupported reference version")
    duration, fps = float(spec["duration_s"]), float(spec["fps"])
    if not np.isfinite([duration, fps]).all() or duration <= 0 or fps <= 0:
        raise ValueError("duration_s and fps must be positive finite numbers")
    if abs(duration*fps - round(duration*fps)) > 1e-6:
        raise ValueError("duration_s must contain an integer number of frame intervals")
    spec["duration_s"], spec["fps"] = duration, fps
    tracks, names = [], set()
    for item in spec["tracks"]:
        name, kind, target = item["name"], item["kind"], item["target"]
        if name in names or kind not in ("site", "body"):
            raise ValueError(f"Duplicate track or invalid frame kind: {name}")
        names.add(name)
        obj = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
        frame_id = mujoco.mj_name2id(model, obj, target)
        if frame_id < 0:
            raise ValueError(f"Unknown {kind}: {target}")
        initial_pos, initial_quat = frame_pose(data, kind, frame_id)
        keys = item["keyframes"]
        times = np.array([k["time_s"] for k in keys], dtype=float)
        if (len(times) == 0 or not np.isfinite(times).all() or times[0] != 0
                or np.any(np.diff(times) <= 0) or times[-1] > duration):
            raise ValueError(f"{name}: key times must start at zero and increase within duration")
        positions, quats = [], []
        for key in keys:
            if ("offset_m" in key) == ("position_m" in key):
                raise ValueError(f"{name}: specify exactly one of offset_m and position_m")
            pos = np.asarray(key.get("position_m", key.get("offset_m")), dtype=float)
            if pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError(f"{name}: position must contain three finite numbers")
            positions.append(initial_pos + pos if "offset_m" in key else pos)
            quats.append(normalize_quat(key.get("quaternion_wxyz", initial_quat)))
        weights = [float(item.get("position_weight", 1)), float(item.get("orientation_weight", .15))]
        if not np.isfinite(weights).all() or min(weights) < 0 or max(weights) <= 0:
            raise ValueError(f"{name}: weights must be nonnegative, with at least one positive")
        color = np.asarray(item.get("color", [1, .5, 0, 1]), dtype=float)
        if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError(f"{name}: color must contain four values in [0, 1]")
        tracks.append(Track(name, kind, target, frame_id, color, *weights,
                            times, np.array(positions), np.array(quats)))
    if not tracks:
        raise ValueError("Reference needs at least one track")
    return spec, tracks


def solve_reference(model, data, spec, tracks, max_iterations=100, *, posture_weight=.003,
                    freeze_static_targets=False):
    """Warm-started damped least-squares IK with joint limits and line search."""
    if not np.isfinite(posture_weight) or posture_weight <= 0:
        raise ValueError("posture_weight must be finite and positive")
    initial = data.qpos.copy()
    times = np.arange(round(spec["duration_s"]*spec["fps"])+1)/spec["fps"]
    shape = (len(times), len(tracks))
    qpos = np.zeros((len(times), model.nq))
    targets = np.zeros((*shape, 3))
    target_quat = np.zeros((*shape, 4))
    actual = np.zeros_like(targets)
    errors = np.zeros(shape)
    angle_errors = np.zeros(shape)
    # Small posture preference resolves redundant solutions without pinning motion.
    regularization = posture_weight
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros_like(jac_pos)
    posture = np.zeros(model.nv)
    collision_frames = []

    def system(poses, with_jacobian=True):
        residuals, rows = [], []
        for track, (p, q) in zip(tracks, poses):
            current_p, current_q = frame_pose(data, track.kind, track.frame_id)
            residuals.extend([(p-current_p)*track.position_weight,
                              rotation_error(q, current_q)*track.orientation_weight])
            if with_jacobian:
                func = mujoco.mj_jacSite if track.kind == "site" else mujoco.mj_jacBody
                func(model, data, jac_pos, jac_rot, track.frame_id)
                rows.extend([jac_pos.copy()*track.position_weight,
                             jac_rot.copy()*track.orientation_weight])
        mujoco.mj_differentiatePos(model, posture, 1, data.qpos, posture_reference)
        residuals.append(regularization*posture.copy())
        if with_jacobian:
            rows.append(regularization*np.eye(model.nv))
        return np.concatenate(residuals), np.vstack(rows) if with_jacobian else None

    previous_poses = None
    for frame, time_s in enumerate(times):
        posture_reference = initial if frame == 0 else qpos[frame-1]
        poses = [track.sample(time_s) for track in tracks]
        static = (freeze_static_targets and previous_poses is not None
                  and all(np.allclose(p, old_p, rtol=0, atol=1e-12)
                          and np.allclose(q, old_q, rtol=0, atol=1e-12)
                          for (p, q), (old_p, old_q) in zip(poses, previous_poses)))
        # A moving posture prior can otherwise drift in redundant joints at rest.
        for _ in range(0 if static else max_iterations):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            residual, jac = system(poses)
            if np.max(np.abs(residual[:-model.nv])) < 1e-7:
                break
            # Levenberg damping also keeps the solve bounded near singular poses.
            # MuJoCo's small dense kernels avoid starting a threaded BLAS runtime.
            normal = np.zeros((model.nv, model.nv))
            rhs = np.zeros(model.nv)
            mujoco.mju_mulMatTMat(normal, jac, jac)
            mujoco.mju_mulMatTVec(rhs, jac, residual)
            normal.flat[::model.nv+1] += 1e-6
            delta = np.zeros(model.nv)
            mujoco.mju_cholFactor(normal, 1e-12)
            mujoco.mju_cholSolve(delta, normal, rhs)
            largest = np.max(np.abs(delta))
            if largest > .12:
                delta *= .12/largest
            previous = data.qpos.copy()
            improved = False
            for scale in (1., .5, .25, .125, .0625):
                data.qpos[:] = previous
                mujoco.mj_integratePos(model, data.qpos, delta, scale)
                for j in range(model.njnt):
                    if model.jnt_limited[j] and model.jnt_type[j] in (
                        mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                        adr = model.jnt_qposadr[j]
                        data.qpos[adr] = np.clip(data.qpos[adr], *model.jnt_range[j])
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                candidate, _ = system(poses, False)
                if np.linalg.norm(candidate) < np.linalg.norm(residual):
                    improved = True
                    break
            if not improved:
                data.qpos[:] = previous
                break
            if np.max(np.abs(scale*delta)) < 1e-8:
                break
        mujoco.mj_forward(model, data)
        qpos[frame] = data.qpos
        for i, (track, (p, q)) in enumerate(zip(tracks, poses)):
            actual_p, actual_q = frame_pose(data, track.kind, track.frame_id)
            targets[frame, i], target_quat[frame, i] = p, q
            actual[frame, i] = actual_p
            errors[frame, i] = np.linalg.norm(p-actual_p)
            angle_errors[frame, i] = np.linalg.norm(rotation_error(q, actual_q))
        pairs = []
        for contact in data.contact:
            if contact.dist >= -.002:
                continue
            a, b = [model.geom(int(i)).name for i in contact.geom]
            # Nominal foot support is expected, but substantial penetration is reported.
            expected_foot = (("foot_omni" in a and "ladder_rung" in b)
                             or ("foot_omni" in b and "ladder_rung" in a))
            if not expected_foot or contact.dist < -.005:
                pairs.append([a, b, float(contact.dist)])
        if pairs:
            collision_frames.append({"frame": frame, "time_s": float(time_s), "pairs": pairs})
        previous_poses = poses
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    active_position = np.array([t.position_weight > 0 for t in tracks])
    active_orientation = np.array([t.orientation_weight > 0 for t in tracks])
    report = {
        "mode": "kinematic reference preview; not dynamically validated",
        "frames": len(times), "duration_s": spec["duration_s"],
        "tracks": {track.name: {"max_position_error_m": float(errors[:, i].max()),
                                "max_orientation_error_deg": float(np.rad2deg(angle_errors[:, i]).max())}
                   for i, track in enumerate(tracks)},
        "collision_frames": collision_frames,
        "tracking_ok": bool(np.all(errors[:, active_position] < .005)
                            and np.all(angle_errors[:, active_orientation] < np.deg2rad(3))),
        "collision_check": "Reports non-support penetration deeper than 2 mm, foot/rung deeper than 5 mm; sampled frames only, not swept collision or force feasibility",
    }
    return {"time_s": times, "qpos": qpos, "target_position": targets,
            "target_quaternion_wxyz": target_quat, "actual_position": actual,
            "position_error_m": errors, "orientation_error_rad": angle_errors}, report
