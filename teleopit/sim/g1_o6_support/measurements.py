"""Signed support accounting and separately declared late-motion assessment."""

import numpy as np
from scipy.spatial.transform import Rotation

from .config import DriftCriteria


def load_shares(row, mass, gravity):
    weight = mass * gravity
    forces = {name: item["force_world"] for name, item in row["supports"].items()}
    per_hand = {side: forces[side + "_hand"][2] / weight for side in ("left", "right")}
    return {
        "weight_N": weight,
        "support_forces_world_N": forces,
        "per_hand_share": per_hand,
        "total_hand_share": sum(per_hand.values()),
    }


def window_motion(time, vectors):
    time, vectors = np.asarray(time), np.asarray(vectors)
    centered = time - time.mean()
    slope = centered @ (vectors - vectors.mean(axis=0)) / (centered @ centered)
    return {
        "slope_vector_per_s": slope.tolist(),
        "slope_per_s": float(np.linalg.norm(slope)),
        "excursion": float(np.linalg.norm(np.ptp(vectors, axis=0))),
    }


def pose_vectors(rows):
    bases = np.array([r["base_pose"] for r in rows])
    rotations = Rotation.from_quat(bases[:, [4, 5, 6, 3]])
    result = {
        "base": (bases[:, :3], (rotations * rotations[0].inv()).as_rotvec()),
        "com": (np.array([r["com"] for r in rows]), None),
    }
    for side in ("left", "right"):
        poses = [r["hand_in_target_rung"][side] for r in rows]
        rotations = Rotation.from_matrix(
            np.array([p["rotation"] for p in poses]).reshape(-1, 3, 3)
        )
        result[side + "_hand"] = (
            np.array([p["position"] for p in poses]),
            (rotations * rotations[0].inv()).as_rotvec(),
        )
    return result


def assess_window(rows, start, end):
    selected = [r for r in rows if start - 1e-8 <= r["hold_time_s"] <= end + 1e-8]
    if (
        len(selected) < 2
        or selected[-1]["hold_time_s"] - selected[0]["hold_time_s"] < end - start - 0.03
    ):
        return {"interval_s": [start, end], "status": "NOT RUN", "signals": {}}
    time = [r["hold_time_s"] for r in selected]
    signals = {}
    for name, (position, angle) in pose_vectors(selected).items():
        item = {"translation": window_motion(time, position)}
        if angle is not None:
            item["rotation"] = window_motion(time, angle)
        signals[name] = item
    passed = all(motion_passes(item) for item in signals.values())
    return {
        "interval_s": [start, end],
        "status": "PASS" if passed else "FAIL",
        "signals": signals,
    }


def motion_passes(item):
    criteria = DriftCriteria()
    p = item["translation"]
    translation = (
        p["slope_per_s"] <= criteria.translation_slope_m_s
        and p["excursion"] <= criteria.translation_excursion_m
    )
    r = item.get("rotation", {"slope_per_s": 0, "excursion": 0})
    return (
        translation
        and r["slope_per_s"] <= criteria.rotation_slope_rad_s
        and r["excursion"] <= criteria.rotation_excursion_rad
    )


def late_drift(rows, requested):
    windows = [
        assess_window(rows, start, start + 30)
        for start in (requested - 60, requested - 30)
        if start >= 0
    ]
    status = windows[-1]["status"] if windows else "NOT RUN"
    return {
        "status": status,
        "criterion": "g1_o6_late_motion_v1",
        "excursion_definition": "Euclidean norm of axis-wise peak-to-peak ranges",
        "windows": windows,
        "asymptotic_stability_proven": False,
    }
