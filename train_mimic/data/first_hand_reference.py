"""Compile the authored first-hand motion into named training reference arrays."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from teleopit.runtime.assets import PROJECT_ROOT, UNITREE_G1_XML
from teleopit.runtime.cartesian_reference import load_reference, solve_reference
from teleopit.runtime.ladder_scene import build_ladder_preview_model

DEFAULT_FIRST_HAND_REFERENCE = str(PROJECT_ROOT / "assets/motions/ladder_first_hand.json")
TRACK_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis", "torso")


def load_first_hand_reference(path=DEFAULT_FIRST_HAND_REFERENCE, robot_xml=None):
    path = Path(path).resolve()
    robot_xml = Path(robot_xml or UNITREE_G1_XML).resolve()
    return _compile(str(path), path.stat().st_mtime_ns,
                    str(robot_xml), robot_xml.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _compile(path, reference_mtime, robot_xml, robot_mtime):
    del reference_mtime, robot_mtime
    model, data = build_ladder_preview_model(robot_xml)
    spec, tracks = load_reference(path, model, data)
    names = [track.name for track in tracks]
    if set(names) != set(TRACK_NAMES):
        raise ValueError(f"First-hand reference requires exactly these tracks: {TRACK_NAMES}")
    if spec["duration_s"] < 4.4:
        raise ValueError("First-hand reference must cover preparation, transfer and final hold (4.4 s)")
    # The command's event gates bind to these authored interval boundaries.
    phases = {p["name"]: p["time_s"] for p in spec.get("phases", [])}
    for name, time in (("Prepare release", 1.0), ("Withdraw left hand", 1.7),
                       ("Approach rung 6", 3.3), ("Grasp and hold", 3.9)):
        if phases.get(name) != time:
            raise ValueError(f"First-hand schedule requires phase {name!r} at {time}s")
    clip, report = solve_reference(model, data, spec, tracks)
    if not report["tracking_ok"] or report["collision_frames"]:
        raise ValueError(f"First-hand reference failed kinematic validation: {report}")
    order = [names.index(name) for name in TRACK_NAMES]
    for i, name in enumerate(TRACK_NAMES):
        track = tracks[order[i]]
        if name in ("right_hand", "left_foot", "right_foot"):
            if not np.allclose(track.positions, track.positions[0], atol=1e-6):
                raise ValueError(f"{name} must remain a stationary support in this task")
            if not np.allclose(np.abs(track.quaternions @ track.quaternions[0]), 1., atol=1e-6):
                raise ValueError(f"{name} orientation must remain stationary in this task")
    joint_ids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    joint_names = tuple(model.joint(j).name for j in joint_ids)
    joints = clip["qpos"][:, model.jnt_qposadr[joint_ids]].astype(np.float32)
    velocity = np.gradient(joints, 1/spec["fps"], axis=0).astype(np.float32)
    signature = hashlib.sha256(Path(path).read_bytes() + Path(robot_xml).read_bytes()
                               + joints.tobytes()).hexdigest()
    return {
        "fps": spec["fps"], "joint_names": joint_names,
        "joint_pos": joints, "joint_vel": velocity,
        "track_names": TRACK_NAMES,
        "position": clip["target_position"][:, order].astype(np.float32),
        "quaternion": clip["target_quaternion_wxyz"][:, order].astype(np.float32),
        "signature": signature, "validation": report,
    }
