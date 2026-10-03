"""Author a right-hand reference from a measured, asymmetric support pose."""
from __future__ import annotations

import re

import mujoco
import numpy as np

from teleopit.runtime.cartesian_reference import frame_pose, rotation_error, slerp


def build_second_hand_spec(model, data, state, meta, design):
    """Retarget in the ladder basis; never mirror or average recorded joint poses."""
    if design.get("version") != 1:
        raise ValueError("Unsupported second-hand design version")
    times = design["times_s"]
    ordered = [times[k] for k in ("settle", "prepare", "release", "withdraw", "lift", "approach", "hold")]
    if not np.isfinite(ordered).all() or not np.all(np.diff([0., *ordered]) > 0):
        raise ValueError("Stage times must be finite, positive and increasing")
    shift = np.asarray(design["body_shift_m"], dtype=float)
    blend = float(design["torso_neutral_blend"])
    withdrawal = float(design["withdrawal_m"])
    clearance = float(design["lateral_clearance_m"])
    if (shift.shape != (3,) or not np.isfinite(shift).all()
            or not 0 <= blend <= 1 or not np.isfinite(withdrawal) or withdrawal <= 0
            or not np.isfinite(clearance) or clearance < 0):
        raise ValueError("Invalid preparation or withdrawal parameters")
    if meta["track_names"] != ["left_hand","right_hand","left_foot","right_foot","pelvis","torso"]:
        raise ValueError("Unexpected pose bank track order")
    if not np.all(state["attached"][-1]) or not np.all(state["foot_contact"][-1]):
        raise ValueError("Start requires both hand attachments and both foot contacts")
    left_rung, right_rung = map(int, state["held_rung"][-1])
    if left_rung != right_rung + 1 or right_rung < 0:
        raise ValueError("Expected left hand one rung above right hand")
    rung_ids = sorted(
        [i for i in range(model.nsite)
         if re.search(r"(?:^|/)left_ladder_rung_\d+_grip$", model.site(i).name)],
        key=lambda i: model.site(i).name,
    )
    if left_rung >= len(rung_ids):
        raise ValueError("Held rung is absent from the scene")
    lower = data.site_xpos[rung_ids[right_rung]].copy()
    upper = data.site_xpos[rung_ids[left_rung]].copy()
    up = (upper - lower) / np.linalg.norm(upper - lower)
    lateral = data.site_xmat[rung_ids[right_rung]].reshape(3, 3)[:, 2].copy()
    # Site capsule direction can be reversed. +lateral must point toward the left grip.
    left_pos = data.site_xpos[meta["hand_site_ids"][0]]
    right_pos = data.site_xpos[meta["hand_site_ids"][1]]
    if np.dot(lateral, left_pos - right_pos) < 0:
        lateral *= -1
    outward = np.cross(up, lateral)
    outward /= np.linalg.norm(outward)
    # Withdraw horizontally away from this ladder face, keeping the original hand height.
    horizontal_outward = outward.copy()
    horizontal_outward[2] = 0
    if np.linalg.norm(horizontal_outward) < 1e-6:
        raise ValueError("Horizontal ladder is not supported by this motion")
    horizontal_outward /= np.linalg.norm(horizontal_outward)
    basis = np.column_stack((-horizontal_outward, lateral, [0., 0., 1.]))
    body_shift = basis @ shift

    frames = [("site", i) for i in meta["hand_site_ids"] + meta["foot_site_ids"]]
    frames += [("body", meta["pelvis_body_id"]), ("body", meta["torso_body_id"])]
    colors = [[1,.55,0,1], [0,.8,1,1], [.45,1,.25,1], [.8,1,.3,1], [.8,.3,1,1], [1,.35,.65,1]]
    # Matches the environment's construction-time torso orientation reference.
    neutral = mujoco.MjData(model)
    mujoco.mj_forward(model, neutral)
    neutral_quat = frame_pose(neutral, "body", meta["torso_body_id"])[1]
    tracks = []
    target = right_pos + upper - lower
    for i, ((kind, frame_id), name) in enumerate(zip(frames, meta["track_names"])):
        p, q = frame_pose(data, kind, frame_id)
        def key(t, position, quat=q):
            return {"time_s": float(t), "position_m": np.asarray(position).tolist(),
                    "quaternion_wxyz": np.asarray(quat).tolist()}
        if name == "right_hand":
            retreat = withdrawal * horizontal_outward - clearance * lateral
            keys = [key(0,p), key(times["release"],p),
                    key(times["withdraw"],p+retreat), key(times["lift"],target+retreat),
                    key(times["approach"],target), key(times["hold"],target)]
        elif name in ("pelvis", "torso"):
            prepared_q = slerp(q, neutral_quat, blend) if name == "torso" else q
            keys = [key(0,p), key(times["settle"],p),
                    key(times["prepare"],p+body_shift,prepared_q),
                    key(times["hold"],p+body_shift,prepared_q)]
        else:
            keys = [key(0,p), key(times["hold"],p)]
        tracks.append({"name": name, "kind": kind, "target": mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY, frame_id),
            "color": colors[i], "position_weight": 0. if name == "torso" else (5. if i in (0,2,3) else 3.),
            "orientation_weight": .5, "keyframes": keys})
    return {
        "version": 1, "name": design["name"], "fps": design["fps"], "duration_s": times["hold"],
        "phases": [{"time_s": t, "name": name} for t, name in [
            (0,"Settle"), (times["settle"],"Prepare torso"), (times["prepare"],"Release right grip"),
            (times["release"],"Withdraw"), (times["withdraw"],"Lift"),
            (times["lift"],"Approach"), (times["approach"],"Grasp and hold")]],
        "contact_schedule": {"left_hand": "attached throughout", "feet": "support throughout",
            "right_hand": {"release_start_s": times["prepare"], "detached_by_s": times["release"],
                           "attach_not_before_s": times["approach"]},
            "execution": "Intent only. A future controller must gate release, attachment and completion on measured support and stability."},
        "geometry": {"source_rung_index": right_rung, "target_rung_index": left_rung,
                     "rung_delta_m": (upper-lower).tolist(), "right_hand_target_m": target.tolist()},
        "neutral_torso_quaternion_wxyz": neutral_quat.tolist(), "tracks": tracks,
    }


def assess_second_hand(model, data, spec, clip, initial_qpos):
    """Additional kinematic checks, including tangent-space velocities and limits."""
    dt = 1. / spec["fps"]
    velocity = np.zeros((len(clip["qpos"]), model.nv))
    for i in range(1, len(velocity)):
        mujoco.mj_differentiatePos(model, velocity[i], dt, clip["qpos"][i-1], clip["qpos"][i])
    margins = []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            q = clip["qpos"][:, model.jnt_qposadr[j]]
            margins.append((float(np.minimum(q-model.jnt_range[j,0], model.jnt_range[j,1]-q).min()), model.joint(j).name))
    release = spec["contact_schedule"]["right_hand"]["release_start_s"]
    torso = next(t for t in spec["tracks"] if t["name"] == "torso")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso["target"])
    data.qpos[:] = clip["qpos"][round(release*spec["fps"])]
    mujoco.mj_forward(model, data)
    orientation = np.linalg.norm(rotation_error(spec["neutral_torso_quaternion_wxyz"], frame_pose(data,"body",torso_id)[1]))
    return {"initial_qpos_max_error": float(np.max(np.abs(clip["qpos"][0]-initial_qpos))),
            "max_joint_speed_rad_s": float(np.abs(velocity[:,6:]).max()),
            "max_joint_acceleration_rad_s2": float(np.abs(np.diff(velocity[:,6:],axis=0)/dt).max()),
            "minimum_joint_limit_margin_rad": min(margins)[0], "closest_joint_limit": min(margins)[1],
            "torso_orientation_at_release_rad": float(orientation),
            "release_orientation_ok": bool(orientation < .25),
            "finite": bool(np.isfinite(clip["qpos"]).all()),
            "joint_limits_ok": bool(min(margins)[0] >= -1e-8)}, velocity
