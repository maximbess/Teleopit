"""Replay the right pad, contact patch, palm and bounded wrist against the saved G6 tread."""

import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1_o6_g6_arm_report import arm_mapping, command_pose
from scripts.verify_g1_o6_g6_grip_feedback import grasp_sample
from scripts.verify_g1_o6_g6_whole_body import static_residual
from scripts.verify_g1_o6_g6_wrist_grip import contact_centroid, joint_step
from teleopit.sim.g1_o6_support.g6_knee_experiment import read_knee_rows
from teleopit.sim.g1_o6_support.pose_contacts import contact_group

try:
    from orjson import loads
except ImportError:
    from json import loads


def edge_margin(points, half_size):
    return np.min(np.asarray(half_size)[:2] - np.abs(np.asarray(points)[..., :2]), axis=-1)


def first_event(times, mask):
    indices = np.flatnonzero(mask)
    return round(float(times[indices[0]]), 6) if indices.size else None


def rotation_in_tread(rest, command, tread):
    return tread.T @ Rotation.from_matrix(command @ rest.T).as_rotvec()


def qpos(row):
    return [*row["base_pose"], *row["joint_positions"]]


def control_trace(path, model, meta):
    data = mujoco.MjData(model)
    ids = arm_mapping(model, meta)["right"]
    limits = model.jnt_range[model.actuator_trnid[ids, 0]]
    mount = meta["model"]["hands"]["right"]["mount"]
    target = meta["support_targets"]["right_hand"]
    samples, landmark, entry = [], None, None
    with (path / "steps.jsonl").open("rb") as stream:
        previous = loads(next(stream))
        rest = np.array(previous["applied_position_targets_rad"])
        for line in stream:
            row = loads(line)
            if row["phase"] == "HOLD":
                if entry is None:
                    entry = qpos(previous)
                state = row["pose_search"]["controller_state"]
                wrist = state["wrist_grip"]
                old = previous["pose_search"]["controller_state"]["arm_correction_rad"]["right"]
                data.qpos[:] = qpos(previous)
                mujoco.mj_kinematics(model, data)
                tread = data.geom(target).xmat.reshape(3, 3).copy()
                if landmark is None and wrist["landmarks"]:
                    point = contact_centroid(previous["contacts"], "right", target)
                    thumb = data.body("right_o6_rh_thumb_distal")
                    landmark = thumb.xmat.reshape(3, 3).T @ (point - thumb.xpos)
                    np.testing.assert_allclose(landmark, wrist["landmarks"]["right"]["local_m"], atol=1e-9)
                origin = command_pose(model, data, mount, ids, rest)[1]
                desired = rest.copy()
                request = wrist["requests"].get("right")
                if request:
                    desired[ids] += joint_step(np.array(old), request["joint_rate_rad_s"], rest[ids], limits,
                                               np.array(previous["effort_fraction"])[ids], meta["config"]["control_dt"])
                else:
                    desired[ids] += state["arm_correction_rad"]["right"]
                # Requested and projected targets belong to this decision. Applied includes actuator delay.
                projected = rest.copy()
                projected[ids] += state["arm_correction_rad"]["right"]
                rotations = [rotation_in_tread(origin, command_pose(model, data, mount, ids, cmd)[1], tread)
                             for cmd in (desired, projected, np.array(row["applied_position_targets_rad"]))]
                samples.append([row["hold_time_s"], *np.concatenate(rotations), *np.asarray(old)[:4]])
            previous = row
    return np.array(samples), entry, landmark


def thumb_patch(row, target):
    contacts = []
    for contact in row["contacts"]:
        if target in contact["geoms"] and "right_hand" in contact["regions"] and contact["normal_force"] >= .05:
            body = contact["bodies"][contact["regions"].index("right_hand")]
            if contact_group(body) == "thumb":
                contacts.append(contact)
    if not contacts:
        return np.full(3, np.nan), np.nan
    weights = [c["normal_force"] for c in contacts]
    return (np.average([c["position"] for c in contacts], weights=weights, axis=0),
            np.average([c["slip_m_s"] for c in contacts], weights=weights))


def native_trace(path, model, meta, entry, landmark):
    data = mujoco.MjData(model)
    knees = read_knee_rows(path, model.nq)
    positions = entry
    samples = []
    weight = model.opt.gravity * model.body_subtreemass[model.body("pelvis").id]
    target = meta["support_targets"]["right_hand"]
    rung, thumb = data.geom(target), data.body("right_o6_rh_thumb_distal")
    palms = [data.site(meta["model"]["hands"][s]["palm"]) for s in ("right", "left")]
    with gzip.open(path / "physics_contacts.jsonl.gz", "rb") as stream:
        rows = (r for r in map(loads, stream) if r["phase"] == "HOLD")
        for knee, row in zip(knees, rows, strict=True):
            if abs(knee[0] - row["time_s"]) > 1e-8:
                raise ValueError("misaligned native contact and position records")
            data.qpos[:] = positions
            mujoco.mj_kinematics(model, data)
            frame = rung.xmat.reshape(3, 3).T
            material = thumb.xpos + thumb.xmat.reshape(3, 3) @ landmark
            center, slip = thumb_patch(row, target)
            points = [material, center, *(p.xpos for p in palms), data.body("pelvis").xpos]
            local = np.concatenate([frame @ (point - rung.xpos) for point in points])
            left, left_opposed = grasp_sample(row, "left")
            right, right_opposed = grasp_sample(row, "right")
            residual = np.linalg.norm(static_residual(row["contacts"], weight))
            samples.append([row["phase_time_s"], *local, slip, left[0]*left_opposed, right[0]*right_opposed,
                            min(left[1:]), residual, knee[1], knee[2]])
            positions = knee[3:]
    return np.asarray(samples)


def replay(path):
    meta = json.loads((path / "resolved.json").read_bytes())
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    controls, entry, landmark = control_trace(path, model, meta)
    if entry is None:
        raise ValueError("no unsupported hold to replay")
    if landmark is None:
        # A five-second screen can end at capture, before a wrist update is issued.
        landmark = np.full(3, np.nan)
    n = native_trace(path, model, meta, entry, landmark)
    half = model.geom(meta["support_targets"]["right_hand"]).size.copy()
    return dict(time=n[:, 0], fixed=n[:, 1:4], centroid=n[:, 4:7], right_palm=n[:, 7:10],
                left_palm=n[:, 10:13], pelvis=n[:, 13:16], slip=n[:, 16], left_thumb=n[:, 17],
                right_thumb=n[:, 18], weak_left=n[:, 19], residual=n[:, 20], knee_gap=n[:, 21],
                knee_contacts=n[:, 22], fixed_margin=edge_margin(n[:, 1:4], half),
                centroid_margin=edge_margin(n[:, 4:7], half), control_time=controls[:, 0],
                requested=controls[:, 1:4], projected=controls[:, 4:7], applied=controls[:, 7:10],
                proximal=controls[:, 10:14])


def describe(a):
    times = a["time"]
    failing = {k: first_event(times, mask) for k, mask in (
        ("right_thumb", a["right_thumb"] < .05), ("left_thumb", a["left_thumb"] < .05),
        ("left_fingers", a["weak_left"] < .05), ("force_balance", a["residual"] >= 25),
        ("knee_clearance", (a["knee_gap"] <= 0) | (a["knee_contacts"] != 0)))}
    ct = a["control_time"]
    applied = np.linalg.norm(a["applied"], axis=1)
    projected = np.linalg.norm(a["projected"], axis=1)
    requested = np.linalg.norm(a["requested"], axis=1)
    # Use the last intact 12 seconds, matched at the same decision before delay.
    stop = min(failing["right_thumb"] or times[-1], times[-1])
    window = (ct >= stop-12) & (ct < stop) & (requested > .030001) & (projected >= .029999)
    direction = None
    if np.any(window):
        error = np.median(a["requested"][window] - a["projected"][window], axis=0)
        direction = (error / np.linalg.norm(error)).tolist()
    keys = ("fixed_margin", "centroid_margin", "fixed", "centroid", "right_palm", "left_palm", "pelvis", "slip")
    snapshots = {}
    for when in (5., 30., 40., 40.56, 50., 52.028, 52.029, 60.):
        i = int(np.argmin(abs(times-when)))
        if abs(times[i]-when) < .00051:
            snapshots[str(when)] = {k: np.asarray(a[k][i]).tolist() for k in keys}
    return dict(native_steps=len(times), end_s=round(times[-1], 6), first=failing,
                saturated_applied_s=first_event(ct, applied >= .029999),
                saturated_requested_s=first_event(ct, requested > .030001),
                saturated_pre_loss_decisions=int(window.sum()), unmet_direction_rung=direction,
                max_proximal_offset_rad=float(np.max(abs(a["proximal"]))), snapshots=snapshots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    signals = replay(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **signals)
    args.output.with_suffix(".json").write_text(json.dumps(describe(signals), indent=2) + "\n")
