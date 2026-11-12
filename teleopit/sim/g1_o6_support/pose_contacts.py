"""Read-only thumb-specific contacts; all vector quantities use world axes."""

import mujoco
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GROUPS = (*FINGERS, "palm", "housing")
LOADED_N = 0.05


def contact_group(body):
    if "_wrist_" in body:
        return "housing"
    return next((name for name in FINGERS if "_" + name + "_" in body), "palm")


def hand_body(body, side):
    return body.startswith(side + "_") and ("_o6_" in body or "_wrist_" in body)


def friction_usage(wrench, friction, dimension):
    force = float(wrench[0])
    if force <= 0:
        return {"translation_only_utilization": None, "torsional_utilization": None}
    denominator = np.maximum(np.array(friction[:2]) * force, 1e-15)
    return {
        "translation_only_utilization": float(
            np.linalg.norm(wrench[1:3] / denominator)
        ),
        "torsional_utilization": float(abs(wrench[3]) / max(friction[2] * force, 1e-15))
        if dimension >= 4
        else None,
    }


def contact_detail(env, index, row, side):
    contact = env.data.contact[index]
    robot_index = next(i for i, b in enumerate(row["bodies"]) if hand_body(b, side))
    sign = 1 if robot_index == 1 else -1
    local = np.zeros(6)
    mujoco.mj_contactForce(env.model, env.data, index, local)
    frame = contact.frame.reshape(3, 3)
    target = env.support_targets[side + "_hand"]
    rung = env.data.geom(target)
    point = rung.xmat.reshape(3, 3).T @ (contact.pos - rung.xpos)
    size = env.model.geom_size[rung.id]
    axis = int(np.argmax(size))
    return {
        "group": contact_group(row["bodies"][robot_index]),
        "geoms": row["geoms"],
        "target": target in row["geoms"],
        "contacted_geometry": row["geoms"][1 - robot_index],
        "normal_force_N": float(local[0]),
        "normal_on_hand_world": (sign * frame[0]).tolist(),
        "force_on_hand_world_N": (sign * frame.T @ local[:3]).tolist(),
        "tangential_force_on_hand_world_N": (sign * frame[1:].T @ local[1:3]).tolist(),
        "contact_axes_world": frame.tolist(),
        "wrench_contact_axes": local.tolist(),
        "friction": contact.friction.tolist(),
        "contact_dimension": int(contact.dim),
        "position_world_m": contact.pos.tolist(),
        "rung_position_m": point.tolist(),
        "rung_axial_axis": axis,
        "rung_axial_edge_margin_m": float(size[axis] - abs(point[axis])),
        "rung_box_face_offsets_m": (size - np.abs(point)).tolist(),
        "slip_m_s": row["slip_m_s"],
        "distance_m": row["distance"],
        **friction_usage(local, contact.friction, contact.dim),
    }


def hand_contacts(env, rows, side):
    selected = []
    for index, row in enumerate(rows):
        if "environment" not in row["regions"]:
            continue
        if any(hand_body(b, side) for b in row["bodies"]):
            selected.append(contact_detail(env, index, row, side))
    return selected


def thumb_opposition(loaded):
    thumb = [c["normal_on_hand_world"] for c in loaded if c["group"] == "thumb"]
    others = [c["normal_on_hand_world"] for c in loaded if c["group"] in FINGERS[1:]]
    return bool(
        thumb and others and np.min(np.array(thumb) @ np.array(others).T) < -0.5
    )


def group_centroid(contacts):
    if not contacts:
        return None
    weights = [c["normal_force_N"] for c in contacts]
    return np.average(
        [c["rung_position_m"] for c in contacts], axis=0, weights=weights
    ).tolist()


def grasp_quality(contacts, height):
    target = [c for c in contacts if c["target"]]
    loaded = [c for c in target if c["normal_force_N"] >= LOADED_N]
    groups = {g: [c for c in loaded if c["group"] == g] for g in GROUPS}
    ordered = all(
        any(
            (c["position_world_m"][2] - height) * (-1 if g == "thumb" else 1) > 0
            for c in groups[g]
        )
        for g in FINGERS
    )
    opposition = thumb_opposition(loaded)
    normal = {g: sum(c["normal_force_N"] for c in groups[g]) for g in GROUPS}
    return {
        "geometric_groups": {g: any(c["group"] == g for c in target) for g in GROUPS},
        "loaded_groups": {g: bool(groups[g]) for g in GROUPS},
        "normal_by_group_N": normal,
        "centroid_by_group_m": {g: group_centroid(groups[g]) for g in GROUPS},
        "thumb_normal_N": normal["thumb"],
        "minimum_finger_normal_N": min(normal[g] for g in FINGERS[1:]),
        "thumb_opposition": opposition,
        "ordered": ordered,
        "valid": opposition and ordered,
    }


def contact_states(hands):
    result = {}
    for side, hand in hands.items():
        quality = hand["quality"]
        for kind in ("geometric_groups", "loaded_groups"):
            result.update(
                {f"{side}:{g}:{kind}": present for g, present in quality[kind].items()}
            )
        result.update(
            {
                f"{side}:{key}": quality[key]
                for key in ("thumb_opposition", "ordered", "valid")
            }
        )
    return result


class ContactEpisodes:
    """Episodes follow anatomical groups, never transient contact IDs."""

    def __init__(self, dt):
        self.dt, self.active, self.rows = dt, {}, []

    def update(self, time, states):
        changed = []
        for name, present in states.items():
            previous = self.active.get(name)
            if previous is None or previous["present"] != present:
                previous = {
                    "signal": name,
                    "present": bool(present),
                    "start_s": time,
                    "duration_s": 0,
                }
                self.active[name] = previous
                self.rows.append(previous)
                changed.append(name)
            previous.update(end_s=time, duration_s=time - previous["start_s"] + self.dt)
        return changed

    def longest_absence(self):
        result = {}
        for row in self.rows:
            if not row["present"]:
                name = row["signal"]
                result[name] = max(result.get(name, 0), row["duration_s"])
        return result
