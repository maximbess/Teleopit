"""Wrist wrenches are internal; only robot/environment contacts enter balance."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.o6_grasp.diagnostics import point_velocity

TARGETS = {
    s + "_" + part: "left_ladder_rung_" + rung
    for s in ("left", "right")
    for part, rung in (("hand", "08"), ("foot", "04"))
}


def region(model, body):
    name = model.body(body).name
    if "_o6_" in name:
        return name.split("_")[0] + "_hand"
    if name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        return name.split("_")[0] + "_foot"
    while body:
        if model.body(body).name == "pelvis":
            return "other_robot"
        body = model.body_parentid[body]
    return "environment"


def contacts(model, data):
    rows = []
    force_sum = np.zeros(3)
    for i, c in enumerate(data.contact):
        bodies = [int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
        regions = [region(model, b) for b in bodies]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        rotation = c.frame.reshape(3, 3)
        force_on_second = rotation.T @ wrench[:3]
        velocity = point_velocity(model, data, c.pos, bodies[1]) - point_velocity(
            model, data, c.pos, bodies[0]
        )
        row = {
            "bodies": [model.body(b).name for b in bodies],
            "regions": regions,
            "geoms": [model.geom(g).name for g in (c.geom1, c.geom2)],
            "normal_force": float(wrench[0]),
            "force_on_second_world": force_on_second.tolist(),
            "normal_on_second_world": rotation[0].tolist(),
            "slip_m_s": float(np.linalg.norm((rotation @ velocity)[1:])),
            "distance": float(c.dist),
            "friction": c.friction.tolist(),
            "position": c.pos.tolist(),
        }
        rows.append(row)
        if (regions[0] == "environment") != (regions[1] == "environment"):
            force_sum += (
                force_on_second if regions[0] == "environment" else -force_on_second
            )
    return rows, force_sum


def support(rows, name, target):
    selected = [
        r
        for r in rows
        if name in r["regions"] and target in r["geoms"] and r["normal_force"] >= 0.05
    ]
    force = np.zeros(3)
    normals = []
    for row in selected:
        sign = 1 if row["regions"][1] == name else -1
        force += sign * np.array(row["force_on_second_world"])
        normals.append(sign * np.array(row["normal_on_second_world"]))
    opposing = bool(normals) and np.min(np.array(normals) @ np.array(normals).T) < -0.5
    return {
        "loaded": bool(selected),
        "opposing": bool(opposing),
        "force_world": force.tolist(),
        "max_slip": max((r["slip_m_s"] for r in selected), default=0.0),
    }


def snapshot(env):
    m, d, _cfg = env.model, env.data, env.config
    rows, total = contacts(m, d)
    supports = {name: support(rows, name, target) for name, target in TARGETS.items()}
    wrist = {}
    poses = {}
    for side, h in env.hands.items():
        rotation = d.site(h["mount"]).xmat.reshape(3, 3)
        wrist[side] = np.r_[
            rotation @ d.sensor(side + "_force").data,
            rotation @ d.sensor(side + "_torque").data,
        ].tolist()
        palm = d.site(h["palm"])
        rung = d.geom("left_ladder_rung_08")
        poses[side] = {
            "position": (palm.xpos - rung.xpos).tolist(),
            "rotation": palm.xmat.tolist(),
        }
    joint_errors = joint_limit_errors(m, d)
    effort = np.abs(d.actuator_force) / np.maximum(
        np.max(np.abs(m.actuator_forcerange), axis=1), 1e-9
    )
    mimic = [
        abs(d.joint(p).qpos[0] - ratio * d.joint(a).qpos[0])
        for h in env.hands.values()
        for p, a, ratio in h["passive"]
    ]
    angle = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]]) * env.reference_rotation.inv()
    return {
        "time": float(d.time),
        "phase": env.phase,
        "wrist_world_on_hand": wrist,
        "supports": supports,
        "contacts": rows,
        "external_contact_force_world": total.tolist(),
        "static_balance_residual_N": (total + [0, 0, -env.mass * 9.81]).tolist(),
        "com": d.subtree_com[m.body("pelvis").id].tolist(),
        "base_pose": d.qpos[:7].tolist(),
        "base_velocity": d.qvel[:6].tolist(),
        "base_drift_m": float(np.linalg.norm(d.qpos[:3] - env.reference_position)),
        "base_angle_rad": float(angle.magnitude()),
        "hand_in_target_rung": poses,
        "actuator_effort": d.actuator_force.tolist(),
        "effort_fraction": effort.tolist(),
        "penetration_m": max((max(0.0, -r["distance"]) for r in rows), default=0.0),
        "joint_positions": d.qpos[7:].tolist(),
        "joint_limit_errors": joint_errors,
        "joint_limit_error_rad": max(joint_errors.values(), default=0.0),
        "mimic_error_rad": float(max(mimic)),
        "numerical_warnings": [int(w.number) for w in d.warning],
        "assistance_world": env.assistance.tolist(),
        "assistance_gain": env.assistance_gain,
        "disturbance_world": env.disturbance.tolist(),
        "unintended_support": unintended(rows),
    }


def joint_limit_errors(model, data):
    q = data.qpos[model.jnt_qposadr[1:]]
    limits = model.jnt_range[1:]
    errors = np.maximum(limits[:, 0] - q, q - limits[:, 1])
    return {model.joint(j + 1).name: float(v) for j, v in enumerate(errors) if v > 1e-9}


def unintended(rows):
    bad = []
    for row in rows:
        if "environment" not in row["regions"] or row["normal_force"] < 1:
            continue
        robot = next((r for r in row["regions"] if r != "environment"), None)
        if robot and TARGETS.get(robot) not in row["geoms"]:
            bad.append(row["geoms"])
    return bad


def failure_reasons(row, cfg, required):
    checks = {
        "drift": row["base_drift_m"] < cfg.max_drift_m,
        "angle": row["base_angle_rad"] < cfg.max_angle_rad,
        "penetration": row["penetration_m"] < cfg.max_penetration_m,
        "joint_limits": row["joint_limit_error_rad"] < cfg.max_joint_error_rad,
        "mimic": row["mimic_error_rad"] < cfg.max_mimic_error_rad,
        "effort": max(row["effort_fraction"]) < cfg.max_effort_fraction,
        "numerical": not any(row["numerical_warnings"]),
        "unintended_support": not row["unintended_support"],
        "velocity": np.linalg.norm(row["base_velocity"][:3]) < 0.05,
        "angular_velocity": np.linalg.norm(row["base_velocity"][3:]) < 0.2,
        "force_balance": np.linalg.norm(row["static_balance_residual_N"]) < 25.0,
    }
    for name in required:
        item = row["supports"][name]
        checks[name] = item["loaded"] and item["max_slip"] < cfg.max_slip_m_s
        if name.endswith("hand"):
            checks[name] &= item["opposing"]
    return [k for k, v in checks.items() if not v]
