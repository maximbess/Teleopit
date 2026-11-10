"""Read-only mechanics. Wrenches are [force N, moment Nm] in world axes."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.o6_grasp.diagnostics import point_velocity


def shift_wrench(wrench, point, origin):
    force, moment = np.asarray(wrench)[:3], np.asarray(wrench)[3:]
    return np.r_[force, moment + np.cross(np.asarray(point) - origin, force)]


def relative_pose(position, rotation, origin, axes):
    return {
        "position_m": (axes.T @ (np.asarray(position) - origin)).tolist(),
        "rotation": (axes.T @ rotation).flatten().tolist(),
    }


def pose_change(position, rotation, reference):
    if reference is None:
        return {"translation_m": None, "angle_rad": None}
    return {
        "translation_m": float(np.linalg.norm(position - reference[0])),
        "angle_rad": float(Rotation.from_matrix(rotation @ reference[1].T).magnitude()),
    }


def subtree(model, root):
    ids = {int(root)}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in ids:
            ids.add(b)
    return sorted(ids)


def momentum(model, data, bodies):
    mass = model.body_mass[bodies]
    center = np.sum(mass[:, None] * data.xipos[bodies], axis=0) / mass.sum()
    linear, angular = np.zeros(3), np.zeros(3)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for b in bodies:
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, b)
        p = model.body_mass[b] * (jacp @ data.qvel)
        r = data.ximat[b].reshape(3, 3)
        linear += p
        angular += r @ (model.body_inertia[b] * (r.T @ (jacr @ data.qvel))) + np.cross(
            data.xipos[b] - center, p
        )
    return {
        "com_world_m": center.tolist(),
        "com_velocity_world_m_s": (linear / mass.sum()).tolist(),
        "linear_kg_m_s": linear.tolist(),
        "angular_kg_m2_s": angular.tolist(),
    }


def contact_rows(model, data):
    result = []
    for i, c in enumerate(data.contact):
        ids = [int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, local)
        r = c.frame.reshape(3, 3)
        velocity = point_velocity(model, data, c.pos, ids[1]) - point_velocity(
            model, data, c.pos, ids[0]
        )
        result.append(
            {
                "body_ids": ids,
                "bodies": [model.body(b).name for b in ids],
                "geoms": [model.geom(g).name for g in (c.geom1, c.geom2)],
                "position_world_m": c.pos.tolist(),
                "normal_world": r[0].tolist(),
                "force_on_second_world_N": (r.T @ local[:3]).tolist(),
                "couple_on_second_world_Nm": (r.T @ local[3:]).tolist(),
                "normal_N": float(local[0]),
                "tangential_N": local[1:3].tolist(),
                "slip_m_s": float(np.linalg.norm((r @ velocity)[1:])),
                "distance_m": float(c.dist),
                "friction": c.friction.tolist(),
            }
        )
    return result


def contact_wrench(rows, bodies, origin):
    total = np.zeros(6)
    for row in rows:
        inside = [b in bodies for b in row["body_ids"]]
        if inside[0] == inside[1]:
            continue
        sign = 1 if inside[1] else -1
        wrench = (
            sign
            * np.r_[row["force_on_second_world_N"], row["couple_on_second_world_Nm"]]
        )
        total += shift_wrench(wrench, row["position_world_m"], origin)
    return total


def support_contacts(rows, bodies, target, minimum):
    return [
        r
        for r in rows
        if target in r["geoms"]
        and set(r["body_ids"]).intersection(bodies)
        and r["normal_N"] >= minimum
    ]


def opposing_contacts(rows, bodies):
    normals = [
        np.array(r["normal_world"]) * (1 if r["body_ids"][1] in bodies else -1)
        for r in rows
    ]
    return bool(normals) and bool(
        np.min(np.array(normals) @ np.array(normals).T) < -0.5
    )


def support_details(rows, bodies, target, origin, minimum=0.05):
    selected = support_contacts(rows, bodies, target, minimum)
    return {
        "loaded": bool(selected),
        "opposing": opposing_contacts(selected, bodies),
        "contact_loss": not bool(selected),
        "wrench_world_about_rung": contact_wrench(selected, bodies, origin).tolist(),
        "normal_N": float(sum(r["normal_N"] for r in selected)),
        "tangential_norm_sum_N": float(
            sum(np.linalg.norm(r["tangential_N"]) for r in selected)
        ),
        "slip_m_s": max((r["slip_m_s"] for r in selected), default=None),
        "positions_world_m": [r["position_world_m"] for r in selected],
        "contact_bodies": [r["bodies"] for r in selected],
    }


def constraint_contributions(model, data):
    kinds = np.array([constraint_kind(model, data, i) for i in range(data.nefc)])
    result = {}
    for kind in ("contact", "joint_limit", "mimic", "other"):
        force = np.where(kinds == kind, data.efc_force[: data.nefc], 0.0)
        generalized = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(model, data, generalized, force)
        result[kind] = generalized
    return result


def constraint_kind(model, data, i):
    kind = int(data.efc_type[i])
    if kind in (5, 6, 7):
        return "contact"
    if kind == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
        return "joint_limit"
    if (
        kind == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        and model.eq_type[data.efc_id[i]] == mujoco.mjtEq.mjEQ_JOINT
    ):
        return "mimic"
    return "other"


def mimic_details(model, data):
    result = []
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        a, b = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        q1 = model.jnt_qposadr[a]
        q2 = model.jnt_qposadr[b]
        x = data.qpos[q2] - model.qpos0[q2]
        value = float(np.polynomial.polynomial.polyval(x, model.eq_data[i, :5]))
        result.append(
            {
                "passive": model.joint(a).name,
                "active": model.joint(b).name,
                "offset_rad": float(model.eq_data[i, 0]),
                "residual_rad": float(data.qpos[q1] - model.qpos0[q1] - value),
            }
        )
    return result


def joint_details(model, data):
    contributions = constraint_contributions(model, data)
    joints = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        joints[model.joint(j).name] = {
            "position_rad": float(data.qpos[q]),
            "velocity_rad_s": float(data.qvel[v]),
            "lower_violation_rad": float(max(0, model.jnt_range[j, 0] - data.qpos[q])),
            "upper_violation_rad": float(max(0, data.qpos[q] - model.jnt_range[j, 1])),
            "actuator_Nm": float(data.qfrc_actuator[v]),
            "total_constraint_Nm": float(data.qfrc_constraint[v]),
            "constraint_Nm": {k: float(f[v]) for k, f in contributions.items()},
        }
    return {"joints": joints, "mimic": mimic_details(model, data)}


def actuator_details(model, data, desired, rate_limited):
    result = {}
    for i in range(model.nu):
        j = model.actuator_trnid[i, 0]
        q = model.jnt_qposadr[j]
        limit = max(abs(model.actuator_forcerange[i]))
        result[model.actuator(i).name] = {
            "joint": model.joint(j).name,
            "desired_rad": float(desired[i]),
            "rate_limited_rad": float(rate_limited[i]),
            "applied_rad": float(data.ctrl[i]),
            "target_error_rad": float(data.ctrl[i] - data.qpos[q]),
            "effort_Nm": float(data.actuator_force[i]),
            "effort_fraction": float(abs(data.actuator_force[i]) / limit),
        }
    return result
