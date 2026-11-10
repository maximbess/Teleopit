"""Named physics-step signals shared by coupled and isolated experiments."""

import mujoco

from .mechanics import contact_rows, mimic_details, support_details


def joint_signals(model, data, cfg):
    values = {}
    thresholds = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[j]:
            continue
        name = model.joint(j).name
        position = data.qpos[model.jnt_qposadr[j]]
        for label, value in [
            ("lower_limit", model.jnt_range[j, 0] - position),
            ("upper_limit", position - model.jnt_range[j, 1]),
        ]:
            key = label + ":" + name
            values[key] = float(max(0, value))
            thresholds[key] = cfg.max_joint_error_rad
    for relation in mimic_details(model, data):
        key = "mimic:" + relation["passive"]
        values[key] = abs(relation["residual_rad"])
        thresholds[key] = cfg.max_mimic_error_rad
    for i in range(model.nu):
        key = "effort:" + model.actuator(i).name
        values[key] = float(
            abs(data.actuator_force[i]) / max(abs(model.actuator_forcerange[i]))
        )
        thresholds[key] = cfg.max_effort_fraction
    return values, thresholds


def hand_signals(h, data, cfg):
    values, thresholds = joint_signals(h.model, data, cfg)
    contacts = contact_rows(h.model, data)
    values["penetration"] = max(
        0.0, max((-r["distance_m"] for r in contacts), default=0.0)
    )
    thresholds["penetration"] = cfg.max_penetration_m
    for name, target in h.telemetry.targets.items():
        support = support_details(
            contacts, h.bodies, target, data.geom(target).xpos, cfg.min_normal_N
        )
        values["contact_loss:" + name] = float(support["contact_loss"])
        thresholds["contact_loss:" + name] = 1.0
        values["opposition_loss:" + name] = float(not support["opposing"])
        thresholds["opposition_loss:" + name] = 1.0
        values["slip:" + name] = support["slip_m_s"]
        thresholds["slip:" + name] = cfg.max_slip_m_s
    return values, thresholds
