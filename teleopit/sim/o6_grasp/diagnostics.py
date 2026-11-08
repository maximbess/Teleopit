"""Privileged contact diagnostics; none of these fields enter the actor."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .controller import Mode


def point_velocity(model, data, point, body):
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, linear, angular, point, body)
    return linear @ data.qvel


def contact_diagnostics(env):
    model, data = env.model, env.data
    origin = data.site("palm_frame").xpos
    wrench = np.zeros(6)  # force then moment, world axes, palm_frame origin
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        geom_ids = (contact.geom1, contact.geom2)
        if env.rung_geom not in geom_ids:
            continue
        hand_geom = geom_ids[1] if geom_ids[0] == env.rung_geom else geom_ids[0]
        sign = 1 if hand_geom == contact.geom2 else -1
        local_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, local_force)
        frame = contact.frame.reshape(3, 3)
        force = sign * frame.T @ local_force[:3]
        torque = sign * frame.T @ local_force[3:]
        wrench[:3] += force
        wrench[3:] += torque + np.cross(contact.pos - origin, force)
        body = model.geom_bodyid[hand_geom]
        velocity = frame @ point_velocity(model, data, contact.pos, body)
        contacts.append(
            {
                "body": model.body(body).name,
                "geom": model.geom(hand_geom).name,
                "position": contact.pos.tolist(),
                "normal_world": (sign * frame[0]).tolist(),
                "normal_force": float(local_force[0]),
                "tangential_force": local_force[1:3].tolist(),
                "force_world": force.tolist(),
                "moment_world": torque.tolist(),
                "slip_speed": float(np.linalg.norm(velocity[1:])),
                "distance": float(contact.dist),
            }
        )
    return contacts, wrench


def relative_palm(env):
    palm, rung = env.data.site("palm_frame"), env.data.site("rung_frame")
    rotation = rung.xmat.reshape(3, 3).T
    return rotation @ (palm.xpos - rung.xpos), rotation @ palm.xmat.reshape(3, 3)


class Verification:
    def __init__(self):
        self.reference = None
        self.stable_time = 0.0
        self.sliding_time = 0.0
        self.bad_time = 0.0
        self.status = "UNVERIFIED"

    def update(self, env, mode, unsupported, contacts, penetration):
        cfg = env.config
        position, rotation = relative_palm(env)
        if self.reference is None or mode in (Mode.OPEN, Mode.CLOSE):
            self.reference = position.copy(), rotation.copy()
        drift = float(np.linalg.norm(position - self.reference[0]))
        angle = float(Rotation.from_matrix(rotation @ self.reference[1].T).magnitude())
        plausible, slip = grasp_surfaces(contacts, cfg.min_normal_force)
        effort = float(np.max(np.abs(env.data.actuator_force)) / cfg.effort_limit)
        reasons = verification_reasons(
            cfg, plausible, drift, angle, slip, effort, penetration
        )
        stable = not reasons
        self.stable_time = (
            self.stable_time + cfg.control_dt if unsupported and stable else 0.0
        )
        self.bad_time = (
            self.bad_time + cfg.control_dt if unsupported and not stable else 0.0
        )
        self.sliding_time = (
            self.sliding_time + cfg.control_dt
            if unsupported and slip > cfg.max_slip_speed
            else 0.0
        )
        self.status = self._status(cfg, mode, unsupported, drift)
        return {
            "verification": self.status,
            "translation_drift": drift,
            "rotation_drift": angle,
            "contact_slip_speed": slip,
            "sustained_sliding": self.sliding_time >= cfg.verification_window,
            "stable_duration": self.stable_time,
            "plausible_contacts": plausible,
            "effort_fraction": effort,
            "verification_reasons": reasons,
        }

    def _status(self, cfg, mode, unsupported, drift):
        if mode == Mode.RELEASE:
            return "RELEASING"
        if not unsupported:
            return "FIXTURE_ASSISTED"
        if drift > cfg.pullout_distance:
            return "FAILED"
        if self.sliding_time >= cfg.verification_window:
            return "SLIPPING"
        if self.bad_time >= cfg.verification_window:
            return "FAILED"
        if self.stable_time >= cfg.verification_window:
            return "VERIFIED"
        return "UNVERIFIED"


def grasp_surfaces(contacts, minimum_force):
    loaded = [c for c in contacts if c["normal_force"] >= minimum_force]
    regions = {c["body"] for c in loaded}
    plausible = len(regions) >= 2 and any("distal" in name for name in regions)
    return plausible, max((c["slip_speed"] for c in loaded), default=0.0)


def verification_reasons(cfg, plausible, drift, angle, slip, effort, penetration):
    checks = {
        "grasp_surfaces": plausible,
        "translation": drift < cfg.max_translation,
        "rotation": angle < cfg.max_rotation,
        "sliding": slip < cfg.max_slip_speed,
        "actuator_loading": effort < cfg.max_effort_fraction,
        "penetration": penetration < cfg.max_penetration,
    }
    return [name for name, passed in checks.items() if not passed]


def reward_terms(mode, info, action_change):
    holding = mode == Mode.HOLD
    moving = mode in (Mode.CLOSE, Mode.HOLD)
    return {
        "acquisition": float(mode == Mode.CLOSE and info["plausible_contacts"]),
        "stable_support": float(holding and info["verification"] == "VERIFIED"),
        "relative_slip": -float(moving) * info["contact_slip_speed"],
        "excessive_effort": -(max(0.0, info["effort_fraction"] - 0.8) ** 2),
        "abrupt_action": -float(action_change),
        "authorized_release": float(mode == Mode.RELEASE and info["release_clear"]),
    }
