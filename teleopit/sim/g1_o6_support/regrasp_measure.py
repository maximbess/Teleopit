"""Native solver samples. External rung forces, fixture and internal wrist loads are separate."""

from collections import defaultdict

import mujoco
import numpy as np

from teleopit.sim.g1_o6.diagnostics import contacts
from teleopit.sim.g1_o6.posture import geom_clearance

from .g6_grip_feedback import SIDES
from .kinematics import Clearance
from .pose_contacts import (
    FINGERS,
    LOADED_N,
    contact_partitions,
    external_hand_contacts,
    ordered_groups,
    thumb_opposition,
)
from .regrasp_actions import FIXTURE

SUPPORTS = ("left_hand", "right_hand", "left_foot", "right_foot")
STATES = ("air", "wrong_surface", "touch", "grasp")
CONTACT_COLUMNS = (
    "step",
    "geom1",
    "geom2",
    "normal_N",
    "Fz_on_geom2_N",
    "normal_x",
    "normal_y",
    "normal_z",
    "world_z_m",
    "slip_m_s",
    "distance_m",
)


def hand_state(details, height):
    target, loaded, groups = contact_partitions(details)
    if len(target) != len(details):
        return "wrong_surface"
    if not target:
        return "air"
    return (
        "grasp"
        if ordered_groups(groups, height) and thumb_opposition(loaded)
        else "touch"
    )


class Gap:
    """Certify all eligible pairs. World AABBs reject distant pairs conservatively."""

    def __init__(self, env, side):
        clearance = Clearance(env, side)
        self.hand = clearance.hand
        self.a, self.b = np.asarray(clearance.pairs).T
        self.model = env.model

    def measure(self, data, centers, extents):
        gap = (
            np.abs(centers[self.a] - centers[self.b])
            - extents[self.a]
            - extents[self.b]
        )
        lower = np.max(gap, axis=1)
        # Infinite ground planes have no finite AABB. Always query them.
        lower[self.model.geom_type[self.b] == mujoco.mjtGeom.mjGEOM_PLANE] = -np.inf
        best = 0.03
        for i in np.flatnonzero(lower < best):
            if lower[i] < best:
                try:
                    distance, _ = geom_clearance(self.model, data, self.a[i], self.b[i])
                except ValueError:
                    distance = 0.0  # Separated mesh with no geometric certificate.
                best = min(best, distance)
        return best


def fixture_wrench(env, fixture):
    wrench = np.zeros(6)
    if fixture >= 0:
        d, m = env.data, env.model
        rows = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (
            d.efc_id == fixture
        )
        generalized = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, generalized, np.where(rows, d.efc_force, 0.0))
        wrench[:3] = generalized[:3]
        wrench[3:] = d.body("pelvis").xmat.reshape(3, 3) @ generalized[3:6]
    return wrench


class NativeTrace:
    def __init__(self, env, side):
        model = env.model
        self.side, self.gap = side, Gap(env, side)
        self.fixture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, FIXTURE)
        self.geom_ids = {model.geom(g).name: g for g in range(model.ngeom)}
        self.limits = model.jnt_range[1:]
        self.addr = model.jnt_qposadr[1:]
        self.effort = np.max(abs(model.actuator_forcerange), axis=1)
        self.passive = [
            (model.joint(p).qposadr[0], model.joint(a).qposadr[0], r)
            for h in env.hands.values()
            for p, a, r in h["passive"]
        ]
        self.samples = defaultdict(list)
        self.detail = []
        self.phase = "setup"
        self.check_gap = False

    def record_contacts(self, env, rows):
        forces, slip, loaded = np.zeros(4), np.zeros(4), np.zeros(4, dtype=bool)
        other, wrong = 0.0, 0
        detail = []
        for row in rows:
            geoms = tuple(self.geom_ids[n] for n in row["geoms"])
            fz = row["force_on_second_world"][2]
            detail.append(
                (
                    len(self.detail),
                    *geoms,
                    row["normal_force"],
                    fz,
                    *row["normal_on_second_world"],
                    row["position"][2],
                    row["slip_m_s"],
                    row["distance"],
                )
            )
            if row["regions"].count("environment") != 1:
                continue
            i = row["regions"].index("environment")
            robot, target = row["regions"][1 - i], row["geoms"][i]
            signed = fz if i == 0 else -fz
            if env.support_targets.get(robot) == target:
                k = SUPPORTS.index(robot)
                forces[k] += signed  # Include sub-threshold contacts in force balance.
                if row["normal_force"] >= LOADED_N:
                    loaded[k] = True
                    slip[k] = max(slip[k], row["slip_m_s"])
            else:
                other += signed
                wrong += int(row["normal_force"] >= 1.0)
        self.detail.append(
            np.asarray(detail, dtype=np.float64).reshape(-1, len(CONTACT_COLUMNS))
        )
        return forces, slip, loaded, other, wrong

    def hand_sample(self, env, rows):
        states, loads, counts, opposed = [], [], [], []
        for side in SIDES:
            selected = external_hand_contacts(env, rows, side)
            _, loaded, groups = contact_partitions(selected)
            opposed.append(thumb_opposition(loaded))
            height = env.data.geom(env.support_targets[side + "_hand"]).xpos[2]
            states.append(STATES.index(hand_state(selected, height)))
            loads.append([sum(c["normal_force_N"] for c in groups[g]) for g in FINGERS])
            counts.append(len(selected))
        return states, loads, counts, opposed

    def clearance(self, env, rows):
        # This counts contacts with the other hand or the robot as well as the ladder.
        ids = self.gap.hand
        external = [
            r["distance"]
            for r in rows
            if sum(self.geom_ids[n] in ids for n in r["geoms"]) == 1
        ]
        if external or not self.check_gap:
            return len(external), min(external, default=np.nan)
        m, d = env.model, env.data
        rotations = d.geom_xmat.reshape(-1, 3, 3)
        centers = d.geom_xpos + np.einsum("gij,gj->gi", rotations, m.geom_aabb[:, :3])
        extents = np.einsum("gij,gj->gi", abs(rotations), m.geom_aabb[:, 3:])
        return 0, self.gap.measure(d, centers, extents)

    def __call__(self, env):
        d, m = env.data, env.model
        rows, total = contacts(m, d)
        forces, slip, loaded, other, wrong = self.record_contacts(env, rows)
        states, loads, hand_count, opposed = self.hand_sample(env, rows)
        count, gap = self.clearance(env, rows)
        pelvis = d.body("pelvis")
        mujoco.mj_subtreeVel(m, d)
        q = d.qpos[self.addr]
        deviation = np.maximum(self.limits[:, 0] - q, q - self.limits[:, 1])
        fixture = fixture_wrench(env, self.fixture)
        angle = pelvis_angle(pelvis.xquat, env.initial_qpos[3:7])
        sample = {
            "time": d.time - env.config.physics_dt,
            "phase": self.phase,
            "fz": forces,
            "other_fz": other,
            "external_force": total,
            "fixture": fixture,
            "state": states,
            "finger_load": loads,
            "hand_count": hand_count,
            "thumb_opposed": opposed,
            "external_count": count,
            "gap": gap,
            "loaded": loaded,
            "slip": slip,
            "wrong": wrong,
            "pelvis": pelvis.xpos.copy(),
            "pelvis_quat": pelvis.xquat.copy(),
            "drift": np.linalg.norm(pelvis.xpos - env.reference_position),
            "tilt": angle,
            "effort": max(abs(d.actuator_force) / self.effort),
            "penetration": max((-r["distance"] for r in rows), default=0.0),
            "joint_excess": max(0.0, max(deviation)),
            "mimic": max(abs(d.qpos[p] - r * d.qpos[a]) for p, a, r in self.passive),
            "warnings": sum(w.number for w in d.warning),
            "base_speed": np.linalg.norm(d.qvel[:3]),
            "base_angular_speed": np.linalg.norm(d.qvel[3:6]),
            "momentum": env.mass * d.subtree_linvel[pelvis.id, 2],
            "applied": max(np.max(abs(d.qfrc_applied)), np.max(abs(d.xfrc_applied))),
            "assist_gain": env.assistance_gain,
            "digits": d.actuator_length[
                np.concatenate([env.hand_act[s] for s in SIDES])
            ]
            .reshape(2, 6)
            .copy(),
            "wrist": np.array([d.site(env.hands[s]["mount"]).xpos for s in SIDES]),
        }
        for name, value in sample.items():
            self.samples[name].append(value)

    def save(self, directory):
        np.savez_compressed(directory / "physics.npz", **self.samples)
        np.savez_compressed(
            directory / "contacts.npz",
            columns=CONTACT_COLUMNS,
            contacts=np.concatenate(self.detail),
        )


def pelvis_angle(quaternion, initial):
    """Quaternion chord length stays accurate even for a microradian fixture rotation."""
    q = np.asarray(quaternion) / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    ref = np.asarray(initial) / np.linalg.norm(initial)
    chord = np.minimum(
        np.linalg.norm(q - ref, axis=-1), np.linalg.norm(q + ref, axis=-1)
    )
    return 4 * np.arcsin(np.clip(chord / 2, 0.0, 1.0))
