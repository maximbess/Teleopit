"""Read-only G6 clearance, contact, effort and force-balance evidence."""

import gzip
import json
from collections import deque

import mujoco
import numpy as np

from teleopit.sim.g1_o6.posture import geom_clearance
from teleopit.sim.o6_diagnostics.events import Events, Saturation
from teleopit.sim.o6_diagnostics.io import write_json

from .g6_gates import required_contact_sides
from .g6_reach import next_tread_reach
from .ladder_observer import LadderObserver


def body_groups(model):
    names = {g: model.geom(g).name for g in range(model.ngeom) if model.geom_contype[g]}
    groups = {
        "torso": [
            g
            for g, n in names.items()
            if any(k in n for k in ("torso", "pelvis", "waist"))
        ]
    }
    for side in ("left", "right"):
        for part in ("knee", "shin", "housing", "forearm"):
            groups[side + "_" + part] = region_geoms(model, names, side, part)
    if not all(groups.values()):
        raise ValueError("required body-clearance geometry is missing")
    return groups


def region_geoms(model, names, side, part):
    patterns = {
        "knee": ("knee",),
        "shin": ("knee",),
        "housing": ("wrist",),
        "forearm": ("elbow", "wrist_roll"),
    }[part]
    selected = [
        g
        for g, n in names.items()
        if n.startswith(side + "_")
        and "_o6_" not in n
        and any(p in n for p in patterns)
    ]
    if part in ("knee", "shin"):
        selected = [
            g for g in selected if (model.geom_pos[g, 2] < -0.1) == (part == "shin")
        ]
    return selected


class BodyClearance:
    def __init__(self, env):
        self.groups = body_groups(env.model)
        self.obstacles = [
            g
            for g in range(env.model.ngeom)
            if "_ladder_" in env.model.geom(g).name and env.model.geom_contype[g]
        ]

    def sample(self, env):
        return {
            name: {
                env.model.geom(o).name: nearest(env, group, o) for o in self.obstacles
            }
            for name, group in self.groups.items()
        }


def nearest(env, group, obstacle):
    measurements = []
    for geom in group:
        lower = (
            np.linalg.norm(env.data.geom_xpos[geom] - env.data.geom_xpos[obstacle])
            - env.model.geom_rbound[geom]
            - env.model.geom_rbound[obstacle]
        )
        if lower >= 0.1:
            measurements.append((0.1, geom, "distance_cap"))
            continue
        try:
            distance, method = geom_clearance(env.model, env.data, geom, obstacle)
        except ValueError:
            distance, method = 0.0, "uncertified_zero"
        measurements.append((distance, geom, method))
    distance, geom, method = min(measurements)
    return {
        "signed_distance_m": distance,
        "robot_geom": env.model.geom(geom).name,
        "method": method,
    }


class G6Observer(LadderObserver):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.body_clearance = BodyClearance(env)
        self.contact_events = Events(
            env.config.physics_dt, env.config.stability_seconds
        )
        self.contact_keys = set()
        self.saturation = Saturation(
            [env.model.actuator(i).name for i in range(env.model.nu)],
            env.config.physics_dt,
            env.config.max_effort_fraction,
        )
        self.contact_stream = self.windows.resources.enter_context(
            gzip.open(path / "physics_contacts.jsonl.gz", "wt")
        )  # noqa: SIM115
        self.minimum_clearance = {}
        self.previous_com = None
        self.selected_side = "left"
        self.phase_token = ""
        self.reachability = next_tread_reach(env)
        self.recent_motion = deque(maxlen=101)

    def __call__(self, env):
        row = super().__call__(env)
        phase = self.phase or env.phase
        self.contact_stream.write(
            json.dumps(
                {
                    "time_s": row["time"],
                    "phase": phase,
                    "phase_time_s": row["time"]
                    - getattr(self, "experiment_phase_start", env.phase_start),
                    "support_targets": env.support_targets,
                    "contacts": row["contacts"],
                }
            )
            + "\n"
        )
        self.observe_body_contacts(row, phase)
        self.saturation.update(
            row["time"],
            dict(zip(self.saturation.rows, row["effort_fraction"])),
            env.phase == "HOLD",
        )
        return row

    def observe_body_contacts(self, row, phase):
        values = {}
        for contact in row["contacts"]:
            if "environment" in contact["regions"] and any(
                "_ladder_" in n for n in contact["geoms"]
            ):
                key = "contact:" + "|".join(contact["geoms"])
                values[key] = max(values.get(key, 0), contact["normal_force"])
        self.contact_keys.update(values)
        self.contact_events.update(
            row["time"],
            phase,
            {k: values.get(k, 0) for k in self.contact_keys},
            {k: 0.05 for k in self.contact_keys},
        )

    def required_contact_sides(self, env):
        return required_contact_sides(self.phase or env.phase, self.selected_side)

    def physics_reasons(self, env, row, hands, states):
        phase_states = {
            k: v if k.split(":")[0] in self.required_contact_sides(env) else True
            for k, v in states.items()
        }
        reasons = super().physics_reasons(env, row, hands, phase_states)
        return (
            [*reasons, "experiment_phase:" + self.phase_token]
            if self.phase_token
            else reasons
        )

    def enrich(self, env, row):
        super().enrich(env, row)
        if env.phase == "HOLD":
            self.recent_motion.append((row["time"], row["base_pose"][:3]))
        clearance = self.body_clearance.sample(env)
        for name, obstacles in clearance.items():
            current = min(v["signed_distance_m"] for v in obstacles.values())
            self.minimum_clearance[name] = min(
                current, self.minimum_clearance.get(name, 0.1)
            )
        row["g6"] = {
            "body_clearance": clearance,
            "force_balance": self.force_balance(env, row),
            "support_targets": dict(env.support_targets),
            "phase": self.phase or env.phase,
        }

    def force_balance(self, env, row):
        jacobian = np.zeros((3, env.model.nv))
        mujoco.mj_jacSubtreeCom(
            env.model, env.data, jacobian, env.model.body("pelvis").id
        )
        velocity = jacobian @ env.data.qvel
        previous = self.previous_com
        self.previous_com = (row["time"], velocity)
        if previous is None or row["time"] <= previous[0]:
            return {"com_velocity_m_s": velocity.tolist(), "dynamic_residual_N": None}
        acceleration = (velocity - previous[1]) / (row["time"] - previous[0])
        residual = (
            np.asarray(row["external_contact_force_world"])
            + env.mass * env.model.opt.gravity
            - env.mass * acceleration
        )
        return {
            "com_velocity_m_s": velocity.tolist(),
            "com_acceleration_m_s2": acceleration.tolist(),
            "dynamic_residual_N": residual.tolist(),
            "interval_s": [previous[0], row["time"]],
            "method": "COM Jacobian velocity difference; current contact force, no wrist internal force counted",
        }

    def report(self):
        return {
            **super().report(),
            "body_minimum_clearance_m": self.minimum_clearance,
            "actuator_saturation": self.saturation.summary(),
            "body_region_convention": "authentic collision meshes; knee/shin split by knee-link local centroid z=-0.1 m; no geometry changed",
            "native_hand_friction": native_friction(self.observed),
            "short_drift": short_drift(self.recent_motion),
            "next_tread_reach": self.reachability,
        }

    def close(self):
        super().close()
        write_json(self.path / "body_contact_events.json", self.contact_events.rows)
        write_json(self.path / "body_clearance_summary.json", self.minimum_clearance)


def native_friction(observed):
    hands = {
        side: {k: v["coefficients"] for k, v in observed.items() if side + "_o6_" in k}
        for side in ("left", "right")
    }
    valid = all(hands.values()) and all(
        np.array_equal(mu, [1.8, 1.8, 0.02, 0.002, 0.002])
        for pairs in hands.values()
        for values in pairs.values()
        for mu in values
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "actual_pairs": hands,
        "expected_effective_vector": [1.8, 1.8, 0.02, 0.002, 0.002],
    }


def short_drift(samples):
    if len(samples) < 2:
        return {"status": "NOT RUN", "maximum_translation_slope_m_s": 1.0}
    times = np.array([s[0] for s in samples])
    positions = np.array([s[1] for s in samples])
    slopes = np.polyfit(times - times[0], positions, 1)[0]
    return {
        "status": "MEASURED",
        "interval_s": [float(times[0]), float(times[-1])],
        "translation_slope_m_s": slopes.tolist(),
        "maximum_translation_slope_m_s": float(np.max(np.abs(slopes))),
        "excursion_m": np.ptp(positions, axis=0).tolist(),
        "scope": "last up to two seconds for ranking only; not a convergence pass",
    }
