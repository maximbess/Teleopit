"""Bounded G6 knee-clearance experiment with native contact and geometric evidence."""

import gzip
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .g6_campaign import STAGES
from .g6_ground_reaction_experiment import (
    GroundReactionObserver, candidates as reaction_candidates, contact_actors, passes as reaction_passes,
)
from .g6_knee import CONFIGURATIONS, GAP_TARGET_M, RUNG, KneeClearance, KneeRung
from .g6_trial import run_candidate
from .g6_whole_body_experiment import WholeBodyCampaign, main

PARENT_TRACE = "df5f1366ec58b2d01c6896550ada5d58d03f84b35934110450f8f1c0c535e5b2"
STANCE = tuple(name for name, (law, _) in CONFIGURATIONS.items() if law == "stance")
COORDINATED = tuple(name for name, (law, _) in CONFIGURATIONS.items() if law == "palms")
NATIVE_FILE = "knee_rung3.f64.gz"


def candidates():
    parent = next(c for c in reaction_candidates() if c.id == "individual_16")
    parent = replace(parent, id="control", parent=None, family="control")
    return [parent, *[replace(parent, id=name, parent="control", family=law,
                              controller=KneeClearance(coordination=law, stance_gain=gain))
                      for name, (law, gain) in CONFIGURATIONS.items()]]


def specification():
    return {"schema": "g6_knee_clearance_v1", "geometry": "G6", "native_mu": 1.8, "stages": STAGES,
            "candidates": {c.id: c.record() for c in candidates()}, "candidate_limit": 4,
            "budgets": {s: (4 if s == "probe" else 3) for s in STAGES}, "parent_probe_steps_sha256": PARENT_TRACE,
            "stance_clearance_target_m": GAP_TARGET_M, "stance": STANCE, "coordinated": COORDINATED,
            "qualification": "positive right knee/rung-3 distance and zero pair contacts at every physics and control sample, both force peaks <25 N, uninterrupted left finger loading and all original gates"}


class ClearanceTrace:
    def __init__(self):
        self.samples, self.contact_steps = 0, 0
        self.minimum_m, self.peak_contact_N = .1, 0.
        self.closest_s, self.first_contact_s = None, None

    def observe(self, elapsed, distance, contacts):
        self.samples += 1
        if not np.isfinite(distance):
            distance = 0.
        if self.closest_s is None or distance < self.minimum_m:
            self.minimum_m, self.closest_s = distance, elapsed
        if contacts:
            self.contact_steps += 1
            self.peak_contact_N = max(self.peak_contact_N, max(c["normal_force"] for c in contacts))
            if self.first_contact_s is None:
                self.first_contact_s = elapsed


def read_knee_rows(path, nq):
    """Little-endian float64 records: model time, integrated gap, contact count, full qpos."""
    with gzip.open(path / NATIVE_FILE, "rb") as stream:
        return np.frombuffer(stream.read(), dtype="<f8").reshape(-1, nq + 3)


class KneeObserver(GroundReactionObserver):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.knee = KneeRung(env.model)
        self.kinematics = mujoco.MjData(env.model)
        self.native, self.control = ClearanceTrace(), ClearanceTrace()
        self.stream = self.windows.resources.enter_context(gzip.open(path / NATIVE_FILE, "wb", compresslevel=1))

    def physics_reasons(self, env, row, hands, states):
        if env.phase == "HOLD":
            # mj_step integrates qpos after its collision solve. Recompute the
            # integrated geometry on a private state and retain the solver's contacts.
            self.kinematics.qpos[:] = env.data.qpos
            mujoco.mj_kinematics(env.model, self.kinematics)
            gap = self.knee.distance(self.kinematics)
            contacts = self.knee.contacts(row["contacts"])
            self.native.observe(row["time"] - env.phase_start, gap, contacts)
            self.stream.write(np.asarray([row["time"], gap, len(contacts), *env.data.qpos], dtype="<f8").tobytes())
        return super().physics_reasons(env, row, hands, states)

    def enrich(self, env, row):
        super().enrich(env, row)
        if row["phase"] == "HOLD":
            gap = row["g6"]["body_clearance"]["right_knee"][RUNG]["signed_distance_m"]
            self.control.observe(row["hold_time_s"], gap, self.knee.contacts(row["contacts"]))

    def report(self):
        return {**super().report(), "knee_rung3": {"rung": RUNG, "geoms": sorted(self.knee.names),
                "native": vars(self.native), "control": vars(self.control)}}


def run_knee(payload, output, seconds, dt):
    return run_candidate(payload, output, seconds, dt, observer_factory=KneeObserver)


def clear(trace, expected_samples):
    gap = trace.get("minimum_m", 0.)
    return (expected_samples > 0 and trace.get("samples") == expected_samples and np.isfinite(gap)
            and gap > 0. and trace.get("contact_steps") == 0)


def passes(result, stage):
    audit = result.get("physics_audit", {})
    knee = audit.get("knee_rung3", {})
    controls = audit.get("ground_reaction", {}).get("control", {}).get("samples", 0)
    return (clear(knee.get("native", {}), audit.get("scored_steps", 0))
            and clear(knee.get("control", {}), controls) and reaction_passes(result, stage))


def actors(record):
    return {**contact_actors(record), "right_knee_to_rung3": record["result"]["physics_audit"]["knee_rung3"]}


class KneeCampaign(WholeBodyCampaign):
    trial_function = staticmethod(run_knee)
    spec_factory = staticmethod(specification)
    passes = staticmethod(passes)
    actors = staticmethod(actors)
    first_pair, second_pair = STANCE, COORDINATED
    audit_key = "stance_only_premise"
    premise = "Foot and knee feedback with a fixed pelvis reference can also settle both palms."
    replacement = "Use the observed common palm pose error to adjust the bounded pelvis wrench reference."
    module = "teleopit.sim.g1_o6_support.g6_knee_experiment"


if __name__ == "__main__":
    main(KneeCampaign, Path("outputs/g1_o6/g6_knee_clearance"))
