"""Carry the wrist-qualified pose's native gates into coupled load and transfer trials."""

import mujoco
import numpy as np

from .g6_ground_reaction import BALANCE_LIMIT_N
from .g6_knee import KneeRung, RUNG
from .g6_observer import G6Observer
from .pose_contacts import LOADED_N


def retention_losses(hands, required, residual, gap, knee_contact):
    losses = []
    for side in sorted(required):
        hand = hands[side]["quality"]
        if not hand["thumb_opposition"] or hand["thumb_normal_N"] < LOADED_N:
            losses.append(side + "_thumb")
    if "left" in required and hands["left"]["quality"]["minimum_finger_normal_N"] < LOADED_N:
        losses.append("left_fingers")
    if not residual < BALANCE_LIMIT_N:
        losses.append("force_balance")
    if not gap > 0 or knee_contact:
        losses.append("right_knee_rung3")
    return losses


class WristTransferObserver(G6Observer):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.knee = KneeRung(env.model)
        self.kinematics = mujoco.MjData(env.model)
        self.first_retention_failure = {}
        self.samples = {"physics": 0, "control": 0}

    def score(self, env, row, hands, gap, source):
        self.samples[source] += 1
        reasons = retention_losses(hands, self.required_contact_sides(env),
                                   np.linalg.norm(row["static_balance_residual_N"]), gap,
                                   bool(self.knee.contacts(row["contacts"])))
        for reason in reasons:
            self.first_retention_failure.setdefault(reason, row["time"])
        return reasons

    def physics_reasons(self, env, row, hands, states):
        reasons = super().physics_reasons(env, row, hands, states)
        if env.phase == "HOLD":
            self.kinematics.qpos[:] = env.data.qpos
            mujoco.mj_kinematics(env.model, self.kinematics)
            reasons += self.score(env, row, hands, self.knee.distance(self.kinematics), "physics")
        return reasons

    def enrich(self, env, row):
        super().enrich(env, row)
        if row["phase"] == "HOLD":
            gap = row["g6"]["body_clearance"]["right_knee"][RUNG]["signed_distance_m"]
            self.score(env, row, row["pose_search"]["hands"], gap, "control")

    def report(self):
        return {**super().report(), "retention_samples": self.samples,
                "first_retention_failure_model_s": self.first_retention_failure}
