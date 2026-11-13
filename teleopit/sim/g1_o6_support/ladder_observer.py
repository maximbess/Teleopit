"""Ladder/material evidence layered on the unchanged anatomical HOLD observer."""

import gzip
import json

import mujoco
import numpy as np

from teleopit.sim.g1_o6.diagnostics import region
from teleopit.sim.o6_diagnostics.io import write_json

from .pose_observer import PoseObserver
from .pose_trial import reset_report


class LadderObserver(PoseObserver):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.applied = self.windows.resources.enter_context(
            gzip.open(path / "applied_forces.jsonl.gz", "wt")  # noqa: SIM115 - owned by the inherited ExitStack
        )
        self.observed = {}
        self.entry_saved = False
        self.previous_phase = None

    def physics_reasons(self, env, row, hands, states):
        reasons = super().physics_reasons(env, row, hands, states)
        if env.phase != self.previous_phase:
            reasons.append("phase_transition:" + env.phase)
            self.previous_phase = env.phase
        return reasons

    def __call__(self, env):
        row = super().__call__(env)
        if env.phase == "HOLD":
            self.applied.write(
                json.dumps(
                    {
                        "time_s": float(env.data.time),
                        "qfrc_applied": env.data.qfrc_applied.tolist(),
                        "xfrc_applied": env.data.xfrc_applied.tolist(),
                    }
                )
                + "\n"
            )
        for contact in row["contacts"]:
            if any("_ladder_rung_" in name for name in contact["geoms"]):
                key = "|".join(sorted(contact["geoms"]))
                record = self.observed.setdefault(
                    key, {"coefficients": [], "samples": 0, "phases": []}
                )
                if contact["friction"] not in record["coefficients"]:
                    record["coefficients"].append(contact["friction"])
                if env.phase not in record["phases"]:
                    record["phases"].append(env.phase)
                record["samples"] += 1
        return row

    def enrich(self, env, row):
        super().enrich(env, row)
        row["ladder_material"] = {"support_wrenches": support_wrenches(env)}
        if env.phase == "HOLD" and not self.entry_saved:
            write_json(
                self.path / "hold_entry.json",
                dict(
                    reset_report(env),
                    root_pose=env.data.qpos[:7].tolist(),
                    relative_poses=row["diagnostic_v1"]["rung_relative_poses_v1"],
                    posture=row["posture"],
                    time_s=float(env.data.time),
                ),
            )
            self.entry_saved = True

    def close(self):
        super().close()
        write_json(self.path / "observed_friction.json", self.observed)


def support_wrenches(env):
    result = {name: np.zeros(6) for name in env.support_targets}
    for index, contact in enumerate(env.data.contact):
        add_support_wrench(env, index, contact, result)
    return {
        name: {
            "force_world_N": wrench[:3].tolist(),
            "moment_world_at_rung_center_Nm": wrench[3:].tolist(),
        }
        for name, wrench in result.items()
    }


def add_support_wrench(env, index, contact, result):
    model, data = env.model, env.data
    geoms = (contact.geom1, contact.geom2)
    names = [model.geom(g).name for g in geoms]
    regions = [region(model, int(model.geom_bodyid[g])) for g in geoms]
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, wrench)
    rotation = contact.frame.reshape(3, 3).T
    for support, target in env.support_targets.items():
        if support in regions and target in names:
            sign = 1 if regions[1] == support else -1
            force = sign * rotation @ wrench[:3]
            moment = sign * rotation @ wrench[3:] + np.cross(
                contact.pos - data.geom(target).xpos, force
            )
            result[support] += np.r_[force, moment]
