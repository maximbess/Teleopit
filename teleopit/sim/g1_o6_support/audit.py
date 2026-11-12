"""Read-only physics-step evidence, retaining native solver-step contact forces."""

import numpy as np

from teleopit.sim.g1_o6.diagnostics import snapshot
from teleopit.sim.g1_o6.posture_report import housing_contact
from teleopit.sim.o6_diagnostics.events import Events
from teleopit.sim.o6_diagnostics.nominal import gate_measurements


class PhysicsAudit:
    def __init__(self, env):
        self.events = Events(env.config.physics_dt, env.config.stability_seconds)
        self.all_events = Events(env.config.physics_dt, env.config.stability_seconds)
        self.optional_supports = set()
        self.scored_steps = 0
        self.applied_steps = 0
        self.prohibited_steps = 0
        self.housing_steps = 0
        self.first_contacts = []
        self.phase = None

    def __call__(self, env):
        # No forward call and no state writes: contact forces are those from the
        # just-integrated solver interval, at resolution dt (at most one dt old).
        row = snapshot(env)
        values, thresholds = gate_measurements(row, env.config)
        values.update(
            numerical=float(any(row["numerical_warnings"])),
            unintended=float(bool(row["unintended_support"])),
            housing=float(housing_contact(row)),
        )
        thresholds.update(numerical=1, unintended=1, housing=1)
        phase = self.phase or env.phase
        self.all_events.update(row["time"], phase, values, thresholds)
        if env.phase == "HOLD":
            for name in self.optional_supports:
                for prefix in ("contact_loss:", "opposition_loss:"):
                    values.pop(prefix + name, None)
            self.scored(env, row, values, thresholds, phase)
        return row

    def scored(self, env, row, values, thresholds, phase):
        self.scored_steps += 1
        applied = bool(np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied))
        values["external_applied_force"] = float(applied)
        thresholds["external_applied_force"] = 1
        self.applied_steps += int(applied)
        self.prohibited_steps += int(bool(row["unintended_support"]))
        self.housing_steps += int(values["housing"])
        if (row["unintended_support"] or values["housing"]) and not self.first_contacts:
            self.first_contacts = row["contacts"]
        self.events.update(row["time"], phase, values, thresholds)

    def report(self):
        return {
            "scored_steps": self.scored_steps,
            "nonzero_applied_steps": self.applied_steps,
            "unintended_support_steps": self.prohibited_steps,
            "housing_contact_steps": self.housing_steps,
            "first_prohibited_contacts": self.first_contacts,
            "first_crossings": self.events.first,
            "first_persistent": self.events.persistent,
            "extrema": self.events.extrema,
            "event_resolution_s": self.events.dt,
            "contact_timing": "native solver interval, at most one physics step old",
            "no_external_support": self.scored_steps > 0 and self.applied_steps == 0,
        }
