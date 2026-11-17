"""Additional acquisition evidence; original G6 scoring remains unchanged."""

from .g6_observer import G6Observer
from .g6_arm import inhibition_reasons


class AcquisitionAudit:
    def __init__(self, dt, required_seconds):
        self.dt, self.required_seconds = dt, required_seconds
        self.phases = {}

    def observe(self, phase, hands, reasons):
        values = self.phases.setdefault(phase, {
            "samples": 0, "valid_samples": 0, "tail_samples": 0,
            "longest_valid_samples": 0, "last_hands": {}, "last_inhibitions": [],
            "peak_thumb_normal_N": {side: 0. for side in hands}})
        valid = all(hand["quality"]["valid"] for hand in hands.values()) and not reasons
        values["samples"] += 1
        values["valid_samples"] += int(valid)
        values["tail_samples"] = values["tail_samples"] + 1 if valid else 0
        values["longest_valid_samples"] = max(values["longest_valid_samples"], values["tail_samples"])
        values["last_hands"] = {side: hand["quality"].copy() for side, hand in hands.items()}
        values["last_inhibitions"] = list(reasons)
        for side, hand in hands.items():
            values["peak_thumb_normal_N"][side] = max(values["peak_thumb_normal_N"][side], hand["quality"]["thumb_normal_N"])

    def ready(self, phase):
        return self.phases.get(phase, {}).get("tail_samples", 0) * self.dt >= self.required_seconds - 1e-9

    def report(self):
        return {"ready_for_removal": self.ready("ACQUIRE"),
                "ready_for_hold": self.ready("REMOVE_ASSISTANCE"),
                "required_continuous_seconds": self.required_seconds,
                "physics_dt": self.dt, "phases": self.phases}


class AcquisitionObserver(G6Observer):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.acquisition = AcquisitionAudit(env.config.physics_dt, env.config.stability_seconds)

    def physics_reasons(self, env, row, hands, states):
        self.acquisition.observe(env.phase, hands, inhibition_reasons(env, row))
        return super().physics_reasons(env, row, hands, states)

    def report(self):
        return {**super().report(), "grasp_acquisition": self.acquisition.report()}
