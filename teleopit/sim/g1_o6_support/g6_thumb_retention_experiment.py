"""Bounded right-thumb experiments against the 52.029 s native G6 failure."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from .g6_campaign import STAGES
from .g6_knee_experiment import experiment_spec
from .g6_pipeline import Pipeline
from .g6_thumb_retention import ThumbRetention
from .g6_whole_body_experiment import main
from .g6_wrist_grip_experiment import WristCampaign, candidates as wrist_candidates, passes as wrist_passes

PARENT_TRACE = "6cc27ee0c3165e6f5a9adc85a9e1756674dfad45af74ea77e083fd1233948797"
BASE_MARGIN = {5.: .033983912923574104, 40.: .026888131193427277}
BASE_CAPTURE_DRIFT_M = .005958194215103542  # tangential material drift, 5.000 to 52.028


def candidates():
    parent = replace(wrist_candidates()[1], id="control", parent=None, family="control")
    inward = [replace(parent, id=f"inward_{mm}mm", parent="control", family="tread_edge",
                      pose=replace(parent.pose, right_grip_offset_m=(mm/1000, 0., 0.))) for mm in (2, 4)]
    arms = [replace(parent, id=name, parent="control", family=name, controller=ThumbRetention(right_refinement=name))
            for name in ("preorient", "recenter")]
    return [parent, *inward, *arms]


def specification():
    spec = experiment_spec("g6_thumb_retention_v1", candidates())
    return {**spec, "stages": {**STAGES, "repeat60": (60, .001)},
            "budgets": {**spec["budgets"], "repeat60": 4}, "parent_probe_steps_sha256": PARENT_TRACE,
            "reference_loss_s": 52.029, "reference_centroid_margin_m": BASE_MARGIN,
            "qualification": "All physical and continuous wrist/grip gates; reproduced and explained 60 s before 120 s."}


def passes(result, stage):
    return wrist_passes(result, "60s" if stage == "repeat60" else stage)


def matched_sample(signals, key, time):
    i = np.argmin(abs(signals["time"] - time))
    if abs(signals["time"][i] - time) > .00051:
        raise ValueError("missing matched native timestamp")
    return signals[key][i]


def mechanism(name, signals):
    from scripts.g1_o6_g6_thumb_replay import first_event

    ct = signals["control_time"]
    angle = np.linalg.norm(signals["applied"], axis=1)
    saturation = first_event(ct, angle >= .029999)
    window = (ct >= 40) & (ct <= 52.029)
    drift = float(np.linalg.norm((matched_sample(signals, "fixed", 52.028)
                                  - matched_sample(signals, "fixed", 5.))[:2]))
    margin = float(matched_sample(signals, "centroid_margin", 40.))
    recovered = bool(np.any(angle[window] < .029) and np.any(angle[ct < 40] >= .024))
    supported = {"preorient": (saturation is None or saturation > 6.34) and drift < BASE_CAPTURE_DRIFT_M,
                 "recenter": recovered, "inward_2mm": margin > BASE_MARGIN[40.],
                 "inward_4mm": margin > BASE_MARGIN[40.]}[name]
    return {"supported": bool(supported), "saturation_s": saturation, "tangential_drift_m": drift,
            "margin_at_40_m": margin, "recovered_wrist_headroom": recovered}


class ThumbCampaign(WristCampaign):
    module = "teleopit.sim.g1_o6_support.g6_thumb_retention_experiment"
    stages = {**STAGES, "repeat60": (60, .001)}
    spec_factory = staticmethod(specification)
    passes = staticmethod(passes)
    first_pair, second_pair = ("inward_2mm", "preorient"), ("recenter",)
    audit_key = "right_acquisition_premise"
    premise = "Changing right-hand acquisition alone can prevent the late unsupported loss."
    replacement = "Allocate a bounded part of right-pad correction to the shoulder and elbow near wrist saturation."

    @staticmethod
    def actors(record):
        result = record["result"]
        audit = result.get("physics_audit", {})
        return {k: audit.get(k) for k in ("grip_support", "ground_reaction", "knee_rung3")} | {
            "strict": result.get("strict", {}).get("failure_reasons", []),
            "reset": result.get("reset_status"), "contact_quality": result.get("contact_quality", {}).get("status")}

    def signals(self, record):
        from scripts.g1_o6_g6_thumb_replay import replay

        return replay(self.root / record["output"])

    def interior_evidence(self):
        probe = self.previous("inward_2mm", "probe")
        if not probe or not self.passes(probe["result"], "probe"):
            return {"eligible": False, "reason": "2 mm acquisition or native probe failed"}
        long = self.previous("inward_2mm", "60s")
        at_40 = long and long["result"]["strict"]["unsupported_hold_seconds"] >= 40.
        when = 40. if at_40 else 5.
        margin = float(matched_sample(self.signals(long if at_40 else probe), "centroid_margin", when))
        return {"eligible": margin > BASE_MARGIN[when], "time_s": when, "margin_m": margin,
                "baseline_margin_m": BASE_MARGIN[when]}

    def validate_stage(self, name, stage, records=None):
        margin = next((d["value"] for d in self.state["decisions"] if d["name"] == "interior_margin"), {})
        if name == "inward_4mm" and not margin.get("eligible"):
            raise ValueError("4 mm requires preserved acquisition and improved 2 mm edge margin")
        if stage == "repeat60":
            prior = self.previous(name, "60s")
            if not prior or not self.passes(prior["result"], "60s"):
                raise ValueError("60 s repeat requires a passed fresh 60 s hold")
        if stage == "120s":
            repeat = self.previous(name, "repeat60")
            if not repeat or not self.passes(repeat["result"], "repeat60") or not self.explained(name):
                raise ValueError("120 s requires reproduced and explained 60 s evidence")
        super().validate_stage(name, stage, records)

    def explained(self, name):
        result = self.decision("mechanism_" + name, lambda: {
            stage: mechanism(name, self.signals(self.previous(name, stage))) for stage in ("60s", "repeat60")})
        return all(r["supported"] for r in result.values())

    def confirm(self, name):
        self.run_many([(name, "repeat60")])
        return (self.passes(self.previous(name, "repeat60")["result"], "repeat60")
                and self.explained(name) and super().confirm(name))

    def run_search(self):
        self.run_many([("control", "probe")])
        candidate = self.run_pair(self.first_pair)
        if not candidate and self.decision("interior_margin", self.interior_evidence)["eligible"]:
            candidate = self.run_pair(("inward_4mm",))
        if not candidate:
            self.audit_first_pair()
            self.run_pair(self.second_pair)
        return Pipeline(self).run()


if __name__ == "__main__":
    main(ThumbCampaign, Path("outputs/g1_o6/g6_thumb_retention"))
