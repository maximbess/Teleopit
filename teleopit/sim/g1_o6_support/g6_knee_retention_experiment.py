"""Single conditional retention trial on the original bounded G6 stage ladder."""

import json
from dataclasses import replace
from pathlib import Path

from .g6_campaign import STAGES
from .g6_knee import CONFIGURATIONS
from .g6_knee_experiment import (
    KneeCampaign, KneeObserver, PARENT_TRACE, candidates as knee_candidates, passes as knee_passes,
    specification as knee_specification,
)
from .g6_knee_retention import CAPTURE_S, RetainedKnee, RetainedReference
from .g6_trial import run_candidate
from .g6_whole_body_experiment import main
from .ladder_campaign import digest, file_hash


def candidates():
    parent = knee_candidates()[0]
    return [parent, replace(parent, id="retained_grip", parent="control", family="retention", controller=RetainedKnee())]


def specification():
    return {"schema": "g6_knee_retention_v1", "geometry": "G6", "native_mu": 1.8, "stages": STAGES,
            "candidates": {c.id: c.record() for c in candidates()}, "candidate_limit": 2,
            "budgets": {stage: (2 if stage == "probe" else 1) for stage in STAGES},
            "parent_probe_steps_sha256": PARENT_TRACE, "prior_specification": digest(knee_specification()),
            "qualification": "original G6 knee-clearance criteria, plus a measured bilateral grasp capture"}


def prior_failures(prior):
    milestone = prior / "milestones.json"
    if not milestone.is_file() or json.loads(milestone.read_text()).get("initial_pose") != "FAIL":
        raise ValueError("the preceding three-configuration knee search must be exhausted")
    state = json.loads((prior / "campaign_state.json").read_text())
    if state["spec_sha256"] != digest(knee_specification()):
        raise ValueError("unexpected preceding knee campaign")
    evidence = {}
    for name in CONFIGURATIONS:
        runs = [r for r in state["runs"] if r["candidate"] == name and r["status"] == "COMPLETE"]
        if not runs or knee_passes(runs[-1]["result"], runs[-1]["stage"]):
            raise ValueError("expected a terminal physical failure for each preceding refinement")
        final = runs[-1]
        raw = prior / final["output"] / "summary.json"
        checksum = file_hash(raw)
        if checksum != final["files_sha256"]["summary.json"] or json.loads(raw.read_text()) != final["result"]:
            raise ValueError("preceding physical failure does not match raw evidence")
        evidence[name] = {"id": final["id"], "identity": final["identity"], "summary_sha256": checksum,
                          "strict": final["result"]["strict"]["failure_reasons"],
                          "native_knee": final["result"]["physics_audit"]["knee_rung3"]["native"]}
    return evidence


class RetentionObserver(KneeObserver):
    def __init__(self, env, path):
        super().__init__(env, path)
        self.controller = env.hold_controller

    def report(self):
        result = super().report()
        if isinstance(self.controller, RetainedReference):
            result["settled_grip"] = {"capture_s": self.controller.grip.capture_s}
        else:
            result["settled_grip"] = {"matched_parent": True}
        return result


def run_retention(payload, output, seconds, dt):
    return run_candidate(payload, output, seconds, dt, observer_factory=RetentionObserver)


def passes(result, stage):
    grip = result.get("physics_audit", {}).get("settled_grip", {})
    capture = grip.get("capture_s")
    duration = result.get("strict", {}).get("unsupported_hold_seconds", 0.)
    exercised = capture is not None and CAPTURE_S - 1e-9 <= capture <= duration
    return knee_passes(result, stage) and (grip.get("matched_parent") is True or exercised)


class RetentionCampaign(KneeCampaign):
    spec_factory = staticmethod(specification)
    trial_function = staticmethod(run_retention)
    passes = staticmethod(passes)
    first_pair, second_pair = ("retained_grip",), ()
    module = "teleopit.sim.g1_o6_support.g6_knee_retention_experiment"

    def prepare(self):
        self.decision("three_prior_physical_failures", lambda: prior_failures(self.root.parent / "g6_knee_clearance"))
        super().prepare()


if __name__ == "__main__":
    main(RetentionCampaign, Path("outputs/g1_o6/g6_knee_retention"))
