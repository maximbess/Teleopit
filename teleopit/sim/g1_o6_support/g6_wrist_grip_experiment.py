"""Three bounded wrist/grip refinements, with the original G6 qualification ladder."""

from dataclasses import replace
from pathlib import Path

from .g6_grip_feedback_experiment import GripCampaign, GripObserver, passes as support_passes
from .g6_grip_preview_experiment import candidates as parent_candidates
from .g6_knee_experiment import experiment_spec
from .g6_trial import run_candidate
from .g6_whole_body_experiment import main
from .g6_wrist_grip import MODES, WristGrip, WristReference

PARENT_TRACE = "7b68775e3d5dcc1ac059162c059ad9509557f9557d914a047be17b59de89fba4"
CONFIGURATIONS = {"right_wrist":"right", "both_wrists":"both", "distributed_arms":"distributed"}


def candidates():
    parent = replace(parent_candidates()[-1], id="control", family="control", parent=None)
    return [parent, *[replace(parent, id=name, parent="control", family=mode, controller=WristGrip(wrist_mode=mode))
                      for name, mode in CONFIGURATIONS.items()]]


def specification():
    return {**experiment_spec("g6_wrist_grip_v1", candidates()), "parent_probe_steps_sha256": PARENT_TRACE,
            "qualification":"Original G6 stable hold, continuous bilateral opposition, four loaded left fingers, "
                            "positive right-knee/rung-3 clearance, residual <25 N and exercised wrist feedback."}


class WristObserver(GripObserver):
    def report(self):
        arm = self.controller.parent
        wrist = ({"mode":arm.mode,"capture_s":arm.capture_s,"updates":arm.updates.copy()}
                 if isinstance(arm, WristReference) else {"mode":"control"})
        return {**super().report(), "wrist_grip":wrist}


def wrist_exercised(audit):
    wrist = audit.get("wrist_grip", {})
    mode = wrist.get("mode")
    if mode == "control":
        return True
    capture = wrist.get("capture_s")
    return (mode in MODES and capture is not None and capture == audit.get("settled_grip",{}).get("capture_s")
            and all(wrist.get("updates",{}).get(s,0)>0 for s in MODES[mode]))


def passes(result, stage):
    return support_passes(result, stage) and wrist_exercised(result["physics_audit"])


def run_wrist(payload, output, seconds, dt):
    return run_candidate(payload, output, seconds, dt, observer_factory=WristObserver)


class WristCampaign(GripCampaign):
    spec_factory = staticmethod(specification)
    trial_function = staticmethod(run_wrist)
    passes = staticmethod(passes)
    first_pair, second_pair = ("right_wrist","both_wrists"), ("distributed_arms",)
    audit_key = "wrist_only_contact_correction"
    premise = "Three wrist joints can maintain pad position and opposition while the proximal arm recenters."
    replacement = "Let all seven joints of each arm correct the contact motion within the same arm limits."
    module = "teleopit.sim.g1_o6_support.g6_wrist_grip_experiment"


if __name__ == "__main__":
    main(WristCampaign, Path("outputs/g1_o6/g6_wrist_grip"))
