"""One conditional native-interval grip preview after the three endpoint controllers."""

from dataclasses import replace
from pathlib import Path

from .g6_grip_feedback_experiment import GripCampaign, LAWS, candidates as feedback_candidates, passes, specification as prior_spec
from .g6_grip_preview import PreviewGrip
from .g6_knee_experiment import experiment_spec
from .g6_knee_retention_experiment import prior_failures
from .g6_whole_body_experiment import main
from .ladder_campaign import digest

PARENT_TRACE = "6e53e3733802b64d954d3a5f8ee75eceb01a9dc2a74e4b47603a1584de0aeddf"


def candidates():
    parent = replace(feedback_candidates()[-1], id="control", parent=None, family="control")
    return [parent, replace(parent, id="native_grip", parent="control", family="preview", controller=PreviewGrip())]


def specification():
    return {**experiment_spec("g6_grip_preview_v1", candidates()), "parent_probe_steps_sha256": PARENT_TRACE,
            "prior_spec_sha256": digest(prior_spec()), "qualification": prior_spec()["qualification"],
            "maximum_additional_hand_previews": 3}


class PreviewCampaign(GripCampaign):
    spec_factory = staticmethod(specification)
    first_pair, second_pair = ("native_grip",), ()
    module = "teleopit.sim.g1_o6_support.g6_grip_preview_experiment"

    def prepare(self):
        self.decision("three_prior_grip_failures", lambda: prior_failures(
            self.root.parent / "g6_grip_feedback", specification=prior_spec, names=LAWS, gate=passes))
        super().prepare()


if __name__ == "__main__":
    main(PreviewCampaign, Path("outputs/g1_o6/g6_grip_preview"))
