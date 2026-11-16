"""Four predeclared feedback trials, with a premise audit between the pairs."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.o6_diagnostics.io import write_json

from .g6_arm import ArmFeedback, MAX_RATE_RAD_S, feedback_limits
from .pose_arm import DEFAULT_RATE_RAD_S
from .g6_campaign import G6Campaign, STAGES, promote
from .g6_candidates import G6Candidate
from .g6_stages import milestone_report
from .ladder_campaign import screen_passes
from .pose_campaign import campaign_lock


def candidates():
    control = G6Candidate("control", GraspPose(hand_rungs=(7, 7), foot_rungs=(2, 2)),
                          reset={"root_offset_x": -.04})
    return [control, *[
        replace(control, id=f"{mode}_{label}", parent="control", family="arm_feedback",
                controller=ArmFeedback(mode, rate))
        for mode in ("retain", "seek")
        for label, rate in (("slow", DEFAULT_RATE_RAD_S), ("fast", MAX_RATE_RAD_S))
    ]]


def specification():
    return {
        "schema": "g1_o6_g6_arm_feedback_v1", "geometry": "G6", "native_mu": 1.8,
        "controller": "explicit bounded acquisition wrapper; original servos and finger law",
        "stages": STAGES,
        "budgets": {"probe": 5, "screen": 4, "60s": 4, "120s": 4, "repeat": 1, "fine": 1},
        "candidate_limit": 5,
        "candidates": {c.id: c.record() for c in candidates()},
        **feedback_limits(),
        "premise_rule": "Seek trials require a recorded audit of two retained-reference failures, observed actors, and the removed assignment.",
        "qualification": "Unchanged physical/contact gates; fresh probe -> 30 -> 60 -> 120 -> nominal repeat -> .0005 confirmation",
    }


class FeedbackCampaign(G6Campaign):
    def __init__(self, root, workers=2):
        super().__init__(root, workers, spec=specification())

    def payload(self, name, stage, provenance):
        payload = super().payload(name, stage, provenance)
        payload["command"] = (f'python -m teleopit.sim.g1_o6_support.g6_feedback_experiment '
                              f'--output "{self.root.as_posix()}" trial {name} {stage}')
        return payload

    def validate_stage(self, name, stage, records=None):
        if name.startswith("seek_") and not any(d["name"] == "premise_audit" for d in self.state["decisions"]):
            raise ValueError("record the two-failure premise audit before seeking contact")
        super().validate_stage(name, stage, records)

    def record_audit(self, audit):
        failed = retained_failures(self.state["runs"])
        if len(failed) != 2 or set(audit["failed_run_ids"]) != {r["id"] for r in failed}:
            raise ValueError("premise audit must identify both completed retained-reference failures")
        for field in ("premise", "removed_assignment", "observed_actors"):
            if not audit.get(field):
                raise ValueError("premise audit is missing " + field)
        frozen = self.decision("premise_audit", lambda: audit)
        if frozen != audit:
            raise ValueError("premise audit is immutable")


def retained_failures(records):
    failed = []
    for name in ("retain_slow", "retain_fast"):
        matches = [r for r in records if r["candidate"] == name and r["status"] == "COMPLETE"
                   and not screen_passes(r["result"])]
        if matches:
            failed.append(matches[-1])
    return failed


def run_pair(campaign, mode):
    names = [c.id for c in candidates() if c.id.startswith(mode + "_")]
    campaign.run_many([(name, "probe") for name in names])
    passed = [name for name in names if screen_passes(campaign.previous(name, "probe")["result"])]
    campaign.run_many([(name, "screen") for name in passed])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("outputs/g1_o6/g6_arm_feedback"))
    result.add_argument("--workers", type=int, default=2)
    sub = result.add_subparsers(dest="action", required=True)
    for name in ("prepare", "promote", "status"):
        sub.add_parser(name)
    control = sub.add_parser("import-control")
    control.add_argument("path", type=Path)
    pair = sub.add_parser("pair")
    pair.add_argument("mode", choices=("retain", "seek"))
    audit = sub.add_parser("audit")
    audit.add_argument("path", type=Path)
    trial = sub.add_parser("trial")
    trial.add_argument("candidate", choices=[c.id for c in candidates()])
    trial.add_argument("stage", choices=list(STAGES))
    retry = sub.add_parser("retry")
    retry.add_argument("candidate", choices=[c.id for c in candidates()])
    retry.add_argument("stage", choices=list(STAGES))
    retry.add_argument("--reason", required=True)
    return result


def dispatch(campaign, args):
    if args.action == "prepare":
        campaign.prepare()
    elif args.action == "import-control":
        campaign.import_control(args.path, stage="probe")
    elif args.action == "pair":
        run_pair(campaign, args.mode)
    elif args.action == "audit":
        campaign.record_audit(json.loads(args.path.read_text()))
    elif args.action == "trial":
        campaign.run_many([(args.candidate, args.stage)])
    elif args.action == "retry":
        campaign.retry(args.candidate, args.stage, args.reason)
    elif args.action == "promote":
        promote(campaign)
    report = milestone_report(campaign)
    write_json(campaign.root / "milestones.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    args = parser().parse_args()
    with campaign_lock(args.output):
        dispatch(FeedbackCampaign(args.output, args.workers), args)
