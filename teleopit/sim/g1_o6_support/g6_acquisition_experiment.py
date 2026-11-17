"""Six bounded grasp-acquisition configurations and two conditional stabilizers."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.o6_diagnostics.io import write_json

from .g6_acquisition import GraspAcquisition, thumb_pose
from .g6_acquisition_audit import AcquisitionObserver
from .g6_arm import MAX_RATE_RAD_S, feedback_limits
from .g6_campaign import G6Campaign, STAGES, rank_record
from .g6_candidates import G6Candidate
from .g6_stages import milestone_report
from .g6_trial import run_candidate
from .ladder_campaign import screen_passes
from .long_hold import long_qualified
from .pose_arm import DEFAULT_RATE_RAD_S
from .pose_campaign import campaign_lock

PAIRS = {"pitch": ("pitch_025", "pitch_035"),
         "first": ("first_025", "first_035"),
         "yaw": ("yaw_020", "yaw_030")}
PRECEDING = {"first": "pitch", "yaw": "first"}


def candidates():
    control = G6Candidate("control", GraspPose(hand_rungs=(7, 7), foot_rungs=(2, 2)),
                          reset={"root_offset_x": -.04})
    rows = [control]
    for pair, pitches, yaw, order in (
        ("pitch", (.25, .35), .5, "together"),
        ("first", (.25, .35), .5, "thumb_first"),
        ("yaw", (.2, .3), 1., "thumb_first"),
    ):
        rows.extend(replace(control, id=name, parent="control", family=pair,
                    pose=thumb_pose(control.pose, pitch=pitch, yaw=yaw),
                    controller=GraspAcquisition(order))
                    for name, pitch in zip(PAIRS[pair], pitches))
    return rows


def specification():
    return {"schema": "g1_o6_g6_grasp_acquisition_v1", "geometry": "G6", "native_mu": 1.8,
            "stages": STAGES, "candidate_limit": 9,
            "budgets": {"probe": 9, "screen": 8, "60s": 4, "120s": 2, "repeat": 1, "fine": 1},
            "candidates": {c.id: c.record() for c in candidates()},
            "predeclared_pairs": PAIRS,
            "stabilization": {"maximum": 2, "activation": "HOLD entry", "rates_rad_s": [DEFAULT_RATE_RAD_S, MAX_RATE_RAD_S], **feedback_limits()},
            "acquisition_prerequisite": "Both anatomical and structural/slip gates for final 0.5 s of removal, or original contact PASS for >=0.5 s unsupported HOLD; no external/prohibited support or material override",
            "premise_rule": "Each later acquisition pair requires two failed preceding candidates and an immutable measured actor/premise audit",
            "qualification": "Fresh 5 -> 30 -> 60 -> 120 s, nominal 120 s repeat and .0005 confirmation; every original physical/contact/convergence gate"}


def run_acquisition_candidate(payload, output, seconds, dt):
    return run_candidate(payload, output, seconds, dt, observer_factory=AcquisitionObserver)


def failed_qualification(record):
    result = record.get("result", {})
    return record["status"] == "COMPLETE" and (
        not screen_passes(result) or (record["stage"] in ("120s", "repeat", "fine") and not long_qualified(result, 120)))


def latest_failures(records, names):
    failed = []
    for name in names:
        matches = [r for r in records if r["candidate"] == name and failed_qualification(r)]
        if matches:
            failed.append(matches[-1]["id"])
    return failed


def acquired(record):
    result = record.get("result", {})
    audit = result.get("physics_audit", {})
    return (record["status"] == "COMPLETE"
            and grasp_established(result)
            and audit.get("no_external_support") is True
            and audit.get("unintended_support_steps") == 0
            and audit.get("housing_contact_steps") == 0
            and audit.get("native_hand_friction", {}).get("status") == "PASS")


def grasp_established(result):
    acquisition = result.get("physics_audit", {}).get("grasp_acquisition", {})
    return acquisition.get("ready_for_hold") is True or (
        result.get("contact_quality", {}).get("status") == "PASS"
        and result.get("strict", {}).get("unsupported_hold_seconds", 0) >= .5)


class AcquisitionCampaign(G6Campaign):
    trial_function = staticmethod(run_acquisition_candidate)

    def __init__(self, root, workers=2):
        super().__init__(root, workers, spec=specification())

    def payload(self, name, stage, provenance):
        result = super().payload(name, stage, provenance)
        result["command"] = (f'python -m teleopit.sim.g1_o6_support.g6_acquisition_experiment '
                             f'--output "{self.root.as_posix()}" trial {name} {stage}')
        return result

    def validate_stage(self, name, stage, records=None):
        preceding = PRECEDING.get(self.candidate(name).family)
        if preceding and not any(d["name"] == "audit_" + preceding for d in self.state["decisions"]):
            raise ValueError("record the preceding two-failure premise audit")
        super().validate_stage(name, stage, records)

    def record_audit(self, pair, audit):
        failed = latest_failures(self.state["runs"], PAIRS[pair])
        if len(failed) != 2 or set(failed) != set(audit["failed_run_ids"]):
            raise ValueError("audit requires two completed failed candidates in the declared pair")
        for field in ("premise", "observed_actors", "removed_assignment"):
            if not audit.get(field):
                raise ValueError("audit is missing " + field)
        if self.decision("audit_" + pair, lambda: audit) != audit:
            raise ValueError("premise audit is immutable")

    def stabilization_parent(self):
        selected = next((d["value"] for d in self.state["decisions"] if d["name"] == "stabilization_parent"), None)
        if selected is None:
            eligible = [r for r in self.state["runs"] if r["candidate"] != "control" and acquired(r)]
            if not eligible:
                raise ValueError("no measured acquired grasp is eligible for stabilization")
            record = min(eligible, key=rank_record)
            selected = self.decision("stabilization_parent", lambda: {"candidate": record["candidate"], "run": record["id"]})
        return self.candidate(selected["candidate"])

    def select_stabilization(self):
        parent = self.stabilization_parent()
        children = [replace(parent, id="stabilize_" + label, parent=parent.id,
                    family="stabilization", controller=replace(parent.controller, arm_rate_rad_s=rate))
                    for label, rate in (("slow", DEFAULT_RATE_RAD_S), ("fast", MAX_RATE_RAD_S))]
        self.register(children)
        return [c.id for c in children]


def run_pair(campaign, names):
    campaign.run_many([(name, "probe") for name in names])
    passed = [name for name in names if screen_passes(campaign.previous(name, "probe")["result"])]
    campaign.run_many([(name, "screen") for name in passed])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("outputs/g1_o6/g6_grasp_acquisition"))
    result.add_argument("--workers", type=int, default=2)
    sub = result.add_subparsers(dest="action", required=True)
    for name in ("prepare", "status", "stabilize"):
        sub.add_parser(name)
    pair = sub.add_parser("pair")
    pair.add_argument("name", choices=list(PAIRS))
    audit = sub.add_parser("audit")
    audit.add_argument("name", choices=list(PAIRS))
    audit.add_argument("path", type=Path)
    for action in ("trial", "retry"):
        trial = sub.add_parser(action)
        trial.add_argument("candidate")
        trial.add_argument("stage", choices=list(STAGES))
        if action == "retry":
            trial.add_argument("--reason", required=True)
    return result


def dispatch(campaign, args):
    actions = {
        "prepare": campaign.prepare,
        "pair": lambda: run_pair(campaign, PAIRS[args.name]),
        "audit": lambda: campaign.record_audit(args.name, json.loads(args.path.read_text())),
        "stabilize": lambda: run_pair(campaign, campaign.select_stabilization()),
        "trial": lambda: campaign.run_many([(args.candidate, args.stage)]),
        "retry": lambda: campaign.retry(args.candidate, args.stage, args.reason),
        "status": lambda: None,
    }
    actions[args.action]()
    report = milestone_report(campaign)
    report["grasp_acquisition"] = {"acquired_candidates": sorted({r["candidate"] for r in campaign.state["runs"] if acquired(r)})}
    write_json(campaign.root / "milestones.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    args = parser().parse_args()
    with campaign_lock(args.output):
        dispatch(AcquisitionCampaign(args.output, args.workers), args)
