"""Bounded whole-body search on the acquired G6 grasp, with fresh HOLD trials."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import write_json

from .g6_acquisition import GraspAcquisition
from .g6_acquisition_experiment import candidates as grasp_candidates, run_acquisition_candidate
from .g6_arm import MAX_RATE_RAD_S
from .g6_campaign import G6Campaign, STAGES, rank_record
from .g6_gates import initial_pose_gate
from .g6_whole_body import BODY_OFFSET_RAD, BODY_RATE_RAD_S, GAINS, WholeBody
from .ladder_campaign import screen_passes
from .long_hold import long_qualified
from .pose_campaign import campaign_lock
from .pose_contacts import LOADED_N
from .qualification import read_run

PARENT_TRACE = "186c3973bcca9af3a76b9a27d50cf513802530ac9078a8926017239fda5c77af"
PAIRS = {"stance": ("stance_low", "stance_high"), "wrench": ("wrench_low", "wrench_high")}
PRECEDING = dict(zip(tuple(STAGES)[1:], tuple(STAGES)[:-1]))


def candidates():
    grasp = next(c for c in grasp_candidates() if c.id == "pitch_035")
    parent = replace(grasp, id="control", family="control", parent=None,
                     controller=GraspAcquisition("together", MAX_RATE_RAD_S))
    return [parent, *[replace(parent, id=name, parent="control", family=law,
                              controller=WholeBody(balance=law, gain=gain))
                      for law, names in PAIRS.items() for name, gain in zip(names, GAINS)]]


def specification():
    return {"schema": "g1_o6_g6_whole_body_v1", "geometry": "G6", "native_mu": 1.8,
            "parent_probe_steps_sha256": PARENT_TRACE, "candidate_limit": 5,
            "candidates": {c.id: c.record() for c in candidates()}, "stages": STAGES,
            "budgets": {s: (5 if s == "probe" else 4) for s in STAGES},
            "pairs": PAIRS, "body_offset_bound_rad": BODY_OFFSET_RAD, "body_rate_rad_s": BODY_RATE_RAD_S,
            "qualification": "original physical, anatomical and convergence gates plus all four left fingers loaded on every HOLD physics step; fresh nominal repeat and half-timestep confirmation"}


def eligible(result, stage):
    audit = result.get("physics_audit", {})
    if audit.get("native_hand_friction", {}).get("status") != "PASS" or not screen_passes(result):
        return False
    minimum = result["contact_quality"].get("minimum_group_normal_N", {}).get("left", {}).get("finger")
    duration = result["strict"]["unsupported_hold_seconds"]
    return (minimum is not None and minimum >= LOADED_N and duration >= STAGES[stage][0] - .021
            and (STAGES[stage][0] < 120 or long_qualified(result, 120)))


def measured_actors(record):
    result = record["result"]
    final = result["strict"]["final_diagnostics"]
    quality = result["contact_quality"]
    return {"duration_s": result["strict"]["unsupported_hold_seconds"],
            "final_support_forces_world_N": {k: v["force_world"] for k, v in final["supports"].items()},
            "minimum_group_normal_N": quality["minimum_group_normal_N"],
            "longest_absence_s": quality.get("longest_absence_s", {}),
            "drift": result["drift_assessment"], "strict_status": result["strict_status"],
            "contact_status": quality["status"]}


class WholeBodyCampaign(G6Campaign):
    trial_function = staticmethod(run_acquisition_candidate)

    def __init__(self, root, workers=2):
        super().__init__(root, workers, spec=specification())

    def payload(self, name, stage, provenance):
        payload = super().payload(name, stage, provenance)
        payload["command"] = (f'python -m teleopit.sim.g1_o6_support.g6_whole_body_experiment '
                              f'--output "{self.root.as_posix()}" trial {name} {stage}')
        return payload

    def validate_stage(self, name, stage, records=None):
        if name == "control" and stage != "probe":
            raise ValueError("the matched control has one fresh probe")
        if name != "control":
            control = self.previous("control", "probe")
            if not control or control.get("files_sha256", {}).get("steps.jsonl") != PARENT_TRACE:
                raise ValueError("matched acquisition control has not reproduced its parent trace")
        if name in PAIRS["wrench"] and not any(d["name"] == "stance_premise" for d in self.state["decisions"]):
            raise ValueError("the second family requires two measured first-family failures")
        super().validate_stage(name, stage, records)

    def validate_preceding(self, name, stage):
        super().validate_preceding(name, stage)
        prior = PRECEDING.get(stage)
        if prior and not eligible(self.previous(name, prior)["result"], prior):
            raise ValueError("preceding stage failed continuous left finger loading or physical qualification")

    def audit_stance(self):
        def measure():
            actors = {}
            for name in PAIRS["stance"]:
                completed = [r for r in self.state["runs"] if r["candidate"] == name and r["status"] == "COMPLETE"]
                if not completed or eligible(completed[-1]["result"], completed[-1]["stage"]):
                    raise ValueError("the stance pair has not produced two terminal physical failures")
                actors[completed[-1]["id"]] = measured_actors(completed[-1])
            return {"premise": "Both legs can realize the required pelvis twist with fixed feet on the tread.",
                    "actors": actors, "removed_assignment": "Replace independent fixed-foot kinematic corrections with a shared desired ground reaction distributed through both native leg servos."}
        return self.decision("stance_premise", measure)

    def run_pair(self, names):
        for stage in ("probe", "screen", "60s"):
            if not names:
                return None
            self.run_many([(name, stage) for name in names])
            names = [name for name in names if eligible(self.previous(name, stage)["result"], stage)]
        for name in sorted(names, key=lambda n: rank_record(self.previous(n, "60s"))):
            if self.confirm(name):
                return name
        return None

    def confirm(self, name):
        for stage in ("120s", "repeat", "fine"):
            self.run_many([(name, stage)])
            if not eligible(self.previous(name, stage)["result"], stage):
                return False
        evidence = [read_run(self.root / self.previous(name, s)["output"]) for s in PRECEDING]
        if not initial_pose_gate(evidence):
            raise ValueError("original initial pose qualification disagrees with individual stages")
        return True

    def run_search(self):
        self.run_many([("control", "probe")])
        candidate = self.run_pair(PAIRS["stance"])
        if not candidate:
            self.audit_stance()
            candidate = self.run_pair(PAIRS["wrench"])
        result = {"initial_pose": "PASS" if candidate else "FAIL", "candidate": candidate}
        write_json(self.root / "milestones.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/g1_o6/g6_whole_body"))
    parser.add_argument("--workers", type=int, default=2)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("run")
    for action in ("trial", "retry"):
        command = sub.add_parser(action)
        command.add_argument("candidate", choices=[c.id for c in candidates()])
        command.add_argument("stage", choices=list(STAGES))
        if action == "retry":
            command.add_argument("--reason", required=True)
    args = parser.parse_args()
    with campaign_lock(args.output):
        campaign = WholeBodyCampaign(args.output, args.workers)
        if args.action == "run":
            print(json.dumps(campaign.run_search()), flush=True)
        elif args.action == "trial":
            campaign.run_many([(args.candidate, args.stage)])
        else:
            campaign.retry(args.candidate, args.stage, args.reason)


if __name__ == "__main__":
    main()
