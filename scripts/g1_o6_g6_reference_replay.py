"""Re-run the archived single right-wrist reference through its original 5, 30 and 60 s stages."""

import argparse
import json
from pathlib import Path

from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign, STAGES
from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
from teleopit.sim.g1_o6_support.g6_wrist_grip_experiment import passes, run_wrist
from teleopit.sim.g1_o6_support.pose_campaign import campaign_lock


class ReferenceReplay(G6Campaign):
    trial_function = staticmethod(run_wrist)

    def payload(self, name, stage, provenance):
        payload = super().payload(name, stage, provenance)
        payload["command"] = self.state["specification"]["command"]
        return payload

    def validate_preceding(self, name, stage):
        super().validate_preceding(name, stage)
        prior = {"screen":"probe", "60s":"screen"}.get(stage)
        if prior and not passes(self.previous(name, prior)["result"], prior):
            raise ValueError("reference did not pass the preceding physical gates")


def repeat_reference(parent, output):
    history = json.loads((parent / "campaign_state.json").read_bytes())
    candidate = G6Candidate.from_dict(history["candidates"]["right_wrist"])
    records = {r["stage"]:r for r in history["runs"] if r["candidate"] == candidate.id and r["status"] == "COMPLETE"}
    stages = ("probe", "screen", "60s")
    candidate = {**candidate.record(), "id":"control", "parent":None, "family":"control"}
    spec = {"schema":"g6_right_wrist_reference_repeat_v1", "native_mu":1.8, "geometry":"G6",
            "candidates":{"control":candidate}, "candidate_limit":1, "budgets":dict.fromkeys(stages, 1),
            "stages":{s:STAGES[s] for s in stages},
            "reference_steps_sha256":{s:records[s]["files_sha256"]["steps.jsonl"] for s in stages},
            "command":f'python -m scripts.g1_o6_g6_reference_replay "{parent.as_posix()}" "{output.as_posix()}"'}
    with campaign_lock(output):
        campaign = ReferenceReplay(output, workers=1, spec=spec)
        for stage in stages:
            campaign.run_many([("control", stage)])
            record = campaign.previous("control", stage)
            if record["files_sha256"]["steps.jsonl"] != spec["reference_steps_sha256"][stage]:
                raise ValueError(f"fresh {stage} did not reproduce the archived command and state trace")
            if not passes(record["result"], stage):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repeat_reference(args.parent.resolve(), args.output.resolve())
