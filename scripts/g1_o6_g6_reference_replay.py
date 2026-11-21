"""Re-run the archived single right-wrist reference through its original 5, 30 and 60 s stages."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign, STAGES
from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
from teleopit.sim.g1_o6_support.g6_wrist_grip_experiment import candidates, passes, run_wrist
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


def reference_inputs(parent):
    if parent.is_file():
        report = json.loads(parent.read_bytes())
        candidate = next(c for c in candidates() if c.id == "right_wrist")
        return candidate, {r["stage"]:r["evidence_sha256"]["steps.jsonl"] for r in report["trials"]
                           if r["candidate"] == candidate.id}
    history = json.loads((parent / "campaign_state.json").read_bytes())
    candidate = G6Candidate.from_dict(history["candidates"]["right_wrist"])
    return candidate, {r["stage"]:r["files_sha256"]["steps.jsonl"] for r in history["runs"]
                       if r["candidate"] == candidate.id and r["status"] == "COMPLETE"}


def decoded_hash(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_native(parent, campaign):
    verified = {}
    for record in campaign.state["runs"]:
        original = parent / "runs" / ("right_wrist_" + record["stage"])
        fresh = campaign.root / record["output"]
        same = {}
        for name in ("physics_contacts.jsonl.gz", "knee_rung3.f64.gz", "applied_forces.jsonl.gz"):
            same[name] = decoded_hash(original / name)
            if same[name] != decoded_hash(fresh / name):
                raise ValueError(f"fresh native records differ: {record['stage']} {name}")
        verified[record["stage"]] = same
    (campaign.root / "native_reference_sha256.json").write_text(json.dumps(verified, indent=2)+"\n")


def repeat_reference(parent, output):
    candidate, expected = reference_inputs(parent)
    stages = ("probe", "screen", "60s")
    candidate = {**candidate.record(), "id":"control", "parent":None, "family":"control"}
    spec = {"schema":"g6_right_wrist_reference_repeat_v1", "native_mu":1.8, "geometry":"G6",
            "candidates":{"control":candidate}, "candidate_limit":1, "budgets":dict.fromkeys(stages, 1),
            "stages":{s:STAGES[s] for s in stages},
            "reference_steps_sha256":{s:expected[s] for s in stages},
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
        if parent.is_dir():
            verify_native(parent, campaign)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repeat_reference(args.parent.resolve(), args.output.resolve())
