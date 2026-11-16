"""Immutable, bounded G6 hypothesis batches with explicit technical recovery."""

import json
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import write_json

from .g6_candidates import G6Candidate, batch_candidates
from .g6_trial import make_environment, run_candidate
from .ladder_campaign import (
    LadderCampaign,
    code_provenance,
    digest,
    physical_identity,
    run_checksums,
    screen_passes,
)
from .long_hold import long_qualified

STAGES = {
    "probe": (5, 0.001),
    "screen": (30, 0.001),
    "60s": (60, 0.001),
    "120s": (120, 0.001),
    "repeat": (120, 0.001),
    "fine": (120, 0.0005),
}


def specification():
    return {
        "schema": "g1_o6_g6_stability_v1",
        "geometry": "G6",
        "native_mu": 1.8,
        "controller": "reference",
        "stages": STAGES,
        "budgets": {
            "probe": 39,
            "screen": 40,
            "60s": 8,
            "120s": 4,
            "repeat": 1,
            "fine": 1,
        },
        "candidate_limit": 40,
        "batch_sizes": {"A": 12, "B": 8, "C": 12, "D": 7},
        "ranking": "physical/contact integrity, base displacement rate, body clearance, effort, next-tread reach; duration only a tie breaker",
        "promotion": "fresh 30 -> 60 -> 120 -> nominal repeat -> half timestep; no failed physical retry",
    }


class G6Campaign(LadderCampaign):
    trial_function = staticmethod(run_candidate)
    stages = STAGES

    def __init__(self, root, workers=3):
        if not 1 <= workers <= 6:
            raise ValueError("workers must be in 1..6")
        self.root, self.workers = Path(root), workers
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "campaign_state.json"
        spec = json.loads(json.dumps(specification()))
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["spec_sha256"] != digest(spec):
                raise ValueError("frozen campaign specification changed")
        else:
            self.state = {
                "schema": spec["schema"],
                "specification": spec,
                "spec_sha256": digest(spec),
                "runs": [],
                "decisions": [],
                "candidates": {"control": G6Candidate("control").record()},
            }
            self.save()

    def candidate(self, name):
        return G6Candidate.from_dict(self.state["candidates"][name])

    def register(self, candidates):
        pending = dict(self.state["candidates"])
        for candidate in candidates:
            old = pending.get(candidate.id)
            if old and G6Candidate.from_dict(old).digest() != candidate.digest():
                raise ValueError("candidate identity is immutable")
            pending[candidate.id] = candidate.record()
        if len(pending) > self.state["specification"]["candidate_limit"]:
            raise ValueError("candidate budget exhausted")
        self.state["candidates"] = pending
        self.save()

    def prepare(self):
        self.decision("source", code_provenance)
        self.decision(
            "reference_physics",
            lambda: physical_identity(
                make_environment(self.candidate("control"), 5, 0.001).model
            ),
        )
        write_json(self.root / "frozen_campaign.json", self.state)

    def freeze_batch(self, batch, parent):
        selected = self.decision(
            "batch_" + batch,
            lambda: {
                "parent": parent,
                "candidates": [
                    c.record() for c in batch_candidates(batch, self.candidate(parent))
                ],
            },
        )
        self.register(G6Candidate.from_dict(c) for c in selected["candidates"])
        return [c["id"] for c in selected["candidates"]]

    def validate_preceding(self, name, stage):
        preceding = {
            "screen": "probe",
            "60s": "screen",
            "120s": "60s",
            "repeat": "120s",
            "fine": "repeat",
        }.get(stage)
        if preceding:
            prior = self.previous(name, preceding)
            if prior is None or not screen_passes(prior.get("result", {})):
                raise ValueError("preceding strict/contact/physical gates failed")
            if stage in ("repeat", "fine") and not long_qualified(prior["result"], 120):
                raise ValueError("preceding 120 s convergence failed")

    def plan_requests(self, requests):
        if len(set(requests)) != len(requests):
            raise ValueError("duplicate request in batch")
        self.prepare()
        frozen = self.decision("source", code_provenance)
        if code_provenance() != frozen:
            raise ValueError("source/assets/runtime changed since campaign freeze")
        preview, plans = list(self.state["runs"]), []
        for name, stage in requests:
            payload = self.payload(name, stage, frozen)
            identity = digest(payload)
            prior = self.previous(name, stage)
            if prior:
                self.verify_resume(prior, identity)
                continue
            self.validate_stage(name, stage, preview)
            plans.append((name, stage, identity, payload))
            preview.append({"candidate": name, "stage": stage})
        return plans

    def payload(self, name, stage, provenance):
        return {
            "candidate": self.candidate(name).record(),
            "stage": stage,
            "seconds": STAGES[stage][0],
            "physics_dt": STAGES[stage][1],
            "provenance": provenance,
            "reference_physics": self.decision("reference_physics", lambda: None),
            "command": f'python -m teleopit.sim.g1_o6_support.g6_experiment --output "{self.root.as_posix()}" trial {name} {stage}',
        }

    def import_control(self, path):
        path = Path(path).resolve()
        checksums = run_checksums(path)
        previous = self.previous("control", "screen")
        if previous:
            if checksums != previous["files_sha256"]:
                raise ValueError("reference evidence changed")
            return
        result = json.loads((path / "summary.json").read_text())
        self.state["runs"].append(
            {
                "id": "control_screen",
                "candidate": "control",
                "stage": "screen",
                "status": "COMPLETE",
                "imported_control": True,
                "output": str(path.relative_to(self.root.resolve())),
                "files_sha256": checksums,
                "result": result,
            }
        )
        self.save()

    def retry(self, name, stage, reason):
        prior = self.previous(name, stage)
        if (
            not reason.strip()
            or prior is None
            or prior["status"] not in ("ERROR", "RUNNING")
        ):
            raise ValueError(
                "technical retry requires interrupted/error evidence and reason"
            )
        current = self.payload(name, stage, code_provenance())
        if digest(current) != prior["identity"]:
            raise ValueError("technical retry identity changed")
        record = self.reserve(name, stage, prior, reason)
        record.update(identity=prior["identity"], payload=prior["payload"])
        self.save()
        self.execute([record], 1)


def rank_record(record):
    result = record.get("result", {})
    strict = result.get("strict", {})
    audit = result.get("physics_audit", {})
    seconds = max(0.02, strict.get("unsupported_hold_seconds", 0))
    drift = strict.get("final_diagnostics", {}).get("base_drift_m", 1)
    clearance = min(audit.get("body_minimum_clearance_m", {"missing": -0.1}).values())
    return (
        not screen_passes(result),
        result.get("reset_status") == "INFEASIBLE",
        result.get("contact_quality", {}).get("status") != "PASS",
        audit.get("short_drift", {}).get(
            "maximum_translation_slope_m_s", drift / seconds
        ),
        -clearance,
        strict.get("peak_effort_fraction", 1),
        audit.get("next_tread_reach", {}).get("status") != "REACHABLE",
        -seconds,
        record["candidate"],
    )


def best_parent(campaign, names):
    records = [
        r
        for r in campaign.state["runs"]
        if r["candidate"] in names and r["status"] == "COMPLETE"
    ]
    return min(records, key=rank_record)["candidate"] if records else "control"


def run_batch(campaign, batch, parent):
    names = campaign.freeze_batch(batch, parent)
    campaign.run_many([(name, "probe") for name in names])
    passed = [
        name
        for name in names
        if screen_passes(campaign.previous(name, "probe")["result"])
    ]
    campaign.run_many([(name, "screen") for name in passed])
    return best_parent(campaign, names)


def promote(campaign):
    for stage, preceding, maximum in (
        ("60s", "screen", 8),
        ("120s", "60s", 4),
        ("repeat", "120s", 1),
        ("fine", "repeat", 1),
    ):
        records = promotion_candidates(campaign.state["runs"], stage, preceding)
        names = campaign.decision(
            "promote_" + stage,
            lambda: [
                r["candidate"] for r in sorted(records, key=rank_record)[:maximum]
            ],
        )
        campaign.run_many([(name, stage) for name in names])


def promotion_candidates(records, stage, preceding):
    eligible = [
        r
        for r in records
        if r["stage"] == preceding and screen_passes(r.get("result", {}))
    ]
    return (
        [r for r in eligible if long_qualified(r["result"], 120)]
        if stage in ("repeat", "fine")
        else eligible
    )
