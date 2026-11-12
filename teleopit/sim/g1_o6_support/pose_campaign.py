"""Single-writer campaign ledger; failed attempts consume the frozen budget."""

import hashlib
import json
import os
import traceback
from contextlib import contextmanager
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import write_json

from .pose_candidates import Candidate, initial_geometry
from .pose_trial import run_candidate

STAGES = {
    "screen": (30, 0.001),
    "60s": (60, 0.001),
    "120s": (120, 0.001),
    "repeat": (120, 0.001),
    "fine": (120, 0.0005),
    "robustness": (120, 0.001),
}
SPEC = Path("docs/g1_o6/pose_search/campaign.json")


def validate_trial(records, candidate, stage, budgets):
    if stage not in STAGES:
        raise ValueError("unknown campaign stage")
    if any(r["candidate"] == candidate and r["stage"] == stage for r in records):
        raise ValueError("candidate/stage already attempted")
    limit = {
        "60s": "promoted_60s",
        "120s": "first_120s",
        "repeat": "finalists",
        "fine": "finalists",
        "robustness": "robustness",
    }.get(stage)
    if (
        limit
        and sum(r["stage"] == stage and not r.get("retry_of") for r in records)
        >= budgets[limit]
    ):
        raise ValueError("campaign trial budget exhausted")
    preceding = {
        "60s": "screen",
        "120s": "60s",
        "repeat": "120s",
        "fine": "repeat",
    }.get(stage)
    if preceding:
        validate_preceding(records, candidate, stage, preceding)


def validate_preceding(records, candidate, stage, preceding):
    prior = [
        r for r in records if r["candidate"] == candidate and r["stage"] == preceding
    ]
    if stage in ("repeat", "fine"):
        if not any(r.get("result", {}).get("progression_justified") for r in prior):
            raise ValueError("preceding 120 s result is not qualified")
    elif not any(r.get("result", {}).get("strict_status") == "PASS" for r in prior):
        raise ValueError("preceding strict duration did not pass")


def lock_stream(stream, unlock=False):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(
            stream.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1
        )
    else:
        import fcntl

        fcntl.flock(stream, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def campaign_lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "campaign.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        lock_stream(stream)
        try:
            yield
        finally:
            lock_stream(stream, unlock=True)


class Campaign:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "campaign_state.json"
        spec = json.loads(SPEC.read_text())
        digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["spec_sha256"] != digest:
                raise ValueError("frozen campaign specification changed")
        else:
            self.state = {
                "schema": "g1_o6_pose_campaign_state_v1",
                "spec_sha256": digest,
                "specification": spec,
                "candidates": {},
                "runs": [],
                "decisions": [],
            }
            self.register(initial_geometry())

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        write_json(temporary, self.state)
        temporary.replace(self.path)

    def register(self, candidates):
        pending = dict(self.state["candidates"])
        for candidate in candidates:
            old = pending.get(candidate.id)
            if old and Candidate.from_dict(old).digest() != candidate.digest():
                raise ValueError("candidate identity is immutable")
            pending[candidate.id] = candidate.record()
        for kind in ("geometry", "controller"):
            count = sum(c["kind"] == kind for c in pending.values())
            if count > self.state["specification"]["budgets"][kind]:
                raise ValueError("candidate budget exhausted")
        self.state["candidates"] = pending
        self.save()

    def candidate(self, name):
        return Candidate.from_dict(self.state["candidates"][name])

    def previous(self, name, stage):
        return next(
            (
                r
                for r in reversed(self.state["runs"])
                if r["candidate"] == name and r["stage"] == stage
            ),
            None,
        )

    def run(self, name, stage, retry_reason=None):
        previous = self.previous(name, stage)
        if previous and previous["status"] == "COMPLETE":
            return previous["result"]
        if previous and not (previous["status"] == "ERROR" and retry_reason):
            raise ValueError(
                "unfinished/error attempt requires a documented technical retry"
            )
        if previous is None:
            validate_trial(
                self.state["runs"], name, stage, self.state["specification"]["budgets"]
            )
        record = self.reserve(name, stage, previous, retry_reason)
        try:
            seconds, dt = STAGES[stage]
            result = run_candidate(
                self.candidate(name),
                self.root / record["output"],
                seconds=seconds,
                physics_dt=dt,
            )
        except Exception:
            record.update(status="ERROR", error=traceback.format_exc())
            self.save()
            raise
        record.update(status="COMPLETE", result=result)
        self.save()
        print(
            json.dumps(
                {
                    "candidate": name,
                    "stage": stage,
                    "strict": result["strict_status"],
                    "contact": result.get("contact_quality", {}).get("status"),
                    "late_motion": result.get("drift_assessment", {}).get("status"),
                    "qualified_interval": result["progression_justified"],
                }
            ),
            flush=True,
        )
        return result

    def reserve(self, name, stage, previous, reason):
        suffix = "_retry_" + str(len(self.state["runs"])) if previous else ""
        run_id = name + "_" + stage + suffix
        record = {
            "id": run_id,
            "candidate": name,
            "stage": stage,
            "status": "RUNNING",
            "configuration_sha256": self.candidate(name).digest(),
            "output": "runs/" + run_id,
        }
        if previous:
            record.update(retry_of=previous["id"], technical_retry_reason=reason)
        self.state["runs"].append(record)
        self.save()
        return record

    def note(self, name, value):
        existing = next((r for r in self.state["decisions"] if r["name"] == name), None)
        if existing is not None and existing["value"] != value:
            raise ValueError("campaign decision already frozen")
        if existing is None:
            self.state["decisions"].append({"name": name, "value": value})
            self.save()


def mandatory(campaign):
    campaign.run("reference", "screen")
    for name in ("rung10", "reference"):
        for stage in ("screen", "60s", "120s"):
            result = campaign.run(name, stage)
            if result["strict_status"] != "PASS":
                break


def initial_screen(campaign):
    for candidate in initial_geometry():
        campaign.run(candidate.id, "screen")
