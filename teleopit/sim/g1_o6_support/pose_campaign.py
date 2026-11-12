"""Single-writer campaign ledger; failed attempts consume the frozen budget."""

import hashlib
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import write_json

from .pose_candidates import Candidate, initial_geometry
from .pose_trial import preflight_geometry, prior_resets, run_candidate
from .pose_validation import validate_robustness

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
    validate_budget(records, stage, budgets)
    preceding = {
        "60s": "screen",
        "120s": "60s",
        "repeat": "120s",
        "fine": "repeat",
    }.get(stage)
    if preceding:
        validate_preceding(records, candidate, stage, preceding)


def validate_budget(records, stage, budgets):
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
        for kind in ("geometry", "controller", "robustness"):
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
            self.validate_stage(name, stage)
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
        self.complete(record, result)
        return result

    def complete(self, record, result):
        record.update(status="COMPLETE", result=result)
        self.save()
        print(
            json.dumps(
                {
                    "candidate": record["candidate"],
                    "stage": record["stage"],
                    "strict": result["strict_status"],
                    "contact": result.get("contact_quality", {}).get("status"),
                    "late_motion": result.get("drift_assessment", {}).get("status"),
                    "qualified_interval": result["progression_justified"],
                }
            ),
            flush=True,
        )

    def run_many(self, requests, workers=3):
        records = self.reserve_many(requests)
        records = self.preflight(records)
        errors = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {
                pool.submit(
                    run_candidate,
                    self.candidate(r["candidate"]),
                    self.root / r["output"],
                    seconds=STAGES[r["stage"]][0],
                    physics_dt=STAGES[r["stage"]][1],
                ): r
                for r in records
            }
            for future in as_completed(jobs):
                record = jobs[future]
                error = future.exception()
                if error is None:
                    self.complete(record, future.result())
                else:
                    record.update(
                        status="ERROR",
                        error="".join(
                            traceback.format_exception(
                                type(error), error, error.__traceback__
                            )
                        ),
                    )
                    errors.append(record["id"])
                    self.save()
        if errors:
            raise RuntimeError("technical errors preserved: " + ", ".join(errors))

    def preflight(self, records):
        references = prior_resets(self.root, self.state["runs"])
        pending = []
        for record in records:
            candidate = self.candidate(record["candidate"])
            result = None
            if record["stage"] == "screen" and candidate.kind == "geometry":
                result = preflight_geometry(
                    candidate, self.root / record["output"], references
                )
            if result is None:
                pending.append(record)
            else:
                self.complete(record, result)
        return pending

    def reserve_many(self, requests):
        pending = []
        preview = list(self.state["runs"])
        for name, stage in requests:
            previous = self.previous(name, stage)
            if previous is not None:
                if previous["status"] != "COMPLETE":
                    raise ValueError(
                        "unfinished/error batch attempt requires explicit retry"
                    )
                continue
            self.validate_stage(name, stage)
            validate_trial(preview, name, stage, self.state["specification"]["budgets"])
            pending.append((name, stage))
            preview.append({"candidate": name, "stage": stage, "status": "RESERVED"})
        return [self.reserve(name, stage, None, None) for name, stage in pending]

    def validate_stage(self, name, stage):
        candidate = self.candidate(name)
        if (stage == "robustness") != (candidate.kind == "robustness"):
            raise ValueError("robustness candidates require the robustness stage")
        if stage == "robustness":
            validate_robustness(self, candidate)

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
        value = json.loads(json.dumps(value))
        existing = next((r for r in self.state["decisions"] if r["name"] == name), None)
        if existing is not None and existing["value"] != value:
            raise ValueError("campaign decision already frozen")
        if existing is None:
            self.state["decisions"].append({"name": name, "value": value})
            self.save()

    def decision(self, name, select):
        existing = next((r for r in self.state["decisions"] if r["name"] == name), None)
        if existing is not None:
            return existing["value"]
        self.note(name, select())
        return self.decision(name, select)


def mandatory(campaign):
    campaign.run("reference", "screen")
    for name in ("rung10", "reference"):
        for stage in ("screen", "60s", "120s"):
            result = campaign.run(name, stage)
            if result["strict_status"] != "PASS":
                break


def initial_screen(campaign):
    requests = [(c.id, "screen") for c in initial_geometry()]
    previous = campaign.previous("reference", "60s")
    if previous and previous.get("result", {}).get("strict_status") == "PASS":
        requests.insert(0, ("reference", "120s"))
    campaign.run_many(requests)
