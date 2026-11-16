"""One writer schedules qualified G6 stages and preserves every attempt."""

import json
from pathlib import Path

from teleopit.sim.o6_diagnostics.io import write_json

from .g6_capacity import capacity_run_paths, run_g6_capacity
from .g6_stages import milestone_report, qualified_pose, transfer_schedule
from .ladder_campaign import code_provenance, digest, run_checksums
from .qualification import Evidence
from .transfer import run_transfer


class Pipeline:
    def __init__(self, campaign):
        self.campaign = campaign
        self.root = campaign.root
        self.path = self.root / "stage_state.json"
        self.book_path = self.root / "evidence.json"
        self.state = (
            json.loads(self.path.read_text()) if self.path.exists() else {"runs": []}
        )

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        write_json(temporary, self.state)
        temporary.replace(self.path)

    def initialize(self, name, runs):
        source = next(r for r in runs if Path(r["path"]).name.endswith("_120s"))
        book = {
            "campaign": "g6_stability",
            "candidate": name,
            "source_run": source["path"],
            "long_holds": [r["path"] for r in runs],
            "capacity_runs": [],
            "transfers": [],
        }
        if self.book_path.exists():
            existing = json.loads(self.book_path.read_text())
            if any(
                existing[k] != book[k]
                for k in ("candidate", "source_run", "long_holds")
            ):
                raise ValueError("frozen qualified source changed")
            return existing
        write_json(self.book_path, book)
        return book

    def attempt(self, name, kind, request):
        identity = digest(
            {"kind": kind, "request": request, "provenance": code_provenance()}
        )
        previous = next((r for r in self.state["runs"] if r["id"] == name), None)
        if previous:
            self.campaign.verify_resume(previous, identity)
            return previous
        record = {
            "id": name,
            "kind": kind,
            "request": request,
            "identity": identity,
            "status": "RUNNING",
            "output": "stages/" + name,
        }
        self.state["runs"].append(record)
        self.save()
        path = self.root / record["output"]
        try:
            result = (
                run_g6_capacity(output=path, **request)
                if kind == "capacity"
                else run_transfer(self.book_path, path, **request)
            )
        except Exception as error:
            record.update(status="ERROR", error=repr(error))
            self.save()
            raise
        record.update(
            status="COMPLETE", result=result, files_sha256=run_checksums(path)
        )
        self.save()
        return record

    def capacity(self, name, source, book, *, planned_share):
        record = self.attempt(
            name,
            "capacity",
            {"source_path": str(source), "planned_share": planned_share, "budget": 192},
        )
        paths = capacity_run_paths(self.root / record["output"])
        book["capacity_runs"] = sorted(set(book["capacity_runs"] + paths))
        write_json(self.book_path, book)
        return record["result"]["capacity_status"] == "PASS"

    def transfer(self, index, request, book):
        name = f"transfer_{index:02d}_{request['mode']}_{request.get('side', 'left')}"
        record = self.attempt(name, "transfer", request)
        if record["result"]["physical_status"] != "PASS":
            return False
        path = str((self.root / record["output"]).resolve())
        book["transfers"] = sorted(set([*book["transfers"], path]))
        write_json(self.book_path, book)
        if not self.capacity(name + "_capacity", path, book, planned_share=0):
            return False
        return any(
            r["path"] == path for r in Evidence(self.book_path).qualified_transfers()
        )

    def run(self):
        name, runs = qualified_pose(self.campaign)
        if name:
            book = self.initialize(name, runs)
            if self.capacity(
                "initial_capacity", book["source_run"], book, planned_share=0.16
            ):
                for index, request in enumerate(transfer_schedule()):
                    if not self.transfer(index, request, book):
                        break
        report = milestone_report(self.campaign)
        write_json(self.root / "milestones.json", report)
        return report
