"""A technical retry cannot erase failure history or change the experiment."""
from copy import deepcopy
import json

import pytest


def attempts():
    from teleopit.sim.g1_o6_support.ladder_campaign import digest
    payload = {"candidate": "pitch_025", "seconds": 120, "physics_dt": .001}
    failed = {"id": "original", "candidate": "pitch_025", "stage": "120s", "output": "runs/original",
              "status": "ERROR", "identity": digest(payload), "payload": payload, "error": "worker startup failed"}
    recovered = {**deepcopy(failed), "id": "recovered", "output": "runs/recovered", "status": "COMPLETE",
                 "retry_of": "original", "technical_retry_reason": "startup failure, same request", "result": {"strict_status": "FAIL"}}
    return [failed, recovered]


def test_verified_history_keeps_errors_and_only_returns_completed_physics():
    from scripts.verify_g1_o6_g6 import verified_history
    records = attempts()
    completed, recoveries = verified_history(records)
    assert completed == [records[1]]
    assert recoveries == [{"failed_attempt": "original", "recovered_by": "recovered", "identity": records[0]["identity"]}]
    assert records[0]["status"] == "ERROR"
    # A technical retry is bookkeeping; its eventual physical FAIL stays FAIL.
    assert completed[0]["result"]["strict_status"] == "FAIL"


@pytest.mark.parametrize("fault", ["unfinished", "physical_retry", "missing_reason", "changed_identity", "changed_payload", "changed_experiment", "changed_prior_payload", "same_output", "duplicate_id"])
def test_verified_history_rejects_unrecovered_or_changed_experiments(fault):
    from scripts.verify_g1_o6_g6 import verified_history
    records = attempts()
    if fault == "unfinished":
        records.pop()
    elif fault == "physical_retry":
        records[0]["status"] = "COMPLETE"
    elif fault == "missing_reason":
        records[1]["technical_retry_reason"] = " "
    elif fault == "changed_identity":
        records[1]["identity"] = "changed"
    elif fault == "changed_payload":
        records[1]["payload"]["seconds"] = 60
    elif fault == "changed_experiment":
        from teleopit.sim.g1_o6_support.ladder_campaign import digest
        records[1]["payload"]["physics_dt"] = .0005
        records[1]["identity"] = digest(records[1]["payload"])
    elif fault == "changed_prior_payload":
        records[0]["payload"]["physics_dt"] = .002
    elif fault == "same_output":
        records[1]["output"] = records[0]["output"]
    elif fault == "duplicate_id":
        records[1]["id"] = records[0]["id"]
    with pytest.raises((AssertionError, ValueError)):
        verified_history(records)


def test_acquisition_report_lists_recovery_without_inventing_error_run_files(tmp_path, monkeypatch):
    from scripts import g1_o6_g6_acquisition_report as reporting
    records = attempts()
    (tmp_path / "report").mkdir()
    (tmp_path / "campaign_state.json").write_text(json.dumps({"runs": records}))
    visited = []
    monkeypatch.setattr(reporting, "report", lambda root: None)

    def verify_run(path, output):
        visited.append(path.name)
        return {"status": "PASS"}

    monkeypatch.setattr(reporting, "verify_run", verify_run)
    reporting.verify(tmp_path)
    evidence = json.loads((tmp_path / "report/acquisition_verification.json").read_text())
    assert visited == ["recovered"]
    assert evidence["runs"] == {"recovered": {"status": "PASS"}}
    assert evidence["technical_recoveries"][0]["failed_attempt"] == "original"
