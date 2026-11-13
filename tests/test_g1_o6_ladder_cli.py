"""Public orchestration preserves all negative conditions and empty promotions."""

from concurrent.futures import ThreadPoolExecutor
import json


def test_all_infeasible_resets_complete_without_any_physics_or_promotions(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support import ladder_campaign as ledger
    from teleopit.sim.g1_o6_support import ladder_experiment as cli

    monkeypatch.setattr(ledger, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(ledger, "select_reset", lambda _: {"status": "INFEASIBLE", "pose": None})
    monkeypatch.setattr(ledger, "code_provenance", lambda: {"code": "software test"})
    monkeypatch.setattr(cli, "import_reference", lambda _: None)
    campaign = ledger.LadderCampaign(tmp_path)
    cli.run_all(campaign)
    assert len(campaign.state["runs"]) == 25
    assert all(r["result"]["strict_status"] == "NOT RUN" for r in campaign.state["runs"])
    assert all(r["stage"] == "screen" for r in campaign.state["runs"])
    assert json.loads((tmp_path / "ranked_60s.json").read_text())["selected"] == []
    assert json.loads((tmp_path / "ranked_120s.json").read_text())["selected"] == []
    # Re-entry uses the frozen negative evidence without new attempts.
    cli.run_all(campaign)
    assert len(campaign.state["runs"]) == 25


def test_freeze_cli_publishes_the_bounded_configuration(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support.ladder_experiment import main

    monkeypatch.setattr("sys.argv", ["experiment", "--output", str(tmp_path), "freeze"])
    main()
    specification = json.loads((tmp_path / "specification.json").read_text())
    assert specification["budgets"] == {"screen": 25, "60s": 8, "120s": 6, "repeat": 2, "fine": 2}
    assert len(specification["conditions"]) == 25
