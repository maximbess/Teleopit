"""Public command and evidence boundaries; synthetic records are test-only."""

import json
from types import SimpleNamespace

import pytest


def test_state_demands_require_both_hands_and_matching_reserve():
    from test_g1_o6_g6_state_evidence import sample
    from teleopit.sim.g1_o6_support.capacity import sampled_state_key
    from teleopit.sim.g1_o6_support.qualification import (
        targeted_demands_covered,
        Evidence,
    )

    row = sample()
    targets = {s + "_hand": "left_ladder_rung_07" for s in ("left", "right")}
    tested = {
        s: {targets[s + "_hand"]: [[0, 0, -12, 0, 0.12, 0]]} for s in ("left", "right")
    }
    assert targeted_demands_covered([row], tested, targets)
    runs = [
        {
            "metadata": {
                "side": s,
                "source_sample_support_targets": targets,
                "source_sample": {"state_sha256": sampled_state_key(row, targets)},
            },
            "summary": {
                "strict_pass": True,
                "plateau_complete": True,
                "physical_retention": True,
                "requested_wrench_world_at_mount": [0, 0, -12, 0, 0.12, 0],
            },
        }
        for s in ("left", "right")
    ]
    assert targeted_demands_covered([row], tested, targets, capacity_runs=runs)
    assert not targeted_demands_covered([row], tested, targets, capacity_runs=runs[:1])
    proof = Evidence.__new__(Evidence)
    proof.source = {"metadata": {"support_targets": targets}}
    proof.is_g6 = True
    assert proof.has_higher_tread()
    targets["left_hand"] = "left_ladder_rung_08"
    assert not proof.has_higher_tread()


def test_g6_strict_observation_exempts_only_selected_reaching_hand():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.posture_report import annotate
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.transfer import TransferTrial
    from teleopit.sim.g1_o6_support.protocol import Step

    env = make_environment(G6Candidate("fixture"), 30, 0.001)
    request = dict(
        mode="load",
        side="left",
        target_share=0.3,
        cycles=1,
        alternating=False,
        three_seconds=1,
        plateau_seconds=10,
        physics_dt=0.001,
    )
    trial = TransferTrial(env, request)
    trial.protocol.steps = [Step("REACH", 15, 30, side="left")]
    row = annotate(env, snapshot(env))
    reasons = trial.strict_reasons(row)
    assert "anatomical_grasp:right" in reasons
    assert "anatomical_grasp:left" not in reasons
    row["pose_search"] = {"hands": {"right": {"quality": {"valid": True}}}}
    assert not any(r.startswith("anatomical_grasp") for r in trial.strict_reasons(row))


def test_pipeline_source_and_attempts_are_immutable(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6_support import g6_pipeline as module
    from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign

    monkeypatch.setattr(module, "code_provenance", lambda: {"source": "fixture"})
    pipeline = module.Pipeline(G6Campaign(tmp_path))
    runs = [{"path": str(tmp_path / "selected_120s")}]
    book = pipeline.initialize("selected", runs)
    assert pipeline.initialize("selected", runs) == book
    with pytest.raises(ValueError, match="source changed"):
        pipeline.initialize("other", runs)
    calls = []

    def transfer(book, path, **request):
        calls.append(request)
        path.mkdir(parents=True)
        (path / "summary.json").write_text("{}")
        return {"physical_status": "BLOCKED"}

    monkeypatch.setattr(module, "run_transfer", transfer)
    first = pipeline.attempt("load", "transfer", {"target_share": 0.3})
    assert pipeline.attempt("load", "transfer", {"target_share": 0.3}) == first
    assert len(calls) == 1
    (tmp_path / first["output"] / "summary.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum"):
        pipeline.attempt("load", "transfer", {"target_share": 0.3})

    def error(*args, **kwargs):
        raise OSError("synthetic interruption")

    monkeypatch.setattr(module, "run_g6_capacity", error)
    with pytest.raises(OSError):
        pipeline.attempt("capacity", "capacity", {"source_path": "fixture"})
    assert module.Pipeline(pipeline.campaign).state["runs"][-1]["status"] == "ERROR"


def test_pipeline_does_not_promote_physical_success_without_capacity(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import g6_pipeline as module
    from teleopit.sim.g1_o6_support.g6_campaign import G6Campaign

    pipeline = module.Pipeline(G6Campaign(tmp_path))
    book = {"transfers": [], "capacity_runs": []}
    record = {"output": "fixture", "result": {"physical_status": "PASS"}}
    monkeypatch.setattr(pipeline, "attempt", lambda *a: record)
    monkeypatch.setattr(pipeline, "capacity", lambda *a, **k: True)
    monkeypatch.setattr(
        module, "Evidence", lambda _: SimpleNamespace(qualified_transfers=lambda: [])
    )
    assert not pipeline.transfer(0, {"mode": "load"}, book)
    assert len(book["transfers"]) == 1
    record["result"]["physical_status"] = "FAIL"
    assert not pipeline.transfer(1, {"mode": "load"}, book)
    assert len(book["transfers"]) == 1


def test_cli_wires_commands_and_preserves_batch_order(tmp_path, monkeypatch, capsys):
    from teleopit.sim.g1_o6_support import g6_experiment as cli

    calls = []
    campaign = SimpleNamespace(
        state={"candidates": {}},
        prepare=lambda: calls.append("prepare"),
        import_control=lambda p: calls.append(str(p)),
        run_many=lambda r: calls.append(r),
        retry=lambda *a: calls.append(a),
        note=lambda *a: calls.append(a),
    )
    monkeypatch.setattr(
        cli, "run_batch", lambda c, b, p: calls.append((b, p)) or b.lower()
    )
    monkeypatch.setattr(cli, "promote", lambda c: calls.append("promote"))
    monkeypatch.setattr(cli, "best_parent", lambda *a: "d")
    monkeypatch.setattr(
        cli,
        "Pipeline",
        lambda c: SimpleNamespace(run=lambda: {"initial_pose": "BLOCKED"}),
    )
    for argv in (
        ["prepare"],
        ["import-control", "control"],
        ["trial", "a_00", "probe"],
        ["retry", "a_00", "probe", "--reason", "storage"],
        ["batch", "A", "--parent", "control"],
        ["search"],
    ):
        cli.dispatch(campaign, cli.parser().parse_args(argv))
    assert calls[-6:-2] == [("A", "control"), ("B", "a"), ("C", "b"), ("D", "c")]
    monkeypatch.setattr("sys.argv", ["experiment", "--output", str(tmp_path), "status"])
    cli.main()
    assert "NOT RUN" in capsys.readouterr().out
