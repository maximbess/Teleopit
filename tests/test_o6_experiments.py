"""Bounded matrix decisions preserve failures and simultaneous signed demands."""

from types import SimpleNamespace

import pytest

from teleopit.sim.o6_diagnostics.config import DiagnosticConfig


def test_matrix_stops_unqualified_and_confirms_fresh_protocols(tmp_path, monkeypatch):
    from teleopit.sim.o6_diagnostics import experiments as e

    calls = []

    def execute(source, path, cfg, case, **options):
        calls.append((case.copy(), options))
        return {
            **case,
            "executed": True,
            "zero_load_qualified": False,
            "strict_pass": False,
            "physical_retention": False,
            "protocol": options.get("protocol", "primary"),
        }

    monkeypatch.setattr(e, "execute", execute)
    source = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=0.001)))
    refs = [{"time_s": 5.2, "wrench_world_at_mount": [1, 2, 3, 4, 5, 6]}]
    status, cases = e.side_matrix(source, tmp_path, DiagnosticConfig(), "left", refs)
    assert status == "not_qualified_at_zero_load"
    assert sum(not c["executed"] for c in cases) == 12
    assert len(calls) == 6
    assert len({c["case"] for c, _ in calls}) == 6
    assert all(c["family"] in ("zero", "combined") for c, _ in calls)
    assert all(
        o.get("protocol") == "seated_diagnostic"
        for c, o in calls
        if c["family"] == "combined"
    )
    assert sum(o.get("dt") == 0.0005 for _, o in calls) == 2


@pytest.mark.parametrize("passes", [True, False])
def test_qualified_matrix_bounds_and_confirmation(passes, tmp_path, monkeypatch):
    from teleopit.sim.o6_diagnostics import experiments as e

    calls = []

    def execute(source, path, cfg, case, **options):
        calls.append((case.copy(), options))
        return {
            **case,
            "executed": True,
            "zero_load_qualified": True,
            "strict_pass": passes and case["scale"] <= 0.5,
            "physical_retention": passes,
            "protocol": "primary",
        }

    monkeypatch.setattr(e, "execute", execute)
    source = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=0.001)))
    refs = [
        {"time_s": t, "wrench_world_at_mount": [1, 2, 3, 4, 5, 6]} for t in (5.2, 5.3)
    ]
    status, cases = e.side_matrix(source, tmp_path, DiagnosticConfig(), "right", refs)
    assert status == ("qualified" if passes else "reserve_failed_diagnostic_only")
    assert all(c["executed"] for c in cases)
    assert len({c["case"] for c, _ in calls}) == len(calls)
    if passes:
        assert any(c["scale"] == 0.625 for c in cases)
        assert any(c["family"] == "combined_confirmation" for c in cases)
        assert sum(o.get("dt") == 0.0005 for _, o in calls) == 6
    else:
        assert len(calls) == 17
        assert sum(o.get("dt") == 0.0005 for _, o in calls) == 2


def test_implicated_passive_joint_maps_to_its_command():
    from teleopit.sim.o6_diagnostics.experiments import implicated

    source = SimpleNamespace(
        hands={
            "left": {
                "active": ["pitch", "yaw", "index"],
                "passive": [("thumb_ip", "pitch", 1)],
            }
        }
    )
    result = {
        "closed_endpoint_errors_rad": {"index_servo": 0.05},
        "saturation": {"yaw_servo": {"longest_s": 1}},
        "first_threshold_violation": {"lower_limit:thumb_ip": {}},
    }
    assert implicated(result, source, "left", DiagnosticConfig()) == [0, 1, 2]


def test_skipped_cases_preserve_unknown_values_and_manifests(tmp_path):
    from teleopit.sim.o6_diagnostics.experiments import save_experiment, skipped
    from teleopit.sim.o6_diagnostics.io import output_directory

    case = {
        "case": "left_unknown",
        "side": "left",
        "scale": 0.25,
        "requested_wrench_world_at_mount": [1, 0, 0, 0, 0, 0],
    }
    result = save_experiment(
        tmp_path, "load_response", [skipped(case, "zero failed")], {}
    )
    assert result["cases"][0]["achieved_force_norm_N"] is None
    assert (tmp_path / "manifest.json").is_file() and (
        tmp_path / "load_response.csv"
    ).is_file()
    with pytest.raises(FileExistsError):
        output_directory(tmp_path)


def test_zero_failure_leaves_primary_matrix_unexecuted():
    from teleopit.sim.o6_diagnostics.experiments import planned_cases, qualification

    cases = planned_cases(
        "left",
        {"time_s": 5.2, "wrench_world_at_mount": [1, 2, 3, 4, 5, 6]},
        DiagnosticConfig(),
    )
    assert len(cases) == 12
    assert len({c["case"] for c in cases}) == 12
    assert (
        qualification({"zero_load_qualified": False, "strict_pass": False})
        == "not_qualified_at_zero_load"
    )
    assert (
        qualification({"zero_load_qualified": True, "strict_pass": False})
        == "reserve_failed_diagnostic_only"
    )
    force = next(c for c in cases if c["family"] == "force_only" and c["scale"] == 0.5)
    assert force["requested_wrench_world_at_mount"] == [0.5, 1, 1.5, 0, 0, 0]
    moment = next(c for c in cases if c["family"] == "moment_only" and c["scale"] == 1)
    assert moment["requested_wrench_world_at_mount"] == [0, 0, 0, 4, 5, 6]


def test_brackets_only_compare_neighbors_in_the_same_direction():
    from teleopit.sim.o6_diagnostics.experiments import brackets

    cases = [
        {"scale": a, "strict_pass": p, "executed": True}
        for a, p in [(0.25, False), (0.5, True), (0.75, False), (1, False)]
    ]
    assert brackets(cases) == [(0.25, 0.5), (0.5, 0.75)]
    assert (
        brackets(
            [
                {"scale": 0.25, "strict_pass": False, "executed": True},
                {"scale": 0.5, "strict_pass": None, "executed": False},
            ]
        )
        == []
    )


@pytest.mark.parametrize("command", ["nominal", "free-close", "load-response"])
def test_documented_cli_dispatches_selected_experiment(
    command, tmp_path, monkeypatch, capsys
):
    import sys

    from teleopit.sim.o6_diagnostics import __main__ as cli
    from teleopit.sim.o6_diagnostics import experiments, nominal

    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        if command == "nominal":
            assert kwargs["pose"].hand_rungs == (11, 11)
        return {"experiment": command}

    monkeypatch.setattr(nominal, "run_nominal", run)
    monkeypatch.setattr(experiments, "free_close", run)
    monkeypatch.setattr(experiments, "load_response", run)
    argv = ["o6_diagnostics", command, "--output", str(tmp_path)]
    if command != "nominal":
        argv += ["--source-run", "explicit_current_run"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert len(calls) == 1 and str(tmp_path) in calls[0]
    assert command in capsys.readouterr().out


def test_load_report_preserves_unknown_cases_and_signed_reference(
    tmp_path, monkeypatch
):
    from teleopit.sim.o6_diagnostics import experiments as e
    from teleopit.sim.o6_diagnostics import plots

    source = SimpleNamespace(path=tmp_path)
    monkeypatch.setattr(
        e, "experiment", lambda *_: (source, tmp_path, DiagnosticConfig())
    )
    sample = {
        "time": 5.2,
        "phase": "HOLD",
        "assistance_gain": 0,
        "wrist_world_on_hand": {"left": [-1, 2, -3, 4, -5, 6]},
    }
    monkeypatch.setattr(e, "read_rows", lambda *_: [sample])
    monkeypatch.setattr(
        e,
        "side_matrix",
        lambda *args: (
            "not_qualified_at_zero_load",
            [
                e.skipped(
                    e.planned_cases("left", args[-1][0], DiagnosticConfig())[0],
                    "zero failed",
                )
            ],
        ),
    )
    monkeypatch.setattr(plots, "load_plot", lambda *_: None)
    result = e.load_response("unused", tmp_path, "left")
    assert result["references"]["left"][0]["wrench_world_at_mount"] == [
        -1,
        2,
        -3,
        4,
        -5,
        6,
    ]
    assert result["cases"][0]["strict_pass"] is None
    assert (tmp_path / "signed_reference_samples.json").is_file()


def test_execute_forwards_signed_world_wrench_without_scaling_gravity(
    tmp_path, monkeypatch
):
    from teleopit.sim.o6_diagnostics import experiments as e

    calls = []

    def run_hand(source, side, path, **kwargs):
        calls.append((side, path, kwargs))
        return {"strict_pass": False}

    monkeypatch.setattr(e, "run_hand", run_hand)
    case = {
        "case": "probe",
        "side": "right",
        "requested_wrench_world_at_mount": [-1, 2, -3, 4, -5, 6],
    }
    result = e.execute(None, tmp_path, DiagnosticConfig(), case, dt=0.0005)
    assert result["executed"] and not result["strict_pass"]
    assert calls[0][2]["wrench"] == case["requested_wrench_world_at_mount"]
    assert calls[0][2]["dt"] == 0.0005
