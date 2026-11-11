"""All current front ends resolve the same optional grasp configuration."""

import json
import sys

import pytest


@pytest.mark.parametrize("mode", ["headless", "viewer", "video", "nominal"])
def test_front_ends_pass_resolved_settings(monkeypatch, tmp_path, mode):
    from teleopit.sim.g1_o6 import __main__ as cli
    from teleopit.sim.g1_o6 import viewer
    from teleopit.sim.o6_diagnostics import __main__ as diagnostic_cli
    from teleopit.sim.o6_diagnostics import nominal

    config = tmp_path / "config.json"
    grasp = tmp_path / "grasp.json"
    config.write_text(json.dumps({"delay_steps": 2}))
    grasp.write_text(json.dumps({"hand_rungs": [10, 11], "finger_preload_Nm": 0.1}))
    calls = []

    def run(cfg, output, *args, pose=None):
        calls.append((cfg, pose))
        return {}

    monkeypatch.setattr(cli, "run_trial", run)
    monkeypatch.setattr(viewer, "show", run)
    monkeypatch.setattr(viewer, "record", run)
    monkeypatch.setattr(nominal, "run_nominal", run)
    argv = ["test"] + (["nominal"] if mode == "nominal" else [])
    argv += [
        "--config",
        str(config),
        "--grasp-config",
        str(grasp),
        "--output",
        str(tmp_path),
    ]
    if mode in ("viewer", "video"):
        argv.append("--" + mode)
    monkeypatch.setattr(sys, "argv", argv)
    (diagnostic_cli if mode == "nominal" else cli).main()
    cfg, pose = calls[0]
    assert cfg.delay_steps == 2
    assert pose.hand_rungs == (10, 11) and pose.finger_preload_Nm == 0.1


@pytest.mark.parametrize("mode", ["", "nominal"])
def test_removed_posture_flag_is_an_argument_error(monkeypatch, mode):
    from teleopit.sim.g1_o6 import __main__ as cli
    from teleopit.sim.o6_diagnostics import __main__ as diagnostic_cli

    argv = ["test"] + ([mode] if mode else [])
    monkeypatch.setattr(
        sys, "argv", argv + ["--posture", "thumb-below-balanced", "--output", "unused"]
    )
    with pytest.raises(SystemExit) as exc:
        (diagnostic_cli if mode else cli).main()
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["free-close", "load-response"])
def test_isolated_cli_requires_explicit_source(monkeypatch, command):
    from teleopit.sim.o6_diagnostics import __main__ as cli

    monkeypatch.setattr(sys, "argv", ["test", command, "--output", "unused"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "values",
    [{"hand_rung": 11}, [], {"hand_rungs": [11, 15]}, {"finger_preload_Nm": 0.36}],
)
def test_invalid_grasp_json_is_rejected(tmp_path, values):
    from teleopit.sim.g1_o6.settings import load_settings

    path = tmp_path / "grasp.json"
    path.write_text(json.dumps(values))
    with pytest.raises((ValueError, TypeError)):
        load_settings(grasp_path=path)
