"""Actual ladder geometry and six-axis replay contracts, without a humanoid."""

import json
from dataclasses import replace

import numpy as np
import pytest

from teleopit.sim.o6_grasp import GraspConfig, O6GraspEnv
from teleopit.sim.o6_grasp.controller import Mode
from teleopit.sim.o6_grasp.presets import ladder_rung_config
from teleopit.sim.o6_grasp.replay import WristLoadReplay, with_replay
from teleopit.sim.o6_grasp.wrenches import apply_wrenches


def trace_document():
    return {
        "schema": "o6_wrist_load_v1",
        "source": {
            "kind": "synthetic",
            "description": "Unit-test fixture, not a measured climb",
        },
        "units": {"time": "s", "force": "N", "torque": "N*m"},
        "frame": "world",
        "application_site": "wrist_mount",
        "point_m": [0, 0, 0],
        "samples": [
            {"time_s": 0, "force_N": [0, 0, 0], "torque_Nm": [0, 0, 0]},
            {"time_s": 1, "force_N": [0, 2, 0], "torque_Nm": [1, 0, 0]},
            {"time_s": 2, "force_N": [0, 0, 0], "torque_Nm": [0, 0, 0]},
        ],
    }


@pytest.fixture
def trace_path(tmp_path):
    path = tmp_path / "wrist.json"
    path.write_text(json.dumps(trace_document()))
    return path


def test_ladder_preset_is_nonpenetrating_and_matches_training_rung():
    from teleopit.sim.ladder_geometry import LADDER_RUNG

    cfg = ladder_rung_config()
    env = O6GraspEnv(cfg)
    _, info = env.reset()
    assert cfg.cross_section == "rectangular"
    assert (cfg.length, cfg.width, cfg.height) == pytest.approx((0.7, 0.11, 0.07))
    assert info["max_penetration"] == 0
    assert env.model.nu == 6 and env.model.neq == 5
    # Standalone local X is the length; training uses world Y as the length.
    rotation = env.data.geom_xmat[env.rung_geom].reshape(3, 3)
    world_half_size = np.abs(rotation) @ env.model.geom_size[env.rung_geom]
    np.testing.assert_allclose(world_half_size, LADDER_RUNG.half_size_xyz)
    np.testing.assert_allclose(
        env.model.geom_friction[env.rung_geom], LADDER_RUNG.friction
    )
    np.testing.assert_allclose(env.model.geom_solref[env.rung_geom], LADDER_RUNG.solref)


def test_interpolation_is_timestamp_based_and_does_not_repeat_tail(trace_path):
    replay = WristLoadReplay.load(trace_path)
    np.testing.assert_allclose(replay.sample(0.5), [0, 1, 0, 0.5, 0, 0])
    np.testing.assert_allclose(replay.sample(1), [0, 2, 0, 1, 0, 0])
    np.testing.assert_array_equal(replay.sample(-0.001), np.zeros(6))
    np.testing.assert_array_equal(replay.sample(2.001), np.zeros(6))
    assert replay.duration == 2


def test_replay_starts_unsupported_and_shifts_wrist_moment_to_palm(trace_path):
    cfg = with_replay(GraspConfig(), trace_path, fade_duration=0)
    env = O6GraspEnv(cfg)
    for mode, elapsed in (
        (Mode.OPEN, 0),
        (Mode.CLOSE, 1),
        (Mode.HOLD, 0.9),
        (Mode.RELEASE, 0.2),
    ):
        _, load, _ = apply_wrenches(env, mode, elapsed)
        np.testing.assert_array_equal(load, np.zeros(6))
    fixture, load, gain = apply_wrenches(env, Mode.HOLD, cfg.fixture_ramp + 1)
    assert gain == 0
    np.testing.assert_array_equal(fixture, np.zeros(6))
    # wrist is .06 m below palm: (-.06 Z) cross (2 Y) = +.12 X.
    np.testing.assert_allclose(load, [0, 2, 0, 1.12, 0, 0])
    first = env.data.qfrc_applied.copy()
    apply_wrenches(env, Mode.HOLD, cfg.fixture_ramp + 1)
    np.testing.assert_array_equal(first, env.data.qfrc_applied)
    np.testing.assert_allclose(first[:6], [0, 2, 0, 1, 0, 0])


def test_replay_fits_hold_and_reset_rewinds_without_reloading(trace_path):
    cfg = with_replay(
        replace(
            GraspConfig(), phase_durations=(0.5, 2.5, 2, 2.5), success_duration=0.5
        ),
        trace_path,
    )
    env = O6GraspEnv(cfg)
    assert env.config.phase_durations[2] >= cfg.fixture_ramp + 2 + cfg.unload_duration
    observation, _ = env.reset(seed=3)
    for _ in range(210):
        env.step(np.zeros(6))
    trace_path.write_text("{}")  # snapshot is owned by this environment
    again, info = env.reset(seed=3)
    np.testing.assert_array_equal(observation, again)
    np.testing.assert_array_equal(info["external_wrench"], np.zeros(6))
    np.testing.assert_allclose(env.replay.sample(1), [0, 2, 0, 1, 0, 0])


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"frame": "sensor"},
        {"application_site": "rung_frame"},
        {"units": {"time": "ms", "force": "N", "torque": "N*m"}},
        {"point_m": [0, 0]},
        {"source": {"kind": "unknown", "description": "test"}},
        {"source": {"kind": "synthetic", "description": " "}},
        {"extra": "unknown top-level field"},
    ],
)
def test_replay_metadata_rejected_at_boundary(trace_path, change):
    document = trace_document()
    document.update(change)
    trace_path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        WristLoadReplay.load(trace_path)


@pytest.mark.parametrize(
    "times", [[0, 0, 2], [1, 2, 3], [0, 2, 1], [0, float("nan"), 2]]
)
def test_replay_timestamps_must_start_at_zero_and_increase(trace_path, times):
    document = trace_document()
    for row, time in zip(document["samples"], times):
        row["time_s"] = time
    trace_path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        WristLoadReplay.load(trace_path)


def test_run_trial_snapshots_trace_and_can_reproduce_from_saved_config(
    trace_path, tmp_path
):
    from teleopit.sim.o6_grasp.config import load_config
    from teleopit.sim.o6_grasp.evaluate import run_trial

    cfg = with_replay(ladder_rung_config(), trace_path)
    output = tmp_path / "result"
    summary = run_trial("ladder", cfg, output)
    assert summary["replay"]["source"]["kind"] == "synthetic"
    assert summary["replay"]["duration_s"] == 2
    assert summary["first_failure"] is not None  # large rung baseline is not validated
    assert len(summary["first_failure"]["external_wrench"]) == 6
    assert (output / "ladder.replay.json").read_bytes() == trace_path.read_bytes()
    trace_path.unlink()
    restored = O6GraspEnv(load_config(output / "ladder.config.json"))
    np.testing.assert_allclose(restored.replay.sample(1), [0, 2, 0, 1, 0, 0])
    rows = [
        json.loads(line) for line in (output / "ladder.jsonl").read_text().splitlines()
    ]
    active = [row for row in rows[1:] if row["wrist_replay"]["active"]]
    assert active and all(row["unsupported"] for row in active)
    assert all(
        np.linalg.norm(row["external_wrench"]) == 0
        for row in rows
        if row["mode"] == "RELEASE"
    )


def test_replay_uses_trace_frame_point_and_explicit_fades(trace_path):
    document = trace_document()
    document.update(frame="palm", point_m=[0.1, 0, 0])
    for row in document["samples"]:
        row["force_N"], row["torque_Nm"] = [0, 2, 0], [1, 0, 0]
    trace_path.write_text(json.dumps(document))
    cfg = with_replay(
        replace(GraspConfig(), palm_quat=(2**-0.5, 0, 0, 2**-0.5)),
        trace_path,
        fade_duration=0.2,
        scale=0.5,
    )
    env = O6GraspEnv(cfg)
    _, load, _ = apply_wrenches(env, Mode.HOLD, cfg.fixture_ramp + 1)
    np.testing.assert_allclose(load, [-1, 0, 0, 0, 0.56, 0.1], atol=1e-12)
    _, halfway, _ = apply_wrenches(env, Mode.HOLD, cfg.fixture_ramp + 0.1)
    np.testing.assert_allclose(halfway, 0.5 * load, atol=1e-12)
    _, end, _ = apply_wrenches(env, Mode.HOLD, cfg.fixture_ramp + 2)
    np.testing.assert_array_equal(end, np.zeros(6))


def test_analytic_suite_does_not_silently_ignore_replay(trace_path, tmp_path):
    from teleopit.sim.o6_grasp.evaluate import run_suite

    with pytest.raises(ValueError, match="replay"):
        run_suite(with_replay(GraspConfig(), trace_path), tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [trace_path.name]


def test_rung_frame_replay_and_physics_substep_clock(trace_path, monkeypatch):
    document = trace_document()
    document["frame"] = "rung"
    document["application_site"] = "palm_frame"
    trace_path.write_text(json.dumps(document))
    env = O6GraspEnv(with_replay(GraspConfig(), trace_path, fade_duration=0))
    _, load, _ = apply_wrenches(env, Mode.HOLD, env.config.fixture_ramp + 1)
    np.testing.assert_allclose(load, [-2, 0, 0, 0, 1, 0], atol=1e-12)
    sampled = []
    original = WristLoadReplay.sample

    def sample(self, time):
        sampled.append(time)
        return original(self, time)

    monkeypatch.setattr(WristLoadReplay, "sample", sample)
    for _ in range(202):
        env.step(np.zeros(6))
    assert len(sampled) >= 10
    np.testing.assert_allclose(np.diff(sampled), env.config.physics_dt, atol=1e-12)


def test_rectangular_suite_actually_varies_dimensions():
    from teleopit.sim.o6_grasp.evaluate import scenarios

    cases = scenarios(ladder_rung_config())
    assert (
        cases["rung_small"].width
        < cases["pullout_small"].width
        < cases["rung_large"].width
    )
    assert (
        cases["rung_small"].height
        < cases["pullout_small"].height
        < cases["rung_large"].height
    )


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [{"time_s": 0}],
        [{"time_s": 0}, {"time_s": 1}],
        [
            {"time_s": 0, "force_N": [0, 0, 0], "torque_Nm": [0, 0, 0]},
            {"time_s": 1, "force_N": [float("inf"), 0, 0], "torque_Nm": [0, 0, 0]},
        ],
    ],
)
def test_malformed_samples_rejected(trace_path, samples):
    document = trace_document()
    document["samples"] = samples
    trace_path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        WristLoadReplay.load(trace_path)


def test_replay_configuration_limits(trace_path):
    for changes in (
        {"load_profile": "replay"},
        {"wrist_load_path": str(trace_path)},
        {"rung_friction": (-1, 0, 0)},
        {"load_profile": "replay", "wrist_load_path": 3},
    ):
        with pytest.raises(ValueError):
            O6GraspEnv(replace(GraspConfig(), **changes))
    for fade, scale in ((-1, 1), (1.1, 1), (0.1, -1)):
        with pytest.raises(ValueError):
            O6GraspEnv(
                with_replay(GraspConfig(), trace_path, fade_duration=fade, scale=scale)
            )


def test_cli_preset_config_relative_trace_and_scale_override(
    trace_path, tmp_path, monkeypatch, capsys
):
    from teleopit.sim.o6_grasp.__main__ import main

    document = trace_document()
    for row in document["samples"]:
        row["time_s"] *= 0.05
    trace_path.write_text(json.dumps(document))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "load_profile": "replay",
                "wrist_load_path": trace_path.name,
                "replay_fade_duration": 0.01,
                "phase_durations": [0.04, 0.04, 0.2, 0.04],
                "fixture_ramp": 0.04,
                "unload_duration": 0.02,
                "verification_window": 0.02,
                "success_duration": 0.04,
            }
        )
    )
    output = tmp_path / "cli"
    monkeypatch.setattr(
        "sys.argv",
        [
            "o6",
            "--preset",
            "ladder-rung",
            "--config",
            str(config),
            "--replay-scale",
            ".25",
            "--output",
            str(output),
        ],
    )
    main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["config"]["width"] == 0.11
    assert summary["replay_result"]["scale"] == 0.25
    assert summary["replay_result"]["fade_duration_s"] == 0.01
    assert (output / "ladder_rung.replay.json").exists()


def test_cli_rejects_incompatible_options(trace_path, monkeypatch):
    from teleopit.sim.o6_grasp.__main__ import main

    for args in (
        ["--suite", "--wrist-load", str(trace_path)],
        ["--replay-scale", ".5"],
    ):
        monkeypatch.setattr("sys.argv", ["o6", *args])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
