"""CLI tests for the ladder-only policy video recorder."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from train_mimic.ladder_playback import configure_ladder_play_phase
from train_mimic.scripts import record_ladder_video


def test_record_ladder_video_cli_has_no_benchmark_arguments() -> None:
    args = record_ladder_video.parse_args(["--checkpoint", "model.pt"])

    assert args.frames == 1000
    assert args.ladder_phase is None
    assert args.width == 1280
    assert args.height == 720
    assert args.camera_distance == 4.5
    assert args.camera_azimuth == 30.0
    assert args.camera_elevation == -5.0
    assert args.camera_lookat == (-0.45, 0.0, 1.45)
    assert args.camera_fovy == 48.0
    assert not hasattr(args, "num_envs")
    assert not hasattr(args, "num_eval_steps")
    assert not hasattr(args, "output_dir")
    assert not hasattr(args, "motion_file")
    assert not hasattr(args, "task")


@pytest.mark.parametrize(
    "phase_name",
    ["stabilize", "first_hand", "second_hand", "first_foot", "second_foot"],
)
def test_record_ladder_video_accepts_phase_prefix(phase_name: str) -> None:
    args = record_ladder_video.parse_args(
        ["--checkpoint", "model.pt", "--ladder_phase", phase_name]
    )

    assert args.ladder_phase == phase_name


@pytest.mark.parametrize(
    ("phase_name", "expected_phase"),
    [
        ("stabilize", 0),
        ("first_hand", 1),
        ("second_hand", 2),
        ("first_foot", 3),
        ("second_foot", 4),
        (None, 4),
    ],
)
def test_record_ladder_video_configures_phase_prefix(
    phase_name: str | None,
    expected_phase: int,
) -> None:
    command_cfg = SimpleNamespace(
        fixed_max_unlocked_phase=None,
        freeze_at_max_unlocked_phase=False,
    )
    env_cfg = SimpleNamespace(commands={"ladder": command_cfg})

    selected = configure_ladder_play_phase(env_cfg, phase_name)

    assert command_cfg.fixed_max_unlocked_phase == expected_phase
    assert command_cfg.freeze_at_max_unlocked_phase is False
    assert selected == (phase_name or "second_foot")


def test_record_ladder_video_freezes_explicit_selected_phase() -> None:
    command_cfg = SimpleNamespace(
        fixed_max_unlocked_phase=None,
        freeze_at_max_unlocked_phase=False,
    )
    env_cfg = SimpleNamespace(commands={"ladder": command_cfg})

    configure_ladder_play_phase(
        env_cfg,
        "stabilize",
        freeze_at_boundary=True,
    )

    assert command_cfg.fixed_max_unlocked_phase == 0
    assert command_cfg.freeze_at_max_unlocked_phase is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frames": 0}, "--frames must be positive"),
        ({"warmup_steps": -1}, "--warmup_steps must be >= 0"),
        ({"width": 0}, "--width and --height must be positive"),
        ({"height": -1}, "--width and --height must be positive"),
        ({"fps": 0}, "--fps must be positive"),
        ({"camera_distance": 0.0}, "--camera_distance must be positive"),
        ({"camera_fovy": 0.0}, "--camera_fovy must be in"),
        ({"camera_fovy": 180.0}, "--camera_fovy must be in"),
    ],
)
def test_record_ladder_video_validates_arguments(
    overrides: dict[str, object],
    message: str,
) -> None:
    args = argparse.Namespace(
        frames=100,
        warmup_steps=0,
        width=1280,
        height=720,
        fps=None,
        camera_distance=4.5,
        camera_fovy=48.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)

    with pytest.raises(ValueError, match=message):
        record_ladder_video._validate_args(args)


def test_record_ladder_video_uses_fixed_outside_ladder_camera() -> None:
    world_origin = object()
    viewer = SimpleNamespace(
        OriginType=SimpleNamespace(WORLD=world_origin),
        origin_type=None,
        entity_name="robot",
        body_name="torso_link",
        lookat=(0.0, 0.0, 0.0),
        distance=4.0,
        azimuth=120.0,
        elevation=-10.0,
        fovy=None,
    )
    env_cfg = SimpleNamespace(viewer=viewer)
    args = SimpleNamespace(
        camera_lookat=(-0.45, 0.0, 1.45),
        camera_distance=4.5,
        camera_azimuth=30.0,
        camera_elevation=-5.0,
        camera_fovy=48.0,
    )

    record_ladder_video._configure_recording_camera(env_cfg, args)

    assert viewer.origin_type is world_origin
    assert viewer.entity_name is None
    assert viewer.body_name is None
    assert viewer.lookat == (-0.45, 0.0, 1.45)
    assert viewer.distance == 4.5
    assert viewer.azimuth == 30.0
    assert viewer.elevation == -5.0
    assert viewer.fovy == 48.0


def test_record_ladder_video_requires_mp4_suffix(tmp_path) -> None:
    args = argparse.Namespace(
        checkpoint=str(tmp_path / "model.pt"),
        output=str(tmp_path / "video.avi"),
    )

    with pytest.raises(ValueError, match="--output must end with .mp4"):
        record_ladder_video._output_path(args)


@pytest.mark.parametrize(
    ("system", "backend", "pyopengl_backend"),
    [
        ("Windows", "glfw", None),
        ("Linux", "egl", "egl"),
    ],
)
def test_record_ladder_video_selects_platform_gl_backend(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    backend: str,
    pyopengl_backend: str | None,
) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.setattr(record_ladder_video.platform, "system", lambda: system)

    record_ladder_video._configure_video_backend()

    assert record_ladder_video.os.environ["MUJOCO_GL"] == backend
    assert record_ladder_video.os.environ.get("PYOPENGL_PLATFORM") == pyopengl_backend


def test_record_ladder_video_missing_checkpoint_fails_before_training_import(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.pt"

    assert record_ladder_video.main(["--checkpoint", str(missing)]) == 1
