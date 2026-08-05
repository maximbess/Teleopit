"""CLI and report-helper tests for the RL-only ladder benchmark."""

from __future__ import annotations

import argparse

import pytest

from train_mimic.scripts import benchmark_ladder


def test_benchmark_ladder_cli_is_ladder_only() -> None:
    args = benchmark_ladder.parse_args(["--checkpoint", "model.pt"])

    assert args.num_envs == 1
    assert args.num_eval_steps == 5000
    assert args.episode_length_s == 20.0
    assert not hasattr(args, "motion_file")
    assert not hasattr(args, "task")

    with pytest.raises(SystemExit):
        benchmark_ladder.parse_args(
            ["--checkpoint", "model.pt", "--motion_file", "motions"]
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_envs": 0}, "--num_envs must be positive"),
        ({"num_eval_steps": 0}, "--num_eval_steps must be positive"),
        ({"warmup_steps": -1}, "--warmup_steps must be >= 0"),
        ({"episode_length_s": 0.0}, "--episode_length_s must be positive"),
        (
            {"video": True, "num_envs": 2},
            "--video requires --num_envs 1",
        ),
        ({"video_length": 0}, "--video_length must be positive"),
    ],
)
def test_benchmark_ladder_validates_rollout_arguments(
    overrides: dict[str, object],
    message: str,
) -> None:
    args = argparse.Namespace(
        num_envs=1,
        num_eval_steps=100,
        warmup_steps=0,
        episode_length_s=20.0,
        video=False,
        video_length=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)

    with pytest.raises(ValueError, match=message):
        benchmark_ladder._validate_args(args)


def test_benchmark_ladder_stats_and_rates_handle_no_completed_episodes() -> None:
    assert benchmark_ladder._rate(0, 0) is None
    assert benchmark_ladder._rate(3, 4) == pytest.approx(0.75)
    assert benchmark_ladder._stats([])["mean"] is None

    stats = benchmark_ladder._stats([0.0, 0.5, 1.0])
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["p50"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(1.0)


def test_benchmark_ladder_missing_checkpoint_fails_before_training_import(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.pt"

    assert benchmark_ladder.main(["--checkpoint", str(missing)]) == 1
