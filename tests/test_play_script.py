"""CLI tests for policy playback."""

from __future__ import annotations

import argparse

import pytest

from train_mimic.scripts import play
from train_mimic.tasks.tracking.config.constants import (
    GENERAL_TRACKING_TASK,
    LADDER_RL_TASK,
)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "task": GENERAL_TRACKING_TASK,
        "motion_file": "data/datasets_precomputed",
        "num_envs": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_play_cli_accepts_ladder_task_without_motion_file() -> None:
    args = play.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--task",
            LADDER_RL_TASK,
        ]
    )

    play._validate_args(args)

    assert args.motion_file is None
    assert not hasattr(args, "ladder_phase")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_envs": 0}, "--num_envs must be positive"),
        ({"motion_file": None}, "--motion_file is required"),
        (
            {"task": LADDER_RL_TASK, "motion_file": "unused"},
            "--motion_file is not used",
        ),
    ],
)
def test_play_validates_task_specific_arguments(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        play._validate_args(_args(**overrides))
