"""CLI and ladder-phase tests for policy playback."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from train_mimic.scripts import play
from train_mimic.ladder_playback import configure_ladder_play_phase
from train_mimic.tasks.tracking.config.constants import (
    GENERAL_TRACKING_TASK,
    LADDER_RL_TASK,
)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "task": GENERAL_TRACKING_TASK,
        "motion_file": "data/datasets_precomputed",
        "ladder_phase": None,
        "num_envs": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_play_cli_accepts_ladder_phase_without_motion_file() -> None:
    args = play.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--task",
            LADDER_RL_TASK,
            "--ladder_phase",
            "second_hand",
        ]
    )

    play._validate_args(args)

    assert args.motion_file is None
    assert args.ladder_phase == "second_hand"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_envs": 0}, "--num_envs must be positive"),
        ({"motion_file": None}, "--motion_file is required"),
        ({"ladder_phase": "stabilize"}, "--ladder_phase is only valid"),
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
def test_play_configures_deepest_ladder_phase(
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
