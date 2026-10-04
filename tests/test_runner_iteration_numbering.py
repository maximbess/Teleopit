from __future__ import annotations

import pytest

import torch

from train_mimic.tasks.tracking.rl.runner import (
    _format_duration,
    _mean_episode_length_s,
    _one_based_iteration_range,
    _ordered_episode_extra_keys,
    _resolve_total_iterations,
    _termination_counts_to_rates,
)


def test_one_based_iteration_range_starts_at_one_for_fresh_run() -> None:
    assert list(_one_based_iteration_range(0, 10)) == list(range(1, 11))


def test_one_based_iteration_range_resumes_from_completed_iteration() -> None:
    assert list(_one_based_iteration_range(10, 12)) == [11, 12]


def test_one_based_iteration_range_is_empty_when_already_at_target() -> None:
    assert list(_one_based_iteration_range(10, 10)) == []


def test_one_based_iteration_range_rejects_target_below_completed() -> None:
    with pytest.raises(ValueError, match='num_learning_iterations'):
        _one_based_iteration_range(11, 10)


def test_resolve_total_iterations_preserves_fresh_run_count() -> None:
    assert _resolve_total_iterations(0, 10) == 10


def test_resolve_total_iterations_adds_requested_iterations_on_resume() -> None:
    assert _resolve_total_iterations(10, 12) == 22


def test_resolve_total_iterations_rejects_negative_requested_iterations() -> None:
    with pytest.raises(ValueError, match='non-negative'):
        _resolve_total_iterations(10, -1)


def test_mean_episode_length_is_reported_in_seconds() -> None:
    assert _mean_episode_length_s([50.0, 70.0], 0.02) == pytest.approx(1.2)


def test_mean_episode_length_rejects_a_non_positive_step() -> None:
    with pytest.raises(ValueError, match="step_dt"):
        _mean_episode_length_s([10.0], 0.0)


def test_format_duration_keeps_hours_above_one_day() -> None:
    assert _format_duration(33 * 3600 + 16 * 60 + 25) == "33:16:25"


def test_episode_extra_keys_include_resets_later_in_rollout() -> None:
    ep_extras = [
        {},
        {"Episode_Reward/height": 1.0},
        {},
        {
            "Episode_Termination/time_out": 1.0,
            "Episode_Reward/height": 2.0,
        },
    ]

    assert _ordered_episode_extra_keys(ep_extras) == (
        "Episode_Reward/height",
        "Episode_Termination/time_out",
    )


def test_termination_counts_become_the_fraction_of_resets() -> None:
    extras = {
        "log": {
            "Episode_Termination/fell_over": 2.0,
            "Episode_Termination/success": torch.tensor(1.0),
            "Episode_Reward/ladder_climb": -0.01,
        }
    }

    _termination_counts_to_rates(extras, torch.tensor([1, 0, 1, 1]))

    assert extras["log"]["Episode_Termination/fell_over"] == pytest.approx(2.0 / 3.0)
    assert extras["log"]["Episode_Termination/success"].item() == pytest.approx(1.0 / 3.0)
    assert extras["log"]["Episode_Reward/ladder_climb"] == -0.01


def test_termination_counts_stay_put_when_nothing_reset() -> None:
    extras = {"log": {"Episode_Termination/fell_over": 4.0}}

    _termination_counts_to_rates(extras, torch.zeros(4))

    assert extras["log"]["Episode_Termination/fell_over"] == 4.0
