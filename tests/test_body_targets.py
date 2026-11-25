"""Sampling and reward math for body-point targets, without a simulator."""

from __future__ import annotations

import torch

from train_mimic.tasks.tracking.mdp.body_targets import (
    TARGET_ALIVE_RATE,
    TARGET_BALL_RADIUS_M,
    TARGET_DISTANCE_CAP_M,
    TARGET_GAIN_SCALE,
    assign_targets,
    point_distance_sum,
    sample_active_indices,
    sample_ball_offsets,
    target_gain_rate,
    target_position_rate,
)


def test_ball_offsets_are_volume_uniform_and_inside_the_radius() -> None:
    generator = torch.Generator().manual_seed(7)
    offsets = sample_ball_offsets(
        4096, TARGET_BALL_RADIUS_M, generator=generator
    )
    radius = torch.linalg.vector_norm(offsets, dim=-1)
    assert torch.all(radius <= TARGET_BALL_RADIUS_M + 1.0e-5)
    cubed = (radius / TARGET_BALL_RADIUS_M).pow(3).mean()
    assert cubed.item() == pytest_approx(0.5, abs=0.03)


def test_one_target_moves_and_the_others_stay() -> None:
    points = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.3, 0.2], [-0.1, 0.0, 0.2], [0.0, 0.0, 1.6]],
            [[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.0, 0.3, 0.1], [0.9, 0.0, 0.1], [1.0, 0.0, 0.5]],
        ]
    )
    active = torch.tensor([4, 0])
    offsets = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.05, 0.1]])
    targets = assign_targets(points, active, offsets)

    moved = torch.linalg.vector_norm(targets - points, dim=-1) > 1.0e-8
    assert moved.sum(dim=-1).tolist() == [1, 1]
    assert moved[0].tolist() == [False, False, False, False, True]
    assert moved[1].tolist() == [True, False, False, False, False]
    torch.testing.assert_close(targets[0, 4], points[0, 4] + offsets[0])
    torch.testing.assert_close(targets[1, 0], points[1, 0] + offsets[1])
    assert point_distance_sum(points, targets).tolist() == pytest_approx(
        [0.1, torch.linalg.vector_norm(offsets[1]).item()]
    )


def test_active_index_covers_each_point() -> None:
    generator = torch.Generator().manual_seed(3)
    indices = sample_active_indices(500, 5, generator=generator)
    assert set(indices.tolist()) == {0, 1, 2, 3, 4}


def test_position_rate_stays_positive_inside_the_ball_and_is_zero_on_the_ground() -> None:
    distance = torch.tensor([0.0, TARGET_BALL_RADIUS_M, TARGET_DISTANCE_CAP_M, 4.0])
    on_ground = torch.tensor([False, False, False, False])
    rate = target_position_rate(distance, on_ground)
    assert rate[1].item() > 0.0
    torch.testing.assert_close(
        rate,
        torch.tensor(
            [
                TARGET_ALIVE_RATE,
                TARGET_ALIVE_RATE - TARGET_BALL_RADIUS_M,
                TARGET_ALIVE_RATE - TARGET_DISTANCE_CAP_M,
                TARGET_ALIVE_RATE - TARGET_DISTANCE_CAP_M,
            ]
        ),
    )
    grounded = target_position_rate(distance, torch.ones(4, dtype=torch.bool))
    assert torch.all(grounded == 0.0)


def test_gain_is_zero_on_the_first_sample_and_then_telescopes() -> None:
    distances = torch.tensor([0.20, 0.10, 0.00])
    previous = torch.zeros(1)
    fresh = torch.tensor([True])
    on_ground = torch.tensor([False])
    dt = 0.02
    paid = []
    for distance in distances:
        current = distance.view(1)
        rate = target_gain_rate(
            previous, current, fresh=fresh, on_ground=on_ground, dt=dt
        )
        paid.append(float((rate * dt).item()))
        previous = current.clone()
        fresh = torch.tensor([False])
    assert paid[0] == 0.0
    assert paid[1] == pytest_approx(TARGET_GAIN_SCALE * (0.20 - 0.10))
    assert paid[2] == pytest_approx(TARGET_GAIN_SCALE * (0.10 - 0.00))
    assert sum(paid) == pytest_approx(TARGET_GAIN_SCALE * 0.20)
    grounded = target_gain_rate(
        torch.tensor([0.2]),
        torch.tensor([0.0]),
        fresh=torch.tensor([False]),
        on_ground=torch.tensor([True]),
        dt=dt,
    )
    assert grounded.item() == 0.0


def pytest_approx(value: float, abs: float | None = None):
    import pytest

    if abs is None:
        return pytest.approx(value)
    return pytest.approx(value, abs=abs)
