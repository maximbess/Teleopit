"""Sampling and reward math for body-point targets, without a simulator."""

from __future__ import annotations

import torch

from train_mimic.tasks.tracking.mdp.body_targets import (
    COMMAND_DIM,
    PRIVILEGED_DIM,
    TARGET_ALIVE_RATE,
    TARGET_BALL_RADIUS_M,
    TARGET_DISTANCE_CAP_M,
    TARGET_GAIN_SCALE,
    assign_targets,
    pack_body_target_command,
    pack_privileged,
    point_distance_sum,
    point_gain_valid,
    sample_active_indices,
    sample_ball_offsets,
    sample_point_weights,
    target_gain_rate,
    target_position_rate,
    weighted_distance_sum,
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
    distances = torch.tensor([[0.20, 0.0, 0.0, 0.0, 0.0], [0.10, 0.0, 0.0, 0.0, 0.0], [0.00, 0.0, 0.0, 0.0, 0.0]])
    weights = torch.ones(1, 5)
    previous = torch.zeros(1, 5)
    valid = torch.zeros(1, 5, dtype=torch.bool)
    on_ground = torch.tensor([False])
    dt = 0.02
    paid = []
    for distance in distances:
        current = distance.view(1, 5)
        rate = target_gain_rate(
            previous,
            current,
            weights=weights,
            valid=valid,
            on_ground=on_ground,
            dt=dt,
        )
        paid.append(float((rate * dt).item()))
        previous = current.clone()
        valid = torch.ones(1, 5, dtype=torch.bool)
    assert paid[0] == 0.0
    assert paid[1] == pytest_approx(TARGET_GAIN_SCALE * (0.20 - 0.10))
    assert paid[2] == pytest_approx(TARGET_GAIN_SCALE * (0.10 - 0.00))
    assert sum(paid) == pytest_approx(TARGET_GAIN_SCALE * 0.20)
    grounded = target_gain_rate(
        torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0]]),
        torch.zeros(1, 5),
        weights=weights,
        valid=torch.ones(1, 5, dtype=torch.bool),
        on_ground=torch.tensor([True]),
        dt=dt,
    )
    assert grounded.item() == 0.0


def test_weight_sampler_keeps_at_most_one_soft_limb_and_can_zero_the_head() -> None:
    generator = torch.Generator().manual_seed(11)
    weights, swing = sample_point_weights(4096, generator=generator)
    assert weights.shape == (4096, 5)
    assert swing.shape == (4096,)
    assert int(swing.min()) == 0 and int(swing.max()) == 3
    assert torch.all((weights >= 0.0) & (weights <= 1.0))
    soft = weights[:, :4] < 0.75
    assert torch.all(soft.sum(dim=-1) <= 1)
    softened = soft.any(dim=-1)
    assert softened.float().mean().item() == pytest_approx(0.5, abs=0.03)
    assert torch.all(soft[softened].to(dtype=torch.int64).argmax(dim=-1) == swing[softened])
    zeros = weights[:, 4] == 0.0
    assert zeros.float().mean().item() == pytest_approx(0.05, abs=0.02)
    assert int(zeros.sum()) > 0


def test_weighted_position_rate_uses_the_weight_outside_each_distance() -> None:
    distances = torch.tensor([[0.20, 0.10, 0.0, 0.0, 0.50]])
    weights = torch.tensor([[1.0, 0.5, 1.0, 0.0, 0.2]])
    weighted = weighted_distance_sum(distances, weights)
    assert weighted.item() == pytest_approx(0.20 + 0.05 + 0.10)
    rate = target_position_rate(weighted, torch.tensor([False]))
    assert rate.item() == pytest_approx(TARGET_ALIVE_RATE - weighted.item())
    grounded = target_position_rate(weighted, torch.tensor([True]))
    assert grounded.item() == 0.0


def test_weight_change_on_a_stationary_body_pays_no_gain() -> None:
    distances = torch.tensor([[0.20, 0.10, 0.05, 0.00, 0.40]])
    valid = point_gain_valid(
        torch.ones(1, 5, dtype=torch.bool),
        torch.tensor([[1.0, 1.0, 0.8, 1.0, 0.2]]),
    )
    rate = target_gain_rate(
        distances,
        distances.clone(),
        weights=torch.tensor([[0.2, 0.9, 1.0, 0.75, 0.0]]),
        valid=valid,
        on_ground=torch.tensor([False]),
        dt=0.02,
    )
    assert rate.item() == 0.0


def test_target_rewrite_pays_no_gain_and_the_next_step_telescopes() -> None:
    previous = torch.tensor([[0.20, 0.0, 0.0, 0.0, 0.0]])
    rewritten = torch.tensor([[0.40, 0.0, 0.0, 0.0, 0.0]])
    weights = torch.ones(1, 5)
    invalid = point_gain_valid(
        torch.zeros(1, 5, dtype=torch.bool),
        torch.ones(1, 5),
    )
    rewrite = target_gain_rate(
        previous,
        rewritten,
        weights=weights,
        valid=invalid,
        on_ground=torch.tensor([False]),
        dt=0.02,
    )
    assert rewrite.item() == 0.0
    arrived = torch.tensor([[0.10, 0.0, 0.0, 0.0, 0.0]])
    following = target_gain_rate(
        rewritten,
        arrived,
        weights=weights,
        valid=torch.ones(1, 5, dtype=torch.bool),
        on_ground=torch.tensor([False]),
        dt=0.02,
    )
    assert (following * 0.02).item() == pytest_approx(TARGET_GAIN_SCALE * (0.40 - 0.10))
    from_zero = point_gain_valid(torch.ones(1, 5, dtype=torch.bool), torch.zeros(1, 5))
    assert not bool(from_zero.any())


def test_command_stays_35d_and_a_zero_weight_hides_the_error() -> None:
    points = torch.zeros(2, 5, 3)
    targets = torch.zeros(2, 5, 3)
    targets[0, 0] = torch.tensor([0.20, 0.0, 0.0])
    targets[1, 4] = torch.tensor([0.0, 0.05, 0.0])
    weights = torch.tensor(
        [
            [0.1, 1.0, 1.0, 1.0, 0.15],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    command = pack_body_target_command(weights, points, targets)
    assert command.shape == (2, COMMAND_DIM)
    torch.testing.assert_close(command[:, :5], weights)
    torch.testing.assert_close(command[0, 5:8], targets[0, 0])
    torch.testing.assert_close(command[0, 20:23], targets[0, 0] - points[0, 0])
    torch.testing.assert_close(command[1, 17:20], points[1, 4])
    torch.testing.assert_close(command[1, 32:35], torch.zeros(3))
    distances = torch.linalg.vector_norm(points - targets, dim=-1)
    privileged = pack_privileged(distances, weights)
    assert privileged.shape == (2, PRIVILEGED_DIM)
    torch.testing.assert_close(privileged[:, :5], distances)
    torch.testing.assert_close(
        privileged[:, 5], weighted_distance_sum(distances, weights)
    )


def pytest_approx(value: float, abs: float | None = None):
    import pytest

    if abs is None:
        return pytest.approx(value)
    return pytest.approx(value, abs=abs)
