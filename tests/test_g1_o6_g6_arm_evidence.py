"""Reconstruct actuator command motion without trusting controller diagnostics."""

import numpy as np
import pytest


def test_applied_command_check_detects_motion_despite_forged_controller_state():
    from scripts.g1_o6_g6_arm_report import command_sample, assert_bounds
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    env = make_environment(candidates()[2], 5, .001)
    row = snapshot(env)
    row["applied_position_targets_rad"] = env.data.ctrl.tolist()
    baseline = np.array(row["applied_position_targets_rad"])
    row["pose_search"] = {"controller_state": {"command_displacement_m": {"left": 0, "right": 0}}}
    indices = env.hold_controller.indices
    mounts = {side: hand["mount"] for side, hand in env.hands.items()}
    sample = command_sample(env.model, mounts, indices, baseline, baseline, row, .02)
    assert_bounds(sample, .02)
    for side in indices:
        row["applied_position_targets_rad"][indices[side][0]] += .03
    sample = command_sample(env.model, mounts, indices, baseline, baseline, row, .02)
    assert sample["translation_m"] > .005
    assert sample["rate_rad_s"] > .02
    with pytest.raises(AssertionError):
        assert_bounds(sample, .02)


def test_arm_evidence_rejects_each_independent_budget_violation():
    from scripts.g1_o6_g6_arm_report import assert_bounds

    valid = {"translation_m": .005, "rotation_rad": .03,
             "offset_rad": .03, "rate_rad_s": .02}
    assert_bounds(valid, .02)
    for key, value in valid.items():
        with pytest.raises(AssertionError):
            assert_bounds({**valid, key: 1.01 * value}, .02)
