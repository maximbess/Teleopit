from dataclasses import replace

import numpy as np
import pytest


def test_wrist_velocity_seats_the_thumb_and_recovers_tangential_contact_location():
    from teleopit.sim.g1_o6_support.g6_wrist_grip import WristGrip, pad_velocity, joint_rate

    zero = np.zeros(3)
    np.testing.assert_array_equal(pad_velocity(zero, .65, True), zero)
    missing = pad_velocity(np.array([.02, -.02, .2]), 1.1, False)
    assert missing[0] > 0 > missing[1] and missing[2] > 0
    assert np.linalg.norm(missing[:2]) == pytest.approx(.002)
    assert missing[2] == pytest.approx(.002)
    assert pad_velocity(zero, 1.0, True)[2] < 0
    j = np.column_stack((np.eye(3), np.zeros(3), np.eye(3)))
    offset = np.full(7, .01)
    for distributed in (False, True):
        rate = joint_rate(j, missing, offset, distributed)
        np.testing.assert_allclose(j @ rate, missing, atol=2e-6)
        if distributed:
            assert not np.allclose(rate[:4], -.4*offset[:4])
        else:
            np.testing.assert_allclose(rate[:4], -.4*offset[:4])
    with pytest.raises(ValueError):
        replace(WristGrip(), wrist_mode="unbounded")


def test_wrist_grip_runs_through_normal_entry_and_keeps_joint_and_cartesian_gates():
    from scripts.g1_o6_g6_arm_report import assert_bounds
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_wrist_grip_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    c = next(c for c in candidates() if c.id == "both_wrists")
    assert G6Candidate.from_dict(c.record()) == c
    env = make_environment(c, 6, .001)
    ref = env.hold_controller
    arm = ref.parent
    previous = {s: v.copy() for s, v in arm.offset.items()}
    while min(arm.updates.values()) < 4:
        env.step()
        assert env.phase not in ("ABORT","HOLD_COMPLETE")
        for side in arm.displacement:
            assert_bounds({"translation_m":arm.displacement[side], "rotation_rad":arm.angle[side],
                           "offset_rad":max(abs(arm.offset[side])),
                           "rate_rad_s":max(abs(arm.offset[side]-previous[side]))/.02}, .02)
        previous = {s: v.copy() for s,v in arm.offset.items()}
    assert arm.capture_s == pytest.approx(ref.grip.capture_s) == pytest.approx(5.)
    assert len(arm.landmarks) == 2 and ref.cap is not None
    assert ref.last.right_thumb_N > .05 and ref.last.finger_N > .05 and ref.last.peak_N < 25
    assert ref.state()["wrist_grip"]["updates"] == arm.updates
    assert env.model.opt.timestep == .001
    assert make_environment(c, 5, .0005).model.opt.timestep == .0005


def test_wrist_campaign_keeps_prior_support_gates_and_requires_active_feedback():
    from teleopit.sim.g1_o6_support.g6_wrist_grip_experiment import wrist_exercised

    audit = {"settled_grip":{"capture_s":5.}}
    assert not wrist_exercised(audit)
    audit["wrist_grip"] = {"mode":"both","capture_s":5.,"updates":{"left":1,"right":0}}
    assert not wrist_exercised(audit)
    audit["wrist_grip"]["updates"]["right"] = 1
    assert wrist_exercised(audit)
    audit["wrist_grip"]["capture_s"] = 4.
    assert not wrist_exercised(audit)
