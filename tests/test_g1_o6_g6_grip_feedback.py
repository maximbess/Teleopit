from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest


def test_pressure_servo_uses_opposition_and_individual_loads_with_bounded_travel():
    from teleopit.sim.g1_o6_support.g6_grip_feedback import GripFeedback, pressure_rates, advance

    profile = GripFeedback(grip_law="pressure", pressure_gain=.12)
    loads = np.array([[.7, .2, 1.4, 1.8, 1.4], [.7, 1.4, 1.4, 1.4, 1.4]])
    rates = pressure_rates(profile, loads, [True, False], np.zeros(2))
    assert rates.shape == (2, 6)
    np.testing.assert_array_equal(rates[:, 1], 0.)
    assert rates[0, 0] == 0. and rates[1, 0] > 0.
    assert rates[0, 2] > 0. and rates[0, 4] < 0. and rates[0, 3] == 0.
    assert np.max(abs(rates)) == pytest.approx(.04)
    anchor = np.full((2, 6), .4)
    cap = anchor.copy()
    for _ in range(300):
        updated = advance(cap, anchor, rates, anchor + 1., .02)
        assert np.max(abs(updated - cap)) <= .0008 + 1e-12
        cap = updated
    assert cap[0, 2] == pytest.approx(.52)
    assert cap[0, 4] == pytest.approx(.28)
    # An already unattainable positive correction must not accumulate.
    blocked = advance(anchor, anchor, rates, anchor - .01, .02)
    assert blocked[0, 2] == anchor[0, 2] and blocked[0, 4] < anchor[0, 4]
    coupled = replace(profile, grip_law="palm")
    motion = pressure_rates(coupled, np.full((2, 5), 1.5), [True, True], np.array([.002, -.002]))
    assert np.all(motion[0, 2:] > 0) and np.all(motion[1, 2:] < 0)
    with pytest.raises(ValueError):
        replace(profile, pressure_gain=10.)


def test_grip_stage_requires_uninterrupted_opposition_at_both_cadences_and_every_previous_gate():
    from teleopit.sim.g1_o6_support.g6_grip_feedback_experiment import SupportTrace, passes
    from test_g1_o6_g6_ground_reaction import passing_hold

    hand = {"thumb_opposition": True, "thumb_normal_N": .5, "minimum_finger_normal_N": 1.1}
    quality = {s: hand.copy() for s in ("left", "right")}
    trace = SupportTrace()
    trace.observe(.001, quality)
    quality["right"]["thumb_opposition"] = False
    trace.observe(.002, quality)
    assert trace.first_loss_s["right_thumb"] == .002 and trace.loss_steps["right_thumb"] == 1
    quality["right"]["thumb_opposition"] = True
    clean = SupportTrace()
    clean.observe(.001, quality)
    good = vars(clean)
    value = passing_hold()
    audit = value["physics_audit"]
    audit["scored_steps"] = 120000
    audit["settled_grip"] = {"capture_s": 5.}
    audit["grip_support"] = {s: {**deepcopy(good), "samples": count} for s, count in (("native", 120000), ("control", 6000))}
    audit["knee_rung3"] = {s: dict(samples=count, minimum_m=.003, contact_steps=0) for s, count in (("native", 120000), ("control", 6000))}
    assert passes(value, "120s")
    for cadence in ("native", "control"):
        for key in ("left_thumb", "right_thumb", "left_fingers"):
            record = audit["grip_support"][cadence]["loss_steps"]
            record[key] = 1
            assert not passes(value, "120s")
            record[key] = 0
    audit["knee_rung3"]["native"]["contact_steps"] = 1
    assert not passes(value, "120s")


def test_real_feedback_replays_through_native_preview_and_restores_at_half_timestep():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_grip_feedback_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import canonical, restore_environment

    c = candidates()[-1]
    assert G6Candidate.from_dict(c.record()) == c
    env = make_environment(c, 6, .001)
    meta = canonical(experiment_metadata(env, snapshot(env), {"mode": "long-hold"}))
    half = restore_environment(meta, .0005)
    assert half.grasp_acquisition == c.controller and half.model.opt.timestep == .0005
    ref = env.hold_controller
    while ref.grip.capture_s is None:
        env.step()
        assert env.phase not in ("HOLD_COMPLETE", "ABORT")
    assert ref.grip.capture_s == pytest.approx(5.)
    forces = []
    def sample(e):
        forces.append(np.linalg.norm(snapshot(e)["static_balance_residual_N"]))
    for _ in range(4):
        forces.clear()
        env.step(sample)
        sample(env)
        assert ref.last.peak_N == pytest.approx(max(forces), abs=1e-8)
    assert np.max(abs(ref.cap - ref.grip.ceiling.reshape(2, 6))) > 0
    assert np.all(env.data.ctrl[ref.fingers] <= ref.uncapped + 1e-12)
