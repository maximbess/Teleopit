import numpy as np
import pytest


def test_grip_preview_favors_an_executed_interval_with_all_four_support_conditions():
    from teleopit.sim.g1_o6_support.g6_grip_preview import GraspForecast, choose_grip, alternatives

    unsafe = GraspForecast(12., 1., 0., .7)
    safe = GraspForecast(20., 1., .30, .42)
    healthy = GraspForecast(12., 1., .58, .54)
    assert choose_grip(healthy, [("unused", 1)], lambda _: pytest.fail("extra simulation")) == ("nominal", None, healthy, 0)
    candidates = [("thumb", 1), ("raise", 2), ("lower", 3)]
    forecasts = {1: GraspForecast(25., 1., 1., 1.), 2: safe, 3: GraspForecast(13., 0., .8, .8)}
    assert choose_grip(unsafe, candidates, forecasts.get) == ("raise", 2, safe, 3)
    assert choose_grip(unsafe, candidates, lambda _: healthy) == ("thumb", 1, healthy, 1)
    anchor = np.full((2,6), .4)
    choices = dict(alternatives(anchor, anchor, anchor, anchor+1., .02))
    assert len(choices)==3
    assert choices["thumb"][0,0] == pytest.approx(.4008)
    assert choices["raise"][1,3] == pytest.approx(.4008)
    assert choices["lower"][0,4] == pytest.approx(.3992)
    assert all(np.array_equal(x[:,1],anchor[:,1]) for x in choices.values())
    near = anchor + .12
    assert np.max(dict(alternatives(near, near, anchor, near+1., .02))["raise"]-anchor) <= .12+1e-12


def test_native_grip_forecast_matches_each_solver_step_and_final_endpoint():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_grip_preview_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from scripts.verify_g1_o6_g6_grip_feedback import grasp_sample

    candidate = candidates()[-1]
    assert G6Candidate.from_dict(candidate.record()) == candidate
    env = make_environment(candidate, 6, .001)
    ref = env.hold_controller
    while ref.grip.capture_s is None:
        env.step()
        assert env.phase not in ("ABORT","HOLD_COMPLETE")
    def sample(e):
        row = snapshot(e)
        row["support_targets"] = e.support_targets
        readings = [grasp_sample(row, s) for s in ("left", "right")]
        actual.append([np.linalg.norm(row["static_balance_residual_N"]), min(readings[0][0][1:]),
                       *[forces[0] if opposed else 0. for forces,opposed in readings]])
    for _ in range(3):
        actual=[]
        env.step(sample)
        sample(env)
        force, finger, left, right = np.asarray(actual).T
        assert [ref.last.peak_N, ref.last.finger_N, ref.last.left_thumb_N, ref.last.right_thumb_N] == pytest.approx(
            [max(force), min(finger), min(left), min(right)],abs=1e-8)
        assert 0 <= ref.extra <= 3 and ref.count <= 21
        assert len(ref.state()["grip_preview"]["tested_forecasts"]) == ref.extra
