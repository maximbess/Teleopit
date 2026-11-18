"""The contact preview must predict the untouched simulator and preserve the gates."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest


def test_preview_prefers_a_safe_command_and_records_when_every_choice_fails():
    from teleopit.sim.g1_o6_support.g6_ground_reaction import Forecast, choose_preview

    options = [("nominal", 0), ("half", 1), ("trim", 2)]
    observed = [Forecast(26., .7), Forecast(15., .03), Forecast(18., .8)]
    choice, forecast, count = choose_preview(options, lambda n: observed[n], 20.)
    assert choice == ("trim", 2) and forecast == observed[2] and count == 3
    assert choose_preview(options, lambda n: Forecast(10., 1.), 20.)[2] == 1
    observed[:] = [Forecast(26., 1.), Forecast(24., .2), Forecast(15., .04)]
    assert choose_preview(options, lambda n: observed[n], 20.)[0] == ("half", 1)
    observed[:] = [Forecast(29., 1.), Forecast(26., 1.), Forecast(28., .03)]
    assert choose_preview(options, lambda n: observed[n], 20.)[0] == ("half", 1)


def test_preview_matches_executed_native_steps_and_endpoint_without_live_state_writes():
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_ground_reaction_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.pose_contacts import grasp_quality, hand_contacts
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate

    candidate = candidates()[1]
    assert G6Candidate.from_dict(candidate.record()) == candidate
    env = make_environment(candidate, .1, .001)
    controller = env.hold_controller
    while env.phase != "HOLD":
        env.step()
        assert controller.last is None
    def measured(env):
        row = snapshot(env)
        quality = grasp_quality(hand_contacts(env, row["contacts"], "left"),
                                env.data.geom(env.support_targets["left_hand"]).xpos[2])
        actual.append((np.linalg.norm(row["static_balance_residual_N"]),quality["minimum_finger_normal_N"]))
    before = [getattr(env.data,k).copy() for k in ("qpos","qvel","ctrl","qfrc_applied","xfrc_applied")]
    controller.preview.evaluate(env, env.data.ctrl.copy())
    for k, original in zip(("qpos","qvel","ctrl","qfrc_applied","xfrc_applied"), before):
        np.testing.assert_array_equal(getattr(env.data,k), original)
    for _ in range(3):
        actual = []
        env.step(measured)
        measured(env)
        assert len(actual)==21
        assert controller.last.peak_N == pytest.approx(max(x[0] for x in actual), abs=1e-8)
        assert controller.last.finger_N == pytest.approx(min(x[1] for x in actual), abs=1e-8)
    env.reset()
    assert env.hold_controller.last is None
    for options in (dict(gain=.8),dict(governor="unknown"),dict(guard_N=25.)):
        with pytest.raises(ValueError):
            replace(candidate.controller, **options)


def test_refinement_requires_both_residual_cadences_all_fingers_and_original_long_gates():
    from teleopit.sim.g1_o6_support.g6_ground_reaction_experiment import passes

    result = {"strict_status":"PASS", "strict":{"unsupported_hold_seconds":120},
              "drift_assessment":{"status":"PASS"},
              "contact_quality":{"status":"PASS","minimum_group_normal_N":{"left":{"finger":.12}}},
              "physics_audit":{"no_external_support":True,"housing_contact_steps":0,"unintended_support_steps":0,
                               "native_hand_friction":{"status":"PASS"}, "extrema":{"force_balance":24.9},
                               "ground_reaction":{"control":{"samples":6000,"peak_N":24.9}}}}
    assert passes(result, "120s")
    for fields, key, bad in ((result["physics_audit"]["extrema"],"force_balance",25.),
                             (result["physics_audit"]["ground_reaction"]["control"],"peak_N",25.),
                             (result["contact_quality"]["minimum_group_normal_N"]["left"],"finger",0.),
                             (result["drift_assessment"],"status","FAIL")):
        original = fields[key]
        fields[key] = bad
        assert not passes(result,"120s")
        fields[key] = original
    del result["physics_audit"]["ground_reaction"]
    assert not passes(result,"probe")
