"""The contact preview must predict the untouched simulator and preserve the gates."""

from dataclasses import replace

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


def passing_hold():
    return {"strict_status":"PASS", "strict":{"unsupported_hold_seconds":120, "final_diagnostics":{"supports":{}}},
              "drift_assessment":{"status":"PASS"},
              "contact_quality":{"status":"PASS","minimum_group_normal_N":{"left":{"finger":.12}}},
              "physics_audit":{"no_external_support":True,"housing_contact_steps":0,"unintended_support_steps":0,
                               "native_hand_friction":{"status":"PASS"}, "extrema":{"force_balance":24.9},
                               "ground_reaction":{"control":{"samples":6000,"peak_N":24.9}}}}


def test_refinement_requires_both_residual_cadences_all_fingers_and_original_long_gates():
    from teleopit.sim.g1_o6_support.g6_ground_reaction_experiment import passes

    result = passing_hold()
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


def test_named_foot_trims_preserve_native_caps_and_restore_the_same_controller(monkeypatch):
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_ground_reaction_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.g6_whole_body import WholeBodyReference
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import canonical, restore_environment

    candidate = candidates()[-1]
    env = make_environment(candidate, 5, .001)
    metadata = canonical(experiment_metadata(env, snapshot(env), {"mode":"long-hold"}))
    env = restore_environment(metadata, .0005)
    assert env.grasp_acquisition == candidate.controller
    ctl = env.hold_controller
    original = WholeBodyReference.leg_targets
    proposals = []
    def record(self, env, levers, force, moment):
        proposals.append((force.copy(), moment.copy()))
        return original(self, env, levers, force, moment)
    monkeypatch.setattr(WholeBodyReference,"leg_targets",record)
    levers = {s:env.data.xpos[ctl.chains[s].body]-env.data.body("pelvis").xpos for s in env.hands}
    ctl.leg_targets(env, levers, np.array([0.,0.,29.9]), np.array([4.99,0.,0.]))
    bounds = np.tile([-.12,.12], (15,1))
    freeze = np.zeros(15, dtype=bool)
    freeze[2] = True
    options = dict(ctl.options(env, np.zeros(15), bounds, freeze))
    assert len(options)==21
    assert max(np.linalg.norm(f) for f, m in proposals) <= 30. + 1e-12
    assert max(np.linalg.norm(m) for f, m in proposals) <= 5. + 1e-12
    assert all(max(abs(v)) <= .06 * .02 + 1e-12 and v[2] == 0 for v in options.values())
    assert not np.array_equal(options["left_-0.5N"], options["right_-0.5N"])


def test_balance_witness_preserves_first_and_worst_actual_supports():
    from teleopit.sim.g1_o6_support.g6_ground_reaction_experiment import BalanceTrace

    trace = BalanceTrace()
    for t, f in enumerate([10.,25.,15.,32.,8.]):
        row = {"static_balance_residual_N":[0.,0.,-f], "supports":{"left_foot":{"force_world":[0.,0.,150.-f]}}}
        trace.observe(row, t * .001)
    assert trace.violations==2 and trace.samples==5 and trace.peak_N==32.
    assert trace.first["hold_s"]==.001 and trace.worst["hold_s"]==.003
    assert trace.worst["supports_world_N"]["left_foot"][2]==118.


def test_original_control_gate_alone_cannot_promote_a_native_force_crossing(tmp_path):
    from teleopit.sim.g1_o6_support.g6_ground_reaction_experiment import GroundReactionCampaign, PARENT_TRACE

    campaign = GroundReactionCampaign(tmp_path)
    campaign.state["runs"].append({"id":"control_probe","candidate":"control","stage":"probe",
                                   "status":"COMPLETE","files_sha256":{"steps.jsonl":PARENT_TRACE}})
    for name in campaign.first_pair:
        result = passing_hold()
        result["physics_audit"]["extrema"]["force_balance"] = 30.
        result["physics_audit"]["ground_reaction"]["physics"] = {"peak_N":30., "first":{"hold_s":.728}}
        campaign.state["runs"].append({"id":name+"_probe","candidate":name,"stage":"probe","status":"COMPLETE","result":result})
        with pytest.raises(ValueError):
            campaign.validate_stage(name,"screen")
    with pytest.raises(ValueError):
        campaign.validate_stage(campaign.second_pair[0],"probe")
    audit = campaign.audit_first_pair()
    assert all(a["balance"]["physics"]["first"]["hold_s"]==.728 for a in audit["actors"].values())
    campaign.validate_stage(campaign.second_pair[0],"probe")
