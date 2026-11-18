import json

import numpy as np
import pytest


def test_refinement_specs_reproduce_both_frozen_campaigns():
    from teleopit.sim.g1_o6_support import g6_knee_experiment, g6_knee_retention_experiment
    from teleopit.sim.g1_o6_support.ladder_campaign import digest

    assert digest(g6_knee_experiment.specification()) == "d57ec28c00c35647589f6e5d89d4a4877dd0eb9b988643d49db4fc34d10327ad"
    assert digest(g6_knee_retention_experiment.specification()) == "f072da13feb29751a1c13c31302b183c71474dc4fa98c3f21dd811d627f96225"


def test_retained_grip_starts_at_verified_grasp_and_permits_the_original_opening():
    from teleopit.sim.g1_o6_support.g6_knee_retention import SettledGrip

    grip=SettledGrip()
    original=np.array([.4,.35,.8,.7])
    applied=np.array([.38,.34,.79,.69])
    valid={s:{'valid':True,'minimum_finger_normal_N':.36} for s in ('left','right')}
    np.testing.assert_array_equal(grip.command(4.99,valid,[],original,applied),original)
    np.testing.assert_array_equal(grip.command(5.,valid,['drift'],original,applied),original)
    valid['left']['minimum_finger_normal_N']=.1
    np.testing.assert_array_equal(grip.command(5.,valid,[],original,applied),original)
    assert grip.capture_s is None
    valid['left']['minimum_finger_normal_N']=.36
    valid['right']['valid']=False
    np.testing.assert_array_equal(grip.command(5.,valid,[],original,applied),original)
    valid['right']['valid']=True
    np.testing.assert_array_equal(grip.command(5.02,valid,[],original,applied),applied)
    applied[:]=1.
    desired=np.array([.3,.35,1.2,.5])
    np.testing.assert_array_equal(grip.command(6.,valid,[],desired,applied),[.3,.34,.79,.5])
    assert grip.capture_s==5.02


def test_bounded_knee_retreat_and_native_control_forecast_after_capture():
    from teleopit.sim.g1_o6_support.g6_knee_retention import retreat_m
    from teleopit.sim.g1_o6_support.g6_knee_retention_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.pose_contacts import grasp_quality, hand_contacts

    assert [retreat_m(d) for d in (.04,.033,.001)] == pytest.approx([0.,.002,.008])
    candidate=candidates()[-1]
    assert G6Candidate.from_dict(candidate.record())==candidate
    env=make_environment(candidate,6,.001)
    c=env.hold_controller
    while c.grip.capture_s is None:
        env.step()
        assert env.phase not in ('HOLD_COMPLETE','ABORT')
    assert c.grip.capture_s==pytest.approx(5.,abs=.021)
    actual=[]
    def measure(e):
        row=snapshot(e)
        fingers=grasp_quality(hand_contacts(e,row['contacts'],'left'),e.data.geom(e.support_targets['left_hand']).xpos[2])
        actual.append([np.linalg.norm(row['static_balance_residual_N']),fingers['minimum_finger_normal_N']])
    for _ in range(3):
        actual.clear()
        env.step(measure); measure(env)
        assert c.last.peak_N==pytest.approx(max(v[0] for v in actual),abs=1e-8)
        assert c.last.finger_N==pytest.approx(min(v[1] for v in actual),abs=1e-8)
        assert np.max(env.data.ctrl[c.fingers]-c.grip.ceiling)<=1e-12


def test_conditional_followup_requires_terminal_physics_and_rejects_an_uncaptured_hold(tmp_path):
    from teleopit.sim.g1_o6_support.g6_knee_retention_experiment import prior_failures, passes
    from test_g1_o6_g6_ground_reaction import passing_hold

    (tmp_path/'milestones.json').write_text(json.dumps({'initial_pose':'PASS','candidate':'stance_soft'}))
    with pytest.raises(ValueError,match='exhausted'):
        prior_failures(tmp_path)
    value=passing_hold()
    a=value['physics_audit']
    a['scored_steps']=120000
    a['knee_rung3']={'native':dict(samples=120000,minimum_m=.003,contact_steps=0),
                     'control':dict(samples=6000,minimum_m=.003,contact_steps=0)}
    a['settled_grip']={'capture_s':None}
    assert not passes(value,'120s')
    for invalid in (0., float('nan'), 121.):
        a['settled_grip']['capture_s']=invalid
        assert not passes(value,'120s')
    a['settled_grip']['capture_s']=5.
    assert passes(value,'120s')
    a['knee_rung3']['native']['minimum_m']=0.
    assert not passes(value,'120s')


def test_retention_requires_the_unchanged_three_terminal_records(tmp_path):
    from teleopit.sim.g1_o6_support.g6_knee_experiment import specification
    from teleopit.sim.g1_o6_support.g6_knee_retention_experiment import prior_failures
    from teleopit.sim.g1_o6_support.ladder_campaign import digest, file_hash

    (tmp_path/'milestones.json').write_text(json.dumps({'initial_pose':'FAIL','candidate':None}))
    state={'runs':[],'spec_sha256':digest(specification())}
    statefile=tmp_path/'campaign_state.json'
    statefile.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='terminal physical'):
        prior_failures(tmp_path)
    for name in ('stance_soft','stance_firm','palm_coordinated'):
        run=tmp_path/name
        run.mkdir()
        result={'strict':{'failure_reasons':['unintended_support']},'physics_audit':{'knee_rung3':{'native':{'contact_steps':1}}}}
        path=run/'summary.json'
        path.write_text(json.dumps(result))
        state['runs'].append(dict(id=name+'_120s', candidate=name, stage='120s', status='COMPLETE',identity='abc',
            output=name,result=result,files_sha256={'summary.json':file_hash(path)}))
    statefile.write_text(json.dumps(state))
    assert len(prior_failures(tmp_path))==3
    path.write_text('{}')
    with pytest.raises(ValueError,match='raw evidence'):
        prior_failures(tmp_path)


def test_knee_guard_moves_the_pelvis_away_and_restores_at_half_timestep(monkeypatch):
    from dataclasses import replace
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_ground_reaction import GroundReactionReference
    from teleopit.sim.g1_o6_support.g6_knee_retention_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import canonical, restore_environment

    candidate=candidates()[-1]
    with pytest.raises(ValueError):
        replace(candidate.controller,grip_retention="always")
    env=make_environment(candidate,5,.001)
    meta=canonical(experiment_metadata(env,snapshot(env),{'mode':'long-hold'}))
    env=restore_environment(meta,.0005)
    ref=env.hold_controller
    assert ref.grip.ceiling is None and env.grasp_acquisition==candidate.controller
    monkeypatch.setattr(GroundReactionReference,'wrench_targets',lambda self,e,p,t,r,l:t)
    monkeypatch.setattr(ref.knee,'distance',lambda d:.031)
    origin=np.array([.01,-.02,.003])
    target=ref.wrench_targets(env,env.data.body('pelvis').id,origin,np.zeros(3),{})
    np.testing.assert_allclose(target,[.006,-.02,.003],atol=1e-12)
    np.testing.assert_array_equal(origin,[.01,-.02,.003])
