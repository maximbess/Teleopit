from dataclasses import replace

import numpy as np
import pytest


def test_sliding_and_pitching_foot_requests_bounded_restoring_reaction():
    from teleopit.sim.g1_o6_support.g6_knee import foot_trim, distribute_trim

    neutral = foot_trim(0., 0., 0., 1.5)
    np.testing.assert_array_equal(neutral, np.zeros(6))
    trim = foot_trim(.1, .4, .02, 1.5)
    assert 0 < trim[0] <= 8. and 0 < trim[4] <= 2.
    np.testing.assert_array_equal(trim[[1, 2, 3, 5]], 0.)
    assert foot_trim(.001, 0., .01, .75)[0] > foot_trim(.001, 0., 0., .75)[0]
    assert foot_trim(-.001, -.01, 0., .75)[0] < 0
    levers = {'left':np.array([0.,.14,-.8]), 'right':np.array([0.,-.14,-.8])}
    trims = {'left':neutral, 'right':trim}
    internal, force, moment = distribute_trim(levers, trims)
    np.testing.assert_allclose(force, trim[:3])
    np.testing.assert_allclose(moment, np.cross(levers['right'], trim[:3]) + trim[3:])
    np.testing.assert_allclose(sum(v[:3] for v in internal.values()), 0., atol=1e-10)
    np.testing.assert_allclose(sum(internal[s][3:] + np.cross(r, internal[s][:3]) for s, r in levers.items()), 0., atol=1e-10)


def test_knee_gate_rejects_touch_between_samples_and_retains_other_gates():
    from teleopit.sim.g1_o6_support.g6_knee_experiment import passes
    from test_g1_o6_g6_ground_reaction import passing_hold

    value = passing_hold()
    audit = value['physics_audit']
    assert not passes(value,'120s')
    audit['scored_steps'] = 120000
    audit['knee_rung3'] = {
        'native':dict(samples=120000, minimum_m=.001, contact_steps=0),
        'control':dict(samples=6000, minimum_m=.001, contact_steps=0)}
    assert passes(value,'120s')
    for cadence in ('native','control'):
        for field, bad in [('samples',1),('minimum_m',0.),('minimum_m',float('nan')),('contact_steps',1)]:
            trace = audit['knee_rung3'][cadence]
            previous, trace[field] = trace[field], bad
            assert not passes(value, '120s')
            trace[field] = previous
    for obj, key, bad in ((audit['extrema'],'force_balance',25.),
                          (value['contact_quality']['minimum_group_normal_N']['left'],'finger',0.),
                          (value['drift_assessment'],'status','FAIL')):
        previous, obj[key] = obj[key], bad
        assert not passes(value,'120s')
        obj[key] = previous


def test_native_run_records_each_integrated_knee_pose_and_preserves_parent_forecast(tmp_path):
    import mujoco
    from teleopit.sim.g1_o6_support.g6_knee import KneeRung
    from teleopit.sim.g1_o6_support.g6_knee_experiment import candidates, KneeObserver, read_knee_rows
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.long_hold import enrich
    from teleopit.sim.o6_diagnostics.nominal import coupled_telemetry

    candidate = candidates()[1]
    assert G6Candidate.from_dict(candidate.record()) == candidate
    with pytest.raises(ValueError):
        replace(candidate.controller, stance_gain=12.)
    env = make_environment(candidate, 5, .001)
    pair = KneeRung(env.model)
    while env.phase != 'HOLD':
        env.step()
    observer = KneeObserver(env, tmp_path)
    poses, actual_forces = [], []
    def sample(e):
        poses.append(e.data.qpos.copy())
        actual_forces.append(np.linalg.norm(snapshot(e)['static_balance_residual_N']))
        observer(e)
    row = env.step(sample)
    enrich(env, row, coupled_telemetry(env), env.phase_start, observer)
    audit = observer.report()['knee_rung3']
    observer.close()
    rows = read_knee_rows(tmp_path, env.model.nq)
    assert rows.shape == (20, env.model.nq+3)
    np.testing.assert_array_equal(rows[:,3:], poses)
    scratch = mujoco.MjData(env.model)
    for record in rows:
        scratch.qpos[:] = record[3:]
        mujoco.mj_kinematics(env.model,scratch)
        assert record[1] == pytest.approx(pair.distance(scratch), abs=1e-10)
    assert audit['native']['minimum_m'] == pytest.approx(min(rows[:,1]))
    assert audit['native']['samples']==20 and audit['control']['samples']==1
    assert audit['native']['minimum_m']>.03 and audit['native']['contact_steps']==0
    actual_forces.append(np.linalg.norm(row['static_balance_residual_N']))
    assert env.hold_controller.last.peak_N == pytest.approx(max(actual_forces), abs=1e-8)


def test_feedback_references_measured_supports_and_bounds_every_preview_trim(monkeypatch):
    from scipy.spatial.transform import Rotation
    from teleopit.sim.g1_o6_support.g6_knee_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.g6_whole_body import WholeBodyReference
    from teleopit.sim.g1_o6_support.scene_contract import canonical, restore_environment
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6.diagnostics import snapshot

    env = make_environment(candidates()[-1], 5, .001)
    meta = canonical(experiment_metadata(env, snapshot(env), {'mode':'long-hold'}))
    env = restore_environment(meta, .0005)
    ctl = env.hold_controller
    pelvis = env.data.body("pelvis")
    levers = {s:env.data.body(ctl.chains[s].body).xpos-pelvis.xpos for s in env.hands}
    ctl.latch_time = env.data.time
    ctl.wrench_targets(env, pelvis.id, np.zeros(3), np.zeros(3), levers)
    assert ctl.trims == {} and set(ctl.feet) == {'left','right'}
    ctl.latch_time -= 3.
    ctl.feet['right'][0][0] -= .020
    for pos, rot in ctl.palms.values():
        pos[0] += 2.
        rot[:] = Rotation.from_rotvec([0.,2.,0.]).as_matrix() @ rot
    realized = []
    original = WholeBodyReference.reaction_targets
    def record(self, e, reactions):
        force = sum(v[:3] for v in reactions.values())
        moment = sum(v[3:]+np.cross(levers[s],v[:3]) for s,v in reactions.items())
        realized.append((force, moment))
        return original(self, e, reactions)
    monkeypatch.setattr(WholeBodyReference, 'reaction_targets', record)
    for _ in range(3):
        ctl.wrench_targets(env,pelvis.id,np.ones(3),np.ones(3),levers)
    assert ctl.trims['right'][0] == 8. and ctl.trims['left'][0] == 0.
    assert ctl.palm_shift[0] == pytest.approx(.012)
    assert ctl.palm_tilt[1] == pytest.approx(.020)
    bounds = np.tile([-.12,.12],(15,1))
    frozen = np.zeros(15,dtype=bool)
    frozen[4] = True
    options = list(ctl.options(env,np.zeros(15),bounds,frozen))
    assert len(options) == 21
    assert all(np.max(abs(v)) <= .06*.02+1e-12 and v[4] == 0 for _,v in options)
    assert max(np.linalg.norm(f) for f,m in realized) == pytest.approx(30.)
    assert max(np.linalg.norm(m) for f,m in realized) == pytest.approx(5.)


def test_geometric_touch_counts_even_without_a_newton_of_knee_load():
    from teleopit.sim.g1_o6_support.g6_knee import KneeRung, RUNG
    from teleopit.sim.g1_o6_support.g6_knee_experiment import candidates, ClearanceTrace
    from teleopit.sim.g1_o6_support.g6_trial import make_environment

    pair = KneeRung(make_environment(candidates()[0],5,.001).model)
    assert len(pair.names)==7
    trace = ClearanceTrace()
    rows = [{'geoms':[RUNG,name],'normal_force':0.} for name in pair.names]
    rows += [{'geoms':['left_ladder_rung_02',name],'normal_force':2.} for name in pair.names]
    assert len(pair.contacts(rows))==len(pair.names)
    trace.observe(.001, .005, [])
    trace.observe(.002, -.0001, pair.contacts(rows))
    trace.observe(.003, .004, [{'normal_force':3.}])
    assert trace.samples==3 and trace.contact_steps==2 and trace.peak_contact_N==3.
    assert trace.closest_s==trace.first_contact_s==.002 and trace.minimum_m==-.0001
    unknown = ClearanceTrace()
    unknown.observe(.001, float('nan'), [])
    assert unknown.minimum_m==0.


def test_palm_refinement_unlocks_only_after_two_measured_stance_failures(tmp_path):
    from teleopit.sim.g1_o6_support.g6_knee_experiment import KneeCampaign, PARENT_TRACE, COORDINATED
    from test_g1_o6_g6_ground_reaction import passing_hold

    campaign = KneeCampaign(tmp_path)
    campaign.state['runs'].append({'id':'control_probe','candidate':'control','stage':'probe','status':'COMPLETE',
                                   'files_sha256':{'steps.jsonl':PARENT_TRACE}})
    for name in campaign.first_pair:
        with pytest.raises(ValueError):
            campaign.validate_stage(COORDINATED[0],'probe')
        result = passing_hold()
        result['physics_audit'].update(scored_steps=120000, knee_rung3={
            'native':dict(samples=120000,minimum_m=-.0001,contact_steps=1),
            'control':dict(samples=6000,minimum_m=.001,contact_steps=0)})
        result['physics_audit']['ground_reaction']['physics'] = {'peak_N':22.}
        campaign.state['runs'].append({'id':name+'_probe','candidate':name,'stage':'probe','status':'COMPLETE','result':result})
        with pytest.raises(ValueError):
            campaign.validate_stage(name,'screen')
    actual = campaign.audit_first_pair()
    assert len(actual['actors'])==2
    assert all(a['right_knee_to_rung3']['native']['contact_steps']==1 for a in actual['actors'].values())
    campaign.validate_stage(COORDINATED[0],'probe')
    assert KneeCampaign(tmp_path).audit_first_pair() == actual
