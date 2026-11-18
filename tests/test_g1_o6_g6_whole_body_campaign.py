"""The experiment requires continuous left finger load as well as the original gates."""

from copy import deepcopy

import pytest


def passed():
    return {'strict_status':'PASS', 'strict':{'unsupported_hold_seconds':120},
            'drift_assessment':{'status':'PASS'},
            'contact_quality':{'status':'PASS','minimum_group_normal_N':{'left':{'finger':.12}}},
            'physics_audit':{'no_external_support':True,'unintended_support_steps':0,
                             'housing_contact_steps':0,'native_hand_friction':{'status':'PASS'}}}


def test_promotion_refuses_even_one_unloaded_finger_and_requires_long_convergence():
    from teleopit.sim.g1_o6_support.g6_whole_body_experiment import eligible
    result = passed()
    for stage in ('probe', 'screen', '60s', '120s', 'repeat', 'fine'):
        assert eligible(result, stage)
        missed = deepcopy(result)
        missed['contact_quality']['minimum_group_normal_N']['left']['finger'] = 0
        assert not eligible(missed, stage)
    for field, bad in [('no_external_support', False), ('housing_contact_steps',1), ('unintended_support_steps',1), ('native_hand_friction',{})]:
        missed = deepcopy(result)
        missed['physics_audit'][field] = bad
        assert not eligible(missed, 'probe')
    result['drift_assessment']['status'] = 'FAIL'
    assert eligible(result, '60s') and not eligible(result, '120s')
    result['strict_status'] = 'FAIL'
    assert not eligible(result, 'probe')


def test_second_pair_requires_two_measured_failures_and_an_immutable_premise(tmp_path):
    from teleopit.sim.g1_o6_support.g6_whole_body_experiment import PARENT_TRACE, WholeBodyCampaign
    campaign = WholeBodyCampaign(tmp_path)
    with pytest.raises(ValueError, match="matched acquisition control"):
        campaign.validate_stage('stance_low', 'probe')
    with pytest.raises(ValueError, match="one fresh probe"):
        campaign.validate_stage('control', 'screen')
    campaign.state['runs'].append({'id':'control_probe','candidate':'control','stage':'probe',
                                   'status':'COMPLETE','files_sha256':{'steps.jsonl':PARENT_TRACE}})
    with pytest.raises(ValueError):
        campaign.validate_stage('wrench_low','probe')
    with pytest.raises(ValueError):
        campaign.audit_stance()
    for name in ('stance_low','stance_high'):
        result = passed()
        result['drift_assessment']['status'] = 'FAIL'
        result['strict']['final_diagnostics'] = {'supports':{s:{'force_world':[0,0,10]} for s in ('left_hand','right_hand','left_foot','right_foot')}}
        campaign.state['runs'].append({'id':name+'_120s','candidate':name,'status':'COMPLETE','stage':'120s','result':result})
    campaign.audit_stance()
    campaign.validate_stage('wrench_low','probe')
    audit = campaign.decision('stance_premise',lambda:None)
    assert set(audit['actors']) == {'stance_low_120s','stance_high_120s'}
    assert audit['premise'] and audit['removed_assignment']
    campaign.audit_stance()
    assert sum(d['name']=='stance_premise' for d in campaign.state['decisions']) == 1


@pytest.mark.parametrize('both_fail', [False, True])
def test_search_promotes_fresh_stages_and_requires_the_full_confirmation(tmp_path, monkeypatch, both_fail):
    from teleopit.sim.g1_o6_support import g6_whole_body_experiment as search

    campaign = search.WholeBodyCampaign(tmp_path)
    def run_trials(requests):
        for name, stage in requests:
            campaign.validate_stage(name, stage)
            result = passed()
            result['strict']['unsupported_hold_seconds'] = search.STAGES[stage][0]
            result['strict']['final_diagnostics'] = {'supports': {}}
            result['physics_audit']['short_drift'] = {'maximum_translation_slope_m_s': .1 if name.endswith('high') else .05}
            if name == 'stance_low' and stage == 'repeat':
                result['drift_assessment']['status'] = 'FAIL'
            if both_fail and name == 'stance_high' and stage == 'fine':
                result['contact_quality']['minimum_group_normal_N']['left']['finger'] = 0
            campaign.state['runs'].append(dict(candidate=name, stage=stage, output=name+'_'+stage,
                id=name+'_'+stage, status='COMPLETE', result=result, files_sha256={'steps.jsonl':search.PARENT_TRACE}))
    def read_evidence(path):
        name, stage = path.name.rsplit('_', 1)
        seconds, dt = search.STAGES[stage]
        return {'path':str(path),'summary':campaign.previous(name, stage)['result'],
                'metadata':{'experiment':{'requested_hold_s':seconds}, 'config':{'physics_dt':dt}}}
    monkeypatch.setattr(campaign, 'run_many', run_trials)
    monkeypatch.setattr(search, 'read_run', read_evidence)
    assert campaign.run_search() == {'initial_pose':'PASS','candidate':'wrench_low' if both_fail else 'stance_high'}
    assert campaign.previous('stance_low', 'fine') is None
    assert campaign.previous('stance_high', 'fine')['status'] == 'COMPLETE'
    assert (campaign.previous('wrench_low', 'probe') is not None) == both_fail
    with pytest.raises(ValueError):
        campaign.validate_preceding('stance_low', 'fine')
