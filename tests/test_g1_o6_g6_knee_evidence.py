import numpy as np
import pytest


def test_native_evidence_rejects_forged_gap_or_missing_geometric_contact():
    from scripts.verify_g1_o6_g6_knee import check_native_sample

    row = {"time_s":7.003, "contacts":[{"geoms":["right_knee_omni_150_collision","left_ladder_rung_03"],
                                      "normal_force":.02}]}
    pose = np.array([1.,2.])
    record = np.r_[7.003, -.0001, 1., pose]
    valid = check_native_sample(record,row,lambda q:-.0001 if np.array_equal(q,pose) else 1.,{"right_knee_omni_150_collision"})
    assert valid == pytest.approx((-.0001,.02))
    for index, value in ((0,7.004),(1,.003),(2,0.),(3,4.)):
        changed = record.copy()
        changed[index] = value
        with pytest.raises(ValueError):
            check_native_sample(changed,row,lambda q:-.0001 if np.array_equal(q,pose) else 1.,{"right_knee_omni_150_collision"})


def test_control_pose_must_be_last_native_sample_and_cover_whole_interval():
    from scripts.verify_g1_o6_g6_knee import check_endpoint

    rows=np.array([[.001,.008,0.,3.,4.],[.002,.007,0.,2.,4.]])
    sample={'time':.002,'base_pose':[2.],'joint_positions':[4.]}
    check_endpoint(sample,rows[-1],.007)
    for changed in ({**sample,'time':.001},{**sample,'joint_positions':[3.]}):
        with pytest.raises(ValueError):
            check_endpoint(changed,rows[-1],.007)
    with pytest.raises(ValueError):
        check_endpoint(sample,rows[-1],-.001)


def test_verifier_rejects_an_incomplete_series_before_reading_force_evidence(tmp_path):
    from scripts.verify_g1_o6_g6_knee import verify_native, verify_controls

    duplicated = np.array([[5.001,.01,0.,1.],[5.001,.01,0.,1.]])
    with pytest.raises(ValueError, match="missing or duplicated"):
        verify_native(tmp_path,duplicated,None,.001)
    with pytest.raises(ValueError, match="incomplete control"):
        verify_controls(tmp_path,duplicated,None,{'config':{'physics_dt':.001,'control_dt':.020}})
