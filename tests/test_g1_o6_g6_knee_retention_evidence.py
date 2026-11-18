import numpy as np
import pytest


def test_grip_ceiling_must_match_previous_applied_targets_and_original_preload():
    from scripts.verify_g1_o6_g6_knee_retention import check_grip

    previous=np.array([.30,.60])
    native=np.array([.29,.61])
    state={'capture_s':5.,'ceiling_rad':[.30,.60],'uncapped_targets_rad':native.tolist()}
    ceiling=check_grip(state,previous,native,np.array([.29,.60]),None,.02)
    np.testing.assert_array_equal(ceiling,previous)
    with pytest.raises(ValueError):
        check_grip({**state,'ceiling_rad':[.4,.6]},previous,native,np.array([.29,.60]),None,.02)
    with pytest.raises(ValueError):
        check_grip(state,previous,native,np.array([.29,.61]),ceiling,.02)
    with pytest.raises(ValueError):
        check_grip({**state,'uncapped_targets_rad':[.29,.8]},previous,native,np.array([.29,.60]),ceiling,.02)
    with pytest.raises(ValueError):
        check_grip({**state,'ceiling_rad':[.29,.6]},previous,native,np.array([.29,.60]),ceiling,.02)


def test_capture_matches_measured_opposition_and_knee_guard():
    from scripts.verify_g1_o6_g6_knee_retention import check_capture, check_retreat

    hand={'quality':{'valid':True,'minimum_finger_normal_N':.36}}
    previous={'hold_time_s':5.,'pose_search':{'hands':{'left':hand,'right':hand}},
              'g6':{'body_clearance':{'right_knee':{'left_ladder_rung_03':{'signed_distance_m':.032}}}}}
    ctl={'settled_grip':{'capture_s':5.,'gap_m':.032,'retreat_m':.003},'inhibition':[]}
    assert check_capture(previous,ctl,None)==5.
    assert check_retreat(previous,ctl)==.003
    hand['quality']['minimum_finger_normal_N']=0
    with pytest.raises(AssertionError):
        check_capture(previous,ctl,None)
    with pytest.raises(ValueError):
        check_capture(previous,ctl,5.02)
    ctl['settled_grip']['retreat_m']=-.003
    with pytest.raises(AssertionError):
        check_retreat(previous,ctl)


def test_first_eligible_capture_and_persistent_ceiling_are_measured():
    from scripts.verify_g1_o6_g6_knee_retention import check_capture, check_grip

    hand = {"quality": {"valid": True, "minimum_finger_normal_N": .35}}
    previous = {"hold_time_s": 5., "pose_search": {"hands": {"left": hand, "right": hand}}}
    control = {"settled_grip": {"capture_s": None}, "inhibition": ["slip"]}
    assert check_capture(previous, control, None) is None
    control["inhibition"] = []
    with pytest.raises(ValueError, match="eligible"):
        check_capture(previous, control, None)

    native = np.array([.12, .21])
    applied = np.array([.11, .20])
    state = {"uncapped_targets_rad": native, "ceiling_rad": None}
    with pytest.raises(ValueError, match="disappeared"):
        check_grip(state, applied, native, applied, applied, .02)
