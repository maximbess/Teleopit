import numpy as np
import pytest

from train_mimic.data.terminal_pose_bank import pose_distances, representatives, stable_history


def test_medoid_is_real_pose_and_extremes_are_covered():
    pos=np.zeros((5,6,3));pos[:,4:,0]=np.array([0.,.001,.002,-.04,.05])[:,None]
    quat=np.zeros((5,6,4));quat[:,:,0]=1
    joints=np.zeros((5,29))
    d=pose_distances(pos,quat,joints)
    ids=representatives(d,3)
    assert ids[0]==1
    assert set(ids[1:])=={3,4}
    np.testing.assert_allclose(d,d.T)
    np.testing.assert_allclose(np.diag(d),0)


def test_quaternion_sign_does_not_change_pose_distance():
    pos=np.zeros((2,6,3));q=np.zeros((2,6,4));q[0,:,0]=1;q[1,:,0]=-1
    np.testing.assert_allclose(pose_distances(pos,q,np.zeros((2,29))),0)
    assert representatives(np.zeros((3,3)),3)==[0,1,2]


def test_only_continuously_stable_same_episode_histories_are_accepted():
    def frame():
        return dict(stable=True,episode_id=4,qpos=np.zeros(4),qvel=np.zeros(3),
                    track_pos=np.zeros((6,3)),track_quat=np.ones((6,4)))
    frames=[frame() for _ in range(5)]
    assert stable_history(frames,5)
    assert not stable_history(frames,6)
    frames[0]['episode_id']=3
    assert not stable_history(frames,5)
    frames[0]['episode_id']=4;frames[2]['stable']=False
    assert not stable_history(frames,5)
    frames[2]['stable']=True;frames[2]['qvel'][0]=np.nan
    assert not stable_history(frames,5)


def test_invalid_selection_count_is_rejected():
    with pytest.raises(ValueError):
        representatives(np.zeros((2,2)),3)
