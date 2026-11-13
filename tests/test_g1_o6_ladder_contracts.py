"""Malformed support arrays fail before model construction or physics."""

import pytest


@pytest.mark.parametrize(
    "values",
    [
        {"hand_rungs": (0, 1)},
        {"hand_rungs": (1, 2.0)},
        {"foot_rungs": (1,)},
        {"foot_rungs": (True, 2)},
        {"grip_offset_m": (0, 0)},
        {"closed_pose": (0,)},
    ],
)
def test_invalid_pose_vectors_are_rejected(values):
    from teleopit.sim.g1_o6.grasp_pose import GraspPose

    with pytest.raises(ValueError):
        GraspPose(**values)
