"""Public default and configuration contract of the supported hold experiment."""

import json
from pathlib import Path

import numpy as np
import pytest

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.grasp_pose import GraspPose


def test_default_reset_matches_qualified_current_reference():
    env = Environment()
    reference = json.loads(
        (Path(__file__).parent / "fixtures/g1_o6_reset.json").read_text()
    )
    assert env.pose == GraspPose()
    np.testing.assert_allclose(env.data.qpos, reference["qpos"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(env.data.ctrl, reference["ctrl"], rtol=0, atol=1e-12)
    assert env.support_targets == reference["support_targets"]
    assert not env.initial_errors


@pytest.mark.parametrize(
    "change",
    [
        {"hand_rungs": [True, 11]},
        {"finger_preload_Nm": 0},
        {"finger_preload_Nm": 0.35},
        {"closed_pose": [0] * 5},
        {"hand_tilt_deg": float("inf")},
        {"grip_offset_m": None},
    ],
)
def test_invalid_grasp_fails_at_configuration_boundary(change):
    with pytest.raises((ValueError, TypeError)):
        GraspPose(**change)


def test_json_grasp_configuration_is_immutable():
    pose = GraspPose(hand_rungs=[11, 12], grip_offset_m=[-0.023, 0, -0.020])
    assert pose.hand_rungs == (11, 12)
    assert isinstance(pose.grip_offset_m, tuple)


def test_wrist_perturbation_moves_current_mount_target():
    from types import SimpleNamespace

    from teleopit.sim.g1_o6.grasp_pose import mount_target

    data = SimpleNamespace(geom=lambda name: SimpleNamespace(xpos=np.zeros(3)))
    normal = SimpleNamespace(data=data, config=Config())
    shifted = SimpleNamespace(data=data, config=Config(wrist_offset_x=0.002))
    a, rotation = mount_target(normal, GraspPose(), "left", 11)
    b, shifted_rotation = mount_target(shifted, GraspPose(), "left", 11)
    np.testing.assert_allclose(b - a, [0.002, 0, 0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(rotation, shifted_rotation)


def test_closing_targets_cannot_exceed_physical_joint_ranges():
    with pytest.raises(ValueError, match="physical joint ranges"):
        Environment(pose=GraspPose(closed_pose=(10,) * 6))


def test_suite_contains_only_current_hold_perturbations():
    from teleopit.sim.g1_o6.run import scenarios

    cases = scenarios(Config())
    assert set(cases) == {
        "nominal",
        "fine_step",
        "body_offset",
        "wrist_offset",
        "friction_low",
        "disturbance",
        "delay",
    }
