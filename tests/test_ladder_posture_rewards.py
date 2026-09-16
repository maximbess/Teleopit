"""Reward arithmetic checks without creating or stepping a simulation."""
from types import SimpleNamespace

import mujoco
import torch
import pytest

from train_mimic.tasks.tracking.config.env import make_g1_ladder_rl_env_cfg
from train_mimic.tasks.tracking.mdp.ladder import (
    LadderStabilizationPoseCost,
    ladder_stabilization_joint_velocity_l2,
    ladder_unwanted_contact_cost,
)


def test_fixed_pose_reference_phase_mask_and_small_motion():
    cfg = make_g1_ladder_rl_env_cfg().rewards["ladder_stabilization_pose"]
    names = ["left_knee_joint", "left_elbow_joint", "waist_roll_joint"]
    reference = torch.tensor([[0.657862, 0.569849, 0.0]])
    robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(
        joint_pos=reference.repeat(3, 1), joint_vel=torch.full((3, 3), 0.1)))
    command = SimpleNamespace(robot=robot, initialized=torch.tensor([True, True, False]),
        finished=torch.zeros(3, dtype=torch.bool),
        is_stabilization_phase=torch.tensor([True, False, True]))
    env = SimpleNamespace(device="cpu", command_manager=SimpleNamespace(get_term=lambda _: command))
    term = LadderStabilizationPoseCost(cfg, env)
    assert term(env, **cfg.params).tolist() == [0.0, 0.0, 0.0]
    robot.data.joint_pos[:, 0] += 0.25
    term.reset()  # A bank reset cannot adopt the displaced knee as the reference.
    assert term(env, **cfg.params).tolist() == pytest.approx([2 / 4.5, 0, 0])
    robot.data.joint_pos.copy_(reference)
    robot.data.joint_pos[:, 1] += 0.25
    assert term(env, **cfg.params).tolist() == pytest.approx([0.5 / 4.5, 0, 0])
    assert ladder_stabilization_joint_velocity_l2(env, "ladder").tolist() == pytest.approx([0.01, 0, 0])


def test_unwanted_contacts_count_current_bodies_not_history_or_match_count():
    sensor = SimpleNamespace(cfg=SimpleNamespace(num_slots=1), data=SimpleNamespace(
        force=torch.tensor([[[3., 0., 0.], [0., 1., 0.], [0., 0., 2.]]]),
        found=torch.tensor([[12, 4, 3]]), force_history=torch.full((1, 3, 4, 3), 100.)))
    env = SimpleNamespace(scene={"bad": sensor})
    assert ladder_unwanted_contact_cost(env, "bad").item() == 2
    env.scene["other_face"] = sensor
    assert ladder_unwanted_contact_cost(env, ("bad", "other_face")).item() == 2
    sensor.data.force.zero_()
    assert ladder_unwanted_contact_cost(env, "bad").item() == 0


def test_posture_reward_and_contact_sensor_contract():
    cfg = make_g1_ladder_rl_env_cfg()
    assert cfg.rewards["ladder_stabilization_pose"].weight == -2
    assert cfg.rewards["ladder_stabilization_joint_velocity"].weight == -0.2
    assert cfg.rewards["ladder_unwanted_contact"].weight == -2
    assert cfg.rewards["action_rate"].weight == -0.1
    sensor = next(s for s in cfg.scene.sensors if s.name == "ladder_unwanted_contact_left")
    assert sensor.num_slots == 1 and sensor.history_length == 0
    assert sensor.secondary.pattern == "left_ladder_body"
    assert set(sensor.primary.exclude) == {
        "left_wrist_yaw_link", "right_wrist_yaw_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
    }
