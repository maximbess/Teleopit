"""Tests for the independent G1 ladder reinforcement-learning pipeline."""

from __future__ import annotations

import argparse
import re
from collections import deque
from types import SimpleNamespace

import mujoco
import pytest
import torch
from mjlab.entity import Entity
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import train_mimic.tasks  # noqa: F401 -- register project tasks
from train_mimic.scripts import train_ladder
from train_mimic.tasks.tracking.config.constants import (
    LADDER_RL_EXPERIMENT_NAME,
    LADDER_RL_TASK,
    SUPPORTED_TASKS,
)
from train_mimic.tasks.tracking.config.env import (
    make_g1_ladder_rl_env_cfg,
    make_g1_ladder_training_robot_cfg,
)
from train_mimic.tasks.tracking.mdp.ladder import (
    LadderClimbCommand,
    LadderPhase,
    LadderTargetProgressReward,
    LadderTorsoAscentReward,
    ladder_cycle_completed,
    ladder_finished,
    ladder_foot_rung_advance,
    ladder_rung_advance,
    ladder_stabilized,
)
from train_mimic.tasks.tracking.rl import LadderOnPolicyRunner


class _DummyLadderCommand(LadderClimbCommand):
    @property
    def active_hand_pos_w(self) -> torch.Tensor:
        return self._test_active_hand_pos

    @property
    def target_pos_w(self) -> torch.Tensor:
        return self._test_active_hand_target

    @property
    def active_hand_vel_w(self) -> torch.Tensor:
        return self._test_active_hand_vel

    @property
    def active_foot_pos_w(self) -> torch.Tensor:
        return self._test_active_foot_pos

    @property
    def active_foot_target_pos_w(self) -> torch.Tensor:
        return self._test_active_foot_target

    @property
    def active_foot_vel_w(self) -> torch.Tensor:
        return self._test_active_foot_vel

    @property
    def foot_contact(self) -> torch.Tensor:
        return self._test_foot_contact

    @property
    def foot_support(self) -> torch.Tensor:
        return self._test_foot_support

    @property
    def torso_com_vel_w(self) -> torch.Tensor:
        return self._test_torso_com_vel

    def _release(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
    ) -> None:
        self.released.append((env_ids.clone(), hand_id))
        self.attached[env_ids, hand_id] = False
        self.held_rung[env_ids, hand_id] = -1

    def _attach(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
        rung_indices: torch.Tensor,
    ) -> None:
        self.attached[env_ids, hand_id] = True
        self.held_rung[env_ids, hand_id] = rung_indices


def _dummy_ladder_command(
    phase: LadderPhase = LadderPhase.STABILIZE,
) -> _DummyLadderCommand:
    command = object.__new__(_DummyLadderCommand)
    command.cfg = SimpleNamespace(
        hand_foot_lead_rungs=3,
        foot_reach_distance=0.10,
        max_foot_speed=0.35,
        attach_distance=0.10,
        max_attach_speed=0.35,
        start_rung=4,
        first_moving_hand="left",
        first_moving_foot="left",
        initial_foot_rung=1,
        curriculum_enabled=True,
        curriculum_success_threshold=0.80,
        curriculum_window_size=100,
        curriculum_min_phase_steps=(100, 100, 100, 100),
        stabilization_dwell_steps=1,
        hand_target_dwell_steps=1,
        foot_target_dwell_steps=1,
        max_stabilization_torso_speed=0.20,
        max_stabilization_joint_speed=1.0,
    )
    command._env = SimpleNamespace(
        device="cpu",
        step_dt=0.02,
        common_step_counter=20_000,
    )
    command.robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.zeros((1, 29)))
    )
    command._rung_site_ids = torch.arange(9)
    command._all_env_ids = torch.arange(1)
    command._unlocked_phase = int(LadderPhase.SECOND_FOOT)
    command._curriculum_phase_start_step = 0
    command._recent_curriculum_outcomes = deque(maxlen=100)
    command._pending_curriculum_outcomes = []
    command.initialized = torch.tensor([True])
    command.attached = torch.tensor([[True, True]])
    command.held_rung = torch.tensor([[4, 4]])
    command.finished = torch.tensor([False])
    command.phase = torch.tensor([int(phase)])
    command._phase_dwell_count = torch.tensor([0])
    command.active_hand = torch.tensor([0])
    command.target_rung = torch.tensor([5])
    command.active_foot = torch.tensor([0])
    command.target_foot_rung = torch.tensor([2])
    command.foot_rung = torch.tensor([[1, 1]])
    command.just_initialized = torch.tensor([False])
    command.just_advanced = torch.tensor([False])
    command.just_foot_advanced = torch.tensor([False])
    command.just_stabilized = torch.tensor([False])
    command.just_cycle_completed = torch.tensor([False])
    command.curriculum_stage_complete = torch.tensor([False])
    command._test_active_hand_pos = torch.zeros((1, 3))
    command._test_active_hand_target = torch.zeros((1, 3))
    command._test_active_hand_vel = torch.zeros((1, 3))
    command._test_active_foot_pos = torch.zeros((1, 3))
    command._test_active_foot_target = torch.zeros((1, 3))
    command._test_active_foot_vel = torch.zeros((1, 3))
    command._test_torso_com_vel = torch.zeros((1, 3))
    command._test_foot_contact = torch.tensor([[False, False]])
    command._test_foot_support = torch.tensor([[False, False]])
    command.released = []
    return command


def test_ladder_task_is_rl_only() -> None:
    cfg = load_env_cfg(LADDER_RL_TASK)

    assert LADDER_RL_TASK in SUPPORTED_TASKS
    assert set(cfg.commands) == {"ladder"}
    assert "motion" not in cfg.commands
    assert set(cfg.observations) == {
        "actor",
        "actor_history",
        "actor_ladder",
        "critic",
        "critic_history",
        "critic_ladder",
    }
    assert set(cfg.observations["actor"].terms) == {
        "ladder_command",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
    }
    assert cfg.observations["actor_history"].history_length == 10
    assert cfg.observations["actor_history"].flatten_history_dim is False
    assert set(cfg.observations["actor_ladder"].terms) == {
        "rung_endpoints_torso"
    }
    assert not any(name.startswith("motion_") for name in cfg.rewards)
    assert {
        "ladder_hand_progress",
        "ladder_foot_progress",
        "ladder_foot_support",
        "ladder_foot_rung_advance",
        "ladder_torso_ascent",
        "ladder_torso_stability",
        "support_joint_velocity",
        "ladder_stabilized",
        "ladder_cycle_completed",
    }.issubset(cfg.rewards)
    assert cfg.rewards["ladder_hand_progress"].func is LadderTargetProgressReward
    assert cfg.rewards["ladder_foot_progress"].func is LadderTargetProgressReward
    assert cfg.rewards["ladder_torso_ascent"].func is LadderTorsoAscentReward
    assert cfg.rewards["ladder_torso_ascent"].weight == 12.0
    assert cfg.rewards["ladder_torso_stability"].weight == 10.0
    assert cfg.rewards["ladder_foot_rung_advance"].weight == 50.0
    assert cfg.rewards["action_rate"].weight == -0.10
    assert cfg.rewards["joint_velocity"].weight == -5.0e-4
    assert cfg.rewards["support_joint_velocity"].weight == -3.0e-3
    assert cfg.terminations["curriculum_stage_complete"].time_out is True
    assert len(cfg.curriculum) == 9
    torso_stages = cfg.curriculum["ladder_torso_ascent_weight"].params["stages"]
    assert torso_stages == [
        {"step": 0, "weight": 12.0},
        {"step": 240_000, "weight": 16.0},
        {"step": 480_000, "weight": 20.0},
    ]
    ladder_cmd = cfg.commands["ladder"]
    assert ladder_cmd.foot_site_names == ("left_foot", "right_foot")
    assert ladder_cmd.foot_contact_sensor_name == "ladder_foot_contact"
    assert ladder_cmd.hand_foot_lead_rungs == 3
    assert ladder_cmd.initialize_on_reset is True
    assert ladder_cmd.start_rung == 4
    assert ladder_cmd.initial_foot_rung == 1
    assert ladder_cmd.curriculum_enabled is True
    assert ladder_cmd.curriculum_success_threshold == 0.80
    assert ladder_cmd.curriculum_window_size == 100
    assert ladder_cmd.curriculum_min_phase_steps == (
        120_000,
        120_000,
        120_000,
        120_000,
    )
    assert ladder_cmd.stabilization_dwell_steps == 5
    assert ladder_cmd.hand_target_dwell_steps == 3
    assert ladder_cmd.foot_target_dwell_steps == 5
    assert cfg.scene.terrain.textures == ()
    assert len(cfg.scene.terrain.materials) == 1
    assert cfg.scene.terrain.materials[0].texture is None
    assert cfg.scene.terrain.materials[0].reflectance == 0.0
    assert cfg.viewer.enable_shadows is False
    assert cfg.viewer.enable_reflections is False
    assert tuple(sensor.name for sensor in cfg.scene.sensors) == (
        "ladder_foot_contact",
    )
    assert load_runner_cls(LADDER_RL_TASK) is LadderOnPolicyRunner

    play_cfg = make_g1_ladder_rl_env_cfg(play=True)
    assert play_cfg.curriculum == {}
    assert play_cfg.commands["ladder"].curriculum_enabled is False
    assert play_cfg.rewards["ladder_torso_ascent"].weight == 20.0


def test_ladder_task_uses_temporal_geometry_ppo_config() -> None:
    rl_cfg = load_rl_cfg(LADDER_RL_TASK)

    assert rl_cfg.experiment_name == LADDER_RL_EXPERIMENT_NAME
    assert rl_cfg.actor.class_name.endswith(":TemporalCNNModel")
    assert rl_cfg.critic.class_name.endswith(":TemporalCNNModel")
    assert rl_cfg.actor.hidden_dims == (2048, 1024, 512, 256, 128)
    assert rl_cfg.critic.hidden_dims == (2048, 1024, 512, 256, 128)
    assert rl_cfg.obs_groups == {
        "actor": ("actor", "actor_history", "actor_ladder"),
        "critic": ("critic", "critic_history", "critic_ladder"),
    }
    assert rl_cfg.actor.distribution_cfg["init_std"] == 0.7
    assert rl_cfg.algorithm.entropy_coef == 0.005
    assert rl_cfg.algorithm.learning_rate == 5.0e-4
    assert rl_cfg.save_interval == 1_000
    assert rl_cfg.max_iterations == 60_000
    assert rl_cfg.upload_model is False


def test_ladder_geometry_observation_uses_torso_frame_and_finite_endpoints() -> None:
    command = object.__new__(LadderClimbCommand)
    command._rung_site_ids = torch.tensor([0, 1], dtype=torch.long)
    command._rung_half_lengths = torch.tensor([0.5, 0.25])
    command._torso_body_id = 0

    identity = torch.eye(3).reshape(9)
    data = SimpleNamespace(
        site_xpos=torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]),
        site_xmat=identity.repeat(1, 2, 1),
        xpos=torch.tensor([[[0.0, 0.0, 0.0]]]),
        xmat=identity.repeat(1, 1, 1),
    )
    command._env = SimpleNamespace(sim=SimpleNamespace(data=data))

    observation = command.rung_endpoints_torso

    assert observation.shape == (1, 2, 7)
    assert observation[0, 0].tolist() == pytest.approx(
        [1.0, 0.0, -0.5, 1.0, 0.0, 0.5, 1.0]
    )
    assert observation[0, 1].tolist() == pytest.approx(
        [1.0, 0.0, 0.75, 1.0, 0.0, 1.25, 1.0]
    )


def test_ladder_robot_augments_canonical_g1_spec() -> None:
    robot_cfg = make_g1_ladder_training_robot_cfg()
    spec = robot_cfg.spec_fn()
    model = Entity(robot_cfg).spec.compile()

    assert robot_cfg.init_state.pos == (-0.911384, 0.0, 1.357509)
    assert len(spec.actuators) == 0
    assert len(spec.keys) == 0
    assert model.nlight == 0
    assert (
        sum(geom_type == mujoco.mjtGeom.mjGEOM_PLANE for geom_type in model.geom_type)
        == 0
    )
    assert (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
        == -1
    )
    assert (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_grip_site",
        )
        >= 0
    )

    # Apply the configured climbing keyframe and verify that both feet contact
    # physical rung 2 while both hand sites are within start-rung attachment
    # range of rung 5.
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = robot_cfg.init_state.pos
    data.qpos[3:7] = robot_cfg.init_state.rot
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        if joint_name == "floating_base_joint":
            continue
        for pattern, value in robot_cfg.init_state.joint_pos.items():
            if re.fullmatch(pattern, joint_name):
                data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)

    foot_rung_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_ladder_rung_02_grip",
    )
    hand_rung_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_ladder_rung_05_grip",
    )
    for foot_name in ("left_foot", "right_foot"):
        foot_site = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            foot_name,
        )
        target = data.site_xpos[foot_rung_site].copy()
        target[1] = data.site_xpos[foot_site, 1]
        assert (
            torch.linalg.vector_norm(
                torch.from_numpy(data.site_xpos[foot_site] - target)
            ).item()
            < 0.10
        )
    for hand_name in ("left_grip_site", "right_grip_site"):
        hand_site = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            hand_name,
        )
        target = data.site_xpos[hand_rung_site].copy()
        target[1] = data.site_xpos[hand_site, 1]
        assert (
            torch.linalg.vector_norm(
                torch.from_numpy(data.site_xpos[hand_site] - target)
            ).item()
            < 0.10
        )

    foot_rung_geom = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_ladder_rung_02",
    )
    contacting_geom_names = set()
    for contact in data.contact:
        geom_pair = (int(contact.geom[0]), int(contact.geom[1]))
        if foot_rung_geom not in geom_pair:
            continue
        other_geom = geom_pair[1] if geom_pair[0] == foot_rung_geom else geom_pair[0]
        contacting_geom_names.add(
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                other_geom,
            )
        )
    assert any(name.startswith("left_foot") for name in contacting_geom_names)
    assert any(name.startswith("right_foot") for name in contacting_geom_names)

    backstop_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_ladder_backstop_collision",
    )
    rung_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_ladder_rung_01",
    )
    rail_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_ladder_rail_01",
    )
    blocker_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_ladder_body_blocker",
    )
    assert backstop_id == -1
    assert rung_id >= 0
    assert rail_id >= 0
    assert blocker_id >= 0
    assert model.geom_type[rung_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_type[rail_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert model.geom_type[blocker_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_size[rung_id].tolist() == pytest.approx([0.055, 0.35, 0.035])
    assert model.geom_contype[rung_id] == 1
    assert model.geom_conaffinity[rung_id] == 1
    assert model.geom_condim[rung_id] == 4
    assert model.geom_contype[rail_id] == 1
    assert model.geom_conaffinity[rail_id] == 1
    assert model.geom_condim[rail_id] == 4
    assert model.geom_solref[rung_id].tolist() == pytest.approx([0.005, 1.0])
    assert model.geom_solref[rail_id].tolist() == pytest.approx([0.005, 1.0])
    assert model.geom_contype[blocker_id] == 2
    assert model.geom_conaffinity[blocker_id] == 2
    assert model.geom_rgba[blocker_id, 3] == 0.0

    torso_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "torso_collision",
    )
    foot_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_foot1_collision",
    )

    def collision_masks_match(first_id: int, second_id: int) -> bool:
        return bool(
            (model.geom_contype[first_id] & model.geom_conaffinity[second_id])
            or (model.geom_contype[second_id] & model.geom_conaffinity[first_id])
        )

    assert model.geom_contype[torso_id] == 3
    assert model.geom_conaffinity[torso_id] == 3
    assert model.geom_contype[foot_id] == 1
    assert model.geom_conaffinity[foot_id] == 1
    assert collision_masks_match(torso_id, blocker_id)
    assert not collision_masks_match(foot_id, blocker_id)
    assert collision_masks_match(foot_id, rung_id)

    # Move the initialized robot's trunk onto the left ladder face and verify
    # that the configured model produces a real blocker contact.  This tests
    # the post-Entity collision configuration used by MJLab/MJWarp, not merely
    # the raw geom attributes emitted by the spec builder.
    blocker_contact_data = mujoco.MjData(model)
    blocker_contact_data.qpos[:] = data.qpos
    blocker_contact_data.qpos[0] = -0.50
    mujoco.mj_forward(model, blocker_contact_data)
    blocker_contact_geoms = set()
    for contact in blocker_contact_data.contact:
        geom_pair = (int(contact.geom[0]), int(contact.geom[1]))
        if blocker_id not in geom_pair:
            continue
        other_geom = geom_pair[1] if geom_pair[0] == blocker_id else geom_pair[0]
        blocker_contact_geoms.add(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
        )
    assert blocker_contact_geoms & {
        "pelvis_collision",
        "torso_collision",
        "head_collision",
    }
    assert not any(name.startswith("left_foot") for name in blocker_contact_geoms)

    left_ladder_geoms = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        ).startswith("left_ladder_")
    }
    assert left_ladder_geoms == {
        "left_ladder_body_blocker",
        "left_ladder_rail_01",
        "left_ladder_rail_02",
        *(f"left_ladder_rung_{index:02d}" for index in range(1, 10)),
    }
    rung_02_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_ladder_rung_02",
    )
    clear_vertical_gap = (
        model.geom_pos[rung_02_id, 2]
        - model.geom_pos[rung_id, 2]
        - model.geom_size[rung_02_id, 2]
        - model.geom_size[rung_id, 2]
    )
    assert clear_vertical_gap > 0.20
    left_ladder_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "left_ladder_body",
    )
    assert left_ladder_body_id >= 0
    assert model.body_dofnum[left_ladder_body_id] == 0
    assert (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_ladder_rung_09_grip",
        )
        >= 0
    )
    assert (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "left_grip_weld",
        )
        >= 0
    )


def test_ladder_fsm_executes_stabilize_hands_then_feet_in_order() -> None:
    command = _dummy_ladder_command()
    command._test_foot_support[:] = True

    command._advance_stabilization_phase(command.phase.clone())
    assert command.just_stabilized.item()
    assert command.phase.item() == int(LadderPhase.FIRST_HAND)
    assert command.held_rung.tolist() == [[-1, 4]]

    command._advance_hand_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.SECOND_HAND)
    assert command.held_rung.tolist() == [[5, -1]]

    command._advance_hand_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.FIRST_FOOT)
    assert command.held_rung.tolist() == [[5, 5]]

    command._test_foot_contact[0, 0] = True
    command._advance_foot_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.SECOND_FOOT)
    assert command.foot_rung.tolist() == [[2, 1]]

    command._test_foot_contact[0, 1] = True
    command._advance_foot_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.STABILIZE)
    assert command.foot_rung.tolist() == [[2, 2]]
    assert command.just_cycle_completed.item()


def test_ladder_foot_phase_requires_real_contact_and_both_hands() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_FOOT)
    command._test_foot_support[0, 1] = True

    command._advance_foot_phase(command.phase.clone())
    assert command.foot_rung.tolist() == [[1, 1]]

    command._test_foot_contact[0, 0] = True
    command.attached[0, 0] = False
    command._advance_foot_phase(command.phase.clone())
    assert command.foot_rung.tolist() == [[1, 1]]

    command.attached[:] = True
    command._advance_foot_phase(command.phase.clone())
    assert command.foot_rung.tolist() == [[2, 1]]


def test_ladder_hand_phase_keeps_both_feet_and_other_hand_supported() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._release(torch.tensor([0]), 0)

    command._advance_hand_phase(command.phase.clone())
    assert command.held_rung.tolist() == [[-1, 4]]

    command._test_foot_support[:] = True
    command.attached[0, 1] = False
    command._advance_hand_phase(command.phase.clone())
    assert command.held_rung.tolist() == [[-1, 4]]

    command.attached[0, 1] = True
    command._advance_hand_phase(command.phase.clone())
    assert command.held_rung.tolist() == [[5, -1]]
    assert command.phase.item() == int(LadderPhase.SECOND_HAND)


def test_ladder_curriculum_ends_prefix_episode_before_locked_phase() -> None:
    command = _dummy_ladder_command()
    command._unlocked_phase = int(LadderPhase.STABILIZE)
    command._env.common_step_counter = 0
    command._test_foot_support[:] = True

    command._advance_stabilization_phase(command.phase.clone())

    assert command.phase.item() == int(LadderPhase.STABILIZE)
    assert command.curriculum_stage_complete.item()


def test_ladder_curriculum_requires_strictly_more_than_80_of_100() -> None:
    command = _dummy_ladder_command()
    command._unlocked_phase = int(LadderPhase.STABILIZE)
    command._curriculum_phase_start_step = 0
    command._env.common_step_counter = 100

    assert not command.update_curriculum([0] * 20 + [1] * 80)
    assert command.curriculum_success_rate == pytest.approx(0.80)
    assert command.max_unlocked_phase == int(LadderPhase.STABILIZE)

    assert command.update_curriculum([1])
    assert command.max_unlocked_phase == int(LadderPhase.FIRST_HAND)
    assert command.curriculum_success_rate == 0.0
    assert command.curriculum_phase_steps == 0


def test_ladder_curriculum_requires_full_window_and_minimum_phase_steps() -> None:
    command = _dummy_ladder_command()
    command._unlocked_phase = int(LadderPhase.STABILIZE)
    command._curriculum_phase_start_step = 0
    command._env.common_step_counter = 99

    assert not command.update_curriculum([1] * 99)
    assert not command.update_curriculum([1])
    assert command.max_unlocked_phase == int(LadderPhase.STABILIZE)

    command._env.common_step_counter = 100
    assert command.update_curriculum([])
    assert command.max_unlocked_phase == int(LadderPhase.FIRST_HAND)


def test_ladder_curriculum_state_round_trip() -> None:
    command = _dummy_ladder_command()
    command._unlocked_phase = int(LadderPhase.SECOND_HAND)
    command._curriculum_phase_start_step = 12_345
    command._recent_curriculum_outcomes.extend([0, 1, 1])
    command._pending_curriculum_outcomes.extend([1, 0])
    state = command.curriculum_state_dict()

    restored = _dummy_ladder_command()
    restored.load_curriculum_state_dict(state)

    assert restored.max_unlocked_phase == int(LadderPhase.SECOND_HAND)
    assert restored._curriculum_phase_start_step == 12_345
    assert list(restored._recent_curriculum_outcomes) == [0, 1, 1]
    assert restored.drain_curriculum_outcomes() == [1, 0]


def test_ladder_runner_saves_curriculum_state(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _dummy_ladder_command()
    command._unlocked_phase = int(LadderPhase.FIRST_FOOT)
    command._recent_curriculum_outcomes.extend([1, 0, 1])
    runner = object.__new__(LadderOnPolicyRunner)
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda _name: command)
        )
    )
    captured: dict[str, object] = {}

    def _capture_save(_runner, path: str, infos=None) -> None:
        captured["path"] = path
        captured["infos"] = infos

    monkeypatch.setattr(MjlabOnPolicyRunner, "save", _capture_save)
    runner.save("curriculum.pt", infos={"existing": 7})

    assert captured["path"] == "curriculum.pt"
    infos = captured["infos"]
    assert isinstance(infos, dict)
    assert infos["existing"] == 7
    state = infos["ladder_curriculum_state"]
    assert isinstance(state, dict)
    assert state["unlocked_phase"] == int(LadderPhase.FIRST_FOOT)
    assert state["recent_outcomes"] == [1, 0, 1]


def test_ladder_runner_rejects_checkpoint_without_curriculum_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _dummy_ladder_command()
    runner = object.__new__(LadderOnPolicyRunner)
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda _name: command)
        )
    )
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", lambda *_args, **_kwargs: {})

    with pytest.raises(RuntimeError, match="adaptive ladder curriculum state"):
        runner.load("fixed-schedule.pt")


def _reward_test_env(command: SimpleNamespace) -> SimpleNamespace:
    command_manager = SimpleNamespace(get_term=lambda _name: command)
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        command_manager=command_manager,
    )


def _reward_test_command() -> SimpleNamespace:
    return SimpleNamespace(
        initialized=torch.tensor([True]),
        finished=torch.tensor([False]),
        is_hand_phase=torch.tensor([False]),
        is_foot_phase=torch.tensor([True]),
        attached=torch.tensor([[True, True]]),
        active_hand=torch.tensor([0]),
        active_hand_pos_w=torch.tensor([[0.20, 0.0, 0.0]]),
        target_pos_w=torch.zeros((1, 3)),
        target_rung=torch.tensor([5]),
        active_foot=torch.tensor([0]),
        active_foot_pos_w=torch.tensor([[0.12, 0.0, 0.0]]),
        active_foot_target_pos_w=torch.zeros((1, 3)),
        target_foot_rung=torch.tensor([2]),
        torso_com_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
        torso_com_vel_w=torch.zeros((1, 3)),
        num_rungs=9,
        just_advanced=torch.tensor([False]),
        just_foot_advanced=torch.tensor([False]),
        just_stabilized=torch.tensor([False]),
        just_cycle_completed=torch.tensor([False]),
    )


def test_ladder_target_progress_reward_does_not_pay_for_hovering() -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)
    cfg = SimpleNamespace(params={"target": "foot", "max_speed": 0.60})
    reward_term = LadderTargetProgressReward(cfg, env)

    assert reward_term(env, "ladder", "foot", 0.60).item() == 0.0

    command.active_foot_pos_w[0, 0] = 0.108
    assert reward_term(env, "ladder", "foot", 0.60).item() == pytest.approx(1.0)
    assert reward_term(env, "ladder", "foot", 0.60).item() == 0.0

    command.active_foot_pos_w[0, 0] = 0.12
    assert reward_term(env, "ladder", "foot", 0.60).item() == pytest.approx(-1.0)

    command.attached[0, 0] = False
    command.active_foot_pos_w[0, 0] = 0.10
    assert reward_term(env, "ladder", "foot", 0.60).item() == 0.0


def test_ladder_torso_ascent_reward_requires_supported_foot_phase() -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)
    cfg = SimpleNamespace(params={"max_speed": 0.40})
    reward_term = LadderTorsoAscentReward(cfg, env)

    assert reward_term(env, "ladder", 0.40).item() == 0.0

    command.torso_com_pos_w[0, 2] = 1.008
    assert reward_term(env, "ladder", 0.40).item() == pytest.approx(1.0)

    command.torso_com_pos_w[0, 2] = 1.004
    assert reward_term(env, "ladder", 0.40).item() == pytest.approx(-0.5, abs=1.0e-4)

    command.attached[:] = False
    command.torso_com_pos_w[0, 2] = 1.012
    assert reward_term(env, "ladder", 0.40).item() == 0.0

    command.attached[:] = True
    command.is_foot_phase[:] = False
    command.torso_com_pos_w[0, 2] = 1.020
    assert reward_term(env, "ladder", 0.40).item() == 0.0


def test_ladder_transition_rewards_are_dt_independent_impulses() -> None:
    command = _reward_test_command()
    command.just_advanced[:] = True
    command.just_foot_advanced[:] = True
    command.just_stabilized[:] = True
    command.just_cycle_completed[:] = True
    command.finished[:] = True
    env = _reward_test_env(command)

    assert ladder_rung_advance(env, "ladder").item() == pytest.approx(50.0)
    assert ladder_foot_rung_advance(env, "ladder").item() == pytest.approx(50.0)
    assert ladder_stabilized(env, "ladder").item() == pytest.approx(50.0)
    assert ladder_cycle_completed(env, "ladder").item() == pytest.approx(50.0)
    assert ladder_finished(env, "ladder").item() == pytest.approx(50.0)


def test_ladder_reset_pose_starts_initialized_on_second_foot_rung() -> None:
    command = _dummy_ladder_command()
    command.initialized[:] = False
    command.attached[:] = False
    command.held_rung = torch.full((1, 2), -1, dtype=torch.long)
    command.foot_rung[:] = 1
    command.target_foot_rung[:] = 2
    command._pending_start_pose_init = torch.tensor([True])
    command.just_initialized = torch.tensor([False])

    command._initialize_from_start_pose()

    assert command.initialized.item()
    assert command.just_initialized.item()
    assert not command._pending_start_pose_init.item()
    assert command.phase.item() == int(LadderPhase.STABILIZE)
    assert command.held_rung.tolist() == [[4, 4]]
    assert command.target_rung.item() == 4
    assert command.foot_rung.tolist() == [[1, 1]]
    assert command.target_foot_rung.item() == 1

    command._test_foot_support[:] = True
    command._advance_stabilization_phase(command.phase.clone())

    assert command.phase.item() == int(LadderPhase.FIRST_HAND)
    assert command.held_rung.tolist() == [[-1, 4]]
    assert command.target_rung.item() == 5


def test_ladder_foot_contact_must_persist_for_configured_dwell() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_FOOT)
    command.cfg.foot_target_dwell_steps = 2
    command._test_foot_support[0, 1] = True

    command._test_foot_contact[0, 0] = True
    command._advance_foot_phase(command.phase.clone())
    assert command.foot_rung.tolist() == [[1, 1]]

    command._test_foot_contact[0, 0] = False
    command._advance_foot_phase(command.phase.clone())
    assert command._phase_dwell_count.item() == 0

    command._test_foot_contact[0, 0] = True
    command._advance_foot_phase(command.phase.clone())
    command._advance_foot_phase(command.phase.clone())
    assert command.foot_rung.tolist() == [[2, 1]]


def test_train_ladder_cli_has_no_motion_arguments() -> None:
    args = train_ladder.parse_args([])

    assert args.logger == "tensorboard"
    assert not hasattr(args, "motion_file")
    assert not hasattr(args, "sampling_mode")
    assert not hasattr(args, "task")
    with pytest.raises(SystemExit):
        train_ladder.parse_args(["--motion_file", "data/datasets_precomputed"])


def test_train_ladder_main_uses_shared_multi_gpu_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    def fake_launch(args: argparse.Namespace, argv: list[str]) -> None:
        called["gpu_ids"] = args.gpu_ids
        called["argv"] = argv

    monkeypatch.setattr(train_ladder.shared_train, "_launch_multi_gpu", fake_launch)
    monkeypatch.setattr(
        train_ladder,
        "_run_worker",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("worker should not run in launcher branch")
        ),
    )

    train_ladder.main(["train_ladder.py", "--gpu_ids", "0", "1", "--num_envs", "1024"])

    assert called == {
        "gpu_ids": [0, 1],
        "argv": [
            "train_ladder.py",
            "--gpu_ids",
            "0",
            "1",
            "--num_envs",
            "1024",
        ],
    }
