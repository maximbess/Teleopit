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
    LadderFootPlacementReward,
    LadderPhase,
    LadderPhaseProgressReward,
    LadderTargetProgressReward,
    LadderTorsoAscentReward,
    LadderUpwardProgressReward,
    ladder_cycle_completed,
    ladder_com_alignment_exp,
    ladder_finished,
    ladder_foot_rung_advance,
    ladder_critic_privileged,
    ladder_failure_penalty,
    ladder_phase_completed,
    ladder_pre_release_stalled,
    ladder_rung_advance,
    ladder_stabilization_orientation_error_l2,
    ladder_stabilized,
    ladder_torso_posture_exp,
    ladder_torso_stability_exp,
)
from train_mimic.tasks.tracking.rl import LadderOnPolicyRunner
from train_mimic.tasks.tracking.rl.runner import (
    _project_distribution_std,
    _set_distribution_std,
)


class _DummyLadderCommand(LadderClimbCommand):
    @property
    def hand_pos_w(self) -> torch.Tensor:
        return self._test_hand_pos

    @property
    def hand_target_pos_w(self) -> torch.Tensor:
        return self._test_hand_target

    @property
    def foot_pos_w(self) -> torch.Tensor:
        return self._test_foot_pos

    @property
    def foot_target_pos_w(self) -> torch.Tensor:
        return self._test_foot_target

    @property
    def torso_rotation_w(self) -> torch.Tensor:
        return self._test_torso_rotation

    @property
    def torso_com_pos_w(self) -> torch.Tensor:
        return self._test_torso_com_pos

    @property
    def pelvis_com_pos_w(self) -> torch.Tensor:
        return self._test_pelvis_com_pos

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

    @property
    def torso_ang_vel_w(self) -> torch.Tensor:
        return self._test_torso_ang_vel

    @property
    def pelvis_ang_vel_w(self) -> torch.Tensor:
        return self._test_pelvis_ang_vel

    @property
    def torso_orientation_error(self) -> torch.Tensor:
        return self._test_torso_orientation_error

    @property
    def torso_support_offset_error(self) -> torch.Tensor:
        return self._test_torso_support_offset_error

    def _release(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
    ) -> None:
        if env_ids.numel() == 0:
            return
        self.released.append((env_ids.clone(), hand_id))
        sim = getattr(self._env, "sim", None)
        if sim is not None and hasattr(self, "_weld_ids"):
            weld_id = int(self._weld_ids[hand_id].item())
            sim.data.eq_active[env_ids, weld_id] = False
        self.attached[env_ids, hand_id] = False
        self.grip_strength[env_ids, hand_id] = 0.0
        self.held_rung[env_ids, hand_id] = -1
        self._reset_release_state(env_ids)

    def _attach(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
        rung_indices: torch.Tensor,
    ) -> None:
        self.attached[env_ids, hand_id] = True
        self.grip_strength[env_ids, hand_id] = 1.0
        self.held_rung[env_ids, hand_id] = rung_indices

    def _restore_weld_parameters(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
    ) -> None:
        del env_ids, hand_id

    def _set_grip_strength(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
        strength: torch.Tensor,
    ) -> None:
        self.grip_strength[env_ids, hand_id] = strength


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
        fixed_max_unlocked_phase=None,
        freeze_at_max_unlocked_phase=False,
        boundary_state_reset_prob=0.0,
        boundary_state_bank_size=0,
        curriculum_success_threshold=0.80,
        curriculum_window_size=100,
        curriculum_min_phase_steps=(100, 100, 100, 100),
        stabilization_dwell_steps=1,
        stabilization_dwell_max_steps=1,
        hand_target_dwell_steps=1,
        foot_target_dwell_steps=1,
        max_stabilization_torso_speed=0.20,
        max_stabilization_joint_speed=1.0,
        max_stabilization_body_angular_speed=0.40,
        max_stabilization_waist_joint_speed=0.60,
        max_stabilization_support_offset_error=0.18,
        max_phase_torso_orientation_error=0.30,
        max_phase_support_offset_error=0.15,
        max_phase_completion_torso_speed=0.20,
        max_phase_completion_joint_speed=1.0,
        first_foot_max_body_drop=0.03,
        cycle_min_body_ascent=0.12,
        release_preload_dwell_steps=1,
        release_ramp_steps=2,
        release_final_dwell_steps=1,
        release_recovery_steps=1,
        pre_release_timeout_steps=300,
        max_release_torso_speed=0.12,
        max_release_torso_orientation_error=0.25,
        max_release_support_offset_error=0.12,
        release_soft_timeconst=0.18,
        release_soft_impedance=0.05,
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
    command._boundary_bank_capacity = 0
    command._boundary_bank_counts = [0] * len(LadderPhase)
    command._boundary_bank_cursors = [0] * len(LadderPhase)
    command.initialized = torch.tensor([True])
    command.attached = torch.tensor([[True, True]])
    command.grip_strength = torch.ones((1, 2))
    command.held_rung = torch.tensor([[4, 4]])
    command.finished = torch.tensor([False])
    command.phase = torch.tensor([int(phase)])
    command.phase_frozen = torch.tensor([False])
    command._phase_dwell_count = torch.tensor([0])
    command._release_active = torch.tensor([False])
    command._release_preload_count = torch.tensor([0])
    command._release_ramp_count = torch.tensor([0])
    command._release_final_count = torch.tensor([0])
    command._release_age_count = torch.tensor([0])
    command._stabilization_dwell_target = torch.tensor([1])
    command._test_torso_com_pos = torch.tensor([[0.0, 0.0, 1.0]])
    command._test_pelvis_com_pos = torch.tensor([[0.0, 0.0, 1.0]])
    command.start_height = torch.tensor([1.0])
    command._cycle_start_body_height = torch.tensor([1.0])
    command._phase_start_body_height = torch.tensor([1.0])
    command.episode_max_body_height = torch.tensor([1.0])
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
    command.pre_release_stalled = torch.tensor([False])
    command._started_from_boundary = torch.tensor([False])
    command._pending_start_pose_init = torch.tensor([False])
    command._pending_boundary_state_init = torch.tensor([False])
    command._test_hand_pos = torch.zeros((1, 2, 3))
    command._test_hand_target = torch.zeros((1, 2, 3))
    command._test_foot_pos = torch.zeros((1, 2, 3))
    command._test_foot_target = torch.zeros((1, 2, 3))
    command._test_torso_rotation = torch.eye(3).unsqueeze(0)
    command._test_active_hand_pos = torch.zeros((1, 3))
    command._test_active_hand_target = torch.zeros((1, 3))
    command._test_active_hand_vel = torch.zeros((1, 3))
    command._test_active_foot_pos = torch.zeros((1, 3))
    command._test_active_foot_target = torch.zeros((1, 3))
    command._test_active_foot_vel = torch.zeros((1, 3))
    command._test_torso_com_vel = torch.zeros((1, 3))
    command._test_torso_ang_vel = torch.zeros((1, 3))
    command._test_pelvis_ang_vel = torch.zeros((1, 3))
    command._waist_joint_ids = torch.tensor([12, 13, 14])
    command._test_torso_orientation_error = torch.zeros(1)
    command._test_torso_support_offset_error = torch.zeros(1)
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
        "critic_privileged",
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
    assert set(cfg.observations["actor_ladder"].terms) == {"rung_tokens_torso"}
    assert set(cfg.observations["critic_privileged"].terms) == {"ladder_privileged"}
    assert cfg.observations["critic_privileged"].history_length is None
    assert not any(name.startswith("motion_") for name in cfg.rewards)
    assert set(cfg.rewards) == {
        "ladder_upward_progress",
        "ladder_phase_completed",
        "ladder_stabilization_orientation",
        "ladder_failure",
        "ladder_finished",
        "ladder_missing_foot_support",
        "ladder_foot_recovery",
        "ladder_stabilization_violation",
        "ladder_stabilization_pose",
        "ladder_stabilization_joint_velocity",
        "ladder_unwanted_contact",
        "action_rate",
        "survival",
        "self_collisions",
        "feet_acc",
        "joint_limits",
    } | {
        f"ladder_{phase.name.lower()}_{component}"
        for phase in LadderPhase
        for component in ("progress", "foot_placement")
    }
    upward_cfg = cfg.rewards["ladder_upward_progress"]
    assert upward_cfg.func is LadderUpwardProgressReward
    assert upward_cfg.weight == 20.0
    foot_placement_cfg = cfg.rewards["ladder_stabilize_foot_placement"]
    assert foot_placement_cfg.func is LadderFootPlacementReward
    assert foot_placement_cfg.weight == 8.0
    assert foot_placement_cfg.params["distance_std"] == 0.06
    assert "loss_scale" not in foot_placement_cfg.params
    progress_cfg = cfg.rewards["ladder_stabilize_progress"]
    assert progress_cfg.func is LadderPhaseProgressReward
    assert progress_cfg.weight == 8.0
    assert progress_cfg.params["first_hand_body_weight"] == 0.0
    assert progress_cfg.params["second_hand_body_weight"] == 0.0
    assert progress_cfg.params["foot_body_weight"] == 0.0
    assert progress_cfg.params["release_progress_weight"] == 1.0
    assert progress_cfg.params["unsupported_progress_scale"] == 0.25
    assert cfg.rewards["ladder_phase_completed"].func is ladder_phase_completed
    assert cfg.rewards["ladder_phase_completed"].weight == 25.0
    assert (
        cfg.rewards["ladder_stabilization_orientation"].func
        is ladder_stabilization_orientation_error_l2
    )
    assert cfg.rewards["ladder_stabilization_orientation"].weight == -1.0
    assert cfg.rewards["ladder_failure"].func is ladder_failure_penalty
    assert cfg.rewards["ladder_failure"].weight == -50.0
    assert cfg.rewards["ladder_finished"].weight == 100.0
    assert cfg.rewards["action_rate"].weight == -0.1
    assert cfg.rewards["ladder_missing_foot_support"].weight == -2.0
    assert cfg.rewards["ladder_foot_recovery"].weight == 4.0
    assert cfg.rewards["ladder_stabilization_violation"].weight == -1.0
    assert cfg.rewards["survival"].weight == 3.0
    assert cfg.rewards["joint_limits"].weight == -10.0
    assert cfg.rewards["feet_acc"].weight == -2.5e-6
    assert cfg.rewards["feet_acc"].params["asset_cfg"].joint_names == r".*ankle.*"
    assert cfg.rewards["self_collisions"].weight == -0.1
    assert cfg.rewards["self_collisions"].params["force_threshold"] == 1.0
    for phase in LadderPhase:
        for component in ("progress", "foot_placement"):
            term = cfg.rewards[f"ladder_{phase.name.lower()}_{component}"]
            assert term.weight == 8.0
            assert term.params["phase"] == int(phase)
    assert cfg.terminations["curriculum_stage_complete"].time_out is True
    assert cfg.terminations["pre_release_stalled"].func is ladder_pre_release_stalled
    assert cfg.curriculum == {}
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
    assert ladder_cmd.boundary_state_reset_prob == 0.50
    assert ladder_cmd.boundary_state_bank_size == 1_024
    assert ladder_cmd.curriculum_min_phase_steps == (
        36_000,
        120_000,
        120_000,
        120_000,
    )
    assert ladder_cmd.stabilization_dwell_steps == 50
    assert ladder_cmd.stabilization_dwell_max_steps == 100
    assert ladder_cmd.max_stabilization_body_angular_speed == 0.40
    assert ladder_cmd.max_stabilization_waist_joint_speed == 0.60
    assert ladder_cmd.max_stabilization_support_offset_error == 0.18
    assert ladder_cmd.max_phase_torso_orientation_error == 0.30
    assert ladder_cmd.max_phase_support_offset_error == 0.15
    assert ladder_cmd.max_phase_completion_torso_speed == 0.20
    assert ladder_cmd.max_phase_completion_joint_speed == 1.0
    assert ladder_cmd.first_foot_max_body_drop == 0.03
    assert ladder_cmd.cycle_min_body_ascent == 0.12
    assert ladder_cmd.release_preload_dwell_steps == 8
    assert ladder_cmd.release_ramp_steps == 20
    assert ladder_cmd.release_final_dwell_steps == 5
    assert ladder_cmd.release_recovery_steps == 2
    assert ladder_cmd.pre_release_timeout_steps == 300
    assert ladder_cmd.max_release_torso_speed == 0.12
    assert ladder_cmd.max_release_torso_orientation_error == 0.25
    assert ladder_cmd.max_release_support_offset_error == 0.12
    assert ladder_cmd.release_soft_timeconst == 0.18
    assert ladder_cmd.release_soft_impedance == 0.05
    assert cfg.events["prepare_ladder_weld_model"].func.model_fields == (
        "eq_solref",
        "eq_solimp",
    )
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
        "ladder_unwanted_contact_left",
        "ladder_unwanted_contact_right",
        "self_collision",
    )
    assert load_runner_cls(LADDER_RL_TASK) is LadderOnPolicyRunner

    play_cfg = make_g1_ladder_rl_env_cfg(play=True)
    assert play_cfg.curriculum == {}
    assert play_cfg.commands["ladder"].curriculum_enabled is False
    assert play_cfg.commands["ladder"].freeze_at_max_unlocked_phase is False
    assert play_cfg.commands["ladder"].boundary_state_reset_prob == 0.0
    assert play_cfg.commands["ladder"].boundary_state_bank_size == 0
    assert play_cfg.rewards["ladder_stabilize_progress"].weight == 8.0
    assert play_cfg.rewards["ladder_phase_completed"].weight == 25.0


def test_ladder_play_checkpoint_restore_preserves_fixed_phase() -> None:
    command = object.__new__(LadderClimbCommand)
    command.cfg = SimpleNamespace(curriculum_enabled=False)
    command._play_unlocked_phase = int(LadderPhase.SECOND_HAND)
    command._unlocked_phase = int(LadderPhase.SECOND_FOOT)
    command._recent_curriculum_outcomes = deque([1, 1], maxlen=100)
    command._pending_curriculum_outcomes = [1]

    command.load_curriculum_state_dict(
        {
            "version": 2,
            "unlocked_phase": int(LadderPhase.SECOND_FOOT),
            "phase_start_step": 0,
            "recent_outcomes": [1] * 100,
            "pending_outcomes": [],
        }
    )

    assert command.max_unlocked_phase == int(LadderPhase.SECOND_HAND)
    assert list(command._recent_curriculum_outcomes) == []
    assert command._pending_curriculum_outcomes == []


def test_ladder_fixed_phase_can_freeze_without_curriculum_termination() -> None:
    command = object.__new__(LadderClimbCommand)
    command.cfg = SimpleNamespace(freeze_at_max_unlocked_phase=True)
    command._unlocked_phase = int(LadderPhase.STABILIZE)
    command.phase_frozen = torch.zeros(1, dtype=torch.bool)
    command.curriculum_stage_complete = torch.zeros(1, dtype=torch.bool)

    command._transition_or_finish_curriculum(
        torch.tensor([0]),
        LadderPhase.FIRST_HAND,
    )

    assert command.phase_frozen.tolist() == [True]
    assert command.curriculum_stage_complete.tolist() == [False]


def test_ladder_task_uses_temporal_geometry_ppo_config() -> None:
    rl_cfg = load_rl_cfg(LADDER_RL_TASK)

    assert rl_cfg.experiment_name == LADDER_RL_EXPERIMENT_NAME
    assert rl_cfg.actor.class_name.endswith(":TemporalCNNModel")
    assert rl_cfg.critic.class_name.endswith(":TemporalCNNModel")
    assert rl_cfg.actor.hidden_dims == (2048, 1024, 512, 256, 128)
    assert rl_cfg.critic.hidden_dims == (2048, 1024, 512, 256, 128)
    assert rl_cfg.obs_groups == {
        "actor": ("actor", "actor_history", "actor_ladder"),
        "critic": (
            "critic",
            "critic_history",
            "critic_ladder",
            "critic_privileged",
        ),
    }
    assert rl_cfg.actor.distribution_cfg["init_std"] == 0.7
    assert rl_cfg.actor.distribution_cfg["std_range"] == (0.25, 1.0)
    assert rl_cfg.algorithm.entropy_coef == 0.005
    assert rl_cfg.algorithm.learning_rate == 5.0e-4
    assert rl_cfg.save_interval == 1_000
    assert rl_cfg.max_iterations == 60_000
    assert rl_cfg.upload_model is False


def test_ladder_geometry_observation_uses_target_aware_torso_tokens() -> None:
    command = object.__new__(LadderClimbCommand)
    command.cfg = SimpleNamespace(start_rung=0)
    command._rung_site_ids = torch.tensor([0, 1], dtype=torch.long)
    command._rung_half_lengths = torch.tensor([0.5, 0.25])
    command._torso_body_id = 0
    command._all_env_ids = torch.arange(1)
    command.initialized = torch.tensor([True])
    command.active_hand = torch.tensor([0])
    command.target_rung = torch.tensor([1])
    command.held_rung = torch.tensor([[0, -1]])
    command.grip_strength = torch.tensor([[0.4, 0.0]])
    command.active_foot = torch.tensor([0])
    command.target_foot_rung = torch.tensor([1])
    command.foot_rung = torch.tensor([[-1, 1]])

    identity = torch.eye(3).reshape(9)
    data = SimpleNamespace(
        site_xpos=torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]),
        site_xmat=identity.repeat(1, 2, 1),
        xpos=torch.tensor([[[0.0, 0.0, 0.0]]]),
        xmat=identity.repeat(1, 1, 1),
    )
    command._env = SimpleNamespace(device="cpu", sim=SimpleNamespace(data=data))

    observation = command.rung_tokens_torso

    assert observation.shape == (1, 2, 15)
    assert observation[0, 0, :7].tolist() == pytest.approx(
        [1.0, 0.0, -0.5, 1.0, 0.0, 0.5, 1.0]
    )
    assert observation[0, 1, :7].tolist() == pytest.approx(
        [1.0, 0.0, 0.75, 1.0, 0.0, 1.25, 1.0]
    )
    # Marker layout: hand targets (L/R), foot targets (L/R), held hands
    # (L/R), and assigned feet (L/R).
    assert observation[0, 0, 7:].tolist() == pytest.approx(
        [0.0, 1.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]
    )
    assert observation[0, 1, 7:].tolist() == pytest.approx(
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )


def test_ladder_command_rotates_target_vectors_into_torso_frame() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._test_torso_rotation = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    command._test_hand_target[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    command._test_hand_target[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    command._test_foot_target[0, 0] = torch.tensor([0.0, 0.0, 1.0])

    observation = command.command

    assert observation.shape == (1, 24)
    assert observation[0, 5:11].tolist() == pytest.approx(
        [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    )
    assert observation[0, 15:21].tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    command.grip_strength[0, 0] = 0.35
    assert command.command[0, 13].item() == pytest.approx(0.35)


def test_ladder_robot_augments_canonical_g1_spec() -> None:
    robot_cfg = make_g1_ladder_training_robot_cfg()
    spec = robot_cfg.spec_fn()
    model = Entity(robot_cfg).spec.compile()

    assert robot_cfg.init_state.pos == (-0.913384, 0.0, 1.364509)
    arm_actuator_effort_limits = {
        actuator_cfg.effort_limit
        for actuator_cfg in robot_cfg.articulation.actuators
        if all(
            any(part in target for part in ("shoulder", "elbow", "wrist"))
            for target in actuator_cfg.target_names_expr
        )
    }
    assert arm_actuator_effort_limits == {3.5, 17.5}
    compiled_arm_effort_limits = set()
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        if actuator_name is None or not any(
            part in actuator_name for part in ("shoulder", "elbow", "wrist")
        ):
            continue
        lower, upper = model.actuator_forcerange[actuator_id]
        assert lower == pytest.approx(-upper)
        compiled_arm_effort_limits.add(float(upper))
    assert compiled_arm_effort_limits == {3.5, 17.5}
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
    assert blocker_id == -1
    assert model.geom_type[rung_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_type[rail_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert model.geom_size[rung_id].tolist() == pytest.approx([0.055, 0.35, 0.035])
    assert model.geom_contype[rung_id] == 1
    assert model.geom_conaffinity[rung_id] == 1
    assert model.geom_condim[rung_id] == 4
    assert model.geom_contype[rail_id] == 1
    assert model.geom_conaffinity[rail_id] == 1
    assert model.geom_condim[rail_id] == 4
    assert model.geom_solref[rung_id].tolist() == pytest.approx([0.005, 1.0])
    assert model.geom_solref[rail_id].tolist() == pytest.approx([0.005, 1.0])
    # Every refined surface remains enabled after the stock G1 editor.
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    refined = [i for i, name in enumerate(names) if "_omni_" in name]
    assert refined
    assert all(model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH for i in refined)
    assert all(model.geom_contype[i] == model.geom_conaffinity[i] == 1 for i in refined)
    assert any(name.startswith("pelvis_contour_omni_") for name in names)
    assert model.geom_priority[rung_id] == 2
    assert model.geom_priority[rail_id] == 2

    left_ladder_geoms = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        ).startswith("left_ladder_")
    }
    assert left_ladder_geoms == {
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
    assert command.held_rung.tolist() == [[4, 4]]
    assert command.is_pre_release.item()

    command._advance_hand_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.FIRST_HAND)
    assert command.grip_strength[0, 0].item() == pytest.approx(0.5)
    command._advance_hand_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.SECOND_HAND)
    assert command.held_rung.tolist() == [[5, 4]]

    command._advance_hand_phase(command.phase.clone())
    command._advance_hand_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.FIRST_FOOT)
    assert command.held_rung.tolist() == [[5, 5]]

    command._test_foot_contact[0, 0] = True
    command._advance_foot_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.SECOND_FOOT)
    assert command.foot_rung.tolist() == [[2, 1]]

    command._test_torso_com_pos[0, 2] = 1.12
    command._test_pelvis_com_pos[0, 2] = 1.12
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


def test_ladder_foot_completion_requires_whole_body_height() -> None:
    first_foot = _dummy_ladder_command(LadderPhase.FIRST_FOOT)
    first_foot._test_foot_support[0, 1] = True
    first_foot._test_foot_contact[0, 0] = True
    first_foot._test_torso_com_pos[0, 2] = 0.96
    first_foot._test_pelvis_com_pos[0, 2] = 0.96

    first_foot._advance_foot_phase(first_foot.phase.clone())
    assert first_foot.foot_rung.tolist() == [[1, 1]]

    second_foot = _dummy_ladder_command(LadderPhase.SECOND_FOOT)
    second_foot.active_foot[:] = 1
    second_foot._test_foot_support[0, 0] = True
    second_foot._test_foot_contact[0, 1] = True
    second_foot._test_torso_com_pos[0, 2] = 1.11
    second_foot._test_pelvis_com_pos[0, 2] = 1.11

    second_foot._advance_foot_phase(second_foot.phase.clone())
    assert second_foot.foot_rung.tolist() == [[1, 1]]

    second_foot._test_torso_com_pos[0, 2] = 1.12
    second_foot._test_pelvis_com_pos[0, 2] = 1.12
    second_foot._advance_foot_phase(second_foot.phase.clone())
    assert second_foot.foot_rung.tolist() == [[1, 2]]


def test_ladder_hand_phase_keeps_both_feet_and_other_hand_supported() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._begin_release(torch.tensor([0]), 0)

    command._advance_hand_phase(command.phase.clone())
    assert command.held_rung.tolist() == [[4, 4]]
    assert command._release_ramp_count.item() == 0

    command._test_foot_support[:] = True
    command.attached[0, 1] = False
    command._advance_hand_phase(command.phase.clone())
    assert command.held_rung.tolist() == [[4, 4]]
    assert command._release_ramp_count.item() == 0

    command.attached[0, 1] = True
    command._advance_hand_phase(command.phase.clone())
    assert command.grip_strength[0, 0].item() == pytest.approx(0.5)
    command._advance_hand_phase(command.phase.clone())
    assert command.held_rung.tolist() == [[5, 4]]
    assert command.phase.item() == int(LadderPhase.SECOND_HAND)


def test_ladder_pre_release_softens_and_recovers_before_detach() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._test_foot_support[:] = True
    command._begin_release(torch.tensor([0]), 0)
    phase_mask = torch.tensor([True])

    command._advance_release_ramp(phase_mask)
    assert command._release_ramp_count.item() == 1
    assert command.grip_strength[0, 0].item() == pytest.approx(0.5)
    assert command.attached[0, 0].item()

    command._test_torso_com_vel[0, 0] = 0.20
    command._advance_release_ramp(phase_mask)
    assert command._release_ramp_count.item() == 0
    assert command.grip_strength[0, 0].item() == pytest.approx(1.0)
    assert command.attached[0, 0].item()

    command._test_torso_com_vel.zero_()
    command._advance_release_ramp(phase_mask)
    command._advance_release_ramp(phase_mask)
    assert not command.attached[0, 0].item()
    assert command.grip_strength[0, 0].item() == 0.0
    assert command.released[-1][1] == 0


def test_ladder_pre_release_cannot_be_vetoed_by_joint_motion() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._test_foot_support[:] = True
    command.robot.data.joint_vel[:] = 100.0
    command._begin_release(torch.tensor([0]), 0)

    command._advance_release_ramp(torch.tensor([True]))

    assert command._release_ramp_count.item() == 1
    assert command.grip_strength[0, 0].item() == pytest.approx(0.5)


def test_ladder_release_conditions_identify_each_failed_gate() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._test_foot_support[:] = True

    conditions = command._release_stability_conditions()
    assert conditions["stable"].item()

    command._test_torso_com_vel[0, 0] = 0.20
    conditions = command._release_stability_conditions()
    assert not conditions["torso_speed_valid"].item()
    assert not conditions["stable"].item()

    command._test_torso_com_vel.zero_()
    command._test_torso_orientation_error[:] = 0.26
    conditions = command._release_stability_conditions()
    assert not conditions["orientation_valid"].item()

    command._test_torso_orientation_error.zero_()
    command._test_torso_support_offset_error[:] = 0.13
    conditions = command._release_stability_conditions()
    assert not conditions["support_offset_valid"].item()

    command._test_torso_support_offset_error.zero_()
    command._test_foot_support[0, 0] = False
    conditions = command._release_stability_conditions()
    assert not conditions["feet_supported"].item()


def test_ladder_release_metrics_compute_during_initial_reset() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    metric_names = (
        "target_distance",
        "rung_progress",
        "attached_hands",
        "foot_target_distance",
        "foot_progress",
        "supported_feet",
        "torso_orientation_error",
        "torso_support_offset_error",
        "active_hand_release_progress",
        "pre_release_age",
        "stabilization_hands_attached",
        "stabilization_feet_supported",
        "stabilization_torso_speed",
        "stabilization_torso_speed_valid",
        "stabilization_joint_speed_rms",
        "stabilization_joint_speed_valid",
        "stabilization_torso_angular_speed",
        "stabilization_pelvis_angular_speed",
        "stabilization_angular_speed_valid",
        "stabilization_waist_joint_speed",
        "stabilization_waist_speed_valid",
        "stabilization_support_offset_valid",
        "stabilization_gate_valid",
        "stabilization_dwell_progress",
        "release_hands_attached",
        "release_feet_supported",
        "release_torso_speed",
        "release_torso_speed_valid",
        "release_orientation_valid",
        "release_support_offset_valid",
        "release_gate_valid",
        "release_preload_progress",
        "release_ramp_progress",
        "release_final_dwell_progress",
        "phase",
        "unlocked_phase",
        "curriculum_success_rate",
        "curriculum_window_fill",
        "curriculum_phase_steps",
    )
    command.metrics = {name: torch.zeros(1) for name in metric_names}

    command._update_metrics()

    assert command.metrics["release_gate_valid"].item() == 0.0
    assert command.metrics["release_ramp_progress"].item() == 0.0


def test_ladder_stabilization_metrics_expose_each_transition_gate() -> None:
    command = _dummy_ladder_command()
    command._test_foot_support[:] = True
    command._phase_dwell_count[:] = 1
    command._stabilization_dwell_target[:] = 2
    command.metrics = {
        name: torch.zeros(1)
        for name in (
            "target_distance",
            "rung_progress",
            "attached_hands",
            "foot_target_distance",
            "foot_progress",
            "supported_feet",
            "torso_orientation_error",
            "torso_support_offset_error",
            "active_hand_release_progress",
            "pre_release_age",
            "stabilization_hands_attached",
            "stabilization_feet_supported",
            "stabilization_torso_speed",
            "stabilization_torso_speed_valid",
            "stabilization_joint_speed_rms",
            "stabilization_joint_speed_valid",
            "stabilization_torso_angular_speed",
            "stabilization_pelvis_angular_speed",
            "stabilization_angular_speed_valid",
            "stabilization_waist_joint_speed",
            "stabilization_waist_speed_valid",
            "stabilization_support_offset_valid",
            "stabilization_gate_valid",
            "stabilization_dwell_progress",
            "release_hands_attached",
            "release_feet_supported",
            "release_torso_speed",
            "release_torso_speed_valid",
            "release_orientation_valid",
            "release_support_offset_valid",
            "release_gate_valid",
            "release_preload_progress",
            "release_ramp_progress",
            "release_final_dwell_progress",
            "phase",
            "unlocked_phase",
            "curriculum_success_rate",
            "curriculum_window_fill",
            "curriculum_phase_steps",
        )
    }

    command._update_metrics()

    assert command.metrics["stabilization_hands_attached"].item() == 1.0
    assert command.metrics["stabilization_feet_supported"].item() == 1.0
    assert command.metrics["stabilization_torso_speed_valid"].item() == 1.0
    assert command.metrics["stabilization_joint_speed_valid"].item() == 1.0
    assert command.metrics["stabilization_angular_speed_valid"].item() == 1.0
    assert command.metrics["stabilization_waist_speed_valid"].item() == 1.0
    assert command.metrics["stabilization_support_offset_valid"].item() == 1.0
    assert command.metrics["stabilization_gate_valid"].item() == 1.0
    assert command.metrics["stabilization_dwell_progress"].item() == 0.5

    command._test_torso_support_offset_error[:] = 0.19
    command._update_metrics()
    assert command.metrics["stabilization_support_offset_valid"].item() == 0.0
    assert command.metrics["stabilization_gate_valid"].item() == 0.0


def test_ladder_pre_release_timeout_terminates() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command.cfg.pre_release_timeout_steps = 2
    command._begin_release(torch.tensor([0]), 0)
    phase_mask = torch.tensor([True])

    command._advance_release_ramp(phase_mask)
    assert not command.pre_release_stalled.item()

    command._advance_release_ramp(phase_mask)
    assert command.pre_release_stalled.item()

    env = _reward_test_env(command)
    assert ladder_pre_release_stalled(env, "ladder").item()


def test_ladder_failure_penalty_excludes_success_and_prefix_completion() -> None:
    terms = {
        "success": torch.tensor([False, True, False, False, False]),
        "curriculum_stage_complete": torch.tensor(
            [False, False, True, False, False]
        ),
    }
    termination_manager = SimpleNamespace(
        dones=torch.tensor([True, True, True, True, False]),
        get_term=lambda name: terms[name],
    )
    env = SimpleNamespace(step_dt=0.02, termination_manager=termination_manager)

    penalty = ladder_failure_penalty(env, "ladder")

    assert penalty.tolist() == pytest.approx([50.0, 0.0, 0.0, 50.0, 0.0])


def test_ladder_grip_strength_maps_to_per_environment_weld_softness() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._weld_ids = torch.tensor([0, 1])
    command._strong_weld_solref = torch.tensor([[0.03, 1.0], [0.03, 1.0]])
    command._strong_weld_solimp = torch.tensor(
        [[0.90, 0.95, 0.001, 0.5, 2.0], [0.90, 0.95, 0.001, 0.5, 2.0]]
    )
    model = SimpleNamespace(
        eq_solref=command._strong_weld_solref.unsqueeze(0).clone(),
        eq_solimp=command._strong_weld_solimp.unsqueeze(0).clone(),
    )
    command._env.sim = SimpleNamespace(model=model)

    LadderClimbCommand._set_grip_strength(
        command,
        torch.tensor([0]),
        0,
        torch.tensor([0.5]),
    )

    assert model.eq_solref[0, 0, 0].item() == pytest.approx(0.105)
    assert model.eq_solref[0, 0, 1].item() == pytest.approx(1.0)
    assert model.eq_solimp[0, 0, 0].item() == pytest.approx(0.475)
    assert model.eq_solimp[0, 0, 1].item() == pytest.approx(0.50)
    torch.testing.assert_close(model.eq_solref[0, 1], torch.tensor([0.03, 1.0]))
    assert command.grip_strength[0, 0].item() == pytest.approx(0.5)


def test_ladder_support_centroid_fades_releasing_hand() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._test_hand_pos[0] = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    command._test_foot_pos.zero_()
    command._test_foot_support[:] = True

    torch.testing.assert_close(
        command.support_centroid_w,
        torch.tensor([[0.5, 0.0, 0.0]]),
    )
    command.grip_strength[0, 0] = 0.0
    torch.testing.assert_close(
        command.support_centroid_w,
        torch.zeros((1, 3)),
    )


def test_ladder_curriculum_ends_prefix_episode_before_locked_phase() -> None:
    command = _dummy_ladder_command()
    command._unlocked_phase = int(LadderPhase.STABILIZE)
    command._env.common_step_counter = 0
    command._test_foot_support[:] = True

    command._advance_stabilization_phase(command.phase.clone())

    assert command.phase.item() == int(LadderPhase.STABILIZE)
    assert command.curriculum_stage_complete.item()


def test_ladder_boundary_bank_restores_physics_and_configures_next_phase() -> None:
    command = _dummy_ladder_command()
    command.cfg.boundary_state_reset_prob = 0.5
    command.cfg.boundary_state_bank_size = 2
    command._anchor_mocap_ids = torch.tensor([0, 1])
    command._weld_ids = torch.tensor([0, 1])
    root_pose = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    root_velocity = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.1]])
    joint_pos = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    joint_vel = torch.full((1, 29), 0.25)
    writes: dict[str, torch.Tensor] = {}

    def write_joint_state_to_sim(
        position: torch.Tensor,
        velocity: torch.Tensor,
        *,
        env_ids: torch.Tensor,
    ) -> None:
        writes["joint_pos"] = position.clone()
        writes["joint_vel"] = velocity.clone()
        writes["joint_env_ids"] = env_ids.clone()

    def write_root_state_to_sim(
        state: torch.Tensor,
        *,
        env_ids: torch.Tensor,
    ) -> None:
        writes["root_state"] = state.clone()
        writes["root_env_ids"] = env_ids.clone()

    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pose_w=root_pose,
            root_link_vel_w=root_velocity,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        ),
        write_joint_state_to_sim=write_joint_state_to_sim,
        write_root_state_to_sim=write_root_state_to_sim,
        clear_state=lambda *, env_ids: writes.update(clear_env_ids=env_ids.clone()),
    )
    sim_data = SimpleNamespace(
        mocap_pos=torch.tensor([[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]]),
        mocap_quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        eq_active=torch.tensor([[True, True]]),
    )
    command._env.sim = SimpleNamespace(data=sim_data)
    command._allocate_boundary_state_bank()
    command._cycle_start_body_height[:] = 0.88
    command._capture_boundary_states(torch.tensor([0]), LadderPhase.FIRST_HAND)
    checkpoint_state = command._boundary_bank_state_dict()
    command._boundary_root_state.zero_()
    command._boundary_cycle_start_body_height.zero_()
    command._boundary_phase_start_body_height.zero_()
    command._boundary_bank_counts = [0] * len(LadderPhase)
    command._load_boundary_bank_state_dict(checkpoint_state)

    command.attached[:] = False
    command.held_rung[:] = -1
    command.foot_rung[:] = -1
    sim_data.mocap_pos.zero_()
    sim_data.eq_active.zero_()
    command._restore_boundary_states(
        torch.tensor([0]),
        torch.tensor([int(LadderPhase.FIRST_HAND)]),
        torch.tensor([0]),
    )

    assert command._boundary_bank_counts[int(LadderPhase.FIRST_HAND)] == 1
    torch.testing.assert_close(
        writes["root_state"],
        torch.cat((root_pose, root_velocity), dim=-1),
    )
    assert torch.equal(writes["joint_pos"], joint_pos)
    assert torch.equal(writes["joint_vel"], joint_vel)
    assert command.phase.item() == int(LadderPhase.FIRST_HAND)
    assert command.target_rung.item() == 5
    assert command.attached.tolist() == [[True, True]]
    assert command.grip_strength.tolist() == [[1.0, 1.0]]
    assert command.held_rung.tolist() == [[4, 4]]
    assert command.foot_rung.tolist() == [[1, 1]]
    assert command._cycle_start_body_height.item() == pytest.approx(0.88)
    assert command._phase_start_body_height.item() == pytest.approx(1.0)
    assert sim_data.eq_active.tolist() == [[True, True]]
    assert command.is_pre_release.item()
    assert command._pending_boundary_state_init.item()
    assert command._started_from_boundary.item()


def test_boundary_started_episode_does_not_pollute_curriculum_window() -> None:
    command = _dummy_ladder_command()
    command._episode_active = torch.tensor([True])
    command._started_from_boundary[:] = True
    command._env.termination_manager = SimpleNamespace(
        dones=torch.tensor([True]),
        get_term=lambda _name: torch.tensor([True]),
    )

    command._record_episode_outcomes(torch.tensor([0]))
    assert command._pending_curriculum_outcomes == []

    command._started_from_boundary[:] = False
    command._record_episode_outcomes(torch.tensor([0]))
    assert command._pending_curriculum_outcomes == [1]


def test_ladder_stabilization_requires_sampled_consecutive_hold() -> None:
    command = _dummy_ladder_command()
    command.cfg.stabilization_dwell_steps = 3
    command.cfg.stabilization_dwell_max_steps = 3
    command._stabilization_dwell_target[:] = 3
    command._test_foot_support[:] = True

    command._advance_stabilization_phase(command.phase.clone())
    command._advance_stabilization_phase(command.phase.clone())

    assert not command.just_stabilized.item()
    assert command.phase.item() == int(LadderPhase.STABILIZE)

    command._advance_stabilization_phase(command.phase.clone())

    assert command.just_stabilized.item()
    assert command.phase.item() == int(LadderPhase.FIRST_HAND)


def test_ladder_stabilization_rejects_invalid_support_offset() -> None:
    command = _dummy_ladder_command()
    command._test_foot_support[:] = True
    command._test_torso_support_offset_error[:] = 0.19

    command._advance_stabilization_phase(command.phase.clone())

    assert command._phase_dwell_count.item() == 0
    assert not command.just_stabilized.item()


def test_ladder_stabilization_transition_ignores_torso_orientation() -> None:
    command = _dummy_ladder_command()
    command._test_foot_support[:] = True
    command._test_torso_orientation_error[:] = 1.0

    command._advance_stabilization_phase(command.phase.clone())

    assert command.just_stabilized.item()
    assert command.phase.item() == int(LadderPhase.FIRST_HAND)


def test_ladder_stabilization_conditions_identify_each_failed_gate() -> None:
    command = _dummy_ladder_command()
    command._test_foot_support[:] = True

    assert command.stabilization_conditions_satisfied.item()

    command.attached[0, 0] = False
    assert not command._stabilization_conditions()["hands_attached"].item()
    command.attached[:] = True

    command._test_foot_support[0, 0] = False
    assert not command._stabilization_conditions()["feet_supported"].item()
    command._test_foot_support[:] = True

    command._test_torso_com_vel[0, 0] = 0.21
    assert not command._stabilization_conditions()["torso_speed_valid"].item()
    command._test_torso_com_vel.zero_()

    command.robot.data.joint_vel[0, 0] = 6.0
    assert not command._stabilization_conditions()["joint_speed_valid"].item()
    command.robot.data.joint_vel.zero_()

    command._test_torso_ang_vel[0, 2] = 0.41
    assert not command._stabilization_conditions()["angular_speed_valid"].item()
    command._test_torso_ang_vel.zero_()

    command._test_pelvis_ang_vel[0, 2] = 0.41
    assert not command._stabilization_conditions()["angular_speed_valid"].item()
    command._test_pelvis_ang_vel.zero_()

    command.robot.data.joint_vel[0, 12] = 0.61
    conditions = command._stabilization_conditions()
    assert conditions["joint_speed_valid"].item()
    assert not conditions["waist_speed_valid"].item()
    command.robot.data.joint_vel.zero_()

    command._test_torso_support_offset_error[:] = 0.19
    assert not command._stabilization_conditions()["support_offset_valid"].item()

    command._test_torso_support_offset_error[:] = 0.17
    command._test_torso_orientation_error[:] = 1.0
    assert not command.phase_support_constraints_satisfied.item()
    assert command.stabilization_conditions_satisfied.item()


def test_ladder_movement_completion_ignores_torso_orientation() -> None:
    command = _dummy_ladder_command(LadderPhase.FIRST_HAND)
    command._test_foot_support[:] = True
    command._test_torso_orientation_error[:] = 1.0

    assert not command.phase_support_constraints_satisfied.item()
    assert command.phase_completion_stable.item()


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


def test_ladder_distribution_projection_keeps_raw_std_trainable() -> None:
    distribution = SimpleNamespace(
        std_type="scalar",
        std_param=torch.nn.Parameter(torch.tensor([0.20, 0.50, 1.20])),
        std_range=(0.25, 1.0),
    )

    parameter, projected = _project_distribution_std(distribution)
    assert parameter is distribution.std_param
    assert projected
    torch.testing.assert_close(
        distribution.std_param,
        torch.tensor([0.25, 0.50, 1.0]),
    )

    parameter = _set_distribution_std(distribution, 0.7)
    assert parameter is distribution.std_param
    torch.testing.assert_close(parameter, torch.full((3,), 0.7))


def test_ladder_curriculum_promotion_restores_std_and_learning_rate() -> None:
    command = SimpleNamespace(
        drain_curriculum_outcomes=lambda: [1],
        update_curriculum=lambda outcomes: outcomes == [1],
        phase_name=lambda _phase: "first_hand",
        max_unlocked_phase=int(LadderPhase.FIRST_HAND),
        curriculum_success_rate=0.0,
        curriculum_window_fill=0.0,
        curriculum_phase_steps=0,
    )
    std_parameter = torch.nn.Parameter(torch.full((29,), 0.25))
    distribution = SimpleNamespace(
        std_type="scalar",
        std_param=std_parameter,
        std_range=(0.25, 1.0),
    )
    optimizer = torch.optim.Adam([std_parameter], lr=1.0e-5)
    optimizer.state[std_parameter]["stale_momentum"] = torch.tensor(1.0)
    policy = SimpleNamespace(distribution=distribution)
    algorithm = SimpleNamespace(
        get_policy=lambda: policy,
        optimizer=optimizer,
        learning_rate=1.0e-5,
    )
    runner = object.__new__(LadderOnPolicyRunner)
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda _name: command)
        )
    )
    runner.alg = algorithm
    runner.cfg = {
        "actor": {"distribution_cfg": {"init_std": 0.7}},
        "algorithm": {"learning_rate": 5.0e-4},
    }
    runner.logger = SimpleNamespace(writer=None)
    runner.is_distributed = False
    runner.gpu_global_rank = 0

    runner._synchronize_ladder_curriculum(iteration=1_500)

    torch.testing.assert_close(std_parameter, torch.full((29,), 0.7))
    assert optimizer.state[std_parameter] == {}
    assert algorithm.learning_rate == pytest.approx(5.0e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)


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


@pytest.mark.parametrize("version", [1, 2])
def test_ladder_curriculum_rejects_old_reward_or_collision_checkpoint(version) -> None:
    command = _dummy_ladder_command()

    with pytest.raises(ValueError, match="Unsupported ladder curriculum"):
        command.load_curriculum_state_dict(
            {
                "version": version,
                "unlocked_phase": int(LadderPhase.STABILIZE),
                "phase_start_step": 0,
            }
        )


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
        grip_strength=torch.ones((1, 2)),
        active_grip_strength=torch.ones(1),
        foot_support=torch.tensor([[True, True]]),
        foot_contact=torch.tensor([[True, True]]),
        foot_pos_w=torch.zeros((1, 2, 3)),
        held_foot_target_pos_w=torch.zeros((1, 2, 3)),
        phase=torch.tensor([int(LadderPhase.STABILIZE)]),
        robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros((1, 29)))),
        active_hand=torch.tensor([0]),
        active_hand_pos_w=torch.tensor([[0.20, 0.0, 0.0]]),
        target_pos_w=torch.zeros((1, 3)),
        target_rung=torch.tensor([5]),
        active_foot=torch.tensor([0]),
        active_foot_pos_w=torch.tensor([[0.12, 0.0, 0.0]]),
        active_foot_target_pos_w=torch.zeros((1, 3)),
        target_foot_rung=torch.tensor([2]),
        torso_com_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
        body_height=torch.tensor([1.0]),
        episode_max_body_height=torch.tensor([1.0]),
        torso_com_vel_w=torch.zeros((1, 3)),
        torso_orientation_error=torch.zeros(1),
        torso_support_offset_error=torch.zeros(1),
        stabilization_conditions_satisfied=torch.tensor([True]),
        phase_support_constraints_satisfied=torch.tensor([True]),
        phase_required_supports_satisfied=torch.tensor([True]),
        _phase_dwell_count=torch.tensor([0]),
        _stabilization_dwell_target=torch.tensor([50]),
        num_rungs=9,
        just_advanced=torch.tensor([False]),
        just_foot_advanced=torch.tensor([False]),
        just_stabilized=torch.tensor([False]),
        just_cycle_completed=torch.tensor([False]),
    )


def test_ladder_phase_progress_is_dense_but_prefers_valid_supports() -> None:
    command = _reward_test_command()
    command.phase[:] = int(LadderPhase.FIRST_FOOT)
    env = _reward_test_env(command)
    params = {
        "rung_spacing": 0.28,
        "reach_distance": 0.35,
        "first_hand_body_weight": 0.0,
        "second_hand_body_weight": 0.0,
        "foot_body_weight": 0.0,
        "release_progress_weight": 1.0,
        "unsupported_progress_scale": 0.25,
        "max_abs_rate": 10.0,
    }
    reward_term = LadderPhaseProgressReward(SimpleNamespace(params=params), env)

    def reward() -> float:
        return reward_term(env, "ladder", **params).item()

    assert reward() == 0.0

    command.active_foot_pos_w[0, 0] = 0.11
    command.body_height[:] = 1.01
    assert reward() == pytest.approx(1.0 / 0.35 / 0.02 * 0.01)

    command.phase_support_constraints_satisfied[:] = False
    command.active_foot_pos_w[0, 0] = 0.10
    assert reward() == pytest.approx(0.25 / 0.35 / 0.02 * 0.01)

    command.active_foot_pos_w[0, 0] = 0.12
    command.body_height[:] = 0.97
    assert reward() < 0.0

    command.phase[:] = int(LadderPhase.SECOND_FOOT)
    assert reward() == 0.0


def test_ladder_stabilization_progress_uses_transition_gates_without_orientation() -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)
    params = {
        "rung_spacing": 0.28,
        "reach_distance": 0.35,
        "first_hand_body_weight": 0.0,
        "second_hand_body_weight": 0.0,
        "foot_body_weight": 0.0,
        "release_progress_weight": 1.0,
        "unsupported_progress_scale": 0.25,
        "max_abs_rate": 10.0,
    }
    reward_term = LadderPhaseProgressReward(SimpleNamespace(params=params), env)

    assert reward_term(env, "ladder", **params).item() == 0.0

    # The generic posture gate may be false because torso orientation is poor,
    # but STABILIZE must use the exact orientation-free transition gates.
    command.phase_support_constraints_satisfied[:] = False
    command.stabilization_conditions_satisfied[:] = True
    command._phase_dwell_count[:] = 1
    expected = (1.0 / 50.0) / env.step_dt
    assert reward_term(env, "ladder", **params).item() == pytest.approx(expected)

    # A reset and rebuilding the already rewarded dwell cannot be farmed.
    command._phase_dwell_count[:] = 0
    assert reward_term(env, "ladder", **params).item() == 0.0
    command._phase_dwell_count[:] = 1
    assert reward_term(env, "ladder", **params).item() == 0.0

    # Only exceeding the previous per-phase record pays another increment.
    command._phase_dwell_count[:] = 2
    assert reward_term(env, "ladder", **params).item() == pytest.approx(expected)


def test_ladder_stabilization_orientation_is_soft_penalty_only() -> None:
    command = _dummy_ladder_command()
    env = _reward_test_env(command)
    command._test_torso_orientation_error[:] = 0.5

    penalty = ladder_stabilization_orientation_error_l2(env, "ladder")

    assert penalty.item() == pytest.approx(0.25)
    assert command.stabilization_conditions_satisfied.item() is False

    command._test_foot_support[:] = True
    assert command.stabilization_conditions_satisfied.item()
    assert ladder_stabilization_orientation_error_l2(
        env,
        "ladder",
    ).item() == pytest.approx(
        0.25,
    )

    command.phase[:] = int(LadderPhase.FIRST_HAND)
    assert ladder_stabilization_orientation_error_l2(env, "ladder").item() == 0.0


def test_ladder_phase_progress_rewards_release_and_penalizes_regrip() -> None:
    command = _reward_test_command()
    command.phase[:] = int(LadderPhase.FIRST_HAND)
    command.is_hand_phase[:] = True
    command.is_foot_phase[:] = False
    env = _reward_test_env(command)
    params = {
        "rung_spacing": 0.28,
        "reach_distance": 0.35,
        "first_hand_body_weight": 0.0,
        "second_hand_body_weight": 0.0,
        "foot_body_weight": 0.0,
        "release_progress_weight": 1.0,
        "unsupported_progress_scale": 0.25,
        "max_abs_rate": 10.0,
    }
    reward_term = LadderPhaseProgressReward(SimpleNamespace(params=params), env)

    assert reward_term(env, "ladder", **params).item() == 0.0

    command.active_grip_strength[:] = 0.9
    release_reward = reward_term(env, "ladder", **params).item()
    assert release_reward == pytest.approx(5.0)

    command.active_grip_strength[:] = 1.0
    regrip_penalty = reward_term(env, "ladder", **params).item()
    assert regrip_penalty == pytest.approx(-5.0)
    assert release_reward + regrip_penalty == pytest.approx(0.0)


def test_ladder_successful_hand_attach_does_not_cancel_release_reward() -> None:
    command = _reward_test_command()
    command.phase[:] = int(LadderPhase.FIRST_HAND)
    command.is_hand_phase[:] = True
    command.is_foot_phase[:] = False
    env = _reward_test_env(command)
    params = {
        "rung_spacing": 0.28,
        "reach_distance": 0.35,
        "first_hand_body_weight": 0.0,
        "second_hand_body_weight": 0.0,
        "foot_body_weight": 0.0,
        "release_progress_weight": 1.0,
        "unsupported_progress_scale": 0.25,
        "max_abs_rate": 10.0,
    }
    reward_term = LadderPhaseProgressReward(SimpleNamespace(params=params), env)

    assert reward_term(env, "ladder", **params).item() == 0.0
    command.active_grip_strength[:] = 0.9
    assert reward_term(env, "ladder", **params).item() == pytest.approx(5.0)

    command.active_grip_strength[:] = 1.0
    command.just_advanced[:] = True
    assert reward_term(env, "ladder", **params).item() == 0.0


def test_ladder_detached_hand_progress_requires_valid_support_and_posture() -> None:
    command = _reward_test_command()
    command.phase[:] = int(LadderPhase.FIRST_HAND)
    command.is_hand_phase[:] = True
    command.is_foot_phase[:] = False
    command.attached[0, 0] = False
    command.phase_support_constraints_satisfied[:] = False
    env = _reward_test_env(command)
    params = {
        "rung_spacing": 0.28,
        "reach_distance": 0.35,
        "first_hand_body_weight": 0.0,
        "second_hand_body_weight": 0.0,
        "foot_body_weight": 0.0,
        "release_progress_weight": 1.0,
        "unsupported_progress_scale": 0.25,
        "max_abs_rate": 10.0,
    }
    reward_term = LadderPhaseProgressReward(SimpleNamespace(params=params), env)

    assert reward_term(env, "ladder", **params).item() == 0.0
    command.active_hand_pos_w[0, 0] = 0.19
    assert reward_term(env, "ladder", **params).item() == 0.0

    command.phase_support_constraints_satisfied[:] = True
    command.active_hand_pos_w[0, 0] = 0.18
    assert reward_term(env, "ladder", **params).item() == pytest.approx(
        1.0 / 0.35 / 0.02 * 0.01
    )


def test_ladder_required_supports_follow_the_active_phase() -> None:
    command = _dummy_ladder_command()
    command._test_foot_support[:] = True

    assert command.phase_required_supports_satisfied.tolist() == [True]

    command.phase[:] = int(LadderPhase.FIRST_HAND)
    command.attached[0, 0] = False
    assert command.phase_required_supports_satisfied.tolist() == [True]

    command._test_foot_support[0, 0] = False
    assert command.phase_required_supports_satisfied.tolist() == [False]

    command.phase[:] = int(LadderPhase.FIRST_FOOT)
    command.attached[:] = True
    assert command.phase_required_supports_satisfied.tolist() == [True]

    command._test_foot_support[0, 1] = False
    assert command.phase_required_supports_satisfied.tolist() == [False]


def test_ladder_upward_progress_only_pays_for_novel_episode_height() -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)
    reward_term = LadderUpwardProgressReward(
        SimpleNamespace(params={"command_name": "ladder"}),
        env,
    )

    def reward() -> float:
        return reward_term(env, "ladder").item()

    assert reward() == 0.0

    command.body_height[:] = 1.01
    assert reward() == pytest.approx(0.5, abs=2.0e-6)

    command.body_height[:] = 0.98
    assert reward() == 0.0

    command.body_height[:] = 1.005
    assert reward() == 0.0

    command.body_height[:] = 1.02
    assert reward() == pytest.approx(0.5, abs=2.0e-6)

    command.phase_required_supports_satisfied[:] = False
    command.body_height[:] = 1.04
    assert reward() == 0.0

    command.phase_required_supports_satisfied[:] = True
    assert reward() == 0.0

    command.body_height[:] = 1.05
    assert reward() == pytest.approx(0.5, abs=2.0e-6)

    reward_term.reset()
    command.body_height[:] = 1.20
    command.episode_max_body_height[:] = command.body_height
    assert reward() == 0.0

    command.body_height[:] = 1.21
    assert reward() == pytest.approx(0.5, abs=2.0e-6)


def test_ladder_critic_privileged_exposes_compact_reward_and_fsm_state() -> None:
    command = _dummy_ladder_command()
    command.episode_max_body_height[:] = 1.10
    command._cycle_start_body_height[:] = 0.90
    command._phase_start_body_height[:] = 0.95
    command._test_torso_com_vel[:] = torch.tensor([[3.0, 4.0, 0.0]])
    command._test_torso_orientation_error[:] = 0.20
    command._test_torso_support_offset_error[:] = 0.30
    command.robot.data.joint_vel[:] = 1.0
    command._test_foot_support[:] = torch.tensor([[True, False]])
    command._release_ramp_count[:] = 1
    command._phase_dwell_count[:] = 1
    command._stabilization_dwell_target[:] = 2
    env = _reward_test_env(command)

    observation = ladder_critic_privileged(env, "ladder")

    assert observation.shape == (1, 14)
    assert observation[0].tolist() == pytest.approx(
        [
            1.00,
            1.10,
            0.10,
            0.10,
            0.05,
            5.00,
            0.20,
            0.30,
            1.00,
            1.00,
            0.00,
            0.00,
            0.50,
            0.50,
        ]
    )


def test_ladder_foot_placement_follows_phase_targets_without_dense_reward() -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)
    params = {"distance_std": 0.06, "max_abs_rate": 50.0}
    reward_term = LadderFootPlacementReward(SimpleNamespace(params=params), env)

    def reward() -> float:
        return reward_term(env, "ladder", **params).item()

    assert reward() == 0.0
    assert reward() == 0.0

    command.foot_contact[0, 0] = False
    contact_loss = reward()
    assert contact_loss == pytest.approx(-25.0)

    command.foot_contact[0, 0] = True
    contact_restore = reward()
    assert contact_restore == pytest.approx(25.0)
    assert contact_loss + contact_restore == pytest.approx(0.0)

    command.phase[:] = int(LadderPhase.FIRST_FOOT)
    command.active_foot_target_pos_w[0] = torch.tensor([0.20, 0.0, 0.0])
    assert reward() == 0.0

    command.foot_contact[0, 0] = False
    assert reward() == pytest.approx(0.0, abs=1.0e-3)

    command.active_foot_pos_w[0] = command.active_foot_target_pos_w[0]
    command.foot_pos_w[0, 0] = command.active_foot_target_pos_w[0]
    command.foot_contact[0, 0] = True
    assert reward() == pytest.approx(25.0)
    assert reward() == 0.0


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


def test_ladder_stability_reward_requires_quiet_four_point_support() -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)

    def reward() -> float:
        return ladder_torso_stability_exp(
            env,
            "ladder",
            torso_speed_std=0.20,
            joint_speed_std=1.0,
            movement_phase_scale=0.35,
        ).item()

    assert reward() == pytest.approx(1.0)

    command.foot_support[0, 0] = False
    assert reward() == 0.0

    command.foot_support[:] = True
    command.robot.data.joint_vel[:] = 1.0
    assert reward() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())

    command.robot.data.joint_vel.zero_()
    command.phase[:] = int(LadderPhase.FIRST_HAND)
    command.attached[0, 0] = False
    assert reward() == pytest.approx(0.35 * 0.75)


def test_ladder_posture_rewards_global_body_shape() -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)

    def posture_reward() -> float:
        return ladder_torso_posture_exp(
            env,
            "ladder",
            orientation_std=0.30,
            movement_phase_scale=0.35,
        ).item()

    def alignment_reward() -> float:
        return ladder_com_alignment_exp(
            env,
            "ladder",
            support_offset_std=0.18,
            movement_phase_scale=0.35,
        ).item()

    assert posture_reward() == pytest.approx(1.0)
    assert alignment_reward() == pytest.approx(1.0)

    command.torso_orientation_error[:] = 0.30
    command.torso_support_offset_error[:] = 0.18
    expected = torch.exp(torch.tensor(-1.0)).item()
    assert posture_reward() == pytest.approx(expected)
    assert alignment_reward() == pytest.approx(expected)

    command.foot_support[0, 0] = False
    assert posture_reward() == 0.0
    assert alignment_reward() == 0.0


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
    assert ladder_phase_completed(env, "ladder").item() == pytest.approx(50.0)
    assert ladder_stabilized(env, "ladder").item() == pytest.approx(50.0)
    assert ladder_cycle_completed(env, "ladder").item() == pytest.approx(50.0)
    assert ladder_finished(env, "ladder").item() == pytest.approx(50.0)


def test_ladder_reset_pose_starts_initialized_on_second_foot_rung() -> None:
    command = _dummy_ladder_command()
    command.initialized[:] = False
    command.attached[:] = False
    command.grip_strength.zero_()
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
    assert command.held_rung.tolist() == [[4, 4]]
    assert command.is_pre_release.item()
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


@pytest.mark.parametrize("component", ["progress", "foot_placement"])
def test_phase_reward_partition_preserves_unsplit_reward(component: str) -> None:
    command = _reward_test_command()
    env = _reward_test_env(command)
    cfg = load_env_cfg(LADDER_RL_TASK)
    template = cfg.rewards[f"ladder_stabilize_{component}"]
    params = {k: v for k, v in template.params.items() if k != "phase"}
    original = template.func(SimpleNamespace(params=params), env)
    partition = [template.func(SimpleNamespace(params=params), env) for _ in LadderPhase]
    nonzero = False
    for cycle in range(2):
        for phase in LadderPhase:
            command.phase[:] = int(phase)
            command.is_hand_phase[:] = phase in (LadderPhase.FIRST_HAND, LadderPhase.SECOND_HAND)
            command.is_foot_phase[:] = phase in (LadderPhase.FIRST_FOOT, LadderPhase.SECOND_FOOT)
            for step in range(8):
                command._phase_dwell_count[:] = step * 5
                command.active_hand_pos_w[:, 0] += 0.002 * (-1 if step % 2 else 1)
                command.active_foot_pos_w[:, 0] += 0.002 * (-1 if step % 2 else 1)
                command.foot_contact[:, 0] = bool(step % 2)
                command.just_advanced[:] = step == 6
                expected = original(env, **params)
                parts = [term(env, **params, phase=int(p))
                         for p, term in zip(LadderPhase, partition)]
                torch.testing.assert_close(torch.stack(parts).sum(0), expected)
                for p, value in zip(LadderPhase, parts):
                    if p != phase:
                        assert torch.count_nonzero(value) == 0
                nonzero |= bool(torch.any(expected != 0))
        original.reset(torch.tensor([0]))
        for term in partition:
            term.reset(torch.tensor([0]))
    assert nonzero


@pytest.mark.parametrize("phase", list(LadderPhase))
def test_ladder_required_support_cost_and_survival(phase: LadderPhase) -> None:
    from train_mimic.tasks.tracking.mdp.ladder import (
        ladder_missing_foot_support, ladder_movement_survival,
    )
    command = _dummy_ladder_command(phase)
    command._test_foot_support[:] = False
    env = _reward_test_env(command)
    expected = 1.0 if command.is_foot_phase.item() else 2.0
    for _ in range(3):
        assert ladder_missing_foot_support(env, "ladder").item() == expected
    assert ladder_movement_survival(env, "ladder").item() == float(phase != LadderPhase.STABILIZE)
    command._test_foot_support[:] = True
    assert ladder_missing_foot_support(env, "ladder").item() == 0.0
    command._test_foot_support[:] = False
    command.finished[:] = True
    assert ladder_missing_foot_support(env, "ladder").item() == 0.0


def test_ladder_foot_recovery_before_contact_and_across_resets() -> None:
    from train_mimic.tasks.tracking.mdp.ladder import LadderFootRecoveryReward
    command = _reward_test_command()
    command.is_foot_phase[:] = False
    command.foot_rung = torch.ones((1, 2), dtype=torch.long)
    command.foot_contact[:] = False
    command.foot_pos_w[:, 0, 0] = 0.20
    env = _reward_test_env(command)
    params = {"command_name": "ladder", "reach_distance": 0.35}
    term = LadderFootRecoveryReward(SimpleNamespace(params=params), env)
    assert term(env, **params).item() == 0.0
    command.foot_pos_w[:, 0, 0] = 0.10
    approach = term(env, **params).item()
    assert approach == pytest.approx(0.10 / (0.70 * env.step_dt))
    command.foot_pos_w[:, 0, 0] = 0.20
    assert term(env, **params).item() == pytest.approx(-approach)
    command.foot_rung[:, 0] += 1
    command.foot_pos_w[:, 0, 0] = 0.10
    assert term(env, **params).item() == 0.0
    term.reset(torch.tensor([0]))
    command.foot_pos_w[:, 0, 0] = 0.0
    assert term(env, **params).item() == 0.0
    command.phase[:] = int(LadderPhase.FIRST_FOOT)
    command.is_foot_phase[:] = True
    command.active_foot[:] = 0
    assert term(env, **params).item() == 0.0
    command.foot_pos_w[:, 0, 0] = 0.50
    assert term(env, **params).item() == 0.0
    command.foot_pos_w[:, 1, 0] = 0.10
    assert term(env, **params).item() < 0.0


@pytest.mark.parametrize("gate", ["torso", "joint", "angular", "waist", "offset"])
def test_ladder_stabilization_cost_rewards_partial_improvement(gate: str) -> None:
    from train_mimic.tasks.tracking.mdp.ladder import ladder_stabilization_violation
    command = _dummy_ladder_command()
    env = _reward_test_env(command)
    assert ladder_stabilization_violation(env, "ladder").item() == 0.0
    if gate == "torso":
        target, limit = command._test_torso_com_vel[:, 0], 0.20
    elif gate == "joint":
        target, limit = command.robot.data.joint_vel, 1.0
    elif gate == "angular":
        target, limit = command._test_pelvis_ang_vel[:, 0], 0.40
    elif gate == "waist":
        target, limit = command.robot.data.joint_vel[:, int(command._waist_joint_ids[0])], 0.60
    else:
        target, limit = command._test_torso_support_offset_error, 0.18
    target[:] = 2 * limit
    worse = ladder_stabilization_violation(env, "ladder").item()
    target[:] = 1.5 * limit
    better = ladder_stabilization_violation(env, "ladder").item()
    assert worse > better > 0.0
    command.phase[:] = int(LadderPhase.FIRST_HAND)
    assert ladder_stabilization_violation(env, "ladder").item() == 0.0


def test_first_hand_recording_prefix_advances_after_stabilization() -> None:
    command = _dummy_ladder_command()
    command.cfg.freeze_at_max_unlocked_phase = True
    command._unlocked_phase = int(LadderPhase.FIRST_HAND)
    command._test_foot_support[:] = True
    command._advance_stabilization_phase(command.phase.clone())
    assert command.phase.item() == int(LadderPhase.FIRST_HAND)
    assert not command.phase_frozen.item()
    assert not command.curriculum_stage_complete.item()
