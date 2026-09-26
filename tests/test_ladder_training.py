"""Tests for the repeated one-rung G1 ladder skill."""

from __future__ import annotations

import argparse
import re
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
    _add_ladder_to_g1_spec,
    _get_g1_training_spec,
    _remove_embedded_ladder_floor,
    _remove_embedded_ladder_lights,
    make_g1_ladder_rl_env_cfg,
    make_g1_ladder_training_robot_cfg,
)
from train_mimic.tasks.tracking.mdp.ladder import (
    FLIGHT_GRACE_STEPS,
    FLIGHT_RATE,
    HOLD_REWARD,
    LadderClimbCommand,
    LadderGripActionCfg,
    ladder_climb,
    ladder_flight,
    ladder_hold,
    ladder_hold_completed,
    limb_axis_features,
)
from train_mimic.tasks.tracking.rl import LadderOnPolicyRunner
from train_mimic.tasks.tracking.rl.runner import (
    _project_distribution_std,
    _set_distribution_std,
)


class _QuietLadder(LadderClimbCommand):
    @property
    def hand_pos_w(self) -> torch.Tensor:
        return self._hands

    @property
    def foot_pos_w(self) -> torch.Tensor:
        return self._feet

    @property
    def foot_contact(self) -> torch.Tensor:
        return self._contact

    @property
    def torso_com_pos_w(self) -> torch.Tensor:
        return self._torso

    @property
    def torso_rotation_w(self) -> torch.Tensor:
        return torch.eye(3).reshape(1, 3, 3)

    @property
    def torso_com_vel_w(self) -> torch.Tensor:
        return self._torso_vel

    @property
    def torso_ang_vel_w(self) -> torch.Tensor:
        return self._torso_spin

    @property
    def pelvis_ang_vel_w(self) -> torch.Tensor:
        return self._pelvis_spin

    def _attach(self, env_ids, hand_id, rung_indices) -> None:
        self.attached[env_ids, hand_id] = True
        self.grip_strength[env_ids, hand_id] = 1.0
        self.held_rung[env_ids, hand_id] = rung_indices


def _quiet_ladder(*, dwell: int = 3, successes: int = 2) -> _QuietLadder:
    command = object.__new__(_QuietLadder)
    command.cfg = SimpleNamespace(
        stabilization_dwell_steps=dwell,
        successes_per_rollout=successes,
        start_rung=4,
        initial_foot_rung=1,
        max_stabilization_torso_speed=0.20,
        max_stabilization_joint_speed=1.0,
        max_stabilization_body_angular_speed=0.40,
        max_stabilization_waist_joint_speed=0.60,
        max_stabilization_support_offset_error=0.18,
    )
    command.robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.zeros(1, 2)),
    )
    command._waist_joint_ids = torch.tensor([0])
    command._hands = torch.zeros(1, 2, 3)
    command._feet = torch.zeros(1, 2, 3)
    command._contact = torch.ones(1, 2, dtype=torch.bool)
    command._torso = torch.zeros(1, 3)
    command._torso_vel = torch.zeros(1, 3)
    command._torso_spin = torch.zeros(1, 3)
    command._pelvis_spin = torch.zeros(1, 3)
    command.attached = torch.ones(1, 2, dtype=torch.bool)
    command.grip_strength = torch.ones(1, 2)
    command.held_rung = torch.tensor([[4, 4]])
    command.foot_rung = torch.tensor([[1, 1]])
    command.baseline_hand_rung = torch.tensor([[4, 4]])
    command.baseline_foot_rung = torch.tensor([[1, 1]])
    command.initialized = torch.tensor([True])
    command.finished = torch.tensor([False])
    command.just_completed = torch.tensor([False])
    command.hold_count = torch.tensor([0])
    command.successes = torch.tensor([0])
    command._pending_start_pose_init = torch.tensor([False])
    command._reference_torso_rotation_w = torch.eye(3).reshape(1, 3, 3)
    command._reference_torso_support_offset_w = torch.zeros(1, 3)
    command._env = SimpleNamespace(device="cpu")
    return command


def test_ladder_task_is_one_repeated_hold() -> None:
    cfg = load_env_cfg(LADDER_RL_TASK)

    assert LADDER_RL_TASK in SUPPORTED_TASKS
    assert set(cfg.commands) == {"ladder"}
    assert set(cfg.actions) == {"joint_pos", "grip"}
    assert isinstance(cfg.actions["grip"], LadderGripActionCfg)
    assert set(cfg.rewards) == {
        "ladder_climb",
        "ladder_hold",
        "ladder_hold_completed",
        "ladder_flight",
    }
    assert cfg.rewards["ladder_climb"].func is ladder_climb
    assert cfg.rewards["ladder_hold"].func is ladder_hold
    assert cfg.rewards["ladder_hold_completed"].func is ladder_hold_completed
    assert cfg.rewards["ladder_flight"].func is ladder_flight
    assert cfg.rewards["ladder_hold_completed"].weight == 1.0
    assert cfg.terminations["time_out"].time_out is False
    assert "curriculum_stage_complete" not in cfg.terminations
    assert "pre_release_stalled" not in cfg.terminations
    assert cfg.curriculum == {}
    ladder_cmd = cfg.commands["ladder"]
    assert ladder_cmd.successes_per_rollout == 4
    assert ladder_cmd.stabilization_dwell_steps == 50
    assert ladder_cmd.initialize_on_reset is True
    assert ladder_cmd.start_rung == 4
    assert ladder_cmd.initial_foot_rung == 1
    assert not hasattr(ladder_cmd, "curriculum_enabled")
    assert not hasattr(ladder_cmd, "fixed_max_unlocked_phase")
    assert "prepare_ladder_weld_model" not in cfg.events
    assert tuple(sensor.name for sensor in cfg.scene.sensors) == (
        "ladder_foot_contact",
    )
    assert load_runner_cls(LADDER_RL_TASK) is LadderOnPolicyRunner

    play_cfg = make_g1_ladder_rl_env_cfg(play=True)
    assert play_cfg.commands["ladder"].successes_per_rollout == 4
    assert set(play_cfg.rewards) == {
        "ladder_climb",
        "ladder_hold",
        "ladder_hold_completed",
        "ladder_flight",
    }


def test_one_attempt_records_the_keyframe_baseline() -> None:
    command = _quiet_ladder()
    command.initialized[:] = False
    command.attached[:] = False
    command.grip_strength.zero_()
    command.held_rung[:] = -1
    command._pending_start_pose_init[:] = True

    command._initialize_from_start_pose()

    assert command.initialized.item()
    assert command.baseline_hand_rung.tolist() == [[4, 4]]
    assert command.baseline_foot_rung.tolist() == [[1, 1]]
    assert command.held_rung.tolist() == [[4, 4]]
    assert command.foot_rung.tolist() == [[1, 1]]
    assert command.successes.item() == 0
    assert command.hold_count.item() == 0
    assert not command.finished.item()
    command._advance_hold()
    assert command.hold_count.item() == 0
    assert not command.just_completed.item()


def test_hold_counter_resets_when_a_gate_breaks() -> None:
    command = _quiet_ladder(dwell=4)
    command.held_rung[:] = 5
    command.foot_rung[:] = 2

    command._advance_hold()
    command._advance_hold()
    assert command.hold_count.item() == 2
    assert not command.just_completed.item()

    command._torso_vel[:] = 1.0
    command._advance_hold()
    assert command.hold_count.item() == 0
    assert not command.just_completed.item()
    assert command.successes.item() == 0


def test_success_pays_only_after_the_full_hold() -> None:
    command = _quiet_ladder(dwell=3, successes=2)
    command.held_rung[:] = 5
    command.foot_rung[:] = 2
    env = SimpleNamespace(
        step_dt=0.02,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )

    command._advance_hold()
    command._advance_hold()
    assert ladder_hold_completed(env, "ladder").item() == 0.0
    assert command.successes.item() == 0

    command._advance_hold()
    assert command.just_completed.item()
    assert command.successes.item() == 1
    assert command.hold_count.item() == 0
    assert ladder_hold_completed(env, "ladder").item() == pytest.approx(50.0)

    command.finished[:] = True
    command.just_completed[:] = False
    assert ladder_hold_completed(env, "ladder").item() == 0.0


def test_limb_axis_distance_caps_at_the_contact_bubble() -> None:
    base = torch.tensor([[0.0, 0.0, 0.0]])
    nxt = torch.tensor([[0.105, 0.0, 0.280]])
    on_rung = torch.tensor([False])

    at_base, lateral = limb_axis_features(base, base, nxt, on_rung)
    assert at_base.item() == pytest.approx(0.299040, abs=1e-4)
    assert lateral.item() == 0.0

    on_next = torch.tensor([True])
    planted, _ = limb_axis_features(nxt, base, nxt, on_next)
    assert planted.item() == 0.0

    hovering, _ = limb_axis_features(nxt, base, nxt, on_rung)
    assert hovering.item() == pytest.approx(0.10)

    beside = nxt.clone()
    beside[:, 1] = 0.10
    _, side_cost = limb_axis_features(beside, base, nxt, on_rung)
    assert side_cost.item() == pytest.approx(0.05)


def _bind_reward_state(command: _QuietLadder) -> None:
    command._climb_distance = torch.zeros(1, 4)
    command._climb_lateral = torch.zeros(1, 4)
    command._climb_valid = torch.zeros(1, dtype=torch.bool)
    command._hold_count_prev = torch.zeros(1, dtype=torch.long)
    command._feet_off_steps = torch.zeros(1, dtype=torch.long)
    command._reward_cache_step = -1
    command._reward_cache = None
    command._env.step_dt = 0.02
    command._env.common_step_counter = 0
    command.attached[:] = True
    command.foot_rung[:] = 1
    command._contact[:] = True


def _reward_env(command: _QuietLadder) -> SimpleNamespace:
    command._env.step_dt = 0.02
    command._env.common_step_counter = 0
    command._env.command_manager = SimpleNamespace(get_term=lambda _name: command)
    return command._env


def test_climb_pays_the_distance_decrease_and_skips_the_first_sample() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    env = _reward_env(command)
    features = [
        (torch.full((1, 4), 0.299), torch.zeros(1, 4)),
        (torch.full((1, 4), 0.149), torch.zeros(1, 4)),
    ]

    def _features() -> tuple[torch.Tensor, torch.Tensor]:
        return features[min(env.common_step_counter, 1)]

    command._limb_features = _features  # type: ignore[method-assign]

    assert ladder_climb(env, "ladder").item() == 0.0
    env.common_step_counter = 1
    # 4 limbs * 0.150 m / 0.02 s
    assert ladder_climb(env, "ladder").item() == pytest.approx(4 * 0.150 / 0.02)


def test_hold_pays_back_a_broken_gate_and_not_a_completion() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    env = _reward_env(command)
    command.hold_count[:] = 49
    command._hold_count_prev[:] = 48
    command._climb_valid[:] = True

    assert ladder_hold(env, "ladder").item() == pytest.approx((HOLD_REWARD / 50) / 0.02)

    env.common_step_counter = 1
    command.hold_count[:] = 0
    command.just_completed[:] = False
    assert ladder_hold(env, "ladder").item() == pytest.approx((-HOLD_REWARD * 49 / 50) / 0.02)

    env.common_step_counter = 2
    command.just_completed[:] = True
    command._hold_count_prev[:] = 49
    assert ladder_hold(env, "ladder").item() == pytest.approx((HOLD_REWARD / 50) / 0.02)
    assert ladder_hold_completed(env, "ladder").item() == pytest.approx(50.0)


def test_flight_penalty_waits_through_the_grace_and_ignores_a_two_hand_reach() -> None:
    command = _quiet_ladder()
    _bind_reward_state(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    env = _reward_env(command)
    command._contact[:] = False
    command.attached[:] = True

    for step in range(FLIGHT_GRACE_STEPS):
        env.common_step_counter = step
        assert ladder_flight(env, "ladder").item() == 0.0

    env.common_step_counter = FLIGHT_GRACE_STEPS
    assert ladder_flight(env, "ladder").item() == pytest.approx(-FLIGHT_RATE)

    env.common_step_counter = FLIGHT_GRACE_STEPS + 1
    command._contact[:] = True
    command.foot_rung[:] = 1
    command.attached[:] = False
    assert ladder_flight(env, "ladder").item() == 0.0

    env.common_step_counter = FLIGHT_GRACE_STEPS + 2
    command._contact[:, 0] = False
    command.attached[:, 1] = True
    assert ladder_flight(env, "ladder").item() == 0.0

    env.common_step_counter = FLIGHT_GRACE_STEPS + 3
    command.attached[:] = False
    assert ladder_flight(env, "ladder").item() == pytest.approx(-FLIGHT_RATE)


def _zero_ladder_metrics(command: _QuietLadder) -> None:
    command._env.step_dt = 0.02
    command.metrics = {
        name: torch.zeros(1)
        for name in (
            "attached_hands",
            "supported_feet",
            "hold_progress",
            "successes",
            "supports_one_rung_higher",
            "stabilization_hands_attached",
            "stabilization_feet_supported",
            "stabilization_torso_speed",
            "stabilization_joint_speed_rms",
            "stabilization_torso_angular_speed",
            "stabilization_pelvis_angular_speed",
            "stabilization_waist_joint_speed",
            "stabilization_support_offset_error",
            "stabilization_gate_valid",
        )
    }


def test_ladder_metrics_sum_the_episode_instead_of_the_last_step() -> None:
    command = _quiet_ladder()
    _zero_ladder_metrics(command)
    command.attached[:] = True
    command.foot_rung[:] = 1
    command._contact[:] = True

    command._update_metrics()
    command._update_metrics()
    assert command.metrics["attached_hands"].item() == pytest.approx(0.08)
    assert command.metrics["supported_feet"].item() == pytest.approx(0.08)

    command.attached[:] = False
    command.foot_rung[:] = -1
    command._update_metrics()
    assert command.metrics["attached_hands"].item() == pytest.approx(0.08)
    assert command.metrics["supported_feet"].item() == pytest.approx(0.08)

    command.just_completed[:] = True
    command._update_metrics()
    command.just_completed[:] = False
    command._update_metrics()
    assert command.metrics["successes"].item() == pytest.approx(1.0)


def test_metric_sums_include_the_state_after_the_command_update() -> None:
    command = _quiet_ladder()
    _zero_ladder_metrics(command)
    command.time_left = torch.tensor([1.0e9])
    command.attached[:] = False
    command.foot_rung[:] = -1

    def _grant_support() -> None:
        command.attached[:] = True
        command.foot_rung[:] = 1
        command.just_completed[:] = True

    command._update_command = _grant_support  # type: ignore[method-assign]
    command.compute(0.02)

    assert command.metrics["attached_hands"].item() == pytest.approx(0.04)
    assert command.metrics["successes"].item() == pytest.approx(1.0)


def test_completed_hold_continues_from_the_same_pose() -> None:
    command = _quiet_ladder(dwell=1, successes=2)
    command.held_rung[:] = 5
    command.foot_rung[:] = 2
    hands_before = command._hands.clone()
    feet_before = command._feet.clone()

    command._advance_hold()

    assert command.just_completed.item()
    assert command.successes.item() == 1
    assert not command.finished.item()
    assert command.baseline_hand_rung.tolist() == [[5, 5]]
    assert command.baseline_foot_rung.tolist() == [[2, 2]]
    assert command.held_rung.tolist() == [[5, 5]]
    assert command.foot_rung.tolist() == [[2, 2]]
    torch.testing.assert_close(command._hands, hands_before)
    torch.testing.assert_close(command._feet, feet_before)

    command.held_rung[:] = 6
    command.foot_rung[:] = 3
    command._advance_hold()
    assert command.finished.item()
    assert command.successes.item() == 2
    assert command.baseline_hand_rung.tolist() == [[6, 6]]


def test_checkpoint_stores_no_curriculum_state(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def save(self, path: str, infos=None) -> None:
        del self
        captured["path"] = path
        captured["infos"] = infos

    monkeypatch.setattr(MjlabOnPolicyRunner, "save", save)
    runner = object.__new__(LadderOnPolicyRunner)
    runner.save("model.pt", infos={"existing": 1})

    assert captured == {"path": "model.pt", "infos": {"existing": 1}}
    assert "ladder_curriculum_state" not in captured["infos"]


def test_checkpoint_without_curriculum_state_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", lambda *args, **kwargs: {"ok": 1})
    runner = object.__new__(LadderOnPolicyRunner)
    runner._project_actor_std = lambda: None

    assert runner.load("model.pt") == {"ok": 1}


def test_grip_request_attaches_only_near_a_rung_and_releases_immediately() -> None:
    command = object.__new__(LadderClimbCommand)
    command.cfg = SimpleNamespace(attach_distance=0.10, max_attach_speed=0.35)
    command._env = SimpleNamespace(device="cpu")
    command.finished = torch.tensor([False])
    command.attached = torch.zeros(1, 2, dtype=torch.bool)
    command.grip_strength = torch.zeros(1, 2)
    command.grip_request = torch.zeros(1, 2, dtype=torch.bool)
    command.held_rung = torch.full((1, 2), -1)
    command._hand_vel_w = torch.zeros(1, 2, 3)
    command._hand_site_ids = torch.tensor([0, 1])
    command._anchor_mocap_ids = torch.tensor([0, 1])
    command._weld_ids = torch.tensor([0, 1])
    command._rung_site_ids = torch.tensor([2, 3])
    command._rung_half_lengths = torch.tensor([0.2, 0.2])
    identity = torch.eye(3).reshape(9)
    site_xpos = torch.zeros(1, 4, 3)
    site_xpos[0, 0] = torch.tensor([0.0, 0.0, 1.0])
    site_xpos[0, 1] = torch.tensor([5.0, 0.0, 0.0])
    site_xpos[0, 3] = torch.tensor([0.0, 0.0, 1.0])
    data = SimpleNamespace(
        site_xpos=site_xpos,
        site_xmat=identity.repeat(1, 4, 1),
        mocap_pos=torch.zeros(1, 2, 3),
        mocap_quat=torch.zeros(1, 2, 4),
        eq_active=torch.zeros(1, 2),
    )
    command._env = SimpleNamespace(
        device="cpu",
        sim=SimpleNamespace(data=data),
        step_dt=0.02,
    )

    command.apply_grip_requests(torch.tensor([[1.0, 1.0]]) > 0.0)
    assert command.attached.tolist() == [[True, False]]
    assert command.held_rung.tolist() == [[1, -1]]
    assert data.eq_active[0, 0].item() == 1.0

    command._hand_vel_w[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    site_xpos[0, 1] = torch.tensor([0.0, 0.0, 1.0])
    command.apply_grip_requests(torch.tensor([[1.0, 1.0]]) > 0.0)
    assert command.attached.tolist() == [[True, False]]

    command.apply_grip_requests(torch.tensor([[-1.0, 1.0]]) > 0.0)
    assert command.attached.tolist() == [[False, False]]
    assert command.held_rung.tolist() == [[-1, -1]]
    assert data.eq_active[0, 0].item() == 0.0


def test_ladder_command_is_hand_attachment_without_a_stage_index() -> None:
    command = object.__new__(LadderClimbCommand)
    command.attached = torch.tensor([[1.0, 0.0]])

    observation = command.command

    assert observation.shape == (1, 2)
    assert observation[0].tolist() == pytest.approx([1.0, 0.0])


def test_overshoot_is_not_exactly_one_rung() -> None:
    command = _quiet_ladder()
    command.held_rung[:] = 6
    command.foot_rung[:] = 2
    assert not command.supports_one_rung_higher.item()

    command.held_rung[:] = 5
    command.foot_rung[:] = 3
    assert not command.supports_one_rung_higher.item()

    command.foot_rung[:] = 2
    assert command.supports_one_rung_higher.item()


def test_ladder_geometry_observation_is_the_fixed_six_rung_window() -> None:
    command = object.__new__(LadderClimbCommand)
    num_rungs = 9
    command._rung_site_ids = torch.arange(num_rungs)
    command._rung_half_lengths = torch.zeros(num_rungs)
    command._torso_body_id = 0
    command.baseline_hand_rung = torch.tensor([[4, 4]])
    command.baseline_foot_rung = torch.tensor([[1, 1]])
    command.held_rung = torch.tensor([[4, 4]])
    command.grip_strength = torch.tensor([[1.0, 1.0]])
    command.foot_rung = torch.tensor([[1, 1]])
    identity = torch.eye(3).reshape(9)
    centers = torch.zeros(1, num_rungs, 3)
    centers[0, :, 2] = torch.arange(num_rungs, dtype=torch.float)
    data = SimpleNamespace(
        site_xpos=centers,
        site_xmat=identity.repeat(1, num_rungs, 1),
        xpos=torch.zeros(1, 1, 3),
        xmat=identity.reshape(1, 1, 9),
    )
    command._env = SimpleNamespace(device="cpu", sim=SimpleNamespace(data=data))

    observation = command.rung_tokens_torso

    assert command.window_rung_indices().tolist() == [[0, 1, 2, 3, 4, 5]]
    assert observation.shape == (1, 6, 15)
    assert observation[0, :, 2].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert observation[0, 5, 7:9].tolist() == pytest.approx([1.0, 1.0])
    assert observation[0, 2, 9:11].tolist() == pytest.approx([1.0, 1.0])
    assert observation[0, 4, 11:13].tolist() == pytest.approx([1.0, 1.0])
    assert observation[0, 1, 13:].tolist() == pytest.approx([1.0, 1.0])

    command.foot_rung[:] = 2
    command.held_rung[:] = 5
    stayed = command.rung_tokens_torso
    assert command.window_rung_indices().tolist() == [[0, 1, 2, 3, 4, 5]]
    assert stayed[0, :, 2].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    command.baseline_foot_rung[:] = 2
    command.baseline_hand_rung[:] = 5
    shifted = command.rung_tokens_torso
    assert command.window_rung_indices().tolist() == [[1, 2, 3, 4, 5, 6]]
    assert shifted[0, :, 2].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_ladder_task_uses_temporal_geometry_ppo_config() -> None:
    rl_cfg = load_rl_cfg(LADDER_RL_TASK)

    assert rl_cfg.experiment_name == LADDER_RL_EXPERIMENT_NAME
    assert rl_cfg.actor.class_name.endswith(":LadderTemporalCNNModel")
    assert rl_cfg.critic.class_name.endswith(":LadderTemporalCNNModel")
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

def test_ladder_box_contacts_keep_zero_margin() -> None:
    robot_cfg = make_g1_ladder_training_robot_cfg()
    for collision in robot_cfg.collisions:
        names = " ".join(collision.geom_names_expr)
        if "ladder_rung_" in names or "ladder_body_blocker" in names:
            assert collision.margin == 0.0
        if "ladder_rail_" in names:
            assert collision.margin == 0.002

    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="left_wrist_yaw_link")
    spec.worldbody.add_body(name="right_wrist_yaw_link")
    _add_ladder_to_g1_spec(spec)
    for collision in robot_cfg.collisions:
        names = " ".join(collision.geom_names_expr)
        if "ladder_" not in names:
            continue
        collision.edit_spec(spec)
    model = spec.compile()
    box_ids = [
        index
        for index in range(model.ngeom)
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX
    ]
    capsule_ids = [
        index
        for index in range(model.ngeom)
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE
    ]
    assert box_ids and capsule_ids
    assert model.geom_margin[box_ids].tolist() == [0.0] * len(box_ids)
    assert model.geom_margin[capsule_ids].tolist() == pytest.approx(
        [0.002] * len(capsule_ids)
    )

def test_composed_ladder_scene_has_one_ground_plane() -> None:
    robot = _get_g1_training_spec()
    _remove_embedded_ladder_floor(robot)
    _remove_embedded_ladder_lights(robot)
    _add_ladder_to_g1_spec(robot)
    scene = mujoco.MjSpec()
    scene.worldbody.add_body(name="terrain").add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0, 0, 0.01),
    )
    scene.attach(robot, prefix="robot/", frame=scene.worldbody.add_frame())
    model = scene.compile()
    plane_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(model.ngeom)
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    assert plane_names == ["terrain"]
    assert model.nlight == 0

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
    assert model.geom_conaffinity[rung_id] == 0
    assert model.geom_condim[rung_id] == 4
    assert model.geom_contype[rail_id] == 1
    assert model.geom_conaffinity[rail_id] == 0
    assert model.geom_condim[rail_id] == 4
    right_rung_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_ladder_rung_01"
    )
    assert right_rung_id >= 0
    assert model.geom_contype[right_rung_id] == model.geom_conaffinity[right_rung_id] == 0
    assert model.geom_solref[rung_id].tolist() == pytest.approx([0.005, 1.0])
    assert model.geom_solref[rail_id].tolist() == pytest.approx([0.005, 1.0])
    # Every refined surface remains enabled after the stock G1 editor.
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    refined = [i for i, name in enumerate(names) if "_omni_" in name]
    assert refined
    assert all(model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH for i in refined)
    assert all(model.geom_contype[i] == 2 and model.geom_conaffinity[i] == 1 for i in refined)
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
