"""Tests for the repeated one-rung G1 ladder skill."""

from __future__ import annotations

import argparse
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
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
from train_mimic.tasks.tracking.mdp import action_rate_l2
from train_mimic.tasks.tracking.mdp.ladder import (
    CLIMB_CONTACT_CAP,
    CLIMB_CONTACT_RATE,
    CLIMB_DISTANCE_PER_M,
    CLIMB_PHASE_FIRST,
    CLIMB_PHASE_LEGS,
    CLIMB_PHASE_SECOND,
    DETACH_COMPLETE,
    DETACH_POTENTIAL,
    ALIVE_RATE,
    ALIVE_WINDOW_S,
    LEG_CROSS_RATE,
    LEG_CROSS_TOTAL,
    LEG_LINK_NAMES,
    LATCH_PRESSURE_ONSET,
    LATCH_PRESSURE_RATE,
    SIDE_CROSS_RATE,
    SIDE_CROSS_TOTAL,
    FOOT_HAND_RUNG_GAP,
    FOOT_LOAD_CAP,
    FOOT_PRESSURE_RATE,
    FOOT_STEP_RATE,
    FOOT_WEIGHT_FRACTION,
    FOOT_SOLE_END_OFFSETS,
    FOOT_TREAD_ABOVE_CENTER_M,
    HOLD_REWARD,
    REFUND_FRACTION,
    RUNG_WINDOW_COUNT,
    EpisodeSumRewardManager,
    foot_bears_load,
    foot_on_tread,
    foot_step_load,
    foot_tread_features,
    install_episode_sum_reward_logs,
    LadderClimbCommand,
    LadderGripActionCfg,
    FLIGHT_RATE,
    FLIGHT_TOTAL,
    ladder_alive,
    ladder_climb,
    ladder_climb_left_arm,
    ladder_climb_left_leg,
    ladder_climb_right_arm,
    ladder_climb_right_leg,
    ladder_detach_left,
    ladder_detach_right,
    ladder_flight,
    latch_pressure,
    ladder_face_axes,
    grip_latch_caption,
    grip_marker_rgba,
    leg_cross,
    projected_legs_cross,
    projected_legs_cross_rails,
    side_cross,
    ladder_hold,
    ladder_hold_completed,
    ladder_ungrip_left,
    ladder_ungrip_right,
    UNGRIP_RATE,
    ladder_critic_privileged,
    limb_axis_features,
)
from train_mimic.tasks.tracking.rl import LadderOnPolicyRunner
from train_mimic.tasks.tracking.rl.runner import (
    _project_distribution_std,
    _recover_nonfinite_ladder_steps,
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
        self.grip_progress[env_ids, hand_id] = 1.0
        self.held_rung[env_ids, hand_id] = rung_indices

    def _hands_in_next_rung_zone(self) -> torch.Tensor:
        return torch.zeros_like(self.attached)

    def foot_bears_weight(self) -> torch.Tensor:
        return self._bearing

    def foot_load_fraction(self) -> torch.Tensor:
        return self._foot_load


def _quiet_ladder(*, dwell: int = 3, successes: int = 2) -> _QuietLadder:
    command = object.__new__(_QuietLadder)
    command.cfg = SimpleNamespace(
        stabilization_dwell_steps=dwell,
        successes_per_rollout=successes,
        start_rung=4,
        initial_foot_rung=1,
        max_stabilization_torso_speed=0.20,
        max_stabilization_body_angular_speed=0.40,
        weld_release_joint_speed=4.0,
        weld_release_speed_factor=4.0,
    )
    command.robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.zeros(1, 2)),
    )
    command._waist_joint_ids = torch.tensor([0])
    command._reference_torso_support_offset_w = torch.zeros(1, 3)
    command._hands = torch.zeros(1, 2, 3)
    command._feet = torch.zeros(1, 2, 3)
    command._contact = torch.ones(1, 2, dtype=torch.bool)
    command._foot_load = torch.zeros(1, 2)
    command._torso = torch.zeros(1, 3)
    command._torso_vel = torch.zeros(1, 3)
    command._torso_spin = torch.zeros(1, 3)
    command._pelvis_spin = torch.zeros(1, 3)
    command.attached = torch.ones(1, 2, dtype=torch.bool)
    command.grip_strength = torch.ones(1, 2)
    command.grip_progress = torch.ones(1, 2)
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
    command.climb_phase = torch.zeros(1, dtype=torch.long)
    command._reward_phase = torch.zeros(1, dtype=torch.long)
    command._lead_hand = 0
    command._trail_hand = 1
    command._env = SimpleNamespace(device="cpu")
    command._init_quiet_window()
    return command


def test_grip_latch_caption_names_the_hand_and_its_progress() -> None:
    assert grip_latch_caption("Left", False, 0.0) == "Left 0% open"
    assert grip_latch_caption("Left", False, 0.42) == "Left 42% open"
    assert grip_latch_caption("Right", True, 1.0) == "Right 100% latched"


def test_viser_hand_latch_panel_follows_each_hand() -> None:
    class _Folder:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> bool:
            return False

    class _Gui:
        def add_folder(self, name: str) -> _Folder:
            assert name == "Hand latch"
            return _Folder()

        def add_markdown(self, text: str) -> SimpleNamespace:
            return SimpleNamespace(content=text)

        def add_progress_bar(self, value: float) -> SimpleNamespace:
            return SimpleNamespace(value=value, color=None)

    command = _quiet_ladder()
    command.cfg.hand_site_names = ("left_grip_site", "right_grip_site")
    command.create_gui("ladder", SimpleNamespace(gui=_Gui()), lambda: 0)
    command.grip_progress[:] = torch.tensor([[0.4, 1.0]])
    command.attached[:] = torch.tensor([[False, True]])
    command._update_grip_progress_gui()

    assert command._grip_gui_bars[0].value == pytest.approx(40.0)
    assert command._grip_gui_bars[0].color == (40, 90, 220)
    assert command._grip_gui_captions[0].content == "Left 40% open"
    assert command._grip_gui_bars[1].value == pytest.approx(100.0)
    assert command._grip_gui_bars[1].color == (220, 50, 50)
    assert command._grip_gui_captions[1].content == "Right 100% latched"


def test_grip_marker_is_red_when_latched_and_blue_when_open() -> None:
    assert grip_marker_rgba(False, 0.0) == pytest.approx((0.0, 0.0, 1.0, 0.10))
    assert grip_marker_rgba(False, 1.0) == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert grip_marker_rgba(True, 1.0) == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert grip_marker_rgba(True, 0.5) == pytest.approx((1.0, 0.0, 0.0, 0.55))


def test_ladder_task_is_one_repeated_hold() -> None:
    cfg = load_env_cfg(LADDER_RL_TASK)

    assert LADDER_RL_TASK in SUPPORTED_TASKS
    assert set(cfg.commands) == {"ladder"}
    assert cfg.commands["ladder"].debug_vis is True
    assert set(cfg.actions) == {"joint_pos", "grip"}
    assert isinstance(cfg.actions["grip"], LadderGripActionCfg)
    assert set(cfg.rewards) == {
        "ladder_climb_left_arm",
        "ladder_climb_right_arm",
        "ladder_climb_left_leg",
        "ladder_climb_right_leg",
        "detach_left",
        "detach_right",
        "ungrip_left",
        "ungrip_right",
        "ladder_hold",
        "ladder_hold_completed",
        "ladder_alive",
        "leg_cross",
        "side_cross",
        "latch_pressure",
        "ladder_flight",
        "action_rate_l2",
    }
    assert cfg.rewards["ladder_climb_left_arm"].func is ladder_climb_left_arm
    assert cfg.rewards["ladder_climb_right_arm"].func is ladder_climb_right_arm
    assert cfg.rewards["ladder_climb_left_leg"].func is ladder_climb_left_leg
    assert cfg.rewards["ladder_climb_right_leg"].func is ladder_climb_right_leg
    assert cfg.rewards["ladder_climb_left_arm"].weight == 1.0
    assert cfg.rewards["ladder_climb_right_arm"].weight == 1.0
    assert cfg.rewards["ladder_climb_left_leg"].weight == 1.0
    assert cfg.rewards["ladder_climb_right_leg"].weight == 1.0
    assert cfg.rewards["detach_left"].func is ladder_detach_left
    assert cfg.rewards["detach_right"].func is ladder_detach_right
    assert cfg.rewards["detach_left"].weight == 1.0
    assert cfg.rewards["detach_right"].weight == 1.0
    assert DETACH_POTENTIAL == pytest.approx(1.0)
    assert DETACH_COMPLETE == pytest.approx(0.5)
    assert cfg.commands["ladder"].lead_hand == "left"
    assert cfg.rewards["ungrip_left"].func is ladder_ungrip_left
    assert cfg.rewards["ungrip_right"].func is ladder_ungrip_right
    assert cfg.rewards["ungrip_left"].weight == 1.0
    assert cfg.rewards["ungrip_right"].weight == 1.0
    assert UNGRIP_RATE == pytest.approx(0.3)
    assert cfg.rewards["ladder_hold"].func is ladder_hold
    assert cfg.rewards["ladder_hold_completed"].func is ladder_hold_completed
    assert cfg.rewards["ladder_alive"].func is ladder_alive
    assert cfg.rewards["ladder_alive"].weight == 1.0
    assert cfg.rewards["leg_cross"].func is leg_cross
    assert cfg.rewards["leg_cross"].weight == 1.0
    assert LEG_CROSS_RATE == -0.5
    assert LEG_CROSS_TOTAL == pytest.approx(-1.2)
    assert cfg.rewards["leg_cross"].params == {"command_name": "ladder"}
    assert cfg.rewards["side_cross"].func is side_cross
    assert cfg.rewards["side_cross"].weight == 1.0
    assert SIDE_CROSS_RATE == pytest.approx(4.0 * LEG_CROSS_RATE)
    assert SIDE_CROSS_TOTAL == pytest.approx(4.0 * LEG_CROSS_TOTAL)
    assert cfg.rewards["side_cross"].params == {"command_name": "ladder"}
    assert cfg.rewards["latch_pressure"].func is latch_pressure
    assert cfg.rewards["latch_pressure"].weight == 1.0
    assert LATCH_PRESSURE_ONSET == pytest.approx(0.5)
    assert LATCH_PRESSURE_RATE == pytest.approx(-32.0)
    assert cfg.rewards["latch_pressure"].params == {"command_name": "ladder"}
    assert cfg.rewards["ladder_flight"].func is ladder_flight
    assert cfg.rewards["ladder_flight"].weight == 1.0
    assert FLIGHT_RATE == -1.0
    assert FLIGHT_TOTAL == pytest.approx(-1.0)
    assert cfg.rewards["ladder_flight"].params == {"command_name": "ladder"}
    assert cfg.rewards["action_rate_l2"].func is action_rate_l2
    assert cfg.rewards["action_rate_l2"].weight == -0.002
    assert cfg.rewards["ladder_hold_completed"].weight == 1.0
    assert cfg.sim.mujoco.gravity == pytest.approx((0.0, 0.0, -9.81 / 3.0))
    assert cfg.commands["ladder"].attach_distance == pytest.approx(0.14)
    assert cfg.terminations["time_out"].time_out is False
    assert set(cfg.terminations) == {"time_out", "success", "touched_ground"}
    assert cfg.terminations["touched_ground"].func.__name__ == "ladder_touched_ground"
    assert "curriculum_stage_complete" not in cfg.terminations
    assert "pre_release_stalled" not in cfg.terminations
    assert cfg.curriculum == {}
    ladder_cmd = cfg.commands["ladder"]
    assert ladder_cmd.successes_per_rollout == 4
    assert ladder_cmd.stabilization_dwell_steps == 50
    assert ladder_cmd.max_stabilization_torso_speed == pytest.approx(0.20)
    assert ladder_cmd.max_stabilization_body_angular_speed == pytest.approx(0.40)
    assert ladder_cmd.weld_release_joint_speed == pytest.approx(4.0)
    assert ladder_cmd.weld_release_speed_factor == 4.0
    assert not hasattr(ladder_cmd, "max_stabilization_joint_speed")
    assert not hasattr(ladder_cmd, "max_stabilization_waist_joint_speed")
    assert not hasattr(ladder_cmd, "max_stabilization_support_offset_error")
    assert ladder_cmd.initialize_on_reset is True
    assert ladder_cmd.start_rung == 7
    assert ladder_cmd.initial_foot_rung == 3
    assert ladder_cmd.start_rung == ladder_cmd.initial_foot_rung + FOOT_HAND_RUNG_GAP
    assert not hasattr(ladder_cmd, "curriculum_enabled")
    assert not hasattr(ladder_cmd, "fixed_max_unlocked_phase")
    assert "prepare_ladder_weld_model" not in cfg.events
    assert tuple(sensor.name for sensor in cfg.scene.sensors) == (
        "ladder_foot_contact",
        "ladder_ground_contact",
    )
    assert cfg.scene.sensors[0].global_frame is True
    assert cfg.sim.contact_sensor_maxmatch == 128
    assert load_runner_cls(LADDER_RL_TASK) is LadderOnPolicyRunner

    play_cfg = make_g1_ladder_rl_env_cfg(play=True)
    assert play_cfg.commands["ladder"].successes_per_rollout == 4
    assert set(play_cfg.rewards) == set(cfg.rewards)


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


def test_hold_counter_resets_when_a_support_leaves() -> None:
    command = _quiet_ladder(dwell=4)
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
    command.held_rung[:] = 5
    command.foot_rung[:] = 2

    command._advance_hold()
    command._advance_hold()
    assert command.hold_count.item() == 2
    assert not command.just_completed.item()

    command._torso_vel[:] = 1.0
    command._advance_hold()
    assert command.hold_count.item() == 3
    assert not command.just_completed.item()

    command.foot_rung[:, 0] = 1
    command._advance_hold()
    assert command.hold_count.item() == 0
    assert not command.just_completed.item()
    assert command.successes.item() == 0


def test_a_fast_planted_window_waits_for_a_quiet_second() -> None:
    command = _quiet_ladder(dwell=3)
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
    command.held_rung[:] = 5
    command.foot_rung[:] = 2
    command._torso_vel[:] = 1.0

    for _ in range(3):
        command._advance_hold()
    assert command.hold_count.item() == 3
    assert not command.just_completed.item()
    assert command.successes.item() == 0

    command._torso_vel.zero_()
    command._advance_hold()
    command._advance_hold()
    assert command.hold_count.item() == 3
    assert not command.just_completed.item()

    command._advance_hold()
    assert command.just_completed.item()
    assert command.hold_count.item() == 0
    assert command.successes.item() == 1
    assert command.baseline_hand_rung.tolist() == [[5, 5]]


def test_success_pays_only_after_the_full_hold() -> None:
    command = _quiet_ladder(dwell=3, successes=2)
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
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


def test_a_foot_counts_only_on_the_top_of_the_rung() -> None:
    center = torch.tensor([0.0, 0.0, 1.0])
    half = 0.35
    on_top = torch.tensor([0.0, 0.12, 1.0 + FOOT_TREAD_ABOVE_CENTER_M])
    assert foot_on_tread(on_top, center, half).item()

    under = torch.tensor([0.0, 0.12, 1.0 - 0.04])
    assert not foot_on_tread(under, center, half).item()

    front_lip = torch.tensor([0.055, 0.12, 1.0])
    assert not foot_on_tread(front_lip, center, half).item()

    on_the_rail = torch.tensor([0.0, 0.50, 1.0 + FOOT_TREAD_ABOVE_CENTER_M])
    assert not foot_on_tread(on_the_rail, center, half).item()


def test_foot_approach_uses_the_sole_point_furthest_down_the_rail() -> None:
    base = torch.zeros(1, 1, 3)
    nxt = torch.tensor([[[0.0, 0.0, 0.20]]])
    standing_z = 0.20 + FOOT_TREAD_ABOVE_CENTER_M
    ankle = torch.tensor([0.0, 0.0, standing_z])
    heel = torch.tensor([0.0, 0.0, standing_z - 0.10])
    points = torch.stack((ankle, heel)).view(1, 1, 2, 3)
    on_rung = torch.tensor([[False]])

    distance, lateral = foot_tread_features(points, base, nxt, on_rung)
    assert distance.item() == pytest.approx(0.10)
    assert lateral.item() == 0.0

    # The ankle is already at the standing height. It does not pay the heel's gap.
    alone, _ = foot_tread_features(ankle.view(1, 1, 1, 3), base, nxt, on_rung)
    assert alone.item() == pytest.approx(0.0)

    planted, _ = foot_tread_features(points, base, nxt, torch.tensor([[True]]))
    assert planted.item() == 0.0

    # Past the standing height, and off the end of the rung, adds nothing.
    ahead = torch.tensor([0.0, 0.50, standing_z + 0.05])
    clear, _ = foot_tread_features(
        torch.stack((ankle, ahead)).view(1, 1, 2, 3),
        base,
        nxt,
        on_rung,
    )
    assert clear.item() == pytest.approx(0.0)

    # The target is above the rung center, so the center itself is still short.
    center_point = torch.tensor([0.0, 0.0, 0.20]).view(1, 1, 1, 3)
    short, _ = foot_tread_features(center_point, base, nxt, on_rung)
    assert short.item() == pytest.approx(FOOT_TREAD_ABOVE_CENTER_M)


def test_limb_axis_distance_falls_until_the_limb_is_on_the_rung() -> None:
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
    assert hovering.item() == pytest.approx(0.0)

    axis = nxt - base
    short = nxt - 0.03 * axis / torch.linalg.vector_norm(axis)
    almost, _ = limb_axis_features(short, base, nxt, on_rung)
    assert almost.item() == pytest.approx(0.03)

    beside = nxt.clone()
    beside[:, 1] = 0.10
    _, side_cost = limb_axis_features(beside, base, nxt, on_rung)
    assert side_cost.item() == pytest.approx(0.05)


def _bind_reward_state(command: _QuietLadder) -> None:
    command._climb_distance = torch.zeros(1, 4)
    command._climb_lateral = torch.zeros(1, 4)
    command._climb_paid = torch.zeros(1, 4)
    command._approach_paid = torch.zeros(1, 4)
    command._approach_peak = torch.zeros(1, 4)
    command._climb_valid = torch.zeros(1, dtype=torch.bool)
    command._hold_count_prev = torch.zeros(1, dtype=torch.long)
    command._bearing = torch.ones(1, 2, dtype=torch.bool)
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
    command._env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    command._env.command_manager = SimpleNamespace(get_term=lambda _name: command)
    return command._env


def _set_climb_phase(command: _QuietLadder, phase: int) -> None:
    command.climb_phase[:] = phase
    command._reward_phase[:] = phase


def test_climb_pays_approach_until_the_contact_and_then_the_bonus() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    env = _reward_env(command)
    gaps = {
        0: 0.20,
        1: 0.05,
        2: 0.02,
    }

    def _features() -> tuple[torch.Tensor, torch.Tensor]:
        gap = gaps.get(env.common_step_counter, 0.02)
        return torch.full((1, 4), gap), torch.zeros(1, 4)

    command._limb_features = _features  # type: ignore[method-assign]
    command.attached[:] = False

    assert ladder_climb(env, "ladder").item() == 0.0
    env.common_step_counter = 1
    # The left-hand phase pays this limb. The following step pays the last 3 cm too.
    closed = 0.15 * CLIMB_DISTANCE_PER_M
    assert ladder_climb(env, "ladder").item() == pytest.approx(closed / 0.02)
    env.common_step_counter = 2
    last = 0.03 * CLIMB_DISTANCE_PER_M
    assert ladder_climb(env, "ladder").item() == pytest.approx(last / 0.02)


def test_approach_floors_at_zero_and_keeps_half_when_the_episode_ends() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    env = _reward_env(command)
    gaps = {0: 0.20, 1: 0.0, 2: 1.0, 3: 1.0}

    def _features() -> tuple[torch.Tensor, torch.Tensor]:
        gap = gaps.get(env.common_step_counter, 1.0)
        return torch.full((1, 4), gap), torch.zeros(1, 4)

    command._limb_features = _features  # type: ignore[method-assign]
    command.attached[:] = False
    command._bearing[:] = False
    command._contact[:] = False

    assert ladder_climb(env, "ladder").item() == 0.0
    env.common_step_counter = 1
    earned = 0.20 * CLIMB_DISTANCE_PER_M
    assert ladder_climb(env, "ladder").item() == pytest.approx(earned / 0.02)

    # Opening far past the start gives back only what was earned.
    env.common_step_counter = 2
    assert ladder_climb(env, "ladder").item() == pytest.approx(-earned / 0.02)
    assert command._approach_paid[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    env.common_step_counter = 3
    assert ladder_climb(env, "ladder").item() == 0.0

    # A wave that returns to the best point and leaves again adds nothing,
    # and the ending step keeps half of the one net drop.
    gaps[4] = 0.0
    gaps[5] = 1.0
    env.common_step_counter = 4
    assert ladder_climb(env, "ladder").item() == pytest.approx(earned / 0.02)
    env.common_step_counter = 5
    assert ladder_climb(env, "ladder").item() == pytest.approx(-earned / 0.02)
    env.common_step_counter = 6
    env.reset_buf = torch.tensor([True])
    assert ladder_climb(env, "ladder").item() == pytest.approx(
        (1.0 - REFUND_FRACTION) * earned / 0.02
    )


def test_foot_bears_load_follows_the_share_of_live_weight() -> None:
    mass = 30.0
    full_g = 9.81
    light_g = full_g / 3.0
    share = FOOT_WEIGHT_FRACTION
    full_weight = mass * full_g
    light_weight = mass * light_g
    full = foot_bears_load(
        torch.tensor([[full_weight * share, full_weight * share - 1.0e-3]]),
        torch.tensor(full_weight),
    )
    light = foot_bears_load(
        torch.tensor([[light_weight * share, light_weight * (share - 0.01)]]),
        torch.tensor(light_weight),
    )

    assert full[0, 0].item()
    assert light[0, 0].item()
    assert not full[0, 1].item()
    assert not light[0, 1].item()


def test_foot_step_load_follows_the_share_of_live_weight() -> None:
    mass = 30.0
    full_g = 9.81
    light_g = full_g / 3.0
    share = 0.20
    full = foot_step_load(
        torch.tensor([[mass * full_g * share, mass * full_g, -5.0]]),
        torch.tensor(mass * full_g),
    )
    light = foot_step_load(
        torch.tensor([[mass * light_g * share, mass * light_g, 0.0]]),
        torch.tensor(mass * light_g),
    )

    assert full[0, 0].item() == pytest.approx(share)
    assert light[0, 0].item() == pytest.approx(share)
    assert full[0, 1].item() == pytest.approx(FOOT_LOAD_CAP)
    assert light[0, 1].item() == pytest.approx(FOOT_LOAD_CAP)
    assert full[0, 2].item() == 0.0
    assert light[0, 2].item() == 0.0


def test_climb_terms_pay_only_the_limb_that_earned_them() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    env = _reward_env(command)
    gaps = {0: torch.tensor([[0.20, 0.20, 0.20, 0.20]])}

    def _features() -> tuple[torch.Tensor, torch.Tensor]:
        gap = gaps.get(env.common_step_counter, gaps[0])
        return gap, torch.zeros(1, 4)

    command._limb_features = _features  # type: ignore[method-assign]
    command.attached[:] = False
    command._bearing[:] = False
    command._contact[:] = False

    def limbs() -> list[float]:
        return [
            ladder_climb_left_arm(env, "ladder").item(),
            ladder_climb_right_arm(env, "ladder").item(),
            ladder_climb_left_leg(env, "ladder").item(),
            ladder_climb_right_leg(env, "ladder").item(),
        ]

    assert limbs() == [0.0, 0.0, 0.0, 0.0]
    env.common_step_counter = 1
    gaps[1] = torch.tensor([[0.0, 0.20, 0.20, 0.20]])
    earned = 0.20 * CLIMB_DISTANCE_PER_M
    assert limbs() == pytest.approx([earned / 0.02, 0.0, 0.0, 0.0])
    assert sum(limbs()) == pytest.approx(ladder_climb(env, "ladder").item())
    assert command._climb_paid[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])

    # The right-hand phase pays that hand. The legs phase pays the foot.
    # Readers of one step share one sample, so the cap moves once.
    _set_climb_phase(command, CLIMB_PHASE_SECOND)
    env.common_step_counter = 2
    gaps[2] = gaps[1]
    command.held_rung[:, 1] = 5
    command.attached[:, 1] = True
    paid = limbs()
    assert paid[0] == 0.0
    assert paid[1] == pytest.approx(CLIMB_CONTACT_RATE)
    assert paid[2] == 0.0
    assert paid[3] == 0.0
    assert command._climb_paid[0].tolist() == pytest.approx(
        [0.0, CLIMB_CONTACT_RATE * env.step_dt, 0.0, 0.0]
    )
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
    env.common_step_counter = 3
    command.foot_rung[:, 1] = 2
    command._contact[:, 1] = True
    paid = limbs()
    assert paid == pytest.approx([0.0, 0.0, 0.0, FOOT_STEP_RATE])
    assert command._climb_paid[0, 3].item() == pytest.approx(FOOT_STEP_RATE * env.step_dt)


def test_climb_pays_a_contact_once_and_ignores_a_reach() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    env = _reward_env(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    step = 0

    def climb() -> float:
        nonlocal step
        env.common_step_counter = step
        step += 1
        return ladder_climb(env, "ladder").item()

    # Baseline hand rung is 4, so the next rung is 5. The first sample pays
    # nothing. A hand pays only while it is welded to that next rung.
    command.held_rung[:, 0] = 5
    command.held_rung[:, 1] = 4
    command._contact[:, 1] = False
    command._bearing[:] = False
    assert climb() == 0.0
    assert climb() == pytest.approx(CLIMB_CONTACT_RATE)

    # The other hand is welded to the current rung, and one rung too high
    # is not the next rung either.
    command.attached[:, 1] = True
    assert climb() == pytest.approx(CLIMB_CONTACT_RATE)
    command.held_rung[:, 0] = 6
    assert climb() == 0.0
    command.held_rung[:, 0] = 5
    command.attached[:, 1] = False

    # Letting go pauses the same cap. Coming back continues it.
    paid = command._climb_paid[0, 0].item()
    command.attached[:, 0] = False
    assert climb() == 0.0
    command.attached[:, 0] = True
    assert climb() == pytest.approx(CLIMB_CONTACT_RATE)
    assert command._climb_paid[0, 0].item() == pytest.approx(
        paid + CLIMB_CONTACT_RATE * env.step_dt
    )

    # An unwelded hand outside the next-rung zone pays nothing. Inside that
    # zone the same hand pays the contact rate times its grip progress.
    command.attached[:] = False
    command.grip_progress[:] = 0.5
    assert climb() == 0.0
    command._hands_in_next_rung_zone = lambda: torch.tensor([[True, False]])  # type: ignore[method-assign]
    assert climb() == pytest.approx(0.5 * CLIMB_CONTACT_RATE)
    command._hands_in_next_rung_zone = lambda: torch.zeros_like(command.attached)  # type: ignore[method-assign]
    command.grip_progress.zero_()

    # The cap is +3. A step can pay at most one rate slice, so the last
    # half-slice pays and the next step pays nothing.
    command.attached[:, 0] = True
    remainder = CLIMB_CONTACT_RATE * env.step_dt * 0.5
    command._climb_paid[0, 0] = CLIMB_CONTACT_CAP - remainder
    assert climb() == pytest.approx(remainder / env.step_dt, abs=1e-4)
    assert command._climb_paid[0, 0].item() == pytest.approx(CLIMB_CONTACT_CAP)
    assert climb() == pytest.approx(0.0, abs=1e-4)
    command.attached[:, 0] = False
    assert climb() == 0.0
    command.attached[:, 0] = True
    assert climb() == pytest.approx(0.0, abs=1e-4)
    command.attached[:] = False

    # The legs phase pays a foot on the next tread 0.10 per second for the
    # contact, plus 1 per second times its load share. The share stops at 0.35.
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
    command.foot_rung[:, 0] = 2
    assert climb() == pytest.approx(FOOT_STEP_RATE)
    command._foot_load[:, 0] = 0.10
    assert climb() == pytest.approx(FOOT_STEP_RATE + FOOT_PRESSURE_RATE * 0.10)
    command._foot_load[:, 0] = 0.80
    foot_rate = FOOT_STEP_RATE + FOOT_PRESSURE_RATE * FOOT_LOAD_CAP
    assert climb() == pytest.approx(foot_rate)
    command.foot_rung[:, 0] = 1
    assert climb() == 0.0
    paid_foot = command._climb_paid[0, 2].item()
    command.foot_rung[:, 0] = 2
    assert climb() == pytest.approx(foot_rate)
    assert command._climb_paid[0, 2].item() == pytest.approx(
        paid_foot + foot_rate * env.step_dt
    )
    foot_remainder = foot_rate * env.step_dt * 0.5
    command._climb_paid[0, 2] = CLIMB_CONTACT_CAP - foot_remainder
    assert climb() == pytest.approx(foot_remainder / env.step_dt, abs=1e-4)
    assert command._climb_paid[0, 2].item() == pytest.approx(CLIMB_CONTACT_CAP)
    assert climb() == pytest.approx(0.0, abs=1e-4)
    command._foot_load[:, 0] = 0.0
    assert climb() == 0.0

    # This foot is already at the cap, so completion adds nothing and clears it.
    command.just_completed[:] = True
    command._foot_load[:, 0] = FOOT_LOAD_CAP
    assert climb() == 0.0
    assert command._climb_paid[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    command.just_completed[:] = False
    assert climb() == pytest.approx(foot_rate)


def test_completion_fills_the_unpaid_contact_caps() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    env = _reward_env(command)
    command.held_rung[:] = 5
    command.foot_rung[:] = 2
    command._foot_load[:] = 0.0
    command.hold_count[:] = 10
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
    open_rate = 2 * FOOT_STEP_RATE

    assert ladder_climb(env, "ladder").item() == 0.0
    env.common_step_counter = 1
    assert ladder_climb(env, "ladder").item() == pytest.approx(open_rate)

    env.common_step_counter = 2
    command.just_completed[:] = True
    remainder = (CLIMB_CONTACT_CAP - command._climb_paid[0, 2:]).sum().item()
    assert ladder_climb(env, "ladder").item() == pytest.approx(remainder / env.step_dt)
    assert command._climb_paid[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])

    env.common_step_counter = 3
    command.just_completed[:] = False
    assert ladder_climb(env, "ladder").item() == pytest.approx(open_rate)


def test_hold_pays_back_a_broken_gate_and_not_a_completion() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    env = _reward_env(command)
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
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

    # The counter was already full, so completion pays the impulse only.
    env.common_step_counter = 3
    command.just_completed[:] = True
    command.hold_count[:] = 0
    command._hold_count_prev[:] = 50
    assert ladder_hold(env, "ladder").item() == pytest.approx(0.0)
    assert ladder_hold_completed(env, "ladder").item() == pytest.approx(50.0)


def test_alive_pays_for_two_seconds_off_the_ground() -> None:
    command = _quiet_ladder()
    _bind_reward_state(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    env = _reward_env(command)

    def alive(step: int) -> float:
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        return ladder_alive(env, "ladder").item()

    assert ALIVE_RATE == pytest.approx(1.0)
    assert ALIVE_WINDOW_S == pytest.approx(2.0)
    assert alive(1) == pytest.approx(ALIVE_RATE)
    assert alive(100) == pytest.approx(ALIVE_RATE)
    assert alive(101) == pytest.approx(0.0)

    env.termination_manager = SimpleNamespace(
        get_term=lambda name: torch.tensor([name == "touched_ground"])
    )
    assert alive(10) == pytest.approx(0.0)


def _ladder_face() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = torch.zeros(1, 3)
    rung_axis = torch.tensor([[0.0, 1.0, 0.0]])
    landing = torch.tensor([[0.0, 0.0, 1.0]])
    origin, lateral, climb, degenerate = ladder_face_axes(origin, landing, rung_axis)
    assert not bool(degenerate.item())
    return origin, lateral, climb


def test_leg_cross_charges_a_held_scissor_and_ignores_a_close_step() -> None:
    origin, lateral, climb = _ladder_face()
    # Thighs cross at their midpoints on the ladder face. Shins stay apart.
    crossed = torch.tensor(
        [
            [
                [0.0, 0.2, 1.0],
                [0.0, -0.2, 0.0],
                [0.0, -0.2, -1.0],
                [0.0, -0.2, 1.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.2, -1.0],
            ]
        ]
    )
    assert bool(projected_legs_cross(crossed, origin, lateral, climb).item())
    # The same scissor, pushed out from the ladder, still crosses on the face.
    pushed = crossed.clone()
    pushed[..., 0] += 0.4
    assert bool(projected_legs_cross(pushed, origin, lateral, climb).item())

    ordered = torch.tensor(
        [
            [
                [0.0, 0.2, 1.0],
                [0.05, 0.2, 0.0],
                [0.05, 0.2, -1.0],
                [0.0, -0.2, 1.0],
                [0.0, -0.2, 0.0],
                [0.0, -0.2, -1.0],
            ]
        ]
    )
    # Ankles are 5 cm apart in depth and still on their own sides.
    assert not bool(projected_legs_cross(ordered, origin, lateral, climb).item())
    parallel = ordered.clone()
    parallel[..., 0] = 0.0
    assert not bool(projected_legs_cross(parallel, origin, lateral, climb).item())

    # A cross near the end of both bones costs the same as one in the middle.
    near_end = torch.tensor(
        [
            [
                [0.0, 0.2, 1.0],
                [0.0, -0.02, 0.0],
                [0.0, -0.02, -1.0],
                [0.0, -0.2, 0.1],
                [0.0, 0.02, 0.0],
                [0.0, 0.02, -1.0],
            ]
        ]
    )
    assert bool(projected_legs_cross(near_end, origin, lateral, climb).item())

    command = SimpleNamespace(
        foot_span_plane=lambda: (origin, lateral, climb, torch.tensor([False])),
        _leg_cross_paid=torch.zeros(1),
        _leg_cross_step=-1,
        _leg_cross_rate=torch.zeros(1),
    )
    env = SimpleNamespace(
        step_dt=1.0,
        common_step_counter=0,
        scene={
            "robot": SimpleNamespace(
                find_bodies=lambda _names, preserve_order=False: (
                    list(range(6)),
                    list(LEG_LINK_NAMES),
                ),
                data=SimpleNamespace(body_link_pos_w=crossed),
            )
        },
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
    assert bool(projected_legs_cross(crossed, origin, lateral, climb).item())
    assert leg_cross(env, "ladder").item() == pytest.approx(LEG_CROSS_RATE)
    assert leg_cross(env, "ladder").item() == pytest.approx(LEG_CROSS_RATE)
    assert command._leg_cross_paid.item() == pytest.approx(LEG_CROSS_RATE)

    env.common_step_counter = 1
    assert leg_cross(env, "ladder").item() == pytest.approx(LEG_CROSS_RATE)
    assert command._leg_cross_paid.item() == pytest.approx(2.0 * LEG_CROSS_RATE)

    env.common_step_counter = 2
    assert leg_cross(env, "ladder").item() == pytest.approx(-0.2)
    assert command._leg_cross_paid.item() == pytest.approx(LEG_CROSS_TOTAL)

    env.common_step_counter = 3
    assert leg_cross(env, "ladder").item() == pytest.approx(0.0)
    assert command._leg_cross_paid.item() == pytest.approx(LEG_CROSS_TOTAL)

    env.common_step_counter = 4
    command.foot_span_plane = lambda: (origin, lateral, climb, torch.tensor([True]))
    command._leg_cross_paid.zero_()
    assert leg_cross(env, "ladder").item() == pytest.approx(0.0)


def _face_rails() -> tuple[torch.Tensor, torch.Tensor]:
    """Climbing-face stiles at y = ±0.35, from z = 0 to z = 2."""

    start = torch.tensor(
        [[[0.0, -0.35, 0.0], [0.0, 0.35, 0.0]]]
    )
    end = torch.tensor(
        [[[0.0, -0.35, 2.0], [0.0, 0.35, 2.0]]]
    )
    return start, end


def test_side_cross_charges_a_leg_through_a_rail_and_ignores_a_step_between() -> None:
    origin, lateral, climb = _ladder_face()
    rail_start, rail_end = _face_rails()
    inside = torch.tensor(
        [
            [
                [0.0, 0.15, 1.2],
                [0.0, 0.15, 0.6],
                [0.0, 0.15, 0.1],
                [0.0, -0.15, 1.2],
                [0.0, -0.15, 0.6],
                [0.0, -0.15, 0.1],
            ]
        ]
    )
    assert not bool(
        projected_legs_cross_rails(
            inside, rail_start, rail_end, origin, lateral, climb
        ).item()
    )
    through = inside.clone()
    through[0, 2, 1] = 0.50
    assert bool(
        projected_legs_cross_rails(
            through, rail_start, rail_end, origin, lateral, climb
        ).item()
    )
    # The same kick, pushed out from the ladder, still crosses on the face.
    pushed = through.clone()
    pushed[..., 0] += 0.4
    assert bool(
        projected_legs_cross_rails(
            pushed, rail_start, rail_end, origin, lateral, climb
        ).item()
    )
    # A cross near the foot of the rail costs the same as one in the middle.
    near_end = inside.clone()
    near_end[0, 1] = torch.tensor([0.0, 0.20, 0.05])
    near_end[0, 2] = torch.tensor([0.0, 0.50, 0.02])
    assert bool(
        projected_legs_cross_rails(
            near_end, rail_start, rail_end, origin, lateral, climb
        ).item()
    )

    command = SimpleNamespace(
        bottom_rung_plane=lambda: (origin, lateral, climb, torch.tensor([False])),
        rail_endpoints=lambda: (rail_start, rail_end),
        _side_cross_paid=torch.zeros(1),
        _side_cross_step=-1,
        _side_cross_rate=torch.zeros(1),
    )
    env = SimpleNamespace(
        step_dt=1.0,
        common_step_counter=0,
        scene={
            "robot": SimpleNamespace(
                find_bodies=lambda _names, preserve_order=False: (
                    list(range(6)),
                    list(LEG_LINK_NAMES),
                ),
                data=SimpleNamespace(body_link_pos_w=through),
            )
        },
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
    assert side_cross(env, "ladder").item() == pytest.approx(SIDE_CROSS_RATE)
    assert side_cross(env, "ladder").item() == pytest.approx(SIDE_CROSS_RATE)
    assert command._side_cross_paid.item() == pytest.approx(SIDE_CROSS_RATE)

    env.common_step_counter = 1
    assert side_cross(env, "ladder").item() == pytest.approx(SIDE_CROSS_RATE)
    assert command._side_cross_paid.item() == pytest.approx(2.0 * SIDE_CROSS_RATE)

    env.common_step_counter = 2
    assert side_cross(env, "ladder").item() == pytest.approx(-0.8)
    assert command._side_cross_paid.item() == pytest.approx(SIDE_CROSS_TOTAL)

    env.common_step_counter = 3
    assert side_cross(env, "ladder").item() == pytest.approx(0.0)
    assert command._side_cross_paid.item() == pytest.approx(SIDE_CROSS_TOTAL)

    env.common_step_counter = 4
    command.bottom_rung_plane = lambda: (origin, lateral, climb, torch.tensor([True]))
    command._side_cross_paid.zero_()
    assert side_cross(env, "ladder").item() == pytest.approx(0.0)

    env.common_step_counter = 5
    command.bottom_rung_plane = lambda: (origin, lateral, climb, torch.tensor([False]))
    env.scene["robot"].data.body_link_pos_w = inside
    assert side_cross(env, "ladder").item() == pytest.approx(0.0)


def test_climb_phase_pays_each_hand_then_the_legs() -> None:
    command = _quiet_ladder(dwell=50)
    _bind_reward_state(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    env = _reward_env(command)
    command.grip_progress[:] = 1.0
    command.attached[:] = True
    command.held_rung[:, 0] = 5
    command.held_rung[:, 1] = 4
    command._contact[:] = False
    command.foot_rung[:] = 1
    command._climb_valid[:] = True
    _set_climb_phase(command, CLIMB_PHASE_FIRST)

    env.common_step_counter = 1
    assert ladder_climb_left_arm(env, "ladder").item() == pytest.approx(CLIMB_CONTACT_RATE)
    assert ladder_climb_right_arm(env, "ladder").item() == 0.0
    assert ladder_ungrip_left(env, "ladder").item() == 0.0
    assert ladder_detach_left(env, "ladder").item() == 0.0
    command.grip_progress[:] = 0.0
    env.common_step_counter = 6
    assert ladder_detach_left(env, "ladder").item() == 0.0
    assert ladder_detach_right(env, "ladder").item() == 0.0
    assert ladder_ungrip_right(env, "ladder").item() == pytest.approx(-UNGRIP_RATE)
    command.grip_progress[:] = 1.0
    command._advance_climb_phase()
    assert command.climb_phase.item() == CLIMB_PHASE_SECOND

    _set_climb_phase(command, CLIMB_PHASE_SECOND)
    command.grip_progress[:, 0] = 0.0
    command.grip_progress[:, 1] = 1.0
    command.held_rung[:, 1] = 5
    env.common_step_counter = 2
    assert ladder_climb_right_arm(env, "ladder").item() == pytest.approx(CLIMB_CONTACT_RATE)
    assert ladder_climb_left_arm(env, "ladder").item() == 0.0
    assert ladder_detach_right(env, "ladder").item() == 0.0
    assert ladder_ungrip_left(env, "ladder").item() == pytest.approx(-UNGRIP_RATE)
    assert ladder_ungrip_right(env, "ladder").item() == 0.0
    command._advance_climb_phase()
    assert command.climb_phase.item() == CLIMB_PHASE_LEGS

    _set_climb_phase(command, CLIMB_PHASE_LEGS)
    command.grip_progress[:, 1] = 0.0
    command.foot_rung[:, 0] = 2
    command._contact[:, 0] = True
    env.common_step_counter = 3
    assert ladder_climb_left_leg(env, "ladder").item() == pytest.approx(FOOT_STEP_RATE)
    assert ladder_climb_right_arm(env, "ladder").item() == 0.0
    assert ladder_ungrip_right(env, "ladder").item() == pytest.approx(-UNGRIP_RATE)

    # Every support is on the next rung. The hold waits for the legs phase.
    command.held_rung[:] = 5
    command.foot_rung[:] = 2
    command._contact[:] = True
    command.attached[:] = True
    _set_climb_phase(command, CLIMB_PHASE_FIRST)
    command._advance_hold()
    assert command.hold_count.item() == 0
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
    command._advance_hold()
    assert command.hold_count.item() == 1

    command.just_completed[:] = True
    command._advance_climb_phase()
    assert command.climb_phase.item() == CLIMB_PHASE_FIRST


def test_detach_pays_the_climbing_hand_for_letting_go() -> None:
    command = _quiet_ladder()
    command.grip_progress[:] = 1.0
    _set_climb_phase(command, CLIMB_PHASE_FIRST)
    env = _reward_env(command)
    dt = env.step_dt

    assert ladder_detach_left(env, "ladder").item() == 0.0
    assert ladder_detach_right(env, "ladder").item() == 0.0
    command.grip_progress[:, 0] = 0.5
    assert ladder_detach_left(env, "ladder").item() == pytest.approx(
        0.5 * DETACH_POTENTIAL / dt
    )
    command.grip_progress[:, 0] = 1.0
    assert ladder_detach_left(env, "ladder").item() == pytest.approx(
        -0.5 * DETACH_POTENTIAL / dt
    )
    command._detach_commanded[:, 0] = True
    command.grip_progress[:, 0] = 0.0
    assert ladder_detach_left(env, "ladder").item() == pytest.approx(
        (DETACH_POTENTIAL + DETACH_COMPLETE) / dt
    )
    assert ladder_detach_left(env, "ladder").item() == 0.0
    command.grip_progress[:, 0] = 1.0
    assert ladder_detach_left(env, "ladder").item() == pytest.approx(-DETACH_POTENTIAL / dt)
    command.grip_progress[:, 0] = 0.0
    assert ladder_detach_left(env, "ladder").item() == 0.0


def test_detach_ignores_a_forced_release_and_a_next_rung_weld() -> None:
    command = _quiet_ladder()
    command.grip_progress[:] = 1.0
    _set_climb_phase(command, CLIMB_PHASE_FIRST)
    env = _reward_env(command)

    assert ladder_detach_left(env, "ladder").item() == 0.0
    command._detach_forced[:, 0] = True
    command._detach_commanded[:, 0] = True
    command.grip_progress[:, 0] = 0.0
    command.held_rung[:, 0] = -1
    command.attached[:, 0] = False
    assert ladder_detach_left(env, "ladder").item() == 0.0

    command.grip_progress[:, 0] = 1.0
    command.held_rung[:, 0] = command.baseline_hand_rung[:, 0] + 1
    command.attached[:, 0] = True
    assert ladder_detach_left(env, "ladder").item() == 0.0
    command.grip_progress[:, 0] = 0.5
    assert ladder_detach_left(env, "ladder").item() == 0.0
    command.grip_progress[:, 0] = 0.0
    assert ladder_detach_left(env, "ladder").item() == 0.0


def test_lead_hand_can_be_the_right_hand() -> None:
    command = _quiet_ladder()
    _bind_reward_state(command)
    command._limb_features = lambda: (torch.zeros(1, 4), torch.zeros(1, 4))  # type: ignore[method-assign]
    env = _reward_env(command)
    command._lead_hand = 1
    command._trail_hand = 0
    command._climb_valid[:] = True
    command.grip_progress[:, 0] = 0.0
    command.grip_progress[:, 1] = 1.0
    command.attached[:] = True
    command.held_rung[:, 1] = 5
    command.held_rung[:, 0] = 4
    _set_climb_phase(command, CLIMB_PHASE_FIRST)

    env.common_step_counter = 1
    assert ladder_climb_right_arm(env, "ladder").item() == pytest.approx(CLIMB_CONTACT_RATE)
    assert ladder_climb_left_arm(env, "ladder").item() == 0.0
    assert ladder_detach_right(env, "ladder").item() == 0.0
    assert ladder_detach_left(env, "ladder").item() == 0.0
    assert ladder_ungrip_left(env, "ladder").item() == pytest.approx(-UNGRIP_RATE)
    assert ladder_ungrip_right(env, "ladder").item() == 0.0
    command._advance_climb_phase()
    assert command.climb_phase.item() == CLIMB_PHASE_SECOND


def test_hand_phase_metrics_count_one_latch_per_rung() -> None:
    command = _quiet_ladder()
    _zero_ladder_metrics(command)
    command.attached[:] = True
    command.held_rung[:, 0] = 5
    command.held_rung[:, 1] = 4
    _set_climb_phase(command, CLIMB_PHASE_FIRST)

    command._reward_phase.copy_(command.climb_phase)
    command._advance_climb_phase()
    command._update_metrics()
    assert command.climb_phase.item() == CLIMB_PHASE_SECOND
    assert command.metrics["first_hand_reached"].item() == pytest.approx(1.0)
    assert command.metrics["first_hand_phase"].item() == pytest.approx(0.02)

    command.held_rung[:, 1] = 5
    command._reward_phase.copy_(command.climb_phase)
    command._advance_climb_phase()
    command._update_metrics()
    assert command.climb_phase.item() == CLIMB_PHASE_LEGS
    assert command.metrics["first_hand_reached"].item() == pytest.approx(1.0)
    assert command.metrics["second_hand_reached"].item() == pytest.approx(1.0)
    assert command.metrics["second_hand_phase"].item() == pytest.approx(0.02)
    assert command.metrics["legs_phase"].item() == 0.0


def test_over_speed_releases_both_welds_and_a_quiet_hold_keeps_them() -> None:
    command = _quiet_ladder()
    released: list[tuple[list[int], int]] = []

    def _release(env_ids: torch.Tensor, hand_id: int) -> None:
        released.append((env_ids.tolist(), hand_id))
        command.attached[env_ids, hand_id] = False

    command._release = _release  # type: ignore[method-assign]

    command._release_over_speed_welds()
    assert released == []
    assert command.attached.all()

    command.robot.data.joint_vel[:] = 5.0
    command._release_over_speed_welds()
    assert released == [([0], 0), ([0], 1)]
    assert not command.attached.any()
    assert command._latch_pressure.item() == pytest.approx(5.0 / 4.0)


def test_latch_pressure_is_free_below_half_the_trip_and_heavy_at_the_trip() -> None:
    command = _quiet_ladder()
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )

    command._release_over_speed_welds()
    assert command._latch_pressure.item() == pytest.approx(0.0)
    assert latch_pressure(env, "ladder").item() == pytest.approx(0.0)

    command.robot.data.joint_vel[:] = 2.0
    command._release_over_speed_welds()
    assert command.attached.all()
    assert command._latch_pressure.item() == pytest.approx(LATCH_PRESSURE_ONSET)
    assert latch_pressure(env, "ladder").item() == pytest.approx(0.0)

    command.robot.data.joint_vel[:] = 2.4
    command._release_over_speed_welds()
    assert command.attached.all()
    assert latch_pressure(env, "ladder").item() == pytest.approx(
        (0.1**2) * LATCH_PRESSURE_RATE
    )

    command.robot.data.joint_vel[:] = 4.0
    command._release_over_speed_welds()
    assert command.attached.all()
    assert latch_pressure(env, "ladder").item() == pytest.approx(
        (0.5**2) * LATCH_PRESSURE_RATE
    )

    command.robot.data.joint_vel.zero_()
    command._pelvis_spin[:] = torch.tensor([1.6, 0.0, 0.0])
    command._release_over_speed_welds()
    assert command.attached.all()
    assert latch_pressure(env, "ladder").item() == pytest.approx(
        (0.5**2) * LATCH_PRESSURE_RATE
    )

    command.attached[:] = False
    command._release_over_speed_welds()
    assert command._latch_pressure.item() == pytest.approx(0.0)
    assert latch_pressure(env, "ladder").item() == pytest.approx(0.0)


def test_nonfinite_step_resets_that_world_and_returns_a_finite_done() -> None:
    reset_ids: list[torch.Tensor] = []
    initialized = {"called": False}

    class _Scene:
        def write_data_to_sim(self) -> None:
            return None

    class _Sim:
        def forward(self) -> None:
            return None

    class _Env:
        unwrapped = None
        scene = _Scene()
        sim = _Sim()

        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            reset_ids.append(env_ids.clone())

    env = _Env()
    env.unwrapped = env
    env.command_manager = SimpleNamespace(
        get_term=lambda _name: SimpleNamespace(
            _initialize_from_start_pose=lambda: initialized.__setitem__("called", True)
        )
    )
    obs = {
        "actor": torch.tensor([[1.0, 2.0], [float("nan"), 0.0]]),
        "actor_ladder": torch.ones(2, 3),
    }
    rewards = torch.tensor([0.2, float("inf")])
    dones = torch.zeros(2, dtype=torch.long)
    extras = {"log": {"Episode_Reward/ladder_climb": float("nan")}}

    cleaned_obs, cleaned_rewards, cleaned_dones = _recover_nonfinite_ladder_steps(
        env, obs, rewards, dones, extras
    )

    assert reset_ids[0].tolist() == [1]
    assert initialized["called"]
    assert torch.isfinite(cleaned_obs["actor"]).all()
    assert cleaned_obs["actor"][1].tolist() == [0.0, 0.0]
    assert cleaned_obs["actor"][0].tolist() == [1.0, 2.0]
    torch.testing.assert_close(cleaned_rewards, torch.tensor([0.2, 0.0]))
    assert cleaned_dones.tolist() == [0, 1]
    assert extras["log"]["Episode_Reward/ladder_climb"] == 0.0


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
            "next_rung_hands",
            "next_rung_feet",
            "first_hand_reached",
            "second_hand_reached",
            "first_hand_phase",
            "second_hand_phase",
            "legs_phase",
            "stabilization_torso_speed",
            "stabilization_torso_angular_speed",
            "stabilization_pelvis_angular_speed",
            "hold_quiet",
        )
    }


def test_reward_logs_are_the_mean_episode_sum() -> None:
    def _zero(env, command_name: str) -> torch.Tensor:
        del command_name
        return torch.zeros(env.num_envs)

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        max_episode_length_s=20.0,
        scene=None,
    )
    manager = RewardManager(
        {
            "ladder_climb": RewardTermCfg(
                func=_zero, weight=1.0, params={"command_name": "ladder"}
            ),
            "ladder_hold": RewardTermCfg(
                func=_zero, weight=1.0, params={"command_name": "ladder"}
            ),
        },
        env,
    )
    env_ids = torch.tensor([0, 1])
    manager._episode_sums["ladder_climb"][:] = torch.tensor([2.0, 4.0])
    manager._episode_sums["ladder_hold"][:] = torch.tensor([1.0, 1.0])

    averaged = RewardManager.reset(manager, env_ids)
    manager._episode_sums["ladder_climb"][:] = torch.tensor([2.0, 4.0])
    manager._episode_sums["ladder_hold"][:] = torch.tensor([1.0, 1.0])
    manager.__class__ = EpisodeSumRewardManager
    totals = manager.reset(env_ids)

    assert averaged["Episode_Reward/ladder_climb"].item() == pytest.approx(3.0 / 20.0)
    assert totals["Episode_Reward/ladder_climb"].item() == pytest.approx(3.0)
    assert totals["Episode_Reward/ladder_hold"].item() == pytest.approx(1.0)
    assert torch.count_nonzero(manager._episode_sums["ladder_climb"]).item() == 0


def test_reset_installs_episode_sum_reward_logs_before_clearing() -> None:
    logged: dict[str, float] = {}

    def _reset_idx(env_ids: torch.Tensor | None = None) -> None:
        value = env.reward_manager.reset(env_ids)["Episode_Reward/ladder_climb"]
        logged["ladder_climb"] = float(value)

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        max_episode_length_s=20.0,
        scene=None,
        _reset_idx=_reset_idx,
    )
    env.reward_manager = RewardManager(
        {
            "ladder_climb": RewardTermCfg(
                func=lambda env, command_name: torch.zeros(env.num_envs),
                weight=1.0,
                params={"command_name": "ladder"},
            )
        },
        env,
    )
    env.reward_manager._episode_sums["ladder_climb"][:] = 2.5
    install_episode_sum_reward_logs(env)
    env._reset_idx(torch.tensor([0]))

    assert logged["ladder_climb"] == pytest.approx(2.5)
    assert env._reset_idx is _reset_idx


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
    assert command.metrics["next_rung_hands"].item() == pytest.approx(0.0)
    assert command.metrics["next_rung_feet"].item() == pytest.approx(0.0)

    command.held_rung[:, 0] = 5
    command.foot_rung[:, 0] = 2
    command._update_metrics()
    assert command.metrics["next_rung_hands"].item() == pytest.approx(0.02)
    assert command.metrics["next_rung_feet"].item() == pytest.approx(0.02)
    command.held_rung[:, 0] = 4
    command.foot_rung[:, 0] = 1

    command.attached[:] = False
    command.foot_rung[:] = -1
    command._update_metrics()
    assert command.metrics["attached_hands"].item() == pytest.approx(0.12)
    assert command.metrics["supported_feet"].item() == pytest.approx(0.12)
    assert command.metrics["next_rung_hands"].item() == pytest.approx(0.02)
    assert command.metrics["next_rung_feet"].item() == pytest.approx(0.02)

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
    _set_climb_phase(command, CLIMB_PHASE_LEGS)
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


def _grip_command() -> tuple[LadderClimbCommand, torch.Tensor]:
    command = object.__new__(LadderClimbCommand)
    command.cfg = SimpleNamespace(attach_distance=0.10, max_attach_speed=0.35)
    command.finished = torch.tensor([False])
    command.attached = torch.zeros(1, 2, dtype=torch.bool)
    command.grip_strength = torch.zeros(1, 2)
    command.grip_progress = torch.zeros(1, 2)
    command.grip_request = torch.zeros(1, 2)
    command.held_rung = torch.full((1, 2), -1)
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
    return command, site_xpos


def test_grip_progress_closes_after_one_second_inside_the_zone() -> None:
    command, site_xpos = _grip_command()

    command.apply_grip_requests(torch.tensor([[1.0, 1.0]]))
    assert command.attached.tolist() == [[False, False]]
    assert command.grip_progress[0, 0].item() == pytest.approx(0.02)
    assert command.grip_progress[0, 1].item() == 0.0

    site_xpos[0, 0] = torch.tensor([5.0, 0.0, 0.0])
    command.apply_grip_requests(torch.tensor([[1.0, 1.0]]))
    assert command.grip_progress[0, 0].item() == 0.0

    site_xpos[0, 0] = torch.tensor([0.0, 0.0, 1.0])
    for _ in range(49):
        command.apply_grip_requests(torch.tensor([[1.0, 0.0]]))
    assert command.attached.tolist() == [[False, False]]
    command.apply_grip_requests(torch.tensor([[1.0, 0.0]]))
    assert command.attached.tolist() == [[True, False]]
    assert command.held_rung.tolist() == [[1, -1]]
    assert command.grip_progress[0, 0].item() == pytest.approx(1.0)
    assert command._env.sim.data.eq_active[0, 0].item() == 1.0

    for _ in range(49):
        command.apply_grip_requests(torch.tensor([[-1.0, 0.0]]))
    assert command.attached.tolist() == [[True, False]]
    command.apply_grip_requests(torch.tensor([[-1.0, 0.0]]))
    assert command.attached.tolist() == [[False, False]]
    assert command.held_rung.tolist() == [[-1, -1]]
    assert command.grip_progress[0, 0].item() == 0.0
    assert command._env.sim.data.eq_active[0, 0].item() == 0.0


def test_ladder_command_reports_the_latch_and_the_progress() -> None:
    command = object.__new__(LadderClimbCommand)
    command.attached = torch.tensor([[1.0, 0.0]])
    command.grip_progress = torch.tensor([[1.0, 0.4]])

    observation = command.command

    assert observation.shape == (1, 4)
    assert observation[0].tolist() == pytest.approx([1.0, 0.0, 1.0, 0.4])


def test_flight_stops_at_one_and_ignores_a_single_reach() -> None:
    command = _quiet_ladder()
    command._flight_paid = torch.zeros(1)
    command._flight_step = -1
    command._flight_rate = torch.zeros(1)
    command.feet_on_flight_rungs = lambda: torch.ones(1, 2, dtype=torch.bool)  # type: ignore[method-assign]
    env = SimpleNamespace(
        step_dt=0.02,
        common_step_counter=0,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )

    command.attached[:, 0] = False
    assert ladder_flight(env, "ladder").item() == 0.0

    command.attached[:] = False
    env.common_step_counter = 1
    assert ladder_flight(env, "ladder").item() == pytest.approx(FLIGHT_RATE)
    command._flight_paid[:] = FLIGHT_TOTAL
    env.common_step_counter = 2
    assert ladder_flight(env, "ladder").item() == 0.0


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


def test_privileged_critic_observation_keeps_the_checkpoint_layout() -> None:
    command = _quiet_ladder(dwell=50, successes=4)
    command.hold_count[:] = 25
    command.successes[:] = 1
    command.held_rung[:] = 5
    command.foot_rung[:] = 2
    command.robot.data.joint_vel[:] = 2.0
    command._torso_vel[:] = torch.tensor([[0.3, 0.0, 0.0]])
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )

    observation = ladder_critic_privileged(env, "ladder")

    assert observation.shape == (1, 13)
    assert observation[0, 0].item() == pytest.approx(0.5)
    assert observation[0, 1].item() == pytest.approx(0.25)
    assert observation[0, 2].item() == pytest.approx(0.3)
    assert observation[0, 3].item() == pytest.approx(2.0)
    assert observation[0, 8].item() == pytest.approx(1.0)
    assert observation[0, 9:11].tolist() == pytest.approx([1.0, 1.0])
    assert observation[0, 11:].tolist() == pytest.approx([1.0, 1.0])


def test_ladder_geometry_observation_spans_one_below_the_feet_to_one_above_the_hands() -> None:
    command = object.__new__(LadderClimbCommand)
    num_rungs = 14
    foot = 1
    hand = foot + FOOT_HAND_RUNG_GAP
    command._rung_site_ids = torch.arange(num_rungs)
    command._rung_half_lengths = torch.zeros(num_rungs)
    command._torso_body_id = 0
    command.baseline_hand_rung = torch.tensor([[hand, hand]])
    command.baseline_foot_rung = torch.tensor([[foot, foot]])
    command.held_rung = torch.tensor([[hand, hand]])
    command.grip_strength = torch.tensor([[1.0, 1.0]])
    command.foot_rung = torch.tensor([[foot, foot]])
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
    window = list(range(foot - 1, foot - 1 + RUNG_WINDOW_COUNT))

    assert RUNG_WINDOW_COUNT == FOOT_HAND_RUNG_GAP + 3
    assert command.window_rung_indices().tolist() == [window]
    assert observation.shape == (1, RUNG_WINDOW_COUNT, 15)
    assert observation[0, :, 2].tolist() == pytest.approx([float(index) for index in window])
    next_hand_row = window.index(hand + 1)
    next_foot_row = window.index(foot + 1)
    hand_row = window.index(hand)
    foot_row = window.index(foot)
    assert observation[0, next_hand_row, 7:9].tolist() == pytest.approx([1.0, 1.0])
    assert observation[0, next_foot_row, 9:11].tolist() == pytest.approx([1.0, 1.0])
    assert observation[0, hand_row, 11:13].tolist() == pytest.approx([1.0, 1.0])
    assert observation[0, foot_row, 13:].tolist() == pytest.approx([1.0, 1.0])

    command.foot_rung[:] = foot + 1
    command.held_rung[:] = hand + 1
    stayed = command.rung_tokens_torso
    assert command.window_rung_indices().tolist() == [window]
    assert stayed[0, :, 2].tolist() == pytest.approx([float(index) for index in window])

    shifted_foot = 2
    shifted_hand = shifted_foot + FOOT_HAND_RUNG_GAP
    command.baseline_foot_rung[:] = shifted_foot
    command.baseline_hand_rung[:] = shifted_hand
    shifted = command.rung_tokens_torso
    shifted_window = list(range(shifted_foot - 1, shifted_foot - 1 + RUNG_WINDOW_COUNT))
    assert command.window_rung_indices().tolist() == [shifted_window]
    assert shifted[0, :, 2].tolist() == pytest.approx(
        [float(index) for index in shifted_window]
    )


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
    assert rl_cfg.actor.distribution_cfg["init_std"] == 0.25
    assert rl_cfg.actor.distribution_cfg["std_range"] == (0.2, 0.5)
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

    assert robot_cfg.init_state.pos == (-1.079482, 0.0, 1.385411)
    assert tuple(robot_cfg.init_state.rot) == (0.983835, 0.0, 0.179079, 0.0)
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
    for side in ("left", "right"):
        hand_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grip_site"
        )
        hand_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_hand_collision"
        )
        assert hand_site >= 0 and hand_geom >= 0
        assert model.site_size[hand_site, 0] == pytest.approx(0.085)
        assert model.geom_size[hand_geom, 0] == pytest.approx(
            model.site_size[hand_site, 0]
        )
        assert model.site_type[hand_site] == mujoco.mjtGeom.mjGEOM_SPHERE

    # Reload the climbing keyframe. Feet are on rung 4 and hands on rung 8
    # (1-based names), with a bent knee and the next rung still in reach.
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

    reset_links = torch.tensor(
        np.array(
            [
                data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
                for name in LEG_LINK_NAMES
            ]
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    foot_rung_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_ladder_rung_04_grip",
    )
    next_foot_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_ladder_rung_05_grip",
    )
    rung_axis = torch.tensor(
        data.site_xmat[foot_rung_site].reshape(3, 3)[:, 2], dtype=torch.float32
    ).view(1, 3)
    face_origin, face_lateral, face_climb, face_degenerate = ladder_face_axes(
        torch.tensor(data.site_xpos[foot_rung_site], dtype=torch.float32).view(1, 3),
        torch.tensor(data.site_xpos[next_foot_site], dtype=torch.float32).view(1, 3),
        rung_axis,
    )
    assert not bool(face_degenerate.item())
    assert not bool(
        projected_legs_cross(
            reset_links, face_origin, face_lateral, face_climb
        ).item()
    )

    foot_rung_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_ladder_rung_04_grip",
    )
    hand_rung_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_ladder_rung_08_grip",
    )
    next_rung_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left_ladder_rung_05_grip",
    )
    rail_a = data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_ladder_rung_01_grip")
    ]
    rail_b = data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_ladder_rung_02_grip")
    ]
    rail = rail_b - rail_a
    rail = rail / np.linalg.norm(rail)
    for foot_name, hip_name in (
        ("left_foot", "left_hip_pitch_link"),
        ("right_foot", "right_hip_pitch_link"),
    ):
        foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot_name)
        hip = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hip_name)]
        target = data.site_xpos[foot_rung_site].copy()
        target[1] = data.site_xpos[foot_site, 1]
        foot_distance = np.linalg.norm(data.site_xpos[foot_site] - target)
        assert foot_distance < 0.055
        assert foot_distance < 0.11
        assert foot_on_tread(
            torch.tensor(data.site_xpos[foot_site], dtype=torch.float32),
            torch.tensor(data.site_xpos[foot_rung_site], dtype=torch.float32),
            0.35,
        ).item()
        hip_to_foot = np.linalg.norm(data.site_xpos[foot_site] - hip)
        assert 0.50 <= hip_to_foot <= 0.62
        sole = torch.tensor(FOOT_SOLE_END_OFFSETS, dtype=torch.float32)
        rotation = torch.tensor(
            data.site_xmat[foot_site].reshape(3, 3),
            dtype=torch.float32,
        )
        site = torch.tensor(data.site_xpos[foot_site], dtype=torch.float32)
        ends = site + sole @ rotation.T
        gap, _ = foot_tread_features(
            ends.view(1, 1, 2, 3),
            torch.tensor(data.site_xpos[foot_rung_site], dtype=torch.float32).view(1, 1, 3),
            torch.tensor(data.site_xpos[next_rung_site], dtype=torch.float32).view(1, 1, 3),
            torch.tensor([[False]]),
        )
        assert 0.12 < gap.item() < 0.28
        next_point = data.site_xpos[next_rung_site].copy()
        next_point[1] = hip[1]
        assert np.linalg.norm(hip - next_point) <= 0.69
    for hand_name in ("left_grip_site", "right_grip_site"):
        hand_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, hand_name)
        target = data.site_xpos[hand_rung_site].copy()
        target[1] = data.site_xpos[hand_site, 1]
        hand_distance = np.linalg.norm(data.site_xpos[hand_site] - target)
        # Clear of the wood, and inside the 0.14 m attach distance.
        assert 0.12 < hand_distance < 0.14
    torso = data.xmat[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    ].reshape(3, 3)
    assert float(torso[:, 2] @ rail) > 0.95
    for contact in data.contact:
        geom_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom[0]) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom[1]) or "",
        }
        assert contact.dist >= -0.001
    lowest_z = min(
        float(data.geom_xpos[geom_id, 2])
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] != 0
        and "ladder" not in (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        )
    )
    assert lowest_z > 0.2

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
    assert all(model.geom_contype[i] == 2 and model.geom_conaffinity[i] == 3 for i in refined)
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
        *(f"left_ladder_rung_{index:02d}" for index in range(1, 15)),
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
    assert clear_vertical_gap == pytest.approx(2.80 / 15 - 0.07, abs=1e-4)
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

def test_ladder_run_log_records_the_code_version(tmp_path) -> None:
    from train_mimic.tasks.tracking.config.ladder_version import (
        LADDER_CODE_VERSION,
        record_ladder_code_version,
    )

    path = record_ladder_code_version(str(tmp_path))
    text = (tmp_path / "code_version.txt").read_text()
    assert path.endswith("code_version.txt")
    assert text.startswith(f"{LADDER_CODE_VERSION}\n")
    assert "ground" in text


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
