from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
import math
import re
from typing import TYPE_CHECKING, Literal, cast

import mujoco
import torch
import torch.nn.functional as F
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


HandName = Literal["left", "right"]


class LadderPhase(IntEnum):
    """Ordered ladder-climbing phases exposed to the policy as a one-hot."""

    STABILIZE = 0
    FIRST_HAND = 1
    SECOND_HAND = 2
    FIRST_FOOT = 3
    SECOND_FOOT = 4


@requires_model_fields("eq_solref", "eq_solimp")
def prepare_ladder_weld_model(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Request per-environment equality parameters for gradual grip release.

    The event itself is intentionally a no-op.  Its model-field declaration makes
    MJLab expand the two equality solver arrays before the ladder command is built,
    so each parallel environment can soften its active hand weld independently.
    """

    del env, env_ids


def _as_rotation_matrix(value: torch.Tensor) -> torch.Tensor:
    """Accept MJWarp site matrices stored as either ``(..., 9)`` or ``(..., 3, 3)``."""

    if value.shape[-2:] == (3, 3):
        return value
    if value.shape[-1] == 9:
        return value.reshape(*value.shape[:-1], 3, 3)
    raise RuntimeError(f"Unexpected site_xmat shape: {tuple(value.shape)}")


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized MuJoCo ``(w, x, y, z)`` quaternions."""

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    # This branch-free form is stable for 180-degree rotations and keeps the
    # code compatible with CUDA graph capture used by MJLab/MuJoCo-Warp.
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)),
        m21 - m12,
    )
    qy = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)),
        m02 - m20,
    )
    qz = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)),
        m10 - m01,
    )
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    return F.normalize(quat, dim=-1, eps=1.0e-8)


class LadderClimbCommand(CommandTerm):
    """Manage an explicit stabilize/hand/hand/foot/foot climbing cycle.

    Both hands and both feet begin on known rungs.  Every cycle first requires
    stable four-point support, then advances the configured first hand, the
    other hand, the configured first foot, and the other foot.  Only the limb
    assigned to the current phase is released or allowed to change support.

    Training keeps a rolling success window for the currently unlocked prefix.
    The next phase opens only after the configured minimum step budget and a
    success rate strictly above the threshold over a full window.  If the
    policy completes the last currently available phase, the command marks the
    short curriculum episode complete instead of waiting for a locked phase.
    """

    cfg: LadderClimbCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(
        self,
        cfg: LadderClimbCommandCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        super().__init__(cfg, env)

        if len(cfg.hand_site_names) != 2:
            raise ValueError("hand_site_names must contain left and right sites")
        if len(cfg.foot_site_names) != 2:
            raise ValueError("foot_site_names must contain left and right sites")
        if len(cfg.anchor_body_names) != 2:
            raise ValueError("anchor_body_names must contain left and right bodies")
        if len(cfg.weld_names) != 2:
            raise ValueError("weld_names must contain left and right welds")
        if not cfg.rung_site_names:
            raise ValueError("rung_site_names must contain at least one rung site")
        if not 0 <= cfg.start_rung < len(cfg.rung_site_names):
            raise ValueError(
                f"start_rung={cfg.start_rung} is outside "
                f"[0, {len(cfg.rung_site_names) - 1}]"
            )
        if cfg.attach_distance <= 0.0:
            raise ValueError("attach_distance must be > 0")
        if cfg.max_attach_speed < 0.0:
            raise ValueError("max_attach_speed must be >= 0")
        if cfg.foot_reach_distance <= 0.0:
            raise ValueError("foot_reach_distance must be > 0")
        if cfg.foot_support_distance <= 0.0:
            raise ValueError("foot_support_distance must be > 0")
        if cfg.max_foot_speed < 0.0:
            raise ValueError("max_foot_speed must be >= 0")
        if cfg.foot_target_height <= 0.0:
            raise ValueError("foot_target_height must be > 0")
        if cfg.hand_foot_lead_rungs < 0:
            raise ValueError("hand_foot_lead_rungs must be >= 0")
        max_initial_foot_rung = max(
            len(cfg.rung_site_names) - 1 - cfg.hand_foot_lead_rungs,
            0,
        )
        if cfg.initialize_on_reset and not (
            0 <= cfg.initial_foot_rung <= max_initial_foot_rung
        ):
            raise ValueError(
                f"initial_foot_rung={cfg.initial_foot_rung} is outside "
                f"[0, {max_initial_foot_rung}] for reset initialization"
            )
        if cfg.stabilization_dwell_steps <= 0:
            raise ValueError("stabilization_dwell_steps must be > 0")
        if (
            cfg.stabilization_dwell_max_steps is not None
            and cfg.stabilization_dwell_max_steps < cfg.stabilization_dwell_steps
        ):
            raise ValueError(
                "stabilization_dwell_max_steps must be >= "
                "stabilization_dwell_steps or None"
            )
        if cfg.hand_target_dwell_steps <= 0:
            raise ValueError("hand_target_dwell_steps must be > 0")
        if cfg.foot_target_dwell_steps <= 0:
            raise ValueError("foot_target_dwell_steps must be > 0")
        if cfg.max_stabilization_torso_speed < 0.0:
            raise ValueError("max_stabilization_torso_speed must be >= 0")
        if cfg.max_stabilization_joint_speed < 0.0:
            raise ValueError("max_stabilization_joint_speed must be >= 0")
        if cfg.max_stabilization_body_angular_speed < 0.0:
            raise ValueError(
                "max_stabilization_body_angular_speed must be >= 0"
            )
        if cfg.max_stabilization_waist_joint_speed < 0.0:
            raise ValueError(
                "max_stabilization_waist_joint_speed must be >= 0"
            )
        if cfg.max_stabilization_support_offset_error <= 0.0:
            raise ValueError("max_stabilization_support_offset_error must be > 0")
        if not 0.0 < cfg.max_phase_torso_orientation_error < math.pi:
            raise ValueError(
                "max_phase_torso_orientation_error must be between 0 and pi"
            )
        if cfg.max_phase_support_offset_error <= 0.0:
            raise ValueError("max_phase_support_offset_error must be > 0")
        if cfg.max_phase_completion_torso_speed < 0.0:
            raise ValueError("max_phase_completion_torso_speed must be >= 0")
        if cfg.max_phase_completion_joint_speed < 0.0:
            raise ValueError("max_phase_completion_joint_speed must be >= 0")
        if cfg.first_foot_max_body_drop < 0.0:
            raise ValueError("first_foot_max_body_drop must be >= 0")
        if cfg.cycle_min_body_ascent <= 0.0:
            raise ValueError("cycle_min_body_ascent must be > 0")
        if cfg.release_preload_dwell_steps <= 0:
            raise ValueError("release_preload_dwell_steps must be > 0")
        if cfg.release_ramp_steps <= 0:
            raise ValueError("release_ramp_steps must be > 0")
        if cfg.release_final_dwell_steps <= 0:
            raise ValueError("release_final_dwell_steps must be > 0")
        if cfg.release_recovery_steps <= 0:
            raise ValueError("release_recovery_steps must be > 0")
        minimum_release_steps = (
            cfg.release_preload_dwell_steps
            + cfg.release_ramp_steps
            + cfg.release_final_dwell_steps
        )
        if cfg.pre_release_timeout_steps < minimum_release_steps:
            raise ValueError(
                "pre_release_timeout_steps must be >= the configured preload, "
                f"ramp, and final dwell total ({minimum_release_steps})"
            )
        if cfg.max_release_torso_speed < 0.0:
            raise ValueError("max_release_torso_speed must be >= 0")
        if not 0.0 < cfg.max_release_torso_orientation_error < math.pi:
            raise ValueError(
                "max_release_torso_orientation_error must be between 0 and pi"
            )
        if cfg.max_release_support_offset_error <= 0.0:
            raise ValueError("max_release_support_offset_error must be > 0")
        if cfg.release_soft_timeconst <= 0.0:
            raise ValueError("release_soft_timeconst must be > 0")
        if not 0.0 < cfg.release_soft_impedance < 1.0:
            raise ValueError("release_soft_impedance must be between 0 and 1")
        if not 0.0 < cfg.curriculum_success_threshold < 1.0:
            raise ValueError(
                "curriculum_success_threshold must be strictly between 0 and 1"
            )
        if cfg.curriculum_window_size <= 0:
            raise ValueError("curriculum_window_size must be > 0")
        if len(cfg.curriculum_min_phase_steps) != len(LadderPhase) - 1:
            raise ValueError(
                "curriculum_min_phase_steps must contain one entry for each "
                "phase transition"
            )
        if any(step <= 0 for step in cfg.curriculum_min_phase_steps):
            raise ValueError("curriculum_min_phase_steps entries must be > 0")
        if cfg.fixed_max_unlocked_phase is not None:
            fixed_phase = int(cfg.fixed_max_unlocked_phase)
            if cfg.curriculum_enabled:
                raise ValueError(
                    "fixed_max_unlocked_phase requires curriculum_enabled=False"
                )
            if (
                not int(LadderPhase.STABILIZE)
                <= fixed_phase
                <= int(LadderPhase.SECOND_FOOT)
            ):
                raise ValueError(
                    "fixed_max_unlocked_phase must be between "
                    f"{int(LadderPhase.STABILIZE)} and "
                    f"{int(LadderPhase.SECOND_FOOT)}, got {fixed_phase}"
                )
        if cfg.freeze_at_max_unlocked_phase and (
            cfg.curriculum_enabled or cfg.fixed_max_unlocked_phase is None
        ):
            raise ValueError(
                "freeze_at_max_unlocked_phase requires curriculum_enabled=False "
                "and fixed_max_unlocked_phase to be set"
            )
        if not 0.0 <= cfg.boundary_state_reset_prob < 1.0:
            raise ValueError("boundary_state_reset_prob must be in [0, 1)")
        if cfg.boundary_state_bank_size < 0:
            raise ValueError("boundary_state_bank_size must be >= 0")
        if cfg.boundary_state_reset_prob > 0.0:
            if cfg.boundary_state_bank_size == 0:
                raise ValueError(
                    "boundary_state_bank_size must be > 0 when boundary resets are enabled"
                )
            if not cfg.initialize_on_reset:
                raise ValueError(
                    "boundary-state resets require initialize_on_reset=True"
                )
        if cfg.grip_half_span is not None and cfg.grip_half_span <= 0.0:
            raise ValueError("grip_half_span must be > 0 or None")

        self.robot = env.scene[cfg.entity_name]
        model = env.sim.mj_model

        hand_site_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_SITE, name)
            for name in cfg.hand_site_names
        ]
        foot_site_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_SITE, name)
            for name in cfg.foot_site_names
        ]
        anchor_body_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in cfg.anchor_body_names
        ]
        weld_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_EQUALITY, name)
            for name in cfg.weld_names
        ]
        rung_site_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_SITE, name)
            for name in cfg.rung_site_names
        ]
        self._torso_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            cfg.torso_body_name,
        )
        self._pelvis_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            cfg.pelvis_body_name,
        )
        torso_link_ids, _ = self.robot.find_bodies(cfg.torso_body_name)
        pelvis_link_ids, _ = self.robot.find_bodies(cfg.pelvis_body_name)
        waist_joint_ids, _ = self.robot.find_joints(r".*waist.*")
        if len(torso_link_ids) != 1:
            raise ValueError(
                f"Expected one robot body matching {cfg.torso_body_name!r}, "
                f"found {len(torso_link_ids)}"
            )
        if len(pelvis_link_ids) != 1:
            raise ValueError(
                f"Expected one robot body matching {cfg.pelvis_body_name!r}, "
                f"found {len(pelvis_link_ids)}"
            )
        if not waist_joint_ids:
            raise ValueError("Ladder robot must contain at least one waist joint")
        self._torso_link_index = torso_link_ids[0]
        self._pelvis_link_index = pelvis_link_ids[0]
        self._waist_joint_ids = torch.tensor(
            waist_joint_ids,
            dtype=torch.long,
            device=self.device,
        )

        anchor_mocap_ids: list[int] = []
        for body_name, body_id in zip(cfg.anchor_body_names, anchor_body_ids):
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(f'Body {body_name!r} must have mocap="true"')
            anchor_mocap_ids.append(mocap_id)

        for weld_name, weld_id in zip(cfg.weld_names, weld_ids):
            if int(model.eq_type[weld_id]) != int(mujoco.mjtEq.mjEQ_WELD):
                raise ValueError(f"Equality {weld_name!r} must be a weld")

        capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        rung_half_lengths: list[float] = []
        for name, site_id in zip(cfg.rung_site_names, rung_site_ids):
            if int(model.site_type[site_id]) != capsule_type:
                raise ValueError(f'Rung site {name!r} must have type="capsule"')
            half_length = float(model.site_size[site_id, 1])
            if half_length <= 0.0:
                raise ValueError(f"Rung site {name!r} has zero capsule length")
            if cfg.grip_half_span is not None:
                half_length = min(half_length, cfg.grip_half_span)
            rung_half_lengths.append(half_length)

        self._hand_site_ids = torch.tensor(
            hand_site_ids, dtype=torch.long, device=self.device
        )
        self._foot_site_ids = torch.tensor(
            foot_site_ids, dtype=torch.long, device=self.device
        )
        self._anchor_mocap_ids = torch.tensor(
            anchor_mocap_ids, dtype=torch.long, device=self.device
        )
        self._weld_ids = torch.tensor(weld_ids, dtype=torch.long, device=self.device)
        weld_solref = env.sim.model.eq_solref
        weld_solimp = env.sim.model.eq_solimp
        expected_model_worlds = self.num_envs
        if (
            weld_solref.shape[0] != expected_model_worlds
            or weld_solimp.shape[0] != expected_model_worlds
        ):
            raise RuntimeError(
                "Ladder grip release requires per-environment eq_solref/eq_solimp; "
                "add the prepare_ladder_weld_model startup event before building "
                "LadderClimbCommand"
            )
        self._strong_weld_solref = weld_solref[0, self._weld_ids].clone()
        self._strong_weld_solimp = weld_solimp[0, self._weld_ids].clone()
        if cfg.release_soft_timeconst <= float(
            self._strong_weld_solref[:, 0].max().item()
        ):
            raise ValueError(
                "release_soft_timeconst must be larger than the configured weld "
                "time constant so the release ramp actually softens the grip"
            )
        self._rung_site_ids = torch.tensor(
            rung_site_ids, dtype=torch.long, device=self.device
        )
        self._rung_half_lengths = torch.tensor(
            rung_half_lengths,
            dtype=torch.float32,
            device=self.device,
        )
        self._all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._play_unlocked_phase = (
            int(cfg.fixed_max_unlocked_phase)
            if cfg.fixed_max_unlocked_phase is not None
            else int(LadderPhase.SECOND_FOOT)
        )
        self._unlocked_phase = (
            int(LadderPhase.STABILIZE)
            if cfg.curriculum_enabled
            else self._play_unlocked_phase
        )
        self._curriculum_phase_start_step = int(getattr(env, "common_step_counter", 0))
        self._recent_curriculum_outcomes: deque[int] = deque(
            maxlen=cfg.curriculum_window_size
        )
        self._pending_curriculum_outcomes: list[int] = []
        self._episode_active = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._started_from_boundary = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self._foot_contact_sensor = env.scene[cfg.foot_contact_sensor_name]
        primary_names = [
            name.rsplit("/", 1)[-1] for name in self._foot_contact_sensor.primary_names
        ]
        try:
            foot_contact_indices = [
                primary_names.index(name) for name in cfg.foot_contact_body_names
            ]
        except ValueError as exc:
            raise ValueError(
                f"Contact sensor {cfg.foot_contact_sensor_name!r} primaries "
                f"{primary_names} do not contain {cfg.foot_contact_body_names}"
            ) from exc
        self._foot_contact_indices = torch.tensor(
            foot_contact_indices,
            dtype=torch.long,
            device=self.device,
        )

        first_hand = 0 if cfg.first_moving_hand == "left" else 1
        self.active_hand = torch.full(
            (self.num_envs,), first_hand, dtype=torch.long, device=self.device
        )
        self.target_rung = torch.full(
            (self.num_envs,), cfg.start_rung, dtype=torch.long, device=self.device
        )
        self.attached = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self.grip_strength = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.held_rung = torch.full(
            (self.num_envs, 2), -1, dtype=torch.long, device=self.device
        )
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_advanced = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_foot_advanced = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_stabilized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.just_cycle_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.curriculum_stage_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.pre_release_stalled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.phase_frozen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pending_start_pose_init = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pending_boundary_state_init = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.phase = torch.full(
            (self.num_envs,),
            int(LadderPhase.STABILIZE),
            dtype=torch.long,
            device=self.device,
        )
        self._phase_dwell_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._release_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._release_preload_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._release_ramp_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._release_final_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._release_age_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stabilization_dwell_target = torch.full(
            (self.num_envs,),
            cfg.stabilization_dwell_steps,
            dtype=torch.long,
            device=self.device,
        )

        first_foot = 0 if cfg.first_moving_foot == "left" else 1
        self.active_foot = torch.full(
            (self.num_envs,), first_foot, dtype=torch.long, device=self.device
        )
        self.target_foot_rung = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.foot_rung = torch.full(
            (self.num_envs, 2), -1, dtype=torch.long, device=self.device
        )
        self._allocate_boundary_state_bank()
        hand_pos = self.hand_pos_w
        self._previous_hand_pos_w = hand_pos.clone()
        self._hand_vel_w = torch.zeros_like(hand_pos)
        foot_pos = self.foot_pos_w
        self._previous_foot_pos_w = foot_pos.clone()
        self._foot_vel_w = torch.zeros_like(foot_pos)
        torso_com_pos = self.torso_com_pos_w
        self._previous_torso_com_pos_w = torso_com_pos.clone()
        self._torso_com_vel_w = torch.zeros_like(torso_com_pos)
        self._reference_torso_rotation_w = self.torso_rotation_w.clone()
        self._reference_torso_support_offset_w = self.torso_support_offset_w.clone()
        body_height = self.body_height.clone()
        self.start_height = body_height.clone()
        self._cycle_start_body_height = body_height.clone()
        self._phase_start_body_height = body_height.clone()
        self.episode_max_body_height = body_height.clone()

        self.metrics["target_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rung_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["attached_hands"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_target_distance"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["foot_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["supported_feet"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["torso_orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["torso_support_offset_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["active_hand_release_progress"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["pre_release_age"] = torch.zeros(
            self.num_envs, device=self.device
        )
        for metric_name in (
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
        ):
            self.metrics[metric_name] = torch.zeros(
                self.num_envs, device=self.device
            )
        for metric_name in (
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
        ):
            self.metrics[metric_name] = torch.zeros(
                self.num_envs, device=self.device
            )
        self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["unlocked_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_success_rate"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["curriculum_window_fill"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["curriculum_phase_steps"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        """Return the 24D ladder command consumed by the RL policy.

        Layout: phase one-hot (5), torso-frame left/right hand-to-target vectors (6),
        attached flags (2), active-hand grip strength (1), normalized hand
        progress (1), torso-frame left/right foot-to-target vectors (6), physical
        foot-contact flags (2), and normalized foot progress (1).  The continuous
        grip value replaces the nearly constant post-reset initialization bit, so
        the command stays 24D while exposing the internal pre-release ramp.
        """

        phase_one_hot = F.one_hot(self.phase, num_classes=len(LadderPhase)).float()
        hand_target_delta = self._vectors_to_torso(
            self.hand_target_pos_w - self.hand_pos_w
        ).flatten(1)
        foot_target_delta = self._vectors_to_torso(
            self.foot_target_pos_w - self.foot_pos_w
        ).flatten(1)
        return torch.cat(
            (
                phase_one_hot,
                hand_target_delta,
                self.attached.float(),
                self.active_grip_strength.unsqueeze(-1),
                self.rung_progress.unsqueeze(-1),
                foot_target_delta,
                self.foot_contact.float(),
                self.foot_progress.unsqueeze(-1),
            ),
            dim=-1,
        )

    @property
    def num_rungs(self) -> int:
        return int(self._rung_site_ids.numel())

    @property
    def hand_pos_w(self) -> torch.Tensor:
        """World positions with shape ``(num_envs, 2, 3)``."""

        return self._env.sim.data.site_xpos[:, self._hand_site_ids]

    @property
    def active_hand_pos_w(self) -> torch.Tensor:
        return self.hand_pos_w[self._all_env_ids, self.active_hand]

    @property
    def active_hand_vel_w(self) -> torch.Tensor:
        return self._hand_vel_w[self._all_env_ids, self.active_hand]

    @property
    def active_grip_strength(self) -> torch.Tensor:
        """Continuous support strength of the hand selected by the current phase."""

        strength = self.grip_strength.gather(
            1,
            self.active_hand[:, None],
        ).squeeze(1)
        return torch.where(self.initialized, strength, 0.0)

    @property
    def foot_pos_w(self) -> torch.Tensor:
        """Sole-site world positions with shape ``(num_envs, 2, 3)``."""

        return self._env.sim.data.site_xpos[:, self._foot_site_ids]

    @property
    def active_foot_pos_w(self) -> torch.Tensor:
        return self.foot_pos_w[self._all_env_ids, self.active_foot]

    @property
    def active_foot_vel_w(self) -> torch.Tensor:
        return self._foot_vel_w[self._all_env_ids, self.active_foot]

    @property
    def torso_com_vel_w(self) -> torch.Tensor:
        return self._torso_com_vel_w

    @property
    def is_stabilization_phase(self) -> torch.Tensor:
        return self.phase == int(LadderPhase.STABILIZE)

    @property
    def is_hand_phase(self) -> torch.Tensor:
        return (self.phase == int(LadderPhase.FIRST_HAND)) | (
            self.phase == int(LadderPhase.SECOND_HAND)
        )

    @property
    def is_pre_release(self) -> torch.Tensor:
        return self.is_hand_phase & self._release_active

    @property
    def is_foot_phase(self) -> torch.Tensor:
        return (self.phase == int(LadderPhase.FIRST_FOOT)) | (
            self.phase == int(LadderPhase.SECOND_FOOT)
        )

    @property
    def max_unlocked_phase(self) -> int:
        return self._unlocked_phase

    @property
    def curriculum_success_rate(self) -> float:
        if not self._recent_curriculum_outcomes:
            return 0.0
        return sum(self._recent_curriculum_outcomes) / len(
            self._recent_curriculum_outcomes
        )

    @property
    def curriculum_window_fill(self) -> float:
        return len(self._recent_curriculum_outcomes) / self.cfg.curriculum_window_size

    @staticmethod
    def phase_name(phase: int) -> str:
        return LadderPhase(phase).name.lower()

    @property
    def curriculum_phase_steps(self) -> int:
        common_step = int(getattr(self._env, "common_step_counter", 0))
        return max(common_step - self._curriculum_phase_start_step, 0)

    def drain_curriculum_outcomes(self) -> list[int]:
        """Return locally completed outcomes since the previous PPO iteration."""

        outcomes = self._pending_curriculum_outcomes
        self._pending_curriculum_outcomes = []
        return outcomes

    def update_curriculum(self, outcomes: Iterable[int]) -> bool:
        """Merge episode outcomes and open at most one new phase.

        This method is called by :class:`LadderOnPolicyRunner` once per PPO
        iteration after multi-GPU outcome synchronization.
        """

        if not self.cfg.curriculum_enabled:
            return False
        for outcome in outcomes:
            value = int(outcome)
            if value not in (0, 1):
                raise ValueError(f"Curriculum outcome must be 0 or 1, got {value}")
            self._recent_curriculum_outcomes.append(value)

        if self._unlocked_phase >= int(LadderPhase.SECOND_FOOT):
            return False
        if len(self._recent_curriculum_outcomes) < self.cfg.curriculum_window_size:
            return False
        minimum_steps = self.cfg.curriculum_min_phase_steps[self._unlocked_phase]
        if self.curriculum_phase_steps < minimum_steps:
            return False
        if self.curriculum_success_rate <= self.cfg.curriculum_success_threshold:
            return False

        self._unlocked_phase += 1
        self._curriculum_phase_start_step = int(
            getattr(self._env, "common_step_counter", 0)
        )
        self._recent_curriculum_outcomes.clear()
        return True

    def _allocate_boundary_state_bank(self) -> None:
        """Allocate a compact per-phase GPU bank of valid transition states."""

        self._boundary_bank_capacity = (
            self.cfg.boundary_state_bank_size
            if self.cfg.boundary_state_reset_prob > 0.0
            else 0
        )
        phase_count = len(LadderPhase)
        capacity = getattr(self, "_boundary_bank_capacity", 0)
        joint_pos = self.robot.data.joint_pos
        float_dtype = joint_pos.dtype
        float_device = joint_pos.device
        self._boundary_bank_counts = [0] * phase_count
        self._boundary_bank_cursors = [0] * phase_count
        self._boundary_root_state = torch.zeros(
            (phase_count, capacity, 13),
            dtype=float_dtype,
            device=float_device,
        )
        self._boundary_joint_pos = torch.zeros(
            (phase_count, capacity, joint_pos.shape[-1]),
            dtype=joint_pos.dtype,
            device=joint_pos.device,
        )
        self._boundary_joint_vel = torch.zeros_like(self._boundary_joint_pos)
        self._boundary_mocap_pos = torch.zeros(
            (phase_count, capacity, 2, 3),
            dtype=float_dtype,
            device=float_device,
        )
        self._boundary_mocap_quat = torch.zeros(
            (phase_count, capacity, 2, 4),
            dtype=float_dtype,
            device=float_device,
        )
        self._boundary_attached = torch.zeros(
            (phase_count, capacity, 2),
            dtype=torch.bool,
            device=float_device,
        )
        self._boundary_held_rung = torch.full(
            (phase_count, capacity, 2),
            -1,
            dtype=torch.long,
            device=float_device,
        )
        self._boundary_foot_rung = torch.full_like(
            self._boundary_held_rung,
            -1,
        )
        self._boundary_cycle_start_body_height = torch.zeros(
            (phase_count, capacity),
            dtype=float_dtype,
            device=float_device,
        )
        self._boundary_phase_start_body_height = torch.zeros_like(
            self._boundary_cycle_start_body_height
        )

    def _capture_boundary_states(
        self,
        env_ids: torch.Tensor,
        next_phase: LadderPhase,
    ) -> None:
        """Store valid support states immediately before a phase begins."""

        capacity = getattr(self, "_boundary_bank_capacity", 0)
        if capacity == 0 or env_ids.numel() == 0:
            return
        if env_ids.numel() > capacity:
            env_ids = env_ids[
                torch.randperm(env_ids.numel(), device=self.device)[:capacity]
            ]

        phase_id = int(next_phase)
        count = env_ids.numel()
        cursor = self._boundary_bank_cursors[phase_id]
        slots = (
            torch.arange(count, dtype=torch.long, device=self.device) + cursor
        ) % capacity
        data = self._env.sim.data
        root_state = torch.cat(
            (
                self.robot.data.root_link_pose_w[env_ids],
                self.robot.data.root_link_vel_w[env_ids],
            ),
            dim=-1,
        )

        self._boundary_root_state[phase_id, slots] = root_state
        self._boundary_joint_pos[phase_id, slots] = self.robot.data.joint_pos[env_ids]
        self._boundary_joint_vel[phase_id, slots] = self.robot.data.joint_vel[env_ids]
        self._boundary_mocap_pos[phase_id, slots] = data.mocap_pos[
            env_ids[:, None], self._anchor_mocap_ids[None, :]
        ]
        self._boundary_mocap_quat[phase_id, slots] = data.mocap_quat[
            env_ids[:, None], self._anchor_mocap_ids[None, :]
        ]
        self._boundary_attached[phase_id, slots] = self.attached[env_ids]
        self._boundary_held_rung[phase_id, slots] = self.held_rung[env_ids]
        self._boundary_foot_rung[phase_id, slots] = self.foot_rung[env_ids]
        self._boundary_cycle_start_body_height[phase_id, slots] = (
            self._cycle_start_body_height[env_ids]
        )
        self._boundary_phase_start_body_height[phase_id, slots] = self.body_height[
            env_ids
        ]
        self._boundary_bank_counts[phase_id] = min(
            capacity,
            self._boundary_bank_counts[phase_id] + count,
        )
        self._boundary_bank_cursors[phase_id] = (cursor + count) % capacity

    def _sample_boundary_resets(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Restore a configured fraction of resets from unlocked phase boundaries."""

        restored = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        if self._boundary_bank_capacity == 0 or env_ids.numel() == 0:
            return restored
        available_phases = [
            phase_id
            for phase_id in range(self.max_unlocked_phase + 1)
            if self._boundary_bank_counts[phase_id] > 0
        ]
        if not available_phases:
            return restored

        restored = (
            torch.rand(env_ids.numel(), device=self.device)
            < self.cfg.boundary_state_reset_prob
        )
        selected_env_ids = env_ids[restored]
        if selected_env_ids.numel() == 0:
            return restored

        available = torch.tensor(
            available_phases,
            dtype=torch.long,
            device=self.device,
        )
        phase_ids = available[
            torch.randint(
                available.numel(),
                (selected_env_ids.numel(),),
                device=self.device,
            )
        ]
        phase_counts = torch.tensor(
            self._boundary_bank_counts,
            dtype=torch.long,
            device=self.device,
        )
        sample_indices = torch.floor(
            torch.rand(selected_env_ids.numel(), device=self.device)
            * phase_counts[phase_ids].to(torch.float32)
        ).to(torch.long)
        self._restore_boundary_states(selected_env_ids, phase_ids, sample_indices)
        return restored

    def _restore_boundary_states(
        self,
        env_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> None:
        """Restore physical/FSM support state and configure the requested phase."""

        root_state = self._boundary_root_state[phase_ids, sample_indices]
        joint_pos = self._boundary_joint_pos[phase_ids, sample_indices]
        joint_vel = self._boundary_joint_vel[phase_ids, sample_indices]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.clear_state(env_ids=env_ids)

        data = self._env.sim.data
        mocap_pos = self._boundary_mocap_pos[phase_ids, sample_indices]
        mocap_quat = self._boundary_mocap_quat[phase_ids, sample_indices]
        attached = self._boundary_attached[phase_ids, sample_indices]
        data.mocap_pos[env_ids[:, None], self._anchor_mocap_ids[None, :]] = mocap_pos
        data.mocap_quat[env_ids[:, None], self._anchor_mocap_ids[None, :]] = mocap_quat
        for hand_id in (0, 1):
            weld_id = int(self._weld_ids[hand_id].item())
            data.eq_active[env_ids, weld_id] = attached[:, hand_id]

        self.attached[env_ids] = attached
        self.grip_strength[env_ids] = attached.float()
        self.held_rung[env_ids] = self._boundary_held_rung[phase_ids, sample_indices]
        self.foot_rung[env_ids] = self._boundary_foot_rung[phase_ids, sample_indices]
        self.initialized[env_ids] = True
        self.finished[env_ids] = False
        self.phase_frozen[env_ids] = False
        self._pending_start_pose_init[env_ids] = False
        self._pending_boundary_state_init[env_ids] = True
        self._started_from_boundary[env_ids] = True
        for hand_id in (0, 1):
            self._restore_weld_parameters(env_ids, hand_id)

        for phase in LadderPhase:
            phase_env_ids = env_ids[phase_ids == int(phase)]
            self._start_phase(
                phase_env_ids,
                phase,
                capture_boundary=False,
            )
        self._cycle_start_body_height[env_ids] = self._boundary_cycle_start_body_height[
            phase_ids, sample_indices
        ]
        self._phase_start_body_height[env_ids] = self._boundary_phase_start_body_height[
            phase_ids, sample_indices
        ]

    def _boundary_bank_state_dict(self) -> dict[str, object] | None:
        if getattr(self, "_boundary_bank_capacity", 0) == 0:
            return None
        return {
            "capacity": self._boundary_bank_capacity,
            "counts": list(self._boundary_bank_counts),
            "cursors": list(self._boundary_bank_cursors),
            "root_state": self._boundary_root_state.detach().cpu().clone(),
            "joint_pos": self._boundary_joint_pos.detach().cpu().clone(),
            "joint_vel": self._boundary_joint_vel.detach().cpu().clone(),
            "mocap_pos": self._boundary_mocap_pos.detach().cpu().clone(),
            "mocap_quat": self._boundary_mocap_quat.detach().cpu().clone(),
            "attached": self._boundary_attached.detach().cpu().clone(),
            "held_rung": self._boundary_held_rung.detach().cpu().clone(),
            "foot_rung": self._boundary_foot_rung.detach().cpu().clone(),
            "cycle_start_body_height": (
                self._boundary_cycle_start_body_height.detach().cpu().clone()
            ),
            "phase_start_body_height": (
                self._boundary_phase_start_body_height.detach().cpu().clone()
            ),
        }

    def _load_boundary_bank_state_dict(self, state: object) -> None:
        if state is None or getattr(self, "_boundary_bank_capacity", 0) == 0:
            self._boundary_bank_counts = [0] * len(LadderPhase)
            self._boundary_bank_cursors = [0] * len(LadderPhase)
            return
        if not isinstance(state, dict):
            raise ValueError("Invalid ladder boundary-state bank checkpoint")
        if int(state.get("capacity", -1)) != self._boundary_bank_capacity:
            raise ValueError(
                "Ladder boundary-state bank capacity does not match the current "
                "configuration; resume with the original boundary_state_bank_size"
            )

        counts = [int(value) for value in cast(list[object], state["counts"])]
        cursors = [int(value) for value in cast(list[object], state["cursors"])]
        if len(counts) != len(LadderPhase) or len(cursors) != len(LadderPhase):
            raise ValueError("Invalid ladder boundary-state bank phase metadata")
        if any(not 0 <= value <= self._boundary_bank_capacity for value in counts):
            raise ValueError("Invalid ladder boundary-state bank count")
        if any(not 0 <= value < self._boundary_bank_capacity for value in cursors):
            raise ValueError("Invalid ladder boundary-state bank cursor")

        tensor_fields = {
            "root_state": self._boundary_root_state,
            "joint_pos": self._boundary_joint_pos,
            "joint_vel": self._boundary_joint_vel,
            "mocap_pos": self._boundary_mocap_pos,
            "mocap_quat": self._boundary_mocap_quat,
            "attached": self._boundary_attached,
            "held_rung": self._boundary_held_rung,
            "foot_rung": self._boundary_foot_rung,
            "cycle_start_body_height": self._boundary_cycle_start_body_height,
            "phase_start_body_height": self._boundary_phase_start_body_height,
        }
        for name, destination in tensor_fields.items():
            source = state.get(name)
            if (
                not isinstance(source, torch.Tensor)
                or source.shape != destination.shape
            ):
                raise ValueError(f"Invalid ladder boundary-state bank tensor {name!r}")
            destination.copy_(source.to(device=self.device, dtype=destination.dtype))
        self._boundary_bank_counts = counts
        self._boundary_bank_cursors = cursors

    def curriculum_state_dict(self) -> dict[str, object]:
        """Serialize adaptive curriculum state for a training checkpoint."""

        return {
            "version": 3,
            "unlocked_phase": self._unlocked_phase,
            "phase_start_step": self._curriculum_phase_start_step,
            "recent_outcomes": list(self._recent_curriculum_outcomes),
            "pending_outcomes": list(self._pending_curriculum_outcomes),
            "boundary_state_bank": self._boundary_bank_state_dict(),
        }

    def load_curriculum_state_dict(self, state: dict[str, object]) -> None:
        """Restore adaptive curriculum state with fail-fast validation."""

        if not self.cfg.curriculum_enabled:
            self._unlocked_phase = self._play_unlocked_phase
            self._recent_curriculum_outcomes.clear()
            self._pending_curriculum_outcomes.clear()
            return
        if state.get("version") != 3:
            raise ValueError(
                "Unsupported ladder curriculum checkpoint version: train a fresh "
                "policy for the refined G1 contact model; old boundary states "
                "and curriculum outcomes are not valid under the new collisions"
            )
        unlocked_phase = int(state["unlocked_phase"])
        if (
            not int(LadderPhase.STABILIZE)
            <= unlocked_phase
            <= int(LadderPhase.SECOND_FOOT)
        ):
            raise ValueError(f"Invalid unlocked ladder phase: {unlocked_phase}")
        phase_start_step = int(state["phase_start_step"])
        if phase_start_step < 0:
            raise ValueError("Ladder curriculum phase_start_step must be non-negative")
        recent = [int(value) for value in state.get("recent_outcomes", [])]
        pending = [int(value) for value in state.get("pending_outcomes", [])]
        if any(value not in (0, 1) for value in (*recent, *pending)):
            raise ValueError("Ladder curriculum outcomes must contain only 0 or 1")
        if len(recent) > self.cfg.curriculum_window_size:
            raise ValueError("Ladder curriculum window is larger than configured")

        self._unlocked_phase = unlocked_phase
        self._curriculum_phase_start_step = phase_start_step
        self._recent_curriculum_outcomes = deque(
            recent,
            maxlen=self.cfg.curriculum_window_size,
        )
        self._pending_curriculum_outcomes = pending
        self._load_boundary_bank_state_dict(state.get("boundary_state_bank"))

    @property
    def foot_contact(self) -> torch.Tensor:
        found = self._foot_contact_sensor.data.found
        if found is None:
            raise RuntimeError(
                f"Contact sensor {self.cfg.foot_contact_sensor_name!r} must "
                "provide the 'found' field"
            )
        return found[:, self._foot_contact_indices] > 0

    @property
    def hand_target_pos_w(self) -> torch.Tensor:
        """Per-hand target positions with shape ``(num_envs, 2, 3)``."""

        target_rungs = self.hand_target_rung_indices
        return torch.stack(
            tuple(
                self._closest_points_on_rungs(
                    self.hand_pos_w[:, hand_id],
                    target_rungs[:, hand_id],
                )
                for hand_id in (0, 1)
            ),
            dim=1,
        )

    @property
    def target_pos_w(self) -> torch.Tensor:
        """Closest allowed point on the active target capsule axis."""

        return self._closest_points_on_rungs(
            self.active_hand_pos_w,
            self.target_rung,
        )

    @property
    def max_foot_rung(self) -> int:
        return max(self.num_rungs - 1 - self.cfg.hand_foot_lead_rungs, 0)

    @property
    def foot_target_rung_indices(self) -> torch.Tensor:
        target_rungs = torch.clamp(self.foot_rung, min=0)
        target_rungs[self._all_env_ids, self.active_foot] = self.target_foot_rung
        return target_rungs

    @property
    def foot_target_pos_w(self) -> torch.Tensor:
        """Per-foot support targets just above the physical rung surfaces."""

        target_rungs = self.foot_target_rung_indices
        return torch.stack(
            tuple(
                self._points_above_rungs(
                    self.foot_pos_w[:, foot_id],
                    target_rungs[:, foot_id],
                )
                for foot_id in (0, 1)
            ),
            dim=1,
        )

    @property
    def active_foot_target_pos_w(self) -> torch.Tensor:
        return self.foot_target_pos_w[self._all_env_ids, self.active_foot]

    @property
    def held_foot_target_pos_w(self) -> torch.Tensor:
        held_rungs = torch.clamp(self.foot_rung, min=0)
        return torch.stack(
            tuple(
                self._points_above_rungs(
                    self.foot_pos_w[:, foot_id],
                    held_rungs[:, foot_id],
                )
                for foot_id in (0, 1)
            ),
            dim=1,
        )

    @property
    def foot_support(self) -> torch.Tensor:
        distance = torch.linalg.vector_norm(
            self.foot_pos_w - self.held_foot_target_pos_w,
            dim=-1,
        )
        return (
            (self.foot_rung >= 0)
            & self.foot_contact
            & (distance <= self.cfg.foot_support_distance)
        )

    @property
    def torso_pos_w(self) -> torch.Tensor:
        return self._env.sim.data.xpos[:, self._torso_body_id]

    @property
    def torso_com_pos_w(self) -> torch.Tensor:
        """World position of the torso body's center of mass."""

        return self._env.sim.data.xipos[:, self._torso_body_id]

    @property
    def pelvis_com_pos_w(self) -> torch.Tensor:
        """World position of the pelvis center of mass."""

        return self._env.sim.data.xipos[:, self._pelvis_body_id]

    @property
    def body_height(self) -> torch.Tensor:
        """Whole-body climbing height represented by pelvis and torso COMs."""

        return 0.5 * (self.pelvis_com_pos_w[:, 2] + self.torso_com_pos_w[:, 2])

    @property
    def torso_rotation_w(self) -> torch.Tensor:
        """World orientation matrices of ``torso_link``."""

        return _as_rotation_matrix(self._env.sim.data.xmat[:, self._torso_body_id])

    @property
    def torso_ang_vel_w(self) -> torch.Tensor:
        """World angular velocity of ``torso_link``."""

        return self.robot.data.body_link_ang_vel_w[:, self._torso_link_index]

    @property
    def pelvis_ang_vel_w(self) -> torch.Tensor:
        """World angular velocity of the pelvis/root body."""

        return self.robot.data.body_link_ang_vel_w[:, self._pelvis_link_index]

    def _vectors_to_torso(self, vectors_w: torch.Tensor) -> torch.Tensor:
        """Rotate batched world-frame vectors into the current torso frame."""

        world_to_torso = self.torso_rotation_w.transpose(-1, -2)
        while world_to_torso.ndim < vectors_w.ndim + 1:
            world_to_torso = world_to_torso.unsqueeze(1)
        return torch.matmul(world_to_torso, vectors_w.unsqueeze(-1)).squeeze(-1)

    @property
    def support_centroid_w(self) -> torch.Tensor:
        """Grip/contact-weighted centroid of the currently load-bearing supports."""

        initialized = self.initialized[:, None]
        hand_weights = torch.where(
            initialized,
            self.grip_strength * self.attached.float(),
            torch.ones_like(self.grip_strength),
        )
        foot_weights = torch.where(
            initialized,
            self.foot_support.float(),
            torch.ones_like(self.grip_strength),
        )
        weighted_sum = (self.hand_pos_w * hand_weights.unsqueeze(-1)).sum(dim=1)
        weighted_sum += (self.foot_pos_w * foot_weights.unsqueeze(-1)).sum(dim=1)
        total_weight = hand_weights.sum(dim=1) + foot_weights.sum(dim=1)
        return weighted_sum / total_weight.clamp_min(1.0).unsqueeze(-1)

    @property
    def torso_support_offset_w(self) -> torch.Tensor:
        """Torso-COM offset from the continuous load-bearing support centroid."""

        return self.torso_com_pos_w - self.support_centroid_w

    @property
    def torso_orientation_error(self) -> torch.Tensor:
        """Full 3-D torso rotation error from the nominal climbing pose, in radians."""

        relative = torch.matmul(
            self._reference_torso_rotation_w.transpose(-1, -2),
            self.torso_rotation_w,
        )
        cosine = torch.clamp(
            0.5 * (relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2] - 1.0),
            min=-1.0,
            max=1.0,
        )
        return torch.acos(cosine)

    @property
    def torso_support_offset_error(self) -> torch.Tensor:
        """Distance from the nominal torso placement inside the support geometry."""

        return torch.linalg.vector_norm(
            self.torso_support_offset_w - self._reference_torso_support_offset_w,
            dim=-1,
        )

    @property
    def phase_support_constraints_satisfied(self) -> torch.Tensor:
        """Whether the non-moving supports and torso pose satisfy the active phase."""

        posture_valid = (
            self.torso_orientation_error <= self.cfg.max_phase_torso_orientation_error
        ) & (self.torso_support_offset_error <= self.cfg.max_phase_support_offset_error)
        return self.phase_required_supports_satisfied & posture_valid

    @property
    def phase_required_supports_satisfied(self) -> torch.Tensor:
        """Whether the physical supports required by the active phase are present."""

        four_supports = self.attached.all(dim=1) & self.foot_support.all(dim=1)

        support_hand = 1 - self.active_hand
        support_hand_attached = self.attached.gather(
            1,
            support_hand[:, None],
        ).squeeze(1)
        hand_phase_supports = support_hand_attached & self.foot_support.all(dim=1)

        support_foot = 1 - self.active_foot
        support_foot_contact = self.foot_support.gather(
            1,
            support_foot[:, None],
        ).squeeze(1)
        foot_phase_supports = self.attached.all(dim=1) & support_foot_contact

        support_valid = torch.where(
            self.phase == int(LadderPhase.STABILIZE),
            four_supports,
            torch.where(self.is_hand_phase, hand_phase_supports, foot_phase_supports),
        )
        return self.initialized & ~self.finished & support_valid

    @property
    def phase_completion_stable(self) -> torch.Tensor:
        """Require valid supports, COM placement, and low speed for completion."""

        torso_speed = torch.linalg.vector_norm(self.torso_com_vel_w, dim=-1)
        joint_speed_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_vel), dim=1)
        )
        return (
            self.phase_required_supports_satisfied
            & (
                self.torso_support_offset_error
                <= self.cfg.max_phase_support_offset_error
            )
            & (torso_speed <= self.cfg.max_phase_completion_torso_speed)
            & (joint_speed_rms <= self.cfg.max_phase_completion_joint_speed)
        )

    def _stabilization_conditions(self) -> dict[str, torch.Tensor]:
        """Return the exact gates used by the stabilization transition."""

        torso_speed = torch.linalg.vector_norm(self.torso_com_vel_w, dim=-1)
        joint_speed_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_vel), dim=1)
        )
        torso_angular_speed = torch.linalg.vector_norm(
            self.torso_ang_vel_w,
            dim=-1,
        )
        pelvis_angular_speed = torch.linalg.vector_norm(
            self.pelvis_ang_vel_w,
            dim=-1,
        )
        body_angular_speed = torch.maximum(
            torso_angular_speed,
            pelvis_angular_speed,
        )
        waist_joint_speed = torch.amax(
            torch.abs(self.robot.data.joint_vel[:, self._waist_joint_ids]),
            dim=1,
        )
        hands_attached = self.attached.all(dim=1)
        feet_supported = self.foot_support.all(dim=1)
        torso_speed_valid = torso_speed <= self.cfg.max_stabilization_torso_speed
        joint_speed_valid = (
            joint_speed_rms <= self.cfg.max_stabilization_joint_speed
        )
        angular_speed_valid = (
            body_angular_speed <= self.cfg.max_stabilization_body_angular_speed
        )
        waist_speed_valid = (
            waist_joint_speed <= self.cfg.max_stabilization_waist_joint_speed
        )
        support_offset_valid = (
            self.torso_support_offset_error
            <= self.cfg.max_stabilization_support_offset_error
        )
        stable = (
            hands_attached
            & feet_supported
            & torso_speed_valid
            & joint_speed_valid
            & angular_speed_valid
            & waist_speed_valid
            & support_offset_valid
        )
        return {
            "hands_attached": hands_attached,
            "feet_supported": feet_supported,
            "torso_speed": torso_speed,
            "torso_speed_valid": torso_speed_valid,
            "joint_speed_rms": joint_speed_rms,
            "joint_speed_valid": joint_speed_valid,
            "torso_angular_speed": torso_angular_speed,
            "pelvis_angular_speed": pelvis_angular_speed,
            "angular_speed_valid": angular_speed_valid,
            "waist_joint_speed": waist_joint_speed,
            "waist_speed_valid": waist_speed_valid,
            "support_offset_valid": support_offset_valid,
            "stable": stable,
        }

    @property
    def stabilization_conditions_satisfied(self) -> torch.Tensor:
        """Whether all transition gates except the consecutive dwell are valid."""

        return self._stabilization_conditions()["stable"]

    @property
    def rung_endpoints_torso(self) -> torch.Tensor:
        """Return all finite rung endpoints in the torso frame.

        The fixed A-frame topology keeps every training rung active, so the
        seventh feature is an explicit validity bit equal to one.  Keeping the
        ``(endpoint_a, endpoint_b, valid)`` contract separate from the phase
        command gives ladder geometry its own encoder and leaves a stable
        insertion point for later layout randomization.

        Returns:
            Tensor shaped ``(num_envs, num_rungs, 7)``.
        """

        data = self._env.sim.data
        centers = data.site_xpos[:, self._rung_site_ids]
        rung_matrices = _as_rotation_matrix(data.site_xmat[:, self._rung_site_ids])
        rung_axes_w = rung_matrices[..., :, 2]
        half_lengths = self._rung_half_lengths.view(1, -1, 1)
        endpoint_a_w = centers - rung_axes_w * half_lengths
        endpoint_b_w = centers + rung_axes_w * half_lengths

        torso_pos_w = data.xpos[:, self._torso_body_id]
        torso_matrix_w = _as_rotation_matrix(data.xmat[:, self._torso_body_id])
        world_to_torso = torso_matrix_w.transpose(-1, -2).unsqueeze(1)

        def to_torso(points_w: torch.Tensor) -> torch.Tensor:
            relative_w = points_w - torso_pos_w.unsqueeze(1)
            return torch.matmul(world_to_torso, relative_w.unsqueeze(-1)).squeeze(-1)

        valid = torch.ones(
            (*centers.shape[:2], 1),
            dtype=centers.dtype,
            device=centers.device,
        )
        return torch.cat(
            (to_torso(endpoint_a_w), to_torso(endpoint_b_w), valid),
            dim=-1,
        )

    @property
    def hand_target_rung_indices(self) -> torch.Tensor:
        """Desired rung index for each hand, including the stationary support hand."""

        start_rungs = torch.full_like(self.held_rung, self.cfg.start_rung)
        target_rungs = torch.where(
            self.initialized[:, None], self.held_rung, start_rungs
        )
        target_rungs[self._all_env_ids, self.active_hand] = self.target_rung
        return torch.where(target_rungs >= 0, target_rungs, start_rungs)

    @property
    def rung_tokens_torso(self) -> torch.Tensor:
        """Return 15D torso-frame rung tokens with target and support markers.

        Each token contains two finite rung endpoints (6), a validity bit (1),
        left/right hand target bits (2), left/right foot target bits (2),
        left/right continuous held-hand support strengths (2), and left/right
        assigned-foot support bits (2).  The moving hand marker fades with its
        weld impedance instead of disappearing discontinuously at detach time.
        """

        endpoints = self.rung_endpoints_torso
        rung_ids = torch.arange(
            self.num_rungs,
            dtype=torch.long,
            device=self.device,
        ).view(1, -1, 1)

        def markers(indices: torch.Tensor) -> torch.Tensor:
            return (rung_ids == indices[:, None, :]).to(endpoints.dtype)

        return torch.cat(
            (
                endpoints,
                markers(self.hand_target_rung_indices),
                markers(self.foot_target_rung_indices),
                markers(self.held_rung) * self.grip_strength[:, None, :],
                markers(self.foot_rung),
            ),
            dim=-1,
        )

    @property
    def top_height(self) -> torch.Tensor:
        top_site_id = int(self._rung_site_ids[-1].item())
        return self._env.sim.data.site_xpos[:, top_site_id, 2]

    @property
    def rung_progress(self) -> torch.Tensor:
        highest_held = torch.max(self.held_rung, dim=1).values
        denominator = max(self.num_rungs - 1 - self.cfg.start_rung, 1)
        return torch.clamp(
            (highest_held - self.cfg.start_rung).float() / denominator,
            min=0.0,
            max=1.0,
        )

    @property
    def foot_progress(self) -> torch.Tensor:
        lowest_foot_rung = torch.min(self.foot_rung, dim=1).values
        denominator = self.max_foot_rung + 1
        return torch.clamp(
            (lowest_foot_rung + 1).float() / denominator,
            min=0.0,
            max=1.0,
        )

    def _update_metrics(self) -> None:
        self.metrics["target_distance"][:] = torch.linalg.vector_norm(
            self.active_hand_pos_w - self.target_pos_w,
            dim=-1,
        )
        self.metrics["rung_progress"][:] = self.rung_progress
        self.metrics["attached_hands"][:] = self.attached.sum(dim=1).float()
        self.metrics["foot_target_distance"][:] = torch.linalg.vector_norm(
            self.active_foot_pos_w - self.active_foot_target_pos_w,
            dim=-1,
        )
        self.metrics["foot_progress"][:] = self.foot_progress
        self.metrics["supported_feet"][:] = self.foot_support.sum(dim=1).float()
        self.metrics["torso_orientation_error"][:] = self.torso_orientation_error
        self.metrics["torso_support_offset_error"][:] = self.torso_support_offset_error
        self.metrics["active_hand_release_progress"][:] = torch.where(
            self.is_hand_phase,
            1.0 - self.active_grip_strength,
            0.0,
        )
        self.metrics["pre_release_age"][:] = self._release_age_count.float()
        stabilization_active = self.is_stabilization_phase
        stabilization_conditions = self._stabilization_conditions()
        zeros = torch.zeros_like(self._phase_dwell_count, dtype=torch.float32)
        for metric_name, condition_name in (
            ("stabilization_hands_attached", "hands_attached"),
            ("stabilization_feet_supported", "feet_supported"),
            ("stabilization_torso_speed_valid", "torso_speed_valid"),
            ("stabilization_joint_speed_valid", "joint_speed_valid"),
            ("stabilization_angular_speed_valid", "angular_speed_valid"),
            ("stabilization_waist_speed_valid", "waist_speed_valid"),
            ("stabilization_support_offset_valid", "support_offset_valid"),
        ):
            self.metrics[metric_name][:] = torch.where(
                stabilization_active,
                stabilization_conditions[condition_name].float(),
                zeros,
            )
        for metric_name, condition_name in (
            ("stabilization_torso_speed", "torso_speed"),
            ("stabilization_joint_speed_rms", "joint_speed_rms"),
            ("stabilization_torso_angular_speed", "torso_angular_speed"),
            ("stabilization_pelvis_angular_speed", "pelvis_angular_speed"),
            ("stabilization_waist_joint_speed", "waist_joint_speed"),
        ):
            self.metrics[metric_name][:] = torch.where(
                stabilization_active,
                stabilization_conditions[condition_name],
                zeros,
            )
        self.metrics["stabilization_gate_valid"][:] = torch.where(
            stabilization_active,
            stabilization_conditions["stable"].float(),
            zeros,
        )
        self.metrics["stabilization_dwell_progress"][:] = torch.where(
            stabilization_active,
            torch.clamp(
                self._phase_dwell_count.float()
                / self._stabilization_dwell_target.clamp_min(1).float(),
                max=1.0,
            ),
            zeros,
        )
        release_active = self.is_pre_release
        release_conditions = self._release_stability_conditions()
        zeros = torch.zeros_like(self._release_age_count, dtype=torch.float32)
        for metric_name, condition_name in (
            ("release_hands_attached", "hands_attached"),
            ("release_feet_supported", "feet_supported"),
            ("release_torso_speed_valid", "torso_speed_valid"),
            ("release_orientation_valid", "orientation_valid"),
            ("release_support_offset_valid", "support_offset_valid"),
        ):
            self.metrics[metric_name][:] = torch.where(
                release_active,
                release_conditions[condition_name].float(),
                zeros,
            )
        self.metrics["release_torso_speed"][:] = torch.where(
            release_active,
            release_conditions["torso_speed"],
            zeros,
        )
        self.metrics["release_gate_valid"][:] = torch.where(
            release_active,
            release_conditions["stable"].float(),
            zeros,
        )
        self.metrics["release_preload_progress"][:] = torch.where(
            release_active,
            torch.clamp(
                self._release_preload_count.float()
                / float(self.cfg.release_preload_dwell_steps),
                max=1.0,
            ),
            zeros,
        )
        self.metrics["release_ramp_progress"][:] = torch.where(
            release_active,
            torch.clamp(
                self._release_ramp_count.float() / float(self.cfg.release_ramp_steps),
                max=1.0,
            ),
            zeros,
        )
        self.metrics["release_final_dwell_progress"][:] = torch.where(
            release_active,
            torch.clamp(
                self._release_final_count.float()
                / float(self.cfg.release_final_dwell_steps),
                max=1.0,
            ),
            zeros,
        )
        self.metrics["phase"][:] = self.phase.float()
        self.metrics["unlocked_phase"][:] = float(self.max_unlocked_phase)
        self.metrics["curriculum_success_rate"][:] = self.curriculum_success_rate
        self.metrics["curriculum_window_fill"][:] = self.curriculum_window_fill
        self.metrics["curriculum_phase_steps"][:] = float(self.curriculum_phase_steps)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Reset FSM and disable both welds for the selected environments."""

        if env_ids.numel() == 0:
            return

        self._record_episode_outcomes(env_ids)
        first_hand = 0 if self.cfg.first_moving_hand == "left" else 1
        self.active_hand[env_ids] = first_hand
        self.target_rung[env_ids] = self.cfg.start_rung
        self.attached[env_ids] = False
        self.grip_strength[env_ids] = 0.0
        self.held_rung[env_ids] = -1
        self.initialized[env_ids] = False
        self.finished[env_ids] = False
        self.just_initialized[env_ids] = False
        self.just_advanced[env_ids] = False
        self.just_foot_advanced[env_ids] = False
        self.just_stabilized[env_ids] = False
        self.just_cycle_completed[env_ids] = False
        self.curriculum_stage_complete[env_ids] = False
        self.pre_release_stalled[env_ids] = False
        self.phase_frozen[env_ids] = False
        self._started_from_boundary[env_ids] = False
        self._pending_boundary_state_init[env_ids] = False
        self._pending_start_pose_init[env_ids] = self.cfg.initialize_on_reset
        self.phase[env_ids] = int(LadderPhase.STABILIZE)
        self._phase_dwell_count[env_ids] = 0
        self._release_active[env_ids] = False
        self._release_preload_count[env_ids] = 0
        self._release_ramp_count[env_ids] = 0
        self._release_final_count[env_ids] = 0
        self._release_age_count[env_ids] = 0
        self._sample_stabilization_dwell_target(env_ids)

        first_foot = 0 if self.cfg.first_moving_foot == "left" else 1
        self.active_foot[env_ids] = first_foot
        if self.cfg.initialize_on_reset:
            self.foot_rung[env_ids] = self.cfg.initial_foot_rung
            self.target_foot_rung[env_ids] = self.cfg.initial_foot_rung
        else:
            self.target_foot_rung[env_ids] = 0
            self.foot_rung[env_ids] = -1
        for hand_id in (0, 1):
            weld_id = int(self._weld_ids[hand_id].item())
            self._env.sim.data.eq_active[env_ids, weld_id] = False
            self._restore_weld_parameters(env_ids, hand_id)

        hand_pos = self.hand_pos_w[env_ids]
        self._previous_hand_pos_w[env_ids] = hand_pos
        self._hand_vel_w[env_ids] = 0.0
        foot_pos = self.foot_pos_w[env_ids]
        self._previous_foot_pos_w[env_ids] = foot_pos
        self._foot_vel_w[env_ids] = 0.0
        torso_com_pos = self.torso_com_pos_w[env_ids]
        self._previous_torso_com_pos_w[env_ids] = torso_com_pos
        self._torso_com_vel_w[env_ids] = 0.0
        body_height = self.body_height[env_ids]
        self.start_height[env_ids] = body_height
        self._cycle_start_body_height[env_ids] = body_height
        self._phase_start_body_height[env_ids] = body_height
        self.episode_max_body_height[env_ids] = body_height
        self._episode_active[env_ids] = True
        self._sample_boundary_resets(env_ids)

    def _record_episode_outcomes(self, env_ids: torch.Tensor) -> None:
        """Queue terminal outcomes for synchronization by the PPO runner."""

        if not self.cfg.curriculum_enabled:
            return
        full_prefix_episode = (
            self._episode_active[env_ids] & ~self._started_from_boundary[env_ids]
        )
        active_ids = env_ids[full_prefix_episode]
        if active_ids.numel() == 0:
            return
        termination_manager = self._env.termination_manager
        done = termination_manager.dones[active_ids]
        completed_ids = active_ids[done]
        if completed_ids.numel() == 0:
            return
        success_term = (
            "curriculum_stage_complete"
            if self._unlocked_phase < int(LadderPhase.SECOND_FOOT)
            else "success"
        )
        successes = termination_manager.get_term(success_term)[completed_ids]
        self._pending_curriculum_outcomes.extend(
            int(value)
            for value in successes.to(device="cpu", dtype=torch.int8).tolist()
        )

    def _update_command(self) -> None:
        """Update velocities and advance one ordered phase per control step."""

        self.just_initialized.zero_()
        self.just_advanced.zero_()
        self.just_foot_advanced.zero_()
        self.just_stabilized.zero_()
        self.just_cycle_completed.zero_()
        self.curriculum_stage_complete.zero_()
        self.pre_release_stalled.zero_()

        hand_pos = self.hand_pos_w
        reset_initialization = (
            self._pending_start_pose_init | self._pending_boundary_state_init
        )
        reset_body_height = self.body_height[reset_initialization]
        self.start_height[reset_initialization] = reset_body_height
        self.episode_max_body_height[reset_initialization] = reset_body_height
        start_pose_init = self._pending_start_pose_init
        start_pose_body_height = self.body_height[start_pose_init]
        self._cycle_start_body_height[start_pose_init] = start_pose_body_height
        self._phase_start_body_height[start_pose_init] = start_pose_body_height
        self._previous_hand_pos_w[reset_initialization] = hand_pos[reset_initialization]
        self._hand_vel_w[:] = (hand_pos - self._previous_hand_pos_w) / self._env.step_dt
        self._previous_hand_pos_w[:] = hand_pos
        foot_pos = self.foot_pos_w
        self._previous_foot_pos_w[reset_initialization] = foot_pos[reset_initialization]
        self._foot_vel_w[:] = (foot_pos - self._previous_foot_pos_w) / self._env.step_dt
        self._previous_foot_pos_w[:] = foot_pos
        torso_com_pos = self.torso_com_pos_w
        self._previous_torso_com_pos_w[reset_initialization] = torso_com_pos[
            reset_initialization
        ]
        self._torso_com_vel_w[:] = (
            torso_com_pos - self._previous_torso_com_pos_w
        ) / self._env.step_dt
        self._previous_torso_com_pos_w[:] = torso_com_pos
        self._pending_boundary_state_init.zero_()

        # Reconcile state if another component or a reset disabled a weld.
        for hand_id in (0, 1):
            weld_id = int(self._weld_ids[hand_id].item())
            weld_active = self._env.sim.data.eq_active[:, weld_id] != 0
            lost = self.attached[:, hand_id] & ~weld_active
            self.attached[lost, hand_id] = False
            self.grip_strength[lost, hand_id] = 0.0
            self.held_rung[lost, hand_id] = -1
            lost_ids = torch.where(lost & (self.active_hand == hand_id))[0]
            self._reset_release_state(lost_ids)

        self._initialize_from_start_pose()
        self._try_initialize()

        # A snapshot prevents an environment from cascading through multiple
        # phases in one control step after a transition changes ``self.phase``.
        phase_at_start = self.phase.clone()
        self._advance_stabilization_phase(phase_at_start)
        self._advance_hand_phase(phase_at_start)
        self._advance_foot_phase(phase_at_start)

    def _initialize_from_start_pose(self) -> None:
        """Attach the reset climbing pose without an approach phase."""

        env_ids = torch.where(self._pending_start_pose_init)[0]
        if env_ids.numel() == 0:
            return
        self._pending_start_pose_init[env_ids] = False
        rung_indices = torch.full(
            (env_ids.numel(),),
            self.cfg.start_rung,
            dtype=torch.long,
            device=self.device,
        )
        self._attach(env_ids, 0, rung_indices)
        self._attach(env_ids, 1, rung_indices)
        self.initialized[env_ids] = True
        self.just_initialized[env_ids] = True
        self.phase[env_ids] = int(LadderPhase.STABILIZE)
        self._phase_dwell_count[env_ids] = 0
        self.target_rung[env_ids] = self.cfg.start_rung
        self.target_foot_rung[env_ids] = self.cfg.initial_foot_rung

    def _try_initialize(self) -> None:
        """Attach both hands to the start rung before stabilization."""

        pending = ~self.initialized
        if not torch.any(pending):
            return

        start_rungs = torch.full(
            (self.num_envs,),
            self.cfg.start_rung,
            dtype=torch.long,
            device=self.device,
        )
        hand_pos = self.hand_pos_w
        left_target = self._closest_points_on_rungs(hand_pos[:, 0], start_rungs)
        right_target = self._closest_points_on_rungs(hand_pos[:, 1], start_rungs)
        distances = torch.stack(
            (
                torch.linalg.vector_norm(hand_pos[:, 0] - left_target, dim=-1),
                torch.linalg.vector_norm(hand_pos[:, 1] - right_target, dim=-1),
            ),
            dim=1,
        )
        speeds = torch.linalg.vector_norm(self._hand_vel_w, dim=-1)
        ready = pending & torch.all(
            (distances <= self.cfg.attach_distance)
            & (speeds <= self.cfg.max_attach_speed),
            dim=1,
        )
        env_ids = torch.where(ready)[0]
        if env_ids.numel() == 0:
            return

        self._complete_initialization(env_ids)

    def _complete_initialization(self, env_ids: torch.Tensor) -> None:
        """Attach both start-rung hands and enter four-point stabilization."""

        rung_indices = torch.full(
            (env_ids.numel(),),
            self.cfg.start_rung,
            dtype=torch.long,
            device=self.device,
        )
        self._attach(env_ids, 0, rung_indices)
        self._attach(env_ids, 1, rung_indices)
        self.initialized[env_ids] = True
        self.just_initialized[env_ids] = True
        self.phase[env_ids] = int(LadderPhase.STABILIZE)
        self._phase_dwell_count[env_ids] = 0
        self.target_rung[env_ids] = self.cfg.start_rung
        self.target_foot_rung[env_ids] = torch.clamp(
            torch.max(self.foot_rung[env_ids], dim=1).values,
            min=0,
            max=self.max_foot_rung,
        )

    def _update_phase_dwell(
        self,
        phase_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        self._phase_dwell_count[phase_mask] = torch.where(
            condition[phase_mask],
            self._phase_dwell_count[phase_mask] + 1,
            0,
        )

    def _advance_stabilization_phase(self, phase_at_start: torch.Tensor) -> None:
        """Require quiet four-point support before either hand is released."""

        phase_mask = (
            (phase_at_start == int(LadderPhase.STABILIZE))
            & self.initialized
            & ~self.finished
            & ~self.phase_frozen
        )
        if not torch.any(phase_mask):
            return

        stable = self.stabilization_conditions_satisfied
        self._update_phase_dwell(phase_mask, stable)
        ready = phase_mask & (
            self._phase_dwell_count >= self._stabilization_dwell_target
        )
        env_ids = torch.where(ready)[0]
        if env_ids.numel() == 0:
            return
        self.just_stabilized[env_ids] = True
        self._transition_or_finish_curriculum(env_ids, LadderPhase.FIRST_HAND)

    def _advance_hand_phase(self, phase_at_start: torch.Tensor) -> None:
        """Unload, release, and reattach one hand without a discontinuous support step."""

        phase_mask = (
            (
                (phase_at_start == int(LadderPhase.FIRST_HAND))
                | (phase_at_start == int(LadderPhase.SECOND_HAND))
            )
            & self.initialized
            & ~self.finished
            & ~self.phase_frozen
        )
        if not torch.any(phase_mask):
            return

        self._advance_release_ramp(phase_mask)

        distance = torch.linalg.vector_norm(
            self.active_hand_pos_w - self.target_pos_w,
            dim=-1,
        )
        speed = torch.linalg.vector_norm(self.active_hand_vel_w, dim=-1)
        moving_attached = self.attached.gather(
            1,
            self.active_hand[:, None],
        ).squeeze(1)
        support_hand = 1 - self.active_hand
        support_hand_attached = self.attached.gather(
            1,
            support_hand[:, None],
        ).squeeze(1)
        reached = (
            ~moving_attached
            & support_hand_attached
            & self.foot_support.all(dim=1)
            & self.phase_completion_stable
            & (distance <= self.cfg.attach_distance)
            & (speed <= self.cfg.max_attach_speed)
        )
        self._update_phase_dwell(phase_mask, reached)
        complete = phase_mask & (
            self._phase_dwell_count >= self.cfg.hand_target_dwell_steps
        )
        env_ids = torch.where(complete)[0]
        if env_ids.numel() == 0:
            return

        moving_hands = self.active_hand[env_ids].clone()
        for hand_id in (0, 1):
            selected = env_ids[moving_hands == hand_id]
            if selected.numel() > 0:
                self._attach(selected, hand_id, self.target_rung[selected])
        self.just_advanced[env_ids] = True

        first_complete = env_ids[phase_at_start[env_ids] == int(LadderPhase.FIRST_HAND)]
        second_complete = env_ids[
            phase_at_start[env_ids] == int(LadderPhase.SECOND_HAND)
        ]
        self._transition_or_finish_curriculum(
            first_complete,
            LadderPhase.SECOND_HAND,
        )
        self._transition_or_finish_curriculum(
            second_complete,
            LadderPhase.FIRST_FOOT,
        )

    def _advance_foot_phase(self, phase_at_start: torch.Tensor) -> None:
        """Accept a foot step only after consecutive real rung contacts."""

        phase_mask = (
            (
                (phase_at_start == int(LadderPhase.FIRST_FOOT))
                | (phase_at_start == int(LadderPhase.SECOND_FOOT))
            )
            & self.initialized
            & self.attached.all(dim=1)
            & ~self.finished
            & ~self.phase_frozen
        )
        if not torch.any(phase_mask):
            return

        distance = torch.linalg.vector_norm(
            self.active_foot_pos_w - self.active_foot_target_pos_w,
            dim=-1,
        )
        speed = torch.linalg.vector_norm(self.active_foot_vel_w, dim=-1)
        active_contact = self.foot_contact.gather(
            1,
            self.active_foot[:, None],
        ).squeeze(1)
        support_foot = 1 - self.active_foot
        support_foot_contact = self.foot_support.gather(
            1,
            support_foot[:, None],
        ).squeeze(1)
        first_foot_height_valid = (
            self.body_height
            >= self._phase_start_body_height - self.cfg.first_foot_max_body_drop
        )
        second_foot_height_valid = (
            self.body_height
            >= self._cycle_start_body_height + self.cfg.cycle_min_body_ascent
        )
        height_valid = torch.where(
            phase_at_start == int(LadderPhase.FIRST_FOOT),
            first_foot_height_valid,
            second_foot_height_valid,
        )
        reached = (
            active_contact
            & support_foot_contact
            & self.phase_completion_stable
            & height_valid
            & (distance <= self.cfg.foot_reach_distance)
            & (speed <= self.cfg.max_foot_speed)
        )
        self._update_phase_dwell(phase_mask, reached)
        complete = phase_mask & (
            self._phase_dwell_count >= self.cfg.foot_target_dwell_steps
        )
        env_ids = torch.where(complete)[0]
        if env_ids.numel() == 0:
            return

        moving_feet = self.active_foot[env_ids].clone()
        self.foot_rung[env_ids, moving_feet] = self.target_foot_rung[env_ids]
        self.just_foot_advanced[env_ids] = True

        first_complete = env_ids[phase_at_start[env_ids] == int(LadderPhase.FIRST_FOOT)]
        second_complete = env_ids[
            phase_at_start[env_ids] == int(LadderPhase.SECOND_FOOT)
        ]
        self._transition_or_finish_curriculum(
            first_complete,
            LadderPhase.SECOND_FOOT,
        )
        if second_complete.numel() == 0:
            return

        self.just_cycle_completed[second_complete] = True
        if self.cfg.freeze_at_max_unlocked_phase and self.max_unlocked_phase == int(
            LadderPhase.SECOND_FOOT
        ):
            self.phase_frozen[second_complete] = True
            return
        hands_at_top = (
            torch.min(self.held_rung[second_complete], dim=1).values
            >= self.num_rungs - 1
        )
        feet_at_top = (
            torch.min(self.foot_rung[second_complete], dim=1).values
            >= self.max_foot_rung
        )
        finished = hands_at_top & feet_at_top
        self.finished[second_complete[finished]] = True
        self._start_phase(
            second_complete[~finished],
            LadderPhase.STABILIZE,
        )

    def _transition_or_finish_curriculum(
        self,
        env_ids: torch.Tensor,
        next_phase: LadderPhase,
    ) -> None:
        if env_ids.numel() == 0:
            return
        if int(next_phase) > self.max_unlocked_phase:
            self._capture_boundary_states(env_ids, next_phase)
            if self.cfg.freeze_at_max_unlocked_phase:
                self.phase_frozen[env_ids] = True
                return
            self.curriculum_stage_complete[env_ids] = True
            return
        self._start_phase(env_ids, next_phase)

    def _start_phase(
        self,
        env_ids: torch.Tensor,
        phase: LadderPhase,
        *,
        capture_boundary: bool = True,
    ) -> None:
        """Configure the single limb allowed to move in ``phase``."""

        if env_ids.numel() == 0:
            return
        if capture_boundary:
            self._capture_boundary_states(env_ids, phase)
        self.phase[env_ids] = int(phase)
        self._phase_dwell_count[env_ids] = 0
        self._phase_start_body_height[env_ids] = self.body_height[env_ids]

        first_hand = 0 if self.cfg.first_moving_hand == "left" else 1
        first_foot = 0 if self.cfg.first_moving_foot == "left" else 1
        if phase == LadderPhase.STABILIZE:
            self._cycle_start_body_height[env_ids] = self.body_height[env_ids]
            self._reset_release_state(env_ids)
            self._sample_stabilization_dwell_target(env_ids)
            self.target_rung[env_ids] = torch.max(self.held_rung[env_ids], dim=1).values
            self.target_foot_rung[env_ids] = torch.max(
                self.foot_rung[env_ids], dim=1
            ).values
            return
        if phase == LadderPhase.FIRST_HAND:
            self.active_hand[env_ids] = first_hand
            self.target_rung[env_ids] = torch.clamp(
                torch.max(self.held_rung[env_ids], dim=1).values + 1,
                max=self.num_rungs - 1,
            )
            self._begin_release(env_ids, first_hand)
            return
        if phase == LadderPhase.SECOND_HAND:
            second_hand = 1 - first_hand
            self.active_hand[env_ids] = second_hand
            self.target_rung[env_ids] = torch.max(self.held_rung[env_ids], dim=1).values
            self._begin_release(env_ids, second_hand)
            return
        if phase == LadderPhase.FIRST_FOOT:
            self._reset_release_state(env_ids)
            self.active_foot[env_ids] = first_foot
            self.target_foot_rung[env_ids] = torch.clamp(
                torch.max(self.foot_rung[env_ids], dim=1).values + 1,
                max=self.max_foot_rung,
            )
            return

        self._reset_release_state(env_ids)
        second_foot = 1 - first_foot
        self.active_foot[env_ids] = second_foot
        self.target_foot_rung[env_ids] = torch.max(
            self.foot_rung[env_ids], dim=1
        ).values

    def _sample_stabilization_dwell_target(self, env_ids: torch.Tensor) -> None:
        """Choose how long each environment must hold quiet four-point support."""

        if env_ids.numel() == 0:
            return
        minimum = self.cfg.stabilization_dwell_steps
        maximum = self.cfg.stabilization_dwell_max_steps
        if maximum is None or maximum == minimum:
            self._stabilization_dwell_target[env_ids] = minimum
            return
        self._stabilization_dwell_target[env_ids] = torch.randint(
            minimum,
            maximum + 1,
            (env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )

    def _reset_release_state(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._release_active[env_ids] = False
        self._release_preload_count[env_ids] = 0
        self._release_ramp_count[env_ids] = 0
        self._release_final_count[env_ids] = 0
        self._release_age_count[env_ids] = 0

    def _restore_weld_parameters(self, env_ids: torch.Tensor, hand_id: int) -> None:
        """Restore the compiled strong weld parameters for selected environments."""

        if env_ids.numel() == 0:
            return
        weld_id = int(self._weld_ids[hand_id].item())
        self._env.sim.model.eq_solref[env_ids, weld_id] = self._strong_weld_solref[
            hand_id
        ]
        self._env.sim.model.eq_solimp[env_ids, weld_id] = self._strong_weld_solimp[
            hand_id
        ]

    def _set_grip_strength(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
        strength: torch.Tensor,
    ) -> None:
        """Map normalized grip strength to per-world equality softness."""

        if env_ids.numel() == 0:
            return
        strength = torch.clamp(strength, min=0.0, max=1.0)
        release = 1.0 - strength
        weld_id = int(self._weld_ids[hand_id].item())
        solref = self._env.sim.model.eq_solref
        solimp = self._env.sim.model.eq_solimp
        strong_ref = self._strong_weld_solref[hand_id]
        strong_imp = self._strong_weld_solimp[hand_id]

        solref[env_ids, weld_id] = strong_ref
        solref[env_ids, weld_id, 0] = strong_ref[0] + release * (
            self.cfg.release_soft_timeconst - strong_ref[0]
        )
        solimp[env_ids, weld_id] = strong_imp
        soft_impedance = torch.as_tensor(
            self.cfg.release_soft_impedance,
            dtype=strong_imp.dtype,
            device=strong_imp.device,
        )
        solimp[env_ids, weld_id, 0] = strong_imp[0] + release * (
            soft_impedance - strong_imp[0]
        )
        solimp[env_ids, weld_id, 1] = strong_imp[1] + release * (
            soft_impedance - strong_imp[1]
        )
        self.grip_strength[env_ids, hand_id] = strength

    def _begin_release(self, env_ids: torch.Tensor, hand_id: int) -> None:
        """Enter PRE_RELEASE while keeping the selected hand physically attached."""

        if env_ids.numel() == 0:
            return
        if torch.any(~self.attached[env_ids, hand_id]):
            raise RuntimeError(
                "Cannot begin ladder PRE_RELEASE for a hand without an active weld"
            )
        self._reset_release_state(env_ids)
        self._release_active[env_ids] = True
        self._restore_weld_parameters(env_ids, hand_id)
        self.grip_strength[env_ids, hand_id] = 1.0

    def _release_stability_conditions(self) -> dict[str, torch.Tensor]:
        """Return every independently logged PRE_RELEASE gate condition."""

        support_hand = 1 - self.active_hand
        support_hand_attached = self.attached.gather(
            1,
            support_hand[:, None],
        ).squeeze(1)
        moving_hand_attached = self.attached.gather(
            1,
            self.active_hand[:, None],
        ).squeeze(1)
        torso_speed = torch.linalg.vector_norm(self.torso_com_vel_w, dim=-1)
        conditions = {
            "hands_attached": support_hand_attached & moving_hand_attached,
            "feet_supported": self.foot_support.all(dim=1),
            "torso_speed": torso_speed,
            "torso_speed_valid": torso_speed <= self.cfg.max_release_torso_speed,
            "orientation_valid": (
                self.torso_orientation_error
                <= self.cfg.max_release_torso_orientation_error
            ),
            "support_offset_valid": (
                self.torso_support_offset_error
                <= self.cfg.max_release_support_offset_error
            ),
        }
        conditions["stable"] = (
            conditions["hands_attached"]
            & conditions["feet_supported"]
            & conditions["torso_speed_valid"]
            & conditions["orientation_valid"]
            & conditions["support_offset_valid"]
        )
        return conditions

    def _release_is_stable(self) -> torch.Tensor:
        """Return the feedback gate used to advance or reverse PRE_RELEASE."""

        return self._release_stability_conditions()["stable"]

    def _advance_release_ramp(self, phase_mask: torch.Tensor) -> None:
        """Transfer load, soften the weld, and detach only after a stable soft hold."""

        active = phase_mask & self._release_active
        if not torch.any(active):
            return
        self._release_age_count[active] += 1
        stable = self._release_is_stable()
        preloading = active & (self._release_ramp_count == 0)
        self._release_preload_count[preloading] = torch.where(
            stable[preloading],
            self._release_preload_count[preloading] + 1,
            0,
        )

        ramping = active & (
            (self._release_ramp_count > 0)
            | (self._release_preload_count >= self.cfg.release_preload_dwell_steps)
        )
        increased = torch.clamp(
            self._release_ramp_count + 1,
            max=self.cfg.release_ramp_steps,
        )
        decreased = torch.clamp(
            self._release_ramp_count - self.cfg.release_recovery_steps,
            min=0,
        )
        self._release_ramp_count[ramping] = torch.where(
            stable[ramping],
            increased[ramping],
            decreased[ramping],
        )
        returned_to_preload = ramping & (self._release_ramp_count == 0) & ~stable
        self._release_preload_count[returned_to_preload] = 0

        u = self._release_ramp_count.float() / float(self.cfg.release_ramp_steps)
        smooth_release = u * u * (3.0 - 2.0 * u)
        strength = 1.0 - smooth_release
        for hand_id in (0, 1):
            selected = torch.where(active & (self.active_hand == hand_id))[0]
            self._set_grip_strength(selected, hand_id, strength[selected])

        fully_soft = active & (self._release_ramp_count >= self.cfg.release_ramp_steps)
        self._release_final_count[active] = torch.where(
            fully_soft[active] & stable[active],
            self._release_final_count[active] + 1,
            0,
        )
        complete = active & (
            self._release_final_count >= self.cfg.release_final_dwell_steps
        )
        for hand_id in (0, 1):
            selected = torch.where(complete & (self.active_hand == hand_id))[0]
            self._release(selected, hand_id)

        stalled = (
            phase_mask
            & self._release_active
            & (self._release_age_count >= self.cfg.pre_release_timeout_steps)
        )
        self.pre_release_stalled[stalled] = True

    def _attach(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
        rung_indices: torch.Tensor,
    ) -> None:
        """Move one mocap anchor to the hand and enable its weld."""

        if env_ids.numel() == 0:
            return

        hand_site_id = int(self._hand_site_ids[hand_id].item())
        mocap_id = int(self._anchor_mocap_ids[hand_id].item())
        weld_id = int(self._weld_ids[hand_id].item())
        data = self._env.sim.data

        hand_pos = data.site_xpos[env_ids, hand_site_id]
        hand_mat = _as_rotation_matrix(data.site_xmat[env_ids, hand_site_id])
        data.mocap_pos[env_ids, mocap_id] = hand_pos
        data.mocap_quat[env_ids, mocap_id] = _matrix_to_quaternion(hand_mat)
        self._restore_weld_parameters(env_ids, hand_id)
        data.eq_active[env_ids, weld_id] = True

        self.attached[env_ids, hand_id] = True
        self.grip_strength[env_ids, hand_id] = 1.0
        self.held_rung[env_ids, hand_id] = rung_indices

    def _release(self, env_ids: torch.Tensor, hand_id: int) -> None:
        """Disable one hand weld.  The other hand is left untouched."""

        if env_ids.numel() == 0:
            return
        weld_id = int(self._weld_ids[hand_id].item())
        self._env.sim.data.eq_active[env_ids, weld_id] = False
        self._restore_weld_parameters(env_ids, hand_id)
        self.attached[env_ids, hand_id] = False
        self.grip_strength[env_ids, hand_id] = 0.0
        self.held_rung[env_ids, hand_id] = -1
        self._reset_release_state(env_ids)

    def _closest_points_on_rungs(
        self,
        points_w: torch.Tensor,
        rung_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Project points onto finite capsule axes for per-environment rungs."""

        clamped_indices = torch.clamp(rung_indices, 0, self.num_rungs - 1)
        site_ids = self._rung_site_ids[clamped_indices]
        data = self._env.sim.data
        centers = data.site_xpos[self._all_env_ids, site_ids]
        matrices = _as_rotation_matrix(data.site_xmat[self._all_env_ids, site_ids])
        # Capsule/site local z-axis expressed in world coordinates.
        axes = matrices[..., :, 2]
        half_spans = self._rung_half_lengths[clamped_indices]
        offset = torch.sum((points_w - centers) * axes, dim=-1)
        offset = torch.maximum(torch.minimum(offset, half_spans), -half_spans)
        return centers + offset.unsqueeze(-1) * axes

    def _points_above_rungs(
        self,
        points_w: torch.Tensor,
        rung_indices: torch.Tensor,
    ) -> torch.Tensor:
        targets = self._closest_points_on_rungs(points_w, rung_indices)
        targets[:, 2] += self.cfg.foot_target_height
        return targets

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        qualified_name = name if "/" in name else f"{self.cfg.entity_name}/{name}"
        object_id = int(
            mujoco.mj_name2id(
                self._env.sim.mj_model,
                object_type,
                qualified_name,
            )
        )
        if object_id < 0:
            raise ValueError(
                f"MJCF object {qualified_name!r} ({object_type.name}) was not found"
            )
        return object_id


@dataclass(kw_only=True)
class LadderClimbCommandCfg(CommandTermCfg):
    """Configuration for :class:`LadderClimbCommand`."""

    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    hand_site_names: tuple[str, str] = (
        "left_grip_site",
        "right_grip_site",
    )
    foot_site_names: tuple[str, str] = (
        "left_foot",
        "right_foot",
    )
    foot_contact_sensor_name: str = "ladder_foot_contact"
    foot_contact_body_names: tuple[str, str] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    anchor_body_names: tuple[str, str] = (
        "left_grip_anchor_body",
        "right_grip_anchor_body",
    )
    weld_names: tuple[str, str] = (
        "left_grip_weld",
        "right_grip_weld",
    )
    rung_site_names: tuple[str, ...] = ()
    torso_body_name: str = "torso_link"
    pelvis_body_name: str = "pelvis"

    first_moving_hand: HandName = "left"
    first_moving_foot: HandName = "left"
    start_rung: int = 0
    initialize_on_reset: bool = False
    initial_foot_rung: int = 0
    curriculum_enabled: bool = True
    curriculum_success_threshold: float = 0.80
    curriculum_window_size: int = 100
    curriculum_min_phase_steps: tuple[int, int, int, int] = (
        36_000,
        120_000,
        120_000,
        120_000,
    )
    fixed_max_unlocked_phase: int | None = None
    freeze_at_max_unlocked_phase: bool = False
    boundary_state_reset_prob: float = 0.0
    boundary_state_bank_size: int = 0
    stabilization_dwell_steps: int = 50
    stabilization_dwell_max_steps: int | None = 100
    hand_target_dwell_steps: int = 3
    foot_target_dwell_steps: int = 5
    max_stabilization_torso_speed: float = 0.20
    max_stabilization_joint_speed: float = 1.0
    max_stabilization_body_angular_speed: float = 0.40
    max_stabilization_waist_joint_speed: float = 0.60
    max_stabilization_support_offset_error: float = 0.18
    max_phase_torso_orientation_error: float = 0.30
    max_phase_support_offset_error: float = 0.15
    max_phase_completion_torso_speed: float = 0.20
    max_phase_completion_joint_speed: float = 1.0
    first_foot_max_body_drop: float = 0.03
    cycle_min_body_ascent: float = 0.12
    release_preload_dwell_steps: int = 8
    release_ramp_steps: int = 20
    release_final_dwell_steps: int = 5
    release_recovery_steps: int = 2
    pre_release_timeout_steps: int = 100
    max_release_torso_speed: float = 0.12
    max_release_torso_orientation_error: float = 0.25
    max_release_support_offset_error: float = 0.12
    release_soft_timeconst: float = 0.18
    release_soft_impedance: float = 0.05
    attach_distance: float = 0.06
    max_attach_speed: float = 0.30
    foot_reach_distance: float = 0.10
    foot_support_distance: float = 0.08
    max_foot_speed: float = 0.60
    foot_target_height: float = 0.045
    hand_foot_lead_rungs: int = 3
    # None uses the full finite capsule axis.  Set a smaller value to keep
    # grips away from rung ends.
    grip_half_span: float | None = None

    def build(self, env: ManagerBasedRlEnv) -> LadderClimbCommand:
        return LadderClimbCommand(self, env)


def _ladder_command(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> LadderClimbCommand:
    return cast(LadderClimbCommand, env.command_manager.get_term(command_name))


class LadderPhaseProgressReward:
    """Target potential with non-farmable novel stabilization progress.

    The potential combines distance to the selected limb target with continuous
    grip release during hand phases.  Positive progress is paid only while the
    non-moving supports and torso pose remain valid; negative progress is never
    masked.  A phase/target key suppresses artificial reward jumps when the FSM
    selects a new target.  During ``STABILIZE`` only a new per-phase maximum of
    normalized dwell is rewarded, so resetting and rebuilding the same dwell
    cannot exploit discounted returns.
    """

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        params = getattr(cfg, "params")
        self._rung_spacing = float(params["rung_spacing"])
        self._reach_distance = float(params["reach_distance"])
        self._first_hand_body_weight = float(params["first_hand_body_weight"])
        self._second_hand_body_weight = float(params["second_hand_body_weight"])
        self._foot_body_weight = float(params["foot_body_weight"])
        self._release_progress_weight = float(params["release_progress_weight"])
        self._unsupported_progress_scale = float(params["unsupported_progress_scale"])
        self._max_abs_rate = float(params["max_abs_rate"])
        if self._rung_spacing <= 0.0:
            raise ValueError("rung_spacing must be > 0")
        if self._reach_distance <= 0.0:
            raise ValueError("reach_distance must be > 0")
        if (
            min(
                self._first_hand_body_weight,
                self._second_hand_body_weight,
                self._foot_body_weight,
            )
            < 0.0
        ):
            raise ValueError("phase body weights must be >= 0")
        if self._release_progress_weight < 0.0:
            raise ValueError("release_progress_weight must be >= 0")
        if self._max_abs_rate <= 0.0:
            raise ValueError("max_abs_rate must be > 0")
        if not 0.0 <= self._unsupported_progress_scale <= 1.0:
            raise ValueError("unsupported_progress_scale must be in [0, 1]")

        self._previous_potential = torch.zeros(env.num_envs, device=env.device)
        self._previous_target_key = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )
        self._valid_previous = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._max_stabilization_potential = torch.zeros(
            env.num_envs,
            device=env.device,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._valid_previous[env_ids] = False
        self._max_stabilization_potential[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        rung_spacing: float,
        reach_distance: float,
        first_hand_body_weight: float,
        second_hand_body_weight: float,
        foot_body_weight: float,
        release_progress_weight: float,
        unsupported_progress_scale: float,
        max_abs_rate: float,
        phase: int | None = None,
    ) -> torch.Tensor:
        del (
            rung_spacing,
            reach_distance,
            first_hand_body_weight,
            second_hand_body_weight,
            foot_body_weight,
            release_progress_weight,
            unsupported_progress_scale,
            max_abs_rate,
        )
        selected_phase = None if phase is None else int(LadderPhase(phase))
        command = _ladder_command(env, command_name)
        phase = command.phase
        is_stabilizing = phase == int(LadderPhase.STABILIZE)

        hand_distance = torch.linalg.vector_norm(
            command.active_hand_pos_w - command.target_pos_w,
            dim=-1,
        )
        foot_distance = torch.linalg.vector_norm(
            command.active_foot_pos_w - command.active_foot_target_pos_w,
            dim=-1,
        )
        target_distance = torch.where(
            command.is_hand_phase, hand_distance, foot_distance
        )
        target_distance = torch.where(
            is_stabilizing,
            torch.zeros_like(target_distance),
            target_distance,
        )

        body_weight = torch.zeros_like(command.body_height)
        body_weight = torch.where(
            phase == int(LadderPhase.FIRST_HAND),
            self._first_hand_body_weight,
            body_weight,
        )
        body_weight = torch.where(
            phase == int(LadderPhase.SECOND_HAND),
            self._second_hand_body_weight,
            body_weight,
        )
        body_weight = torch.where(
            command.is_foot_phase,
            self._foot_body_weight,
            body_weight,
        )
        movement_potential = (
            body_weight * command.body_height / self._rung_spacing
            - target_distance / self._reach_distance
        )
        release_potential = torch.where(
            command.is_hand_phase,
            1.0 - command.active_grip_strength,
            0.0,
        )
        movement_potential = (
            movement_potential + self._release_progress_weight * release_potential
        )
        dwell_potential = (
            command._phase_dwell_count.float()
            / command._stabilization_dwell_target.clamp_min(1).float()
        )
        potential = torch.where(is_stabilizing, dwell_potential, movement_potential)

        phase_target_key = phase * (2 * command.num_rungs + 1)
        hand_target_key = command.active_hand * command.num_rungs + command.target_rung
        foot_target_key = (
            command.active_foot * command.num_rungs + command.target_foot_rung
        )
        phase_target_key = phase_target_key + torch.where(
            command.is_hand_phase,
            hand_target_key,
            torch.where(command.is_foot_phase, foot_target_key, 0),
        )

        valid = command.initialized & ~command.finished
        same_target = self._valid_previous & (
            phase_target_key == self._previous_target_key
        )
        # Reattaching restores grip strength to one.  When the next curriculum
        # phase is still locked, the public phase/target key does not change,
        # so suppress that successful completion discontinuity explicitly.
        same_target &= ~command.just_advanced
        target_changed = ~self._valid_previous | (
            phase_target_key != self._previous_target_key
        )
        stabilization_record = torch.where(
            target_changed,
            potential,
            self._max_stabilization_potential,
        )
        novel_stabilization_progress = torch.clamp(
            potential - stabilization_record,
            min=0.0,
        )
        signed_progress = potential - self._previous_potential
        progress = torch.where(
            is_stabilizing,
            novel_stabilization_progress,
            signed_progress,
        )
        progress_rate = progress / env.step_dt
        progress_rate = torch.clamp(
            progress_rate,
            min=-self._max_abs_rate,
            max=self._max_abs_rate,
        )
        positive_constraints_satisfied = torch.where(
            is_stabilizing,
            command.stabilization_conditions_satisfied,
            command.phase_support_constraints_satisfied,
        )
        positive_scale = (
            self._unsupported_progress_scale
            + (1.0 - self._unsupported_progress_scale)
            * positive_constraints_satisfied.float()
        )
        active_hand_attached = command.attached.gather(
            1,
            command.active_hand[:, None],
        ).squeeze(1)
        detached_hand_phase = command.is_hand_phase & ~active_hand_attached
        positive_scale = torch.where(
            detached_hand_phase,
            command.phase_support_constraints_satisfied.float(),
            positive_scale,
        )
        gated_rate = torch.where(
            progress_rate > 0.0,
            progress_rate * positive_scale,
            progress_rate,
        )
        reward = torch.where(valid & same_target, gated_rate, 0.0)

        self._previous_potential[:] = potential
        self._previous_target_key[:] = phase_target_key
        self._valid_previous[:] = valid
        self._max_stabilization_potential[:] = torch.where(
            is_stabilizing,
            torch.maximum(stabilization_record, potential),
            torch.zeros_like(self._max_stabilization_potential),
        )
        # Update history for every environment before masking the output.
        # This preserves potential differences across phase changes and resets.
        if selected_phase is not None:
            reward = torch.where(command.phase == selected_phase, reward, 0.0)
        return reward


class LadderUpwardProgressReward:
    """Reward only novel whole-body height reached during an episode.

    The per-environment maximum is anchored to the first valid height after a
    reset.  Returning the novel height increment as a rate makes the integrated
    reward independent of the control timestep::

        sum(weight * reward * dt) = weight * (max_t(height_t) - height_0)

    Lowering the body and climbing back to an already rewarded height therefore
    cannot farm reward.  ``body_height`` averages pelvis and torso COM height,
    so the term cannot be maximized by lifting only one body point.
    """

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        del cfg, env

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        command = _ladder_command(env, command_name)
        height = command.body_height
        active = command.initialized
        previous_max = command.episode_max_body_height
        new_max = torch.maximum(previous_max, height)
        novel_height = torch.where(
            active,
            new_max - previous_max,
            0.0,
        )
        command.episode_max_body_height[:] = torch.where(
            active,
            new_max,
            previous_max,
        )
        supported = command.phase_required_supports_satisfied
        return novel_height * supported.float() / env.step_dt


class LadderFootPlacementReward:
    """Reward improvements in phase-appropriate physical foot placement.

    Stabilization and hand phases keep both feet on their assigned support
    rungs.  During a foot phase, the non-moving foot keeps its assigned rung
    while the selected foot is evaluated against the new target rung.  The
    signed potential difference rewards establishing the desired contact and
    penalizes losing it without paying a dense reward for standing still.
    """

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        params = getattr(cfg, "params")
        self._distance_std = float(params["distance_std"])
        self._max_abs_rate = float(params["max_abs_rate"])
        if self._distance_std <= 0.0:
            raise ValueError("distance_std must be > 0")
        if self._max_abs_rate <= 0.0:
            raise ValueError("max_abs_rate must be > 0")

        self._previous_quality = torch.zeros(env.num_envs, device=env.device)
        self._previous_target_key = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )
        self._valid_previous = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._valid_previous[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        distance_std: float,
        max_abs_rate: float,
        phase: int | None = None,
    ) -> torch.Tensor:
        del distance_std, max_abs_rate
        selected_phase = None if phase is None else int(LadderPhase(phase))
        command = _ladder_command(env, command_name)

        foot_ids = torch.arange(
            command.foot_pos_w.shape[1],
            device=env.device,
        )
        moving_foot = command.is_foot_phase[:, None] & (
            foot_ids[None, :] == command.active_foot[:, None]
        )
        desired_pos_w = torch.where(
            moving_foot[:, :, None],
            command.active_foot_target_pos_w[:, None, :],
            command.held_foot_target_pos_w,
        )

        distance = torch.linalg.vector_norm(
            command.foot_pos_w - desired_pos_w,
            dim=-1,
        )
        placement_quality = (
            torch.exp(-torch.square(distance / self._distance_std))
            * command.foot_contact.float()
        )
        quality = placement_quality.mean(dim=1)

        target_key = command.phase * (2 * command.num_rungs + 1)
        foot_target_key = (
            command.active_foot * command.num_rungs + command.target_foot_rung
        )
        target_key = target_key + torch.where(
            command.is_foot_phase,
            foot_target_key,
            0,
        )

        valid = command.initialized & ~command.finished
        same_target = self._valid_previous & (target_key == self._previous_target_key)
        quality_delta = quality - self._previous_quality
        progress_rate = quality_delta / env.step_dt
        progress_rate = torch.clamp(
            progress_rate,
            min=-self._max_abs_rate,
            max=self._max_abs_rate,
        )
        reward = torch.where(valid & same_target, progress_rate, 0.0)

        self._previous_quality[:] = quality
        self._previous_target_key[:] = target_key
        self._valid_previous[:] = valid
        # Update history for every environment before masking the output.
        # This preserves potential differences across phase changes and resets.
        if selected_phase is not None:
            reward = torch.where(command.phase == selected_phase, reward, 0.0)
        return reward


def _ladder_required_feet(command: LadderClimbCommand) -> torch.Tensor:
    """Exclude only the selected moving foot during foot phases."""
    foot_ids = torch.arange(2, device=command.phase.device)
    return ~(command.is_foot_phase[:, None] & (
        foot_ids[None, :] == command.active_foot[:, None]
    ))


def ladder_movement_survival(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Keep the survival term outside stabilization only."""
    command = _ladder_command(env, command_name)
    return (command.phase != int(LadderPhase.STABILIZE)).float()


def ladder_missing_foot_support(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Count missing required physical foot supports on every active step."""
    command = _ladder_command(env, command_name)
    missing = (_ladder_required_feet(command) & ~command.foot_support).sum(dim=1)
    return missing.float() * (command.initialized & ~command.finished).float()


class LadderFootRecoveryReward:
    """Signed approach rate to held foot rungs, including before contact.

    Histories are anchored again on reset or phase/held-rung changes. A selected
    moving foot is excluded; its existing phase progress handles its new target.
    """

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        self._reach_distance = float(getattr(cfg, "params")["reach_distance"])
        if self._reach_distance <= 0:
            raise ValueError("reach_distance must be > 0")
        self._previous_distance = torch.zeros(env.num_envs, device=env.device)
        self._previous_key = torch.zeros((env.num_envs, 3), dtype=torch.long, device=env.device)
        self._valid_previous = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._valid_previous[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env: ManagerBasedRlEnv, command_name: str,
                 reach_distance: float) -> torch.Tensor:
        del reach_distance
        command = _ladder_command(env, command_name)
        required = _ladder_required_feet(command)
        distances = torch.linalg.vector_norm(
            command.foot_pos_w - command.held_foot_target_pos_w, dim=-1
        )
        # Use a fixed divisor so losing a support cannot change normalization.
        distance = (distances * required.float()).sum(dim=1) / (2 * self._reach_distance)
        key = torch.cat((command.phase[:, None], command.foot_rung), dim=1)
        valid = command.initialized & ~command.finished
        same_target = self._valid_previous & (key == self._previous_key).all(dim=1)
        rate = (self._previous_distance - distance) / env.step_dt
        reward = torch.where(valid & same_target, rate, 0.0)
        self._previous_distance[:] = distance
        self._previous_key[:] = key
        self._valid_previous[:] = valid
        return reward


class LadderStabilizationPoseCost:
    """Deviation from the fixed climbing keyframe, never from a reset-bank pose."""

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        params = getattr(cfg, "params")
        command = _ladder_command(env, params["command_name"])
        names = command.robot.joint_names
        reference = params["reference_joint_pos"]
        weights = params["joint_weights"]

        def resolve(mapping: dict[str, float], default: float) -> torch.Tensor:
            values = []
            for name in names:
                matches = [value for pattern, value in mapping.items() if re.fullmatch(pattern, name)]
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous ladder pose parameter for {name}")
                values.append(matches[0] if matches else default)
            return torch.tensor(values, device=env.device)

        self._reference = resolve(reference, 0.0)
        self._weights = resolve(weights, 0.5)
        self._sigma = float(params["sigma"])
        if self._sigma <= 0 or not bool((self._weights > 0).all()):
            raise ValueError("Ladder pose sigma and joint weights must be positive")

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # The reference is intentionally independent of reset and bank sampling.
        pass

    def __call__(self, env: ManagerBasedRlEnv, command_name: str,
                 reference_joint_pos: dict[str, float], joint_weights: dict[str, float],
                 sigma: float) -> torch.Tensor:
        command = _ladder_command(env, command_name)
        error = ((command.robot.data.joint_pos - self._reference) / self._sigma).square()
        active = command.initialized & ~command.finished & command.is_stabilization_phase
        return (error * self._weights).sum(-1) / self._weights.sum() * active.float()


def ladder_stabilization_joint_velocity_l2(
    env: ManagerBasedRlEnv, command_name: str,
) -> torch.Tensor:
    """Mean squared joint speed in units of 1 rad/s, with no dead zone."""
    command = _ladder_command(env, command_name)
    active = command.initialized & ~command.finished & command.is_stabilization_phase
    return command.robot.data.joint_vel.square().mean(-1) * active.float()


def ladder_unwanted_contact_cost(
    env: ManagerBasedRlEnv, sensor_name: str | tuple[str, ...], force_threshold: float = 1.0,
) -> torch.Tensor:
    """Count bodies with a current ladder contact, not historical contact points."""
    names = (sensor_name,) if isinstance(sensor_name, str) else sensor_name
    hits = []
    for name in names:
        sensor = env.scene[name]
        if sensor.cfg.num_slots != 1:
            raise ValueError("Unwanted ladder contacts require one slot per body")
        force = sensor.data.force
        assert force is not None
        hits.append(torch.linalg.vector_norm(force, dim=-1) > force_threshold)
    # Both face sensors use the same ordered primary bodies. Count a body once
    # even when it touches both faces simultaneously.
    return torch.stack(hits).any(dim=0).sum(-1).float()


def ladder_stabilization_violation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Dense squared threshold excess; improvement reduces the penalty.

    The five continuous gates contribute equally. Already valid gates cost zero,
    while hands and feet are handled by attachment mechanics and support costs.
    """
    command = _ladder_command(env, command_name)
    gates = command._stabilization_conditions()
    values = (
        (gates["torso_speed"], command.cfg.max_stabilization_torso_speed),
        (gates["joint_speed_rms"], command.cfg.max_stabilization_joint_speed),
        (torch.maximum(gates["torso_angular_speed"], gates["pelvis_angular_speed"]),
         command.cfg.max_stabilization_body_angular_speed),
        (gates["waist_joint_speed"], command.cfg.max_stabilization_waist_joint_speed),
        (command.torso_support_offset_error, command.cfg.max_stabilization_support_offset_error),
    )
    # A zero configured threshold uses unit scale rather than dividing by zero.
    errors = [torch.square(torch.clamp(value - limit, min=0) / (limit if limit > 0 else 1.0))
              for value, limit in values]
    active = command.initialized & ~command.finished & command.is_stabilization_phase
    return torch.stack(errors).mean(dim=0) * active.float()


class LadderTargetProgressReward:
    """Reward signed progress toward the active hand or foot target.

    Absolute proximity rewards can be collected forever by hovering near a
    target.  This stateful term instead returns normalized closing speed.  It
    yields zero when the target or movement phase changes, positive reward
    while approaching, zero while stationary, and negative reward while
    retreating.
    """

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        params = getattr(cfg, "params")
        self._target: Literal["hand", "foot"] = params["target"]
        self._max_speed = float(params["max_speed"])
        if self._target not in ("hand", "foot"):
            raise ValueError("target must be either 'hand' or 'foot'")
        if self._max_speed <= 0.0:
            raise ValueError("max_speed must be > 0")

        self._previous_distance = torch.zeros(env.num_envs, device=env.device)
        self._previous_target_key = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )
        self._valid_previous = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._valid_previous[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        target: Literal["hand", "foot"],
        max_speed: float,
    ) -> torch.Tensor:
        del target, max_speed
        command = _ladder_command(env, command_name)

        if self._target == "hand":
            distance = torch.linalg.vector_norm(
                command.active_hand_pos_w - command.target_pos_w,
                dim=-1,
            )
            moving_hand_attached = command.attached.gather(
                1,
                command.active_hand[:, None],
            ).squeeze(1)
            eligible = (
                command.initialized
                & ~command.finished
                & command.is_hand_phase
                & ~moving_hand_attached
            )
            target_key = command.active_hand * command.num_rungs + command.target_rung
        else:
            distance = torch.linalg.vector_norm(
                command.active_foot_pos_w - command.active_foot_target_pos_w,
                dim=-1,
            )
            eligible = (
                command.initialized
                & ~command.finished
                & command.is_foot_phase
                & command.attached.all(dim=1)
            )
            target_key = (
                command.active_foot * command.num_rungs + command.target_foot_rung
            )

        same_target = self._valid_previous & (target_key == self._previous_target_key)
        closing_speed = (self._previous_distance - distance) / env.step_dt
        reward = torch.clamp(closing_speed / self._max_speed, min=-1.0, max=1.0)
        reward = torch.where(eligible & same_target, reward, 0.0)

        self._previous_distance[:] = distance
        self._previous_target_key[:] = target_key
        self._valid_previous[:] = eligible
        return reward


class LadderTorsoAscentReward:
    """Reward signed torso-COM ascent during the supported foot phases."""

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        params = getattr(cfg, "params")
        self._max_speed = float(params["max_speed"])
        if self._max_speed <= 0.0:
            raise ValueError("max_speed must be > 0")
        self._previous_height = torch.zeros(env.num_envs, device=env.device)
        self._valid_previous = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._valid_previous[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        max_speed: float,
    ) -> torch.Tensor:
        del max_speed
        command = _ladder_command(env, command_name)
        height = command.torso_com_pos_w[:, 2]
        gripping = (
            command.initialized
            & command.is_foot_phase
            & command.attached.all(dim=1)
            & ~command.finished
        )
        ascent_speed = (height - self._previous_height) / env.step_dt
        reward = torch.clamp(ascent_speed / self._max_speed, min=-1.0, max=1.0)
        reward = torch.where(gripping & self._valid_previous, reward, 0.0)

        self._previous_height[:] = height
        self._valid_previous[:] = gripping
        return reward


def ladder_support_hand(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = _ladder_command(env, command_name)
    attached = command.grip_strength * command.attached.float()
    moving_hand = command.active_hand[:, None]
    support_during_hand_phase = attached.scatter(1, moving_hand, 0.0).sum(dim=1)
    return torch.where(
        command.is_hand_phase,
        support_during_hand_phase,
        attached.mean(dim=1),
    )


def ladder_rung_advance(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return _ladder_command(env, command_name).just_advanced.float() / env.step_dt


def ladder_foot_support(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward real foot contact near the rung recorded by the foot FSM."""

    command = _ladder_command(env, command_name)
    return command.foot_support.float().mean(dim=1)


def ladder_torso_stability_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    torso_speed_std: float,
    joint_speed_std: float,
    movement_phase_scale: float,
) -> torch.Tensor:
    """Reward quiet support, with a strict four-point stabilization signal.

    During ``STABILIZE`` the reward is non-zero only while both welded hands
    and both physically contacting feet support the robot.  Later movement
    phases retain a smaller stability signal based on the fraction of active
    supports, so the selected limb can move without removing the incentive to
    keep the rest of the body quiet.
    """

    command = _ladder_command(env, command_name)
    torso_speed_sq = torch.sum(torch.square(command.torso_com_vel_w), dim=-1)
    joint_speed_rms_sq = torch.mean(
        torch.square(command.robot.data.joint_vel),
        dim=1,
    )
    quiet = torch.exp(
        -torso_speed_sq / torso_speed_std**2 - joint_speed_rms_sq / joint_speed_std**2
    )

    hand_support = (command.grip_strength * command.attached.float()).mean(dim=1)
    foot_support = command.foot_support.float().mean(dim=1)
    support_fraction = 0.5 * (hand_support + foot_support)
    four_point_support = (
        command.attached.all(dim=1) & command.foot_support.all(dim=1)
    ).float()
    stabilization_phase = command.phase == int(LadderPhase.STABILIZE)
    support_quality = torch.where(
        stabilization_phase,
        four_point_support,
        movement_phase_scale * support_fraction,
    )
    valid = command.initialized & ~command.finished
    return quiet * support_quality * valid.float()


def _ladder_posture_support_quality(
    command: LadderClimbCommand,
    movement_phase_scale: float,
) -> torch.Tensor:
    hand_support = (command.grip_strength * command.attached.float()).mean(dim=1)
    foot_support = command.foot_support.float().mean(dim=1)
    support_fraction = 0.5 * (hand_support + foot_support)
    four_point_support = (
        command.attached.all(dim=1) & command.foot_support.all(dim=1)
    ).float()
    return torch.where(
        command.phase == int(LadderPhase.STABILIZE),
        four_point_support,
        movement_phase_scale * support_fraction,
    )


def ladder_torso_posture_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    orientation_std: float,
    movement_phase_scale: float,
) -> torch.Tensor:
    """Reward the nominal full torso orientation instead of individual joint angles."""

    command = _ladder_command(env, command_name)
    posture = torch.exp(
        -torch.square(command.torso_orientation_error) / orientation_std**2
    )
    support_quality = _ladder_posture_support_quality(command, movement_phase_scale)
    valid = command.initialized & ~command.finished
    return posture * support_quality * valid.float()


def ladder_stabilization_orientation_error_l2(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Softly discourage torso rotation without making orientation a hard gate."""

    command = _ladder_command(env, command_name)
    active = (
        command.initialized
        & ~command.finished
        & (command.phase == int(LadderPhase.STABILIZE))
    )
    return torch.square(command.torso_orientation_error) * active.float()


def ladder_com_alignment_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    support_offset_std: float,
    movement_phase_scale: float,
) -> torch.Tensor:
    """Reward torso-COM placement relative to the current four-support geometry."""

    command = _ladder_command(env, command_name)
    alignment = torch.exp(
        -torch.square(command.torso_support_offset_error) / support_offset_std**2
    )
    support_quality = _ladder_posture_support_quality(command, movement_phase_scale)
    valid = command.initialized & ~command.finished
    return alignment * support_quality * valid.float()


class LadderSupportJointVelocityPenalty:
    """Penalize motion of the waist and limbs acting as supports.

    The active hand or foot is excluded only during its own movement phase.
    This makes the penalty stronger on joints that should currently carry the
    robot without discouraging the selected limb from reaching its target.
    """

    _ARM_PARTS = ("shoulder", "elbow", "wrist")
    _LEG_PARTS = ("hip", "knee", "ankle")

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        params = getattr(cfg, "params")
        entity_name = params.get("entity_name", "robot")
        self._robot = env.scene[entity_name]
        names = tuple(name.rsplit("/", 1)[-1] for name in self._robot.joint_names)

        def ids_for(side: str, parts: tuple[str, ...]) -> torch.Tensor:
            ids = [
                index
                for index, name in enumerate(names)
                if name.startswith(f"{side}_") and any(part in name for part in parts)
            ]
            if not ids:
                raise ValueError(f"No {side} support joints found in {names}")
            return torch.tensor(ids, dtype=torch.long, device=env.device)

        waist_ids = [index for index, name in enumerate(names) if "waist" in name]
        if not waist_ids:
            raise ValueError(f"No waist support joints found in {names}")
        self._waist_ids = torch.tensor(
            waist_ids,
            dtype=torch.long,
            device=env.device,
        )
        self._arm_ids = (
            ids_for("left", self._ARM_PARTS),
            ids_for("right", self._ARM_PARTS),
        )
        self._leg_ids = (
            ids_for("left", self._LEG_PARTS),
            ids_for("right", self._LEG_PARTS),
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        entity_name: str = "robot",
    ) -> torch.Tensor:
        del entity_name
        command = _ladder_command(env, command_name)
        joint_vel_sq = torch.square(self._robot.data.joint_vel)
        waist_cost = joint_vel_sq[:, self._waist_ids].sum(dim=1)
        arm_cost = torch.stack(
            tuple(joint_vel_sq[:, ids].sum(dim=1) for ids in self._arm_ids),
            dim=1,
        )
        leg_cost = torch.stack(
            tuple(joint_vel_sq[:, ids].sum(dim=1) for ids in self._leg_ids),
            dim=1,
        )

        support_arms = torch.ones_like(command.attached, dtype=torch.bool)
        support_arms.scatter_(
            1,
            command.active_hand[:, None],
            ~command.is_hand_phase[:, None],
        )
        support_legs = torch.ones_like(command.foot_contact, dtype=torch.bool)
        support_legs.scatter_(
            1,
            command.active_foot[:, None],
            ~command.is_foot_phase[:, None],
        )
        cost = (
            waist_cost
            + (arm_cost * support_arms.float()).sum(dim=1)
            + (leg_cost * support_legs.float()).sum(dim=1)
        )
        return cost * (command.initialized & ~command.finished).float()


def ladder_foot_rung_advance(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return _ladder_command(env, command_name).just_foot_advanced.float() / env.step_dt


def ladder_phase_completed(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Return one fixed impulse for any ordered FSM phase transition."""

    command = _ladder_command(env, command_name)
    completed = (
        command.just_stabilized | command.just_advanced | command.just_foot_advanced
    )
    return completed.float() / env.step_dt


def ladder_failure_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Return one impulse for any terminal outcome that is not task progress."""

    del command_name
    termination_manager = env.termination_manager
    successful = termination_manager.get_term("success").bool()
    prefix_completed = termination_manager.get_term(
        "curriculum_stage_complete"
    ).bool()
    failed = termination_manager.dones.bool() & ~successful & ~prefix_completed
    return failed.float() / env.step_dt


def ladder_stabilized(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return _ladder_command(env, command_name).just_stabilized.float() / env.step_dt


def ladder_cycle_completed(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = _ladder_command(env, command_name)
    return command.just_cycle_completed.float() / env.step_dt


def ladder_finished(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return _ladder_command(env, command_name).finished.float() / env.step_dt


def ladder_both_hands_free(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = _ladder_command(env, command_name)
    return ((~command.attached.any(dim=1)) & command.initialized).float()


def ladder_upright_exp(
    env: ManagerBasedRlEnv,
    std: float,
    entity_name: str = "robot",
) -> torch.Tensor:
    projected_gravity = env.scene[entity_name].data.projected_gravity_b
    error = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    return torch.exp(-error / std**2)


def ladder_success(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate an episode once the final rung has been reached."""

    return _ladder_command(env, command_name).finished


def ladder_pre_release_stalled(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate an episode that exceeds the bounded PRE_RELEASE duration."""

    return _ladder_command(env, command_name).pre_release_stalled


def ladder_curriculum_stage_complete(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """End a short prefix episode when its next phase is still locked."""

    return _ladder_command(env, command_name).curriculum_stage_complete


def ladder_rung_endpoints_torso(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Expose fixed ladder geometry as ``(B, num_rungs, 7)`` observations."""

    return _ladder_command(env, command_name).rung_endpoints_torso


def ladder_rung_tokens_torso(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Expose target-aware ladder geometry as ``(B, num_rungs, 15)`` tokens."""

    return _ladder_command(env, command_name).rung_tokens_torso


def ladder_critic_privileged(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Expose compact reward/FSM state to the asymmetric training critic only."""

    command = _ladder_command(env, command_name)
    body_height = command.body_height
    torso_speed = torch.linalg.vector_norm(command.torso_com_vel_w, dim=-1)
    joint_speed_rms = torch.sqrt(
        torch.mean(torch.square(command.robot.data.joint_vel), dim=1)
    )
    height_below_record = torch.clamp(
        command.episode_max_body_height - body_height,
        min=0.0,
    )
    release_progress = command._release_ramp_count.float() / float(
        command.cfg.release_ramp_steps
    )
    dwell_target = torch.where(
        command.is_stabilization_phase,
        command._stabilization_dwell_target,
        torch.where(
            command.is_hand_phase,
            torch.full_like(
                command._phase_dwell_count,
                command.cfg.hand_target_dwell_steps,
            ),
            torch.full_like(
                command._phase_dwell_count,
                command.cfg.foot_target_dwell_steps,
            ),
        ),
    )
    dwell_progress = torch.clamp(
        command._phase_dwell_count.float() / dwell_target.clamp_min(1).float(),
        min=0.0,
        max=1.0,
    )

    scalar_terms = (
        body_height,
        command.episode_max_body_height,
        height_below_record,
        body_height - command._cycle_start_body_height,
        body_height - command._phase_start_body_height,
        torso_speed,
        command.torso_orientation_error,
        command.torso_support_offset_error,
        joint_speed_rms,
    )
    return torch.cat(
        (
            *(term.unsqueeze(-1) for term in scalar_terms),
            command.foot_support.float(),
            command.phase_required_supports_satisfied.float().unsqueeze(-1),
            release_progress.unsqueeze(-1),
            dwell_progress.unsqueeze(-1),
        ),
        dim=-1,
    )
