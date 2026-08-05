from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Literal, cast

import mujoco
import torch
import torch.nn.functional as F
from mjlab.managers import CommandTerm, CommandTermCfg

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
        if cfg.hand_target_dwell_steps <= 0:
            raise ValueError("hand_target_dwell_steps must be > 0")
        if cfg.foot_target_dwell_steps <= 0:
            raise ValueError("foot_target_dwell_steps must be > 0")
        if cfg.max_stabilization_torso_speed < 0.0:
            raise ValueError("max_stabilization_torso_speed must be >= 0")
        if cfg.max_stabilization_joint_speed < 0.0:
            raise ValueError("max_stabilization_joint_speed must be >= 0")
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
        self._unlocked_phase = (
            int(LadderPhase.STABILIZE)
            if cfg.curriculum_enabled
            else int(LadderPhase.SECOND_FOOT)
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
        self._pending_start_pose_init = torch.zeros(
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
        hand_pos = self.hand_pos_w
        self._previous_hand_pos_w = hand_pos.clone()
        self._hand_vel_w = torch.zeros_like(hand_pos)
        foot_pos = self.foot_pos_w
        self._previous_foot_pos_w = foot_pos.clone()
        self._foot_vel_w = torch.zeros_like(foot_pos)
        torso_com_pos = self.torso_com_pos_w
        self._previous_torso_com_pos_w = torso_com_pos.clone()
        self._torso_com_vel_w = torch.zeros_like(torso_com_pos)
        self.start_height = self.torso_pos_w[:, 2].clone()

        self.metrics["target_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rung_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["attached_hands"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["foot_target_distance"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["foot_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["supported_feet"] = torch.zeros(self.num_envs, device=self.device)
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

        Layout: phase one-hot (5), left/right hand-to-target vectors (6),
        attached flags (2), initialized flag (1), normalized hand progress (1),
        left/right foot-to-target vectors (6), physical foot-contact flags (2),
        and normalized foot progress (1).  This preserves the existing 24D
        command and therefore the 117D/120D actor/critic dimensions.
        """

        phase_one_hot = F.one_hot(self.phase, num_classes=len(LadderPhase)).float()
        hand_target_delta = (self.hand_target_pos_w - self.hand_pos_w).flatten(1)
        foot_target_delta = (self.foot_target_pos_w - self.foot_pos_w).flatten(1)
        return torch.cat(
            (
                phase_one_hot,
                hand_target_delta,
                self.attached.float(),
                self.initialized.float().unsqueeze(-1),
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

    def curriculum_state_dict(self) -> dict[str, object]:
        """Serialize adaptive curriculum state for a training checkpoint."""

        return {
            "version": 1,
            "unlocked_phase": self._unlocked_phase,
            "phase_start_step": self._curriculum_phase_start_step,
            "recent_outcomes": list(self._recent_curriculum_outcomes),
            "pending_outcomes": list(self._pending_curriculum_outcomes),
        }

    def load_curriculum_state_dict(self, state: dict[str, object]) -> None:
        """Restore adaptive curriculum state with fail-fast validation."""

        if not self.cfg.curriculum_enabled:
            self._unlocked_phase = int(LadderPhase.SECOND_FOOT)
            self._recent_curriculum_outcomes.clear()
            self._pending_curriculum_outcomes.clear()
            return
        if state.get("version") != 1:
            raise ValueError("Unsupported ladder curriculum checkpoint version")
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

        start_rungs = torch.full_like(self.held_rung, self.cfg.start_rung)
        target_rungs = torch.where(
            self.initialized[:, None], self.held_rung, start_rungs
        )
        target_rungs[self._all_env_ids, self.active_hand] = self.target_rung
        target_rungs = torch.where(target_rungs >= 0, target_rungs, start_rungs)
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
        self.held_rung[env_ids] = -1
        self.initialized[env_ids] = False
        self.finished[env_ids] = False
        self.just_initialized[env_ids] = False
        self.just_advanced[env_ids] = False
        self.just_foot_advanced[env_ids] = False
        self.just_stabilized[env_ids] = False
        self.just_cycle_completed[env_ids] = False
        self.curriculum_stage_complete[env_ids] = False
        self._pending_start_pose_init[env_ids] = self.cfg.initialize_on_reset
        self.phase[env_ids] = int(LadderPhase.STABILIZE)
        self._phase_dwell_count[env_ids] = 0

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

        hand_pos = self.hand_pos_w[env_ids]
        self._previous_hand_pos_w[env_ids] = hand_pos
        self._hand_vel_w[env_ids] = 0.0
        foot_pos = self.foot_pos_w[env_ids]
        self._previous_foot_pos_w[env_ids] = foot_pos
        self._foot_vel_w[env_ids] = 0.0
        torso_com_pos = self.torso_com_pos_w[env_ids]
        self._previous_torso_com_pos_w[env_ids] = torso_com_pos
        self._torso_com_vel_w[env_ids] = 0.0
        self.start_height[env_ids] = self.torso_pos_w[env_ids, 2]
        self._episode_active[env_ids] = True

    def _record_episode_outcomes(self, env_ids: torch.Tensor) -> None:
        """Queue terminal outcomes for synchronization by the PPO runner."""

        if not self.cfg.curriculum_enabled:
            return
        active_ids = env_ids[self._episode_active[env_ids]]
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

        hand_pos = self.hand_pos_w
        self.start_height[self._pending_start_pose_init] = self.torso_pos_w[
            self._pending_start_pose_init,
            2,
        ]
        self._previous_hand_pos_w[self._pending_start_pose_init] = hand_pos[
            self._pending_start_pose_init
        ]
        self._hand_vel_w[:] = (hand_pos - self._previous_hand_pos_w) / self._env.step_dt
        self._previous_hand_pos_w[:] = hand_pos
        foot_pos = self.foot_pos_w
        self._previous_foot_pos_w[self._pending_start_pose_init] = foot_pos[
            self._pending_start_pose_init
        ]
        self._foot_vel_w[:] = (foot_pos - self._previous_foot_pos_w) / self._env.step_dt
        self._previous_foot_pos_w[:] = foot_pos
        torso_com_pos = self.torso_com_pos_w
        self._previous_torso_com_pos_w[self._pending_start_pose_init] = torso_com_pos[
            self._pending_start_pose_init
        ]
        self._torso_com_vel_w[:] = (
            torso_com_pos - self._previous_torso_com_pos_w
        ) / self._env.step_dt
        self._previous_torso_com_pos_w[:] = torso_com_pos

        # Reconcile state if another component or a reset disabled a weld.
        for hand_id in (0, 1):
            weld_id = int(self._weld_ids[hand_id].item())
            weld_active = self._env.sim.data.eq_active[:, weld_id] != 0
            lost = self.attached[:, hand_id] & ~weld_active
            self.attached[lost, hand_id] = False
            self.held_rung[lost, hand_id] = -1

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
        )
        if not torch.any(phase_mask):
            return

        torso_speed = torch.linalg.vector_norm(self.torso_com_vel_w, dim=-1)
        joint_speed_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_vel), dim=1)
        )
        stable = (
            self.attached.all(dim=1)
            & self.foot_support.all(dim=1)
            & (torso_speed <= self.cfg.max_stabilization_torso_speed)
            & (joint_speed_rms <= self.cfg.max_stabilization_joint_speed)
        )
        self._update_phase_dwell(phase_mask, stable)
        ready = phase_mask & (
            self._phase_dwell_count >= self.cfg.stabilization_dwell_steps
        )
        env_ids = torch.where(ready)[0]
        if env_ids.numel() == 0:
            return
        self.just_stabilized[env_ids] = True
        self._transition_or_finish_curriculum(env_ids, LadderPhase.FIRST_HAND)

    def _advance_hand_phase(self, phase_at_start: torch.Tensor) -> None:
        """Attach one hand after it remains close and slow for several steps."""

        phase_mask = (
            (
                (phase_at_start == int(LadderPhase.FIRST_HAND))
                | (phase_at_start == int(LadderPhase.SECOND_HAND))
            )
            & self.initialized
            & ~self.finished
        )
        if not torch.any(phase_mask):
            return

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
        reached = (
            active_contact
            & support_foot_contact
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
            self.curriculum_stage_complete[env_ids] = True
            return
        self._start_phase(env_ids, next_phase)

    def _start_phase(
        self,
        env_ids: torch.Tensor,
        phase: LadderPhase,
    ) -> None:
        """Configure the single limb allowed to move in ``phase``."""

        if env_ids.numel() == 0:
            return
        self.phase[env_ids] = int(phase)
        self._phase_dwell_count[env_ids] = 0

        first_hand = 0 if self.cfg.first_moving_hand == "left" else 1
        first_foot = 0 if self.cfg.first_moving_foot == "left" else 1
        if phase == LadderPhase.STABILIZE:
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
            self._release(env_ids, first_hand)
            return
        if phase == LadderPhase.SECOND_HAND:
            second_hand = 1 - first_hand
            self.active_hand[env_ids] = second_hand
            self.target_rung[env_ids] = torch.max(self.held_rung[env_ids], dim=1).values
            self._release(env_ids, second_hand)
            return
        if phase == LadderPhase.FIRST_FOOT:
            self.active_foot[env_ids] = first_foot
            self.target_foot_rung[env_ids] = torch.clamp(
                torch.max(self.foot_rung[env_ids], dim=1).values + 1,
                max=self.max_foot_rung,
            )
            return

        second_foot = 1 - first_foot
        self.active_foot[env_ids] = second_foot
        self.target_foot_rung[env_ids] = torch.max(
            self.foot_rung[env_ids], dim=1
        ).values

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
        data.eq_active[env_ids, weld_id] = True

        self.attached[env_ids, hand_id] = True
        self.held_rung[env_ids, hand_id] = rung_indices

    def _release(self, env_ids: torch.Tensor, hand_id: int) -> None:
        """Disable one hand weld.  The other hand is left untouched."""

        if env_ids.numel() == 0:
            return
        weld_id = int(self._weld_ids[hand_id].item())
        self._env.sim.data.eq_active[env_ids, weld_id] = False
        self.attached[env_ids, hand_id] = False
        self.held_rung[env_ids, hand_id] = -1

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

    first_moving_hand: HandName = "left"
    first_moving_foot: HandName = "left"
    start_rung: int = 0
    initialize_on_reset: bool = False
    initial_foot_rung: int = 0
    curriculum_enabled: bool = True
    curriculum_success_threshold: float = 0.80
    curriculum_window_size: int = 100
    curriculum_min_phase_steps: tuple[int, int, int, int] = (
        120_000,
        120_000,
        120_000,
        120_000,
    )
    stabilization_dwell_steps: int = 5
    hand_target_dwell_steps: int = 3
    foot_target_dwell_steps: int = 5
    max_stabilization_torso_speed: float = 0.20
    max_stabilization_joint_speed: float = 1.0
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
    attached = command.attached.float()
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
    std: float,
) -> torch.Tensor:
    """Reward low torso-COM speed throughout an initialized climb."""

    command = _ladder_command(env, command_name)
    error = torch.sum(torch.square(command.torso_com_vel_w), dim=-1)
    stable = torch.exp(-error / std**2)
    return stable * (command.initialized & ~command.finished).float()


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
