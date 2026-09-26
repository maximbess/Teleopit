from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import mujoco
import torch
import torch.nn.functional as F
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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


# Rung indices are 0-based: index 0 is the lowest rung. The hands stay
# this many rungs above the feet, matching the climbing keyframe.
FOOT_HAND_RUNG_GAP = 3
# One rung below the feet through one rung above the hands.
RUNG_WINDOW_COUNT = FOOT_HAND_RUNG_GAP + 3


class LadderClimbCommand(CommandTerm):
    """Repeat one skill: move all four supports up exactly one rung and hold.

    The first attempt in a rollout starts on the climbing keyframe. Each
    attempt records the 0-based rung under the feet and the rung three above
    it under the hands. It completes only after every support is on the next
    rung and the stabilize gates stay true for a fixed number of frames. A
    broken gate clears that counter. The completed pose is kept, those rungs
    become the next baseline, and the same policy is asked for another rung.
    After ``successes_per_rollout`` holds the episode ends. The command never
    chooses a limb or a phase.
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
        if cfg.successes_per_rollout <= 0:
            raise ValueError("successes_per_rollout must be > 0")
        if cfg.attach_distance <= 0.0:
            raise ValueError("attach_distance must be > 0")
        if cfg.max_attach_speed < 0.0:
            raise ValueError("max_attach_speed must be >= 0")
        if cfg.foot_support_distance <= 0.0:
            raise ValueError("foot_support_distance must be > 0")
        if cfg.initialize_on_reset and not (
            0 <= cfg.initial_foot_rung < len(cfg.rung_site_names)
        ):
            raise ValueError(
                f"initial_foot_rung={cfg.initial_foot_rung} is outside "
                f"[0, {len(cfg.rung_site_names) - 1}]"
            )
        if cfg.start_rung != cfg.initial_foot_rung + FOOT_HAND_RUNG_GAP:
            raise ValueError(
                f"start_rung={cfg.start_rung} must be initial_foot_rung + "
                f"{FOOT_HAND_RUNG_GAP} "
                f"({cfg.initial_foot_rung + FOOT_HAND_RUNG_GAP})"
            )
        last_hand_rung = cfg.start_rung + cfg.successes_per_rollout - 1
        if cfg.initial_foot_rung < 1 or last_hand_rung + 1 >= len(cfg.rung_site_names):
            raise ValueError(
                "0-indexed window from one below the feet through one above "
                "the hands does not fit every attempt on this ladder"
            )
        if cfg.stabilization_dwell_steps <= 0:
            raise ValueError("stabilization_dwell_steps must be > 0")
        if cfg.max_stabilization_torso_speed < 0.0:
            raise ValueError("max_stabilization_torso_speed must be >= 0")
        if cfg.max_stabilization_joint_speed < 0.0:
            raise ValueError("max_stabilization_joint_speed must be >= 0")
        if cfg.max_stabilization_body_angular_speed < 0.0:
            raise ValueError("max_stabilization_body_angular_speed must be >= 0")
        if cfg.max_stabilization_waist_joint_speed < 0.0:
            raise ValueError("max_stabilization_waist_joint_speed must be >= 0")
        if cfg.max_stabilization_support_offset_error <= 0.0:
            raise ValueError("max_stabilization_support_offset_error must be > 0")
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
        self._rung_site_ids = torch.tensor(
            rung_site_ids, dtype=torch.long, device=self.device
        )
        self._rung_half_lengths = torch.tensor(
            rung_half_lengths,
            dtype=torch.float32,
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

        self.attached = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self.grip_strength = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.grip_request = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self.held_rung = torch.full(
            (self.num_envs, 2), -1, dtype=torch.long, device=self.device
        )
        self.foot_rung = torch.full(
            (self.num_envs, 2), -1, dtype=torch.long, device=self.device
        )
        self.baseline_hand_rung = torch.full(
            (self.num_envs, 2), cfg.start_rung, dtype=torch.long, device=self.device
        )
        self.baseline_foot_rung = torch.full(
            (self.num_envs, 2),
            cfg.initial_foot_rung,
            dtype=torch.long,
            device=self.device,
        )
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.hold_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._climb_distance = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._climb_lateral = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._climb_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._hold_count_prev = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._feet_off_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reward_cache_step = -1
        self._reward_cache: dict[str, torch.Tensor] | None = None
        self._pending_start_pose_init = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._skip_foot_detection = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
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
        self._reference_torso_rotation_w = self.torso_rotation_w.clone()
        self._reference_torso_support_offset_w = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )

        self.metrics["attached_hands"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["supported_feet"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hold_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["successes"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["supports_one_rung_higher"] = torch.zeros(
            self.num_envs, device=self.device
        )
        for metric_name in (
            "stabilization_hands_attached",
            "stabilization_feet_supported",
            "stabilization_torso_speed",
            "stabilization_joint_speed_rms",
            "stabilization_torso_angular_speed",
            "stabilization_pelvis_angular_speed",
            "stabilization_waist_joint_speed",
            "stabilization_support_offset_error",
            "stabilization_gate_valid",
        ):
            self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Whether each hand is attached. There is no phase or stage index."""

        return self.attached.float()

    @property
    def num_rungs(self) -> int:
        return int(self._rung_site_ids.numel())

    @property
    def hand_pos_w(self) -> torch.Tensor:
        """World positions with shape ``(num_envs, 2, 3)``."""

        return self._env.sim.data.site_xpos[:, self._hand_site_ids]

    @property
    def foot_pos_w(self) -> torch.Tensor:
        return self._env.sim.data.site_xpos[:, self._foot_site_ids]

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
    def foot_support(self) -> torch.Tensor:
        """Feet that are touching the ladder and standing on a rung."""

        return self.foot_contact & (self.foot_rung >= 0)

    @property
    def next_hand_rung(self) -> torch.Tensor:
        return self._next_rung(self.baseline_hand_rung)

    @property
    def next_foot_rung(self) -> torch.Tensor:
        return self._next_rung(self.baseline_foot_rung)

    @property
    def torso_pos_w(self) -> torch.Tensor:
        return self._env.sim.data.xpos[:, self._torso_body_id]

    @property
    def torso_com_pos_w(self) -> torch.Tensor:
        return self._env.sim.data.xipos[:, self._torso_body_id]

    @property
    def pelvis_com_pos_w(self) -> torch.Tensor:
        return self._env.sim.data.xipos[:, self._pelvis_body_id]

    @property
    def torso_rotation_w(self) -> torch.Tensor:
        return _as_rotation_matrix(self._env.sim.data.xmat[:, self._torso_body_id])

    @property
    def torso_com_vel_w(self) -> torch.Tensor:
        return self._torso_com_vel_w

    @property
    def torso_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self._torso_link_index]

    @property
    def pelvis_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self._pelvis_link_index]

    def _vectors_to_torso(self, vectors_w: torch.Tensor) -> torch.Tensor:
        world_to_torso = self.torso_rotation_w.transpose(-1, -2)
        while world_to_torso.ndim < vectors_w.ndim + 1:
            world_to_torso = world_to_torso.unsqueeze(1)
        return torch.matmul(world_to_torso, vectors_w.unsqueeze(-1)).squeeze(-1)

    def _points_to_torso(self, points_w: torch.Tensor) -> torch.Tensor:
        torso = self.torso_pos_w
        while torso.ndim < points_w.ndim:
            torso = torso.unsqueeze(1)
        return self._vectors_to_torso(points_w - torso)

    @property
    def support_centroid_w(self) -> torch.Tensor:
        hand_weights = self.attached.float()
        foot_weights = self.foot_support.float()
        weighted_sum = (self.hand_pos_w * hand_weights.unsqueeze(-1)).sum(dim=1)
        weighted_sum += (self.foot_pos_w * foot_weights.unsqueeze(-1)).sum(dim=1)
        total_weight = hand_weights.sum(dim=1) + foot_weights.sum(dim=1)
        return weighted_sum / total_weight.clamp_min(1.0).unsqueeze(-1)

    @property
    def torso_support_offset_w(self) -> torch.Tensor:
        return self.torso_com_pos_w - self.support_centroid_w

    @property
    def torso_orientation_error(self) -> torch.Tensor:
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
        return torch.linalg.vector_norm(
            self.torso_support_offset_w - self._reference_torso_support_offset_w,
            dim=-1,
        )

    @property
    def supports_one_rung_higher(self) -> torch.Tensor:
        """Whether every support is on the rung exactly one above its baseline."""

        hands = (self.held_rung == self.baseline_hand_rung + 1).all(dim=1)
        feet = (self.foot_rung == self.baseline_foot_rung + 1).all(dim=1)
        return self.attached.all(dim=1) & self.foot_support.all(dim=1) & hands & feet

    def window_rung_indices(self) -> torch.Tensor:
        """0-based rungs from one below the feet through one above the hands.

        Both feet share the foot baseline and both hands share the rung
        ``FOOT_HAND_RUNG_GAP`` above it. The indices come from those baselines,
        so they stay fixed until a completed hold moves the baselines up.
        """

        foot_rung = self.baseline_foot_rung[:, 0]
        start = foot_rung - 1
        offsets = torch.arange(
            RUNG_WINDOW_COUNT,
            dtype=torch.long,
            device=foot_rung.device,
        )
        return start[:, None] + offsets

    def _stabilization_conditions(self) -> dict[str, torch.Tensor]:
        """Gates that must stay true for the whole hold."""

        torso_speed = torch.linalg.vector_norm(self.torso_com_vel_w, dim=-1)
        joint_speed_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_vel), dim=1)
        )
        torso_angular_speed = torch.linalg.vector_norm(self.torso_ang_vel_w, dim=-1)
        pelvis_angular_speed = torch.linalg.vector_norm(self.pelvis_ang_vel_w, dim=-1)
        body_angular_speed = torch.maximum(torso_angular_speed, pelvis_angular_speed)
        waist_joint_speed = torch.amax(
            torch.abs(self.robot.data.joint_vel[:, self._waist_joint_ids]),
            dim=1,
        )
        hands_attached = self.attached.all(dim=1)
        feet_supported = self.foot_support.all(dim=1)
        torso_speed_valid = torso_speed <= self.cfg.max_stabilization_torso_speed
        joint_speed_valid = joint_speed_rms <= self.cfg.max_stabilization_joint_speed
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
            "support_offset_error": self.torso_support_offset_error,
            "support_offset_valid": support_offset_valid,
            "stable": stable,
        }

    @property
    def rung_endpoints_torso(self) -> torch.Tensor:
        """Return all finite rung endpoints in the torso frame.

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
        endpoints = torch.cat(
            (
                self._points_to_torso(endpoint_a_w),
                self._points_to_torso(endpoint_b_w),
                torch.ones(
                    (*centers.shape[:2], 1),
                    dtype=centers.dtype,
                    device=centers.device,
                ),
            ),
            dim=-1,
        )
        return endpoints

    @property
    def rung_tokens_torso(self) -> torch.Tensor:
        """Return the six 15D torso-frame tokens around the current stance.

        Rung indices are 0-based. The rows are one below the feet, the foot
        rung, the next foot rung, the rung between, the hand rung, and one
        above the hands. That set is fixed for the attempt.

        Each token contains two endpoints (6), a validity bit (1), left/right
        hand next-rung bits (2), left/right foot next-rung bits (2), left/right
        held-hand strengths (2), and left/right current foot-rung bits (2).
        """

        endpoints = self.rung_endpoints_torso
        rung_ids = torch.arange(
            self.num_rungs,
            dtype=torch.long,
            device=self.device,
        ).view(1, -1, 1)

        def markers(indices: torch.Tensor) -> torch.Tensor:
            return (rung_ids == indices[:, None, :]).to(endpoints.dtype)

        tokens = torch.cat(
            (
                endpoints,
                markers(self.next_hand_rung),
                markers(self.next_foot_rung),
                markers(self.held_rung) * self.grip_strength[:, None, :],
                markers(self.foot_rung),
            ),
            dim=-1,
        )
        window = self.window_rung_indices()
        index = window.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return torch.gather(tokens, 1, index)

    def compute(self, dt: float) -> None:
        """Update the climb, then add that state to the episode sums.

        The base command samples metrics before ``_update_command``, so the
        reset frame is recorded before the hands are welded. Episode end then
        logs whatever the last sample was, which for a fall is an empty ladder.
        """

        self.time_left -= dt
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        self._update_command()
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Integrate this control step. Reset logs the sums, then clears them.

        State signals are multiplied by the control step, so the log is in
        seconds (hand-seconds and foot-seconds for the two counts).
        ``successes`` adds one per completed hold and is not scaled by time.
        """

        dt = float(self._env.step_dt)
        conditions = self._stabilization_conditions()
        self.metrics["attached_hands"] += self.attached.sum(dim=1).float() * dt
        self.metrics["supported_feet"] += self.foot_support.sum(dim=1).float() * dt
        self.metrics["hold_progress"] += (
            torch.clamp(
                self.hold_count.float() / float(self.cfg.stabilization_dwell_steps),
                max=1.0,
            )
            * dt
        )
        self.metrics["successes"] += self.just_completed.float()
        self.metrics["supports_one_rung_higher"] += (
            self.supports_one_rung_higher.float() * dt
        )
        for metric_name, condition_name in (
            ("stabilization_hands_attached", "hands_attached"),
            ("stabilization_feet_supported", "feet_supported"),
            ("stabilization_torso_speed", "torso_speed"),
            ("stabilization_joint_speed_rms", "joint_speed_rms"),
            ("stabilization_torso_angular_speed", "torso_angular_speed"),
            ("stabilization_pelvis_angular_speed", "pelvis_angular_speed"),
            ("stabilization_waist_joint_speed", "waist_joint_speed"),
            ("stabilization_support_offset_error", "support_offset_error"),
            ("stabilization_gate_valid", "stable"),
        ):
            value = conditions[condition_name]
            sample = value.float() if value.dtype == torch.bool else value
            self.metrics[metric_name] += sample * dt

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.attached[env_ids] = False
        self.grip_strength[env_ids] = 0.0
        self.grip_request[env_ids] = False
        self.held_rung[env_ids] = -1
        self.initialized[env_ids] = False
        self.finished[env_ids] = False
        self.just_completed[env_ids] = False
        self.hold_count[env_ids] = 0
        self.successes[env_ids] = 0
        self._climb_valid[env_ids] = False
        self._hold_count_prev[env_ids] = 0
        self._feet_off_steps[env_ids] = 0
        self._reward_cache_step = -1
        self.baseline_hand_rung[env_ids] = self.cfg.start_rung
        self.baseline_foot_rung[env_ids] = self.cfg.initial_foot_rung
        self._pending_start_pose_init[env_ids] = self.cfg.initialize_on_reset
        if self.cfg.initialize_on_reset:
            self.foot_rung[env_ids] = self.cfg.initial_foot_rung
        else:
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

    def _update_command(self) -> None:
        self.just_completed.zero_()
        self._refresh_velocities(self._pending_start_pose_init)
        self._skip_foot_detection[:] = self._pending_start_pose_init
        self._initialize_from_start_pose()
        self._detect_foot_rungs()
        self._advance_hold()

    def _refresh_velocities(self, reset_ids: torch.Tensor) -> None:
        hand_pos = self.hand_pos_w
        self._previous_hand_pos_w[reset_ids] = hand_pos[reset_ids]
        self._hand_vel_w[:] = (hand_pos - self._previous_hand_pos_w) / self._env.step_dt
        self._previous_hand_pos_w[:] = hand_pos
        foot_pos = self.foot_pos_w
        self._previous_foot_pos_w[reset_ids] = foot_pos[reset_ids]
        self._foot_vel_w[:] = (foot_pos - self._previous_foot_pos_w) / self._env.step_dt
        self._previous_foot_pos_w[:] = foot_pos
        torso_com_pos = self.torso_com_pos_w
        self._previous_torso_com_pos_w[reset_ids] = torso_com_pos[reset_ids]
        self._torso_com_vel_w[:] = (
            torso_com_pos - self._previous_torso_com_pos_w
        ) / self._env.step_dt
        self._previous_torso_com_pos_w[:] = torso_com_pos

    def _initialize_from_start_pose(self) -> None:
        """Attach both hands on the climbing keyframe and record that baseline."""

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
        self.foot_rung[env_ids] = self.cfg.initial_foot_rung
        self.baseline_hand_rung[env_ids] = self.cfg.start_rung
        self.baseline_foot_rung[env_ids] = self.cfg.initial_foot_rung
        self.initialized[env_ids] = True
        self.hold_count[env_ids] = 0
        self.successes[env_ids] = 0
        self.finished[env_ids] = False
        self._reference_torso_rotation_w[env_ids] = self.torso_rotation_w[env_ids]
        hand_pos = self.hand_pos_w[env_ids]
        foot_pos = self.foot_pos_w[env_ids]
        centroid = (hand_pos.sum(dim=1) + foot_pos.sum(dim=1)) / 4.0
        self._reference_torso_support_offset_w[env_ids] = (
            self.torso_com_pos_w[env_ids] - centroid
        )

    def _detect_foot_rungs(self) -> None:
        for foot_id in (0, 1):
            indices, distances = self._nearest_rung(self.foot_pos_w[:, foot_id])
            supported = self.foot_contact[:, foot_id] & (
                distances <= self.cfg.foot_support_distance
            )
            detected = torch.where(supported, indices, torch.full_like(indices, -1))
            self.foot_rung[:, foot_id] = torch.where(
                self._skip_foot_detection,
                self.foot_rung[:, foot_id],
                detected,
            )

    def _advance_hold(self) -> None:
        """Count a fixed hold, then continue from the pose that completed it."""

        self.just_completed.zero_()
        conditions = self._stabilization_conditions()
        ready = (
            self.initialized
            & ~self.finished
            & self.supports_one_rung_higher
            & conditions["stable"]
        )
        self.hold_count = torch.where(
            ready,
            self.hold_count + 1,
            torch.zeros_like(self.hold_count),
        )
        complete = self.hold_count >= self.cfg.stabilization_dwell_steps
        if not torch.any(complete):
            return
        self.just_completed[complete] = True
        self.baseline_hand_rung[complete] = self.held_rung[complete]
        self.baseline_foot_rung[complete] = self.foot_rung[complete]
        self.hold_count[complete] = 0
        self.successes[complete] += 1
        self.finished[complete] = self.successes[complete] >= self.cfg.successes_per_rollout

    def _limb_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(d, ell)`` for both hands and both feet, shape ``(E, 4)``."""

        centers = self._env.sim.data.site_xpos[:, self._rung_site_ids]
        limbs = torch.cat((self.hand_pos_w, self.foot_pos_w), dim=1)
        baseline = torch.cat((self.baseline_hand_rung, self.baseline_foot_rung), dim=1)
        nxt = torch.clamp(baseline + 1, min=0, max=self.num_rungs - 1)
        on_hand = self.attached & (self.held_rung == self.baseline_hand_rung + 1)
        on_foot = self.foot_support & (self.foot_rung == self.baseline_foot_rung + 1)
        on_target = torch.cat((on_hand, on_foot), dim=1)
        return limb_axis_features(
            limbs,
            _gather_rung_centers(centers, baseline),
            _gather_rung_centers(centers, nxt),
            on_target,
        )

    def reward_terms(self) -> dict[str, torch.Tensor]:
        """Agent-visible ``r_climb``, ``r_hold``, and the flight rate for this step.

        Cached on ``common_step_counter`` so the four reward terms share one sample.
        """

        step_id = int(getattr(self._env, "common_step_counter", -1))
        if self._reward_cache is not None and self._reward_cache_step == step_id:
            return self._reward_cache

        distance, lateral = self._limb_features()
        refresh = ~self._climb_valid | self.just_completed
        climb = (self._climb_distance - distance).sum(dim=-1)
        climb = climb + (self._climb_lateral - lateral).sum(dim=-1)
        climb = torch.where(refresh, torch.zeros_like(climb), climb)
        self._climb_distance.copy_(distance)
        self._climb_lateral.copy_(lateral)
        self._climb_valid[:] = True

        dwell = float(self.cfg.stabilization_dwell_steps)
        hold_step = torch.where(
            self.just_completed,
            torch.ones_like(self.hold_count),
            self.hold_count - self._hold_count_prev,
        )
        hold = HOLD_REWARD * hold_step.float() / dwell
        self._hold_count_prev.copy_(self.hold_count)

        both_feet_off = (~self.foot_support).all(dim=-1)
        both_hands_off = (~self.attached).all(dim=-1)
        a_foot_off = (~self.foot_support).any(dim=-1)
        self._feet_off_steps = torch.where(
            both_feet_off,
            self._feet_off_steps + 1,
            torch.zeros_like(self._feet_off_steps),
        )
        flight = both_feet_off & (self._feet_off_steps > FLIGHT_GRACE_STEPS)
        drop = both_hands_off & a_foot_off & ~both_feet_off
        lost = flight | drop
        flight_rate = torch.where(
            lost,
            torch.full_like(climb, -FLIGHT_RATE),
            torch.zeros_like(climb),
        )
        self._reward_cache = {"climb": climb, "hold": hold, "flight": flight_rate}
        self._reward_cache_step = step_id
        return self._reward_cache

    def apply_grip_requests(self, request: torch.Tensor) -> None:
        """Attach a hand that asks for a nearby rung, and release when it stops."""

        request = request.to(device=self.device, dtype=torch.bool)
        self.grip_request = request
        for hand_id in (0, 1):
            release_ids = torch.where(self.attached[:, hand_id] & ~request[:, hand_id])[0]
            self._release(release_ids, hand_id)
            pending = request[:, hand_id] & ~self.attached[:, hand_id] & ~self.finished
            if not torch.any(pending):
                continue
            indices, distances = self._nearest_rung(self.hand_pos_w[:, hand_id])
            speeds = torch.linalg.vector_norm(self._hand_vel_w[:, hand_id], dim=-1)
            ready = (
                pending
                & (distances <= self.cfg.attach_distance)
                & (speeds <= self.cfg.max_attach_speed)
            )
            env_ids = torch.where(ready)[0]
            self._attach(env_ids, hand_id, indices[env_ids])

    def _attach(
        self,
        env_ids: torch.Tensor,
        hand_id: int,
        rung_indices: torch.Tensor,
    ) -> None:
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
        self.grip_strength[env_ids, hand_id] = 1.0
        self.held_rung[env_ids, hand_id] = rung_indices

    def _release(self, env_ids: torch.Tensor, hand_id: int) -> None:
        if env_ids.numel() == 0:
            return
        weld_id = int(self._weld_ids[hand_id].item())
        self._env.sim.data.eq_active[env_ids, weld_id] = False
        self.attached[env_ids, hand_id] = False
        self.grip_strength[env_ids, hand_id] = 0.0
        self.held_rung[env_ids, hand_id] = -1

    def _next_rung(self, baseline: torch.Tensor) -> torch.Tensor:
        return torch.clamp(baseline + 1, min=0, max=self.num_rungs - 1)

    def _nearest_rung(
        self,
        points_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the closest rung index and distance for each environment."""

        data = self._env.sim.data
        centers = data.site_xpos[:, self._rung_site_ids]
        matrices = _as_rotation_matrix(data.site_xmat[:, self._rung_site_ids])
        axes = matrices[..., :, 2]
        half_spans = self._rung_half_lengths.view(1, -1)
        offset = torch.sum((points_w.unsqueeze(1) - centers) * axes, dim=-1)
        offset = torch.maximum(torch.minimum(offset, half_spans), -half_spans)
        closest = centers + offset.unsqueeze(-1) * axes
        distances = torch.linalg.vector_norm(points_w.unsqueeze(1) - closest, dim=-1)
        indices = torch.argmin(distances, dim=-1)
        chosen = distances.gather(1, indices[:, None]).squeeze(1)
        return indices, chosen

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

    hand_site_names: tuple[str, str] = ("left_grip_site", "right_grip_site")
    foot_site_names: tuple[str, str] = ("left_foot", "right_foot")
    foot_contact_sensor_name: str = "ladder_foot_contact"
    foot_contact_body_names: tuple[str, str] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    anchor_body_names: tuple[str, str] = (
        "left_grip_anchor_body",
        "right_grip_anchor_body",
    )
    weld_names: tuple[str, str] = ("left_grip_weld", "right_grip_weld")
    rung_site_names: tuple[str, ...] = ()
    torso_body_name: str = "torso_link"
    pelvis_body_name: str = "pelvis"

    start_rung: int = 0
    initialize_on_reset: bool = False
    initial_foot_rung: int = 0
    successes_per_rollout: int = 4
    stabilization_dwell_steps: int = 50
    max_stabilization_torso_speed: float = 0.20
    max_stabilization_joint_speed: float = 1.0
    max_stabilization_body_angular_speed: float = 0.40
    max_stabilization_waist_joint_speed: float = 0.60
    max_stabilization_support_offset_error: float = 0.18
    attach_distance: float = 0.06
    max_attach_speed: float = 0.30
    foot_support_distance: float = 0.08
    grip_half_span: float | None = None

    def build(self, env: ManagerBasedRlEnv) -> LadderClimbCommand:
        return LadderClimbCommand(self, env)


@dataclass(kw_only=True)
class LadderGripActionCfg(ActionTermCfg):
    """Two grip requests, one for each hand, appended after the joint targets."""

    command_name: str = "ladder"
    threshold: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> LadderGripAction:
        return LadderGripAction(self, env)


class LadderGripAction(ActionTerm):
    """Forward each hand's grip request to the ladder command."""

    cfg: LadderGripActionCfg

    def __init__(self, cfg: LadderGripActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        command = cast(
            LadderClimbCommand,
            self._env.command_manager.get_term(self.cfg.command_name),
        )
        command.apply_grip_requests(self._raw_actions > self.cfg.threshold)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


def _ladder_command(env: ManagerBasedRlEnv, command_name: str) -> LadderClimbCommand:
    return cast(LadderClimbCommand, env.command_manager.get_term(command_name))


# One rung step on this ladder is 0.299 m. Four limbs are 1.196 m.
# PPO sees these values after the reward manager multiplies by the 0.02 s step.
CLIMB_CONTACT_BUBBLE_M = 0.10
LATERAL_COEFF = 0.5
RUNG_HEIGHT_BAND_M = 0.08
HOLD_REWARD = 0.5
FLIGHT_RATE = 2.0
FLIGHT_GRACE_STEPS = 4


def _gather_rung_centers(centers: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe = indices.clamp(0, centers.shape[1] - 1)
    return torch.gather(centers, 1, safe.unsqueeze(-1).expand(-1, -1, 3))


def limb_axis_features(
    limb: torch.Tensor,
    base_center: torch.Tensor,
    next_center: torch.Tensor,
    on_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance cap ``d`` and lateral cost ``ell`` for each limb.

    ``s`` is the remaining distance along the baseline-rung to next-rung axis.
    A limb on exactly that next rung has ``d = 0``. Otherwise

        d = max(s, 0.10)

    ``ell = 0.5 * lateral`` when the limb is within 0.08 m of the next rung
    height, and 0 when it is above that band.
    """

    axis = next_center - base_center
    length = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    direction = axis / length.clamp_min(1.0e-8)
    degenerate = length.squeeze(-1) < 1.0e-6
    remaining = ((next_center - limb) * direction).sum(dim=-1)
    remaining = torch.where(degenerate, torch.zeros_like(remaining), remaining.clamp_min(0.0))
    offset = limb - base_center
    along_base = (offset * direction).sum(dim=-1, keepdim=True)
    lateral = torch.linalg.vector_norm(offset - along_base * direction, dim=-1)
    lateral = torch.where(degenerate, torch.zeros_like(lateral), lateral)
    at_height = (limb[..., 2] - next_center[..., 2]).abs() <= RUNG_HEIGHT_BAND_M
    ell = torch.where(at_height, LATERAL_COEFF * lateral, torch.zeros_like(lateral))
    bubble = torch.full_like(remaining, CLIMB_CONTACT_BUBBLE_M)
    distance = torch.where(
        on_target,
        torch.zeros_like(remaining),
        torch.maximum(remaining, bubble),
    )
    return distance, ell


def ladder_climb(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Along-axis progress toward the next rung, plus lateral cost at rung height.

    For limbs ``i`` ordered left hand, right hand, left foot, right foot:

        r_climb = Σ_i (d_i⁻ − d_i) + Σ_i (ell_i⁻ − ell_i)

    ``d_i = 0`` when that limb is on exactly baseline + 1, and
    ``d_i = max(s_i, 0.10)`` otherwise. ``s_i`` is the remaining distance
    along the rung-to-rung axis. The first sample after a reset, and the step
    that records a completed hold, store the new distances and contribute 0.
    The reward manager multiplies by ``dt``, so this returns ``r_climb / dt``.
    """

    command = _ladder_command(env, command_name)
    return command.reward_terms()["climb"] / env.step_dt


def ladder_hold(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Potential on the dwell counter. One finished quiet second is worth 0.5.

        r_hold = 0.5 * (n_t − n_{t−1}) / N_dwell

    A broken gate sets ``n_t = 0``, which pays the accumulated count back.
    Completion also clears ``n``, but that step pays ``+0.5 / N_dwell`` instead
    of the payback. ``r_done`` carries the separate ``+1`` impulse. The reward
    manager multiplies by ``dt``, so this returns ``r_hold / dt``.
    """

    command = _ladder_command(env, command_name)
    return command.reward_terms()["hold"] / env.step_dt


def ladder_flight(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Lost-stance rate. The manager scales by ``dt``, so a step in flight is −0.04.

    Both feet off is free for 4 steps (0.08 s), then costs 2 per second. Both
    hands open while a foot is off costs the same rate immediately. Two hands
    moving with both feet still supported costs nothing.
    """

    command = _ladder_command(env, command_name)
    return command.reward_terms()["flight"]


def ladder_hold_completed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Completion impulse. The manager scales by ``dt``, so the buffer receives 1.

        r_done = 1

    when the dwell just finished, and 0 on a fall or a deadline.
    """

    command = _ladder_command(env, command_name)
    return command.just_completed.float() / env.step_dt


def ladder_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """End the rollout after N completed holds."""

    return _ladder_command(env, command_name).finished


def ladder_rung_endpoints_torso(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return _ladder_command(env, command_name).rung_endpoints_torso


def ladder_rung_tokens_torso(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    return _ladder_command(env, command_name).rung_tokens_torso


def ladder_remaining_time(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Normalized time to the terminal deadline, current critic frame only."""

    return (
        (1.0 - env.episode_length_buf.float() / env.max_episode_length)
        .clamp(0.0, 1.0)
        .unsqueeze(-1)
    )


def ladder_critic_privileged(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Hold progress and the stabilize gates, with no stage index."""

    command = _ladder_command(env, command_name)
    gates = command._stabilization_conditions()
    dwell = float(command.cfg.stabilization_dwell_steps)
    limit = float(command.cfg.successes_per_rollout)
    scalar_terms = (
        command.hold_count.float() / dwell,
        command.successes.float() / limit,
        gates["torso_speed"],
        gates["joint_speed_rms"],
        gates["torso_angular_speed"],
        gates["pelvis_angular_speed"],
        gates["waist_joint_speed"],
        command.torso_support_offset_error,
        command.supports_one_rung_higher.float(),
    )
    return torch.cat(
        (
            *(term.unsqueeze(-1) for term in scalar_terms),
            command.attached.float(),
            command.foot_support.float(),
        ),
        dim=-1,
    )
