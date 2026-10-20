from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.reward_manager import RewardManager
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def grip_marker_rgba(latched: bool, progress: float) -> tuple[float, float, float, float]:
    """Red while the weld holds, blue while the hand is open.

    Opacity follows grip progress from 10% at 0 to 100% at 1.
    """

    clamped = min(max(float(progress), 0.0), 1.0)
    alpha = 0.10 + 0.90 * clamped
    if latched:
        return (1.0, 0.0, 0.0, alpha)
    return (0.0, 0.0, 1.0, alpha)


def grip_latch_caption(side: str, latched: bool, progress: float) -> str:
    """Short latch readout, for example ``Left 40% open``."""

    clamped = min(max(float(progress), 0.0), 1.0)
    state = "latched" if latched else "open"
    return f"{side} {clamped * 100:.0f}% {state}"


def episode_reward_caption(total: float, last: float | None = None) -> str:
    """Viser readout for the running episode return."""

    text = f"**Episode total:** {float(total):.3f}"
    if last is None:
        return text
    return f"{text}\n\n**Last episode:** {float(last):.3f}"


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
FOOT_HAND_RUNG_GAP = 4
# One rung below the feet through one rung above the hands.
RUNG_WINDOW_COUNT = FOOT_HAND_RUNG_GAP + 3

_EPISODE_REWARD_LOG_PREFIX = "Episode_Reward/"


class EpisodeSumRewardManager(RewardManager):
    """Log the mean summed return of each reward term.

    MJLab reports ``mean(episode sum) / max_episode_length_s``, the return
    spread across the deadline. These logs are the mean of the summed return
    over the environments reset on this step.
    """

    def reset(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> dict[str, torch.Tensor]:
        extras = RewardManager.reset(self, env_ids)
        horizon_s = float(self._env.max_episode_length_s)
        for key, value in extras.items():
            if key.startswith(_EPISODE_REWARD_LOG_PREFIX):
                extras[key] = value * horizon_s
        return extras


def install_episode_sum_reward_logs(env: ManagerBasedRlEnv) -> None:
    """Make the next reset log reward totals, then leave ``_reset_idx`` alone.

    The ladder command is built before the reward manager, so the swap waits
    until the first environment reset.
    """

    if getattr(env, "_reward_logs_episode_sums", False):
        return
    original_reset_idx = env._reset_idx

    def _reset_idx(env_ids: torch.Tensor | None = None) -> None:
        manager = getattr(env, "reward_manager", None)
        if isinstance(manager, RewardManager) and not getattr(
            manager, "_logs_episode_sums", False
        ):
            manager.reset = MethodType(EpisodeSumRewardManager.reset, manager)  # type: ignore[method-assign]
            manager._logs_episode_sums = True
            env._reset_idx = original_reset_idx  # type: ignore[method-assign]
            env._reward_logs_episode_sums = True
        return original_reset_idx(env_ids)

    env._reset_idx = _reset_idx  # type: ignore[method-assign]
    env._reward_logs_episode_sums = True


class LadderClimbCommand(CommandTerm):
    """Repeat one skill: move all four supports up exactly one rung and hold.

    The first attempt in a rollout starts on the climbing keyframe. Each
    attempt records the 0-based rung under the feet and the hand rung
    ``FOOT_HAND_RUNG_GAP`` above it. The hold counts while every support
    stays on the next rung. It completes when that trailing window's mean
    torso speed and mean torso and pelvis angular speeds are inside the
    limits. Leaving a support clears the counter. The completed pose is kept,
    those rungs become the next baseline, and the same policy is asked for
    another rung.
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
        if cfg.lead_hand not in ("left", "right"):
            raise ValueError("lead_hand must be 'left' or 'right'")
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
        if cfg.max_stabilization_body_angular_speed < 0.0:
            raise ValueError("max_stabilization_body_angular_speed must be >= 0")
        if cfg.weld_release_speed_factor <= 0.0:
            raise ValueError("weld_release_speed_factor must be > 0")
        if cfg.weld_release_joint_speed < 0.0:
            raise ValueError("weld_release_joint_speed must be >= 0")
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
        self._hand_marker_radius = float(model.site_size[hand_site_ids[0], 0])
        self._hand_site_body_ids = np.asarray(
            [int(model.site_bodyid[site_id]) for site_id in hand_site_ids],
            dtype=np.int32,
        )
        self._hand_site_local_pos = np.asarray(
            [model.site_pos[site_id] for site_id in hand_site_ids],
            dtype=np.float64,
        )
        self._grip_marker_handle = None
        self._grip_marker_count = 0
        self._grip_marker_vertices: np.ndarray | None = None
        self._grip_marker_faces: np.ndarray | None = None
        self._grip_marker_latched: np.ndarray | None = None
        self._grip_marker_progress: np.ndarray | None = None
        self._grip_label_handles: dict[tuple[int, int], object] = {}
        self._grip_gui_bars: list[object] = []
        self._grip_gui_captions: list[object] = []
        self._grip_gui_env = None
        self._reward_total_caption = None
        self._reward_total_shown = 0.0
        self._reward_last_episode: float | None = None
        self._reward_total_env = -1
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
        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        rung_geom_ids: list[int] = []
        rung_box_half: list[list[float]] = []
        for name in cfg.rung_site_names:
            if not name.endswith("_grip"):
                raise ValueError(
                    f"Rung site {name!r} must be named after its box geom plus '_grip'"
                )
            geom_name = name[: -len("_grip")]
            geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if int(model.geom_type[geom_id]) != box_type:
                raise ValueError(f'Rung geom {geom_name!r} must have type="box"')
            half = model.geom_size[geom_id]
            if float(half[0]) <= 0.0 or float(half[2]) <= 0.0:
                raise ValueError(f"Rung geom {geom_name!r} has no cross-section")
            rung_geom_ids.append(geom_id)
            rung_box_half.append([float(half[0]), float(half[1]), float(half[2])])
        self._rung_geom_ids = torch.tensor(
            rung_geom_ids, dtype=torch.long, device=self.device
        )
        self._rung_box_half = torch.tensor(
            rung_box_half, dtype=torch.float32, device=self.device
        )
        rail_geom_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in SIDE_RAIL_NAMES
        ]
        rail_half_lengths: list[float] = []
        for name, geom_id in zip(SIDE_RAIL_NAMES, rail_geom_ids):
            if int(model.geom_type[geom_id]) != capsule_type:
                raise ValueError(f'Ladder rail {name!r} must have type="capsule"')
            half_length = float(model.geom_size[geom_id, 1])
            if half_length <= 0.0:
                raise ValueError(f"Ladder rail {name!r} has zero capsule length")
            rail_half_lengths.append(half_length)
        self._rail_geom_ids = torch.tensor(
            rail_geom_ids, dtype=torch.long, device=self.device
        )
        self._rail_half_lengths = torch.tensor(
            rail_half_lengths,
            dtype=torch.float32,
            device=self.device,
        )
        self._foot_sole_offsets = torch.tensor(
            FOOT_SOLE_END_OFFSETS,
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
        self.grip_progress = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.grip_request = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
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
        self._init_quiet_window()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # 0 first hand, 1 second hand, 2 legs. ``lead_hand`` chooses which
        # hand is first. Rewards read ``_reward_phase``.
        self._lead_hand = 0 if cfg.lead_hand == "left" else 1
        self._trail_hand = 1 - self._lead_hand
        self.climb_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reward_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._climb_distance = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._climb_lateral = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._climb_paid = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._leg_cross_paid = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._leg_cross_step = -1
        self._leg_cross_rate = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._side_cross_paid = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._side_cross_step = -1
        self._side_cross_rate = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._latch_pressure = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._flight_paid = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._flight_step = -1
        self._flight_rate = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._approach_paid = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._approach_peak = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._climb_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._hold_count_prev = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Baseline for the climbing-hand detach potential. The first sample
        # of a phase pays 0; later samples pay the change in (1 - progress).
        self._detach_progress_prev = self.grip_progress.clone()
        self._detach_tracked = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._detach_held_prev = self.held_rung.clone()
        self._detach_forced = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._detach_commanded = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._detach_done = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._robot_mass = self._robot_mass_kg()
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
        self.metrics["next_rung_hands"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["next_rung_feet"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["first_hand_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["second_hand_reached"] = torch.zeros(
            self.num_envs, device=self.device
        )
        for phase_name in (
            "first_hand_phase",
            "second_hand_phase",
            "lead_leg_phase",
            "trail_leg_phase",
            "hold_phase",
        ):
            self.metrics[phase_name] = torch.zeros(self.num_envs, device=self.device)
        for metric_name in (
            "stabilization_torso_speed",
            "stabilization_torso_angular_speed",
            "stabilization_pelvis_angular_speed",
            "hold_quiet",
        ):
            self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

        install_episode_sum_reward_logs(env)

    @property
    def command(self) -> torch.Tensor:
        """Latched hands, then grip progress. There is no phase index.

        The first two values are the weld. The last two are the ramp that
        closes the weld at 1 and opens it at 0.
        """

        return torch.cat((self.attached.float(), self.grip_progress), dim=-1)

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
    def foot_sole_points_w(self) -> torch.Tensor:
        """Heel and toe on the sole, shape ``(num_envs, 2, 2, 3)``.

        The offsets are in the foot-site frame. The site frame matches the
        ankle frame, so the climb axis picks whichever end is still behind.
        """

        rotation = _as_rotation_matrix(
            self._env.sim.data.site_xmat[:, self._foot_site_ids]
        )
        offset = torch.einsum("efij,pj->efpi", rotation, self._foot_sole_offsets)
        return self.foot_pos_w.unsqueeze(2) + offset

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

    def foot_bears_weight(self) -> torch.Tensor:
        """Whether each foot's upward ladder force carries 20% of body weight.

        The contact sensor reports the force on the foot. With the contact
        frame rotated into the world, a foot standing on a tread has a
        positive ``z`` component. Body weight is mass times the simulator
        gravity, so the same stance crosses this line when gravity changes.
        The step reward uses ``foot_load_fraction`` instead.
        """

        force = self._foot_contact_sensor.data.force
        if force is None:
            return torch.zeros(
                (self.num_envs, 2), dtype=torch.bool, device=self.device
            )
        upward = force[:, self._foot_contact_indices, 2]
        weight = self._robot_mass * self._gravity_magnitude()
        return foot_bears_load(
            upward,
            torch.as_tensor(weight, device=upward.device, dtype=upward.dtype),
        )

    def foot_load_fraction(self) -> torch.Tensor:
        """Share of body weight on each foot, clamped to ``[0, 0.35]``.

        The denominator is body mass times the simulator's gravity, so the
        same stance is the same share when gravity changes.
        """

        force = self._foot_contact_sensor.data.force
        if force is None:
            return torch.zeros(
                (self.num_envs, 2), dtype=torch.float32, device=self.device
            )
        upward = force[:, self._foot_contact_indices, 2]
        weight = self._robot_mass * self._gravity_magnitude()
        return foot_step_load(
            upward,
            torch.as_tensor(weight, device=upward.device, dtype=upward.dtype),
        )

    def _robot_mass_kg(self) -> float:
        """Body mass, excluding the ladder, from the unrandomized model."""

        model = self._env.sim.mj_model
        mass = 0.0
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if "ladder" in name:
                continue
            mass += float(model.body_mass[body_id])
        return mass

    def _gravity_magnitude(self) -> float:
        """Live gravity magnitude from the simulator, in m/s²."""

        gravity = torch.as_tensor(
            self._env.sim.mj_model.opt.gravity, dtype=torch.float32
        )
        return float(torch.linalg.vector_norm(gravity).clamp_min(1.0e-8).item())

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
        """Centroid of the hands and feet that are currently supporting."""

        hand_weights = self.attached.float()
        foot_weights = self.foot_support.float()
        weighted_sum = (self.hand_pos_w * hand_weights.unsqueeze(-1)).sum(dim=1)
        weighted_sum += (self.foot_pos_w * foot_weights.unsqueeze(-1)).sum(dim=1)
        total_weight = hand_weights.sum(dim=1) + foot_weights.sum(dim=1)
        return weighted_sum / total_weight.clamp_min(1.0).unsqueeze(-1)

    @property
    def torso_support_offset_error(self) -> torch.Tensor:
        """Torso offset from the support centroid, compared with the reset pose.

        Kept for the critic observation. Checkpoints trained before the hold
        change read this slot, and the hold itself does not.
        """

        return torch.linalg.vector_norm(
            self.torso_com_pos_w
            - self.support_centroid_w
            - self._reference_torso_support_offset_w,
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

    def _init_quiet_window(self) -> None:
        """Trailing speed samples for the planted hold. One row per environment."""

        dwell = int(self.cfg.stabilization_dwell_steps)
        envs = int(self.hold_count.shape[0])
        device = self.hold_count.device
        self._torso_speed_window = torch.zeros(
            (envs, dwell), dtype=torch.float32, device=device
        )
        self._torso_ang_window = torch.zeros(
            (envs, dwell), dtype=torch.float32, device=device
        )
        self._pelvis_ang_window = torch.zeros(
            (envs, dwell), dtype=torch.float32, device=device
        )
        self._torso_speed_sum = torch.zeros(envs, dtype=torch.float32, device=device)
        self._torso_ang_sum = torch.zeros(envs, dtype=torch.float32, device=device)
        self._pelvis_ang_sum = torch.zeros(envs, dtype=torch.float32, device=device)
        self._quiet_fill = torch.zeros(envs, dtype=torch.long, device=device)
        self._quiet_cursor = torch.zeros(envs, dtype=torch.long, device=device)
        self._hold_quiet = torch.zeros(envs, dtype=torch.bool, device=device)

    def _clear_quiet_window(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._quiet_fill[env_ids] = 0
        self._quiet_cursor[env_ids] = 0
        self._torso_speed_window[env_ids] = 0.0
        self._torso_ang_window[env_ids] = 0.0
        self._pelvis_ang_window[env_ids] = 0.0
        self._torso_speed_sum[env_ids] = 0.0
        self._torso_ang_sum[env_ids] = 0.0
        self._pelvis_ang_sum[env_ids] = 0.0

    def _record_quiet_sample(
        self,
        active: torch.Tensor,
        torso_speed: torch.Tensor,
        torso_ang: torch.Tensor,
        pelvis_ang: torch.Tensor,
    ) -> None:
        """Push one speed sample for each planted environment."""

        ids = torch.where(active)[0]
        if ids.numel() == 0:
            return
        dwell = int(self.cfg.stabilization_dwell_steps)
        cursor = self._quiet_cursor
        full = self._quiet_fill >= dwell
        cols = cursor[ids]
        self._torso_speed_window[ids, cols] = torso_speed[ids]
        self._torso_ang_window[ids, cols] = torso_ang[ids]
        self._pelvis_ang_window[ids, cols] = pelvis_ang[ids]
        self._torso_speed_sum[ids] = self._torso_speed_window[ids].sum(dim=1)
        self._torso_ang_sum[ids] = self._torso_ang_window[ids].sum(dim=1)
        self._pelvis_ang_sum[ids] = self._pelvis_ang_window[ids].sum(dim=1)
        self._quiet_fill = torch.where(
            active & ~full, self._quiet_fill + 1, self._quiet_fill
        )
        self._quiet_cursor = torch.where(
            active, (cursor + 1) % dwell, self._quiet_cursor
        )

    def _quiet_window_satisfied(self) -> torch.Tensor:
        """Whether the trailing planted second is slow enough to finish."""

        dwell = float(self.cfg.stabilization_dwell_steps)
        full = self._quiet_fill >= int(self.cfg.stabilization_dwell_steps)
        mean_torso = self._torso_speed_sum / dwell
        mean_torso_ang = self._torso_ang_sum / dwell
        mean_pelvis_ang = self._pelvis_ang_sum / dwell
        linear_limit = float(self.cfg.max_stabilization_torso_speed)
        angular_limit = float(self.cfg.max_stabilization_body_angular_speed)
        return (
            full
            & (mean_torso <= linear_limit)
            & (mean_torso_ang <= angular_limit)
            & (mean_pelvis_ang <= angular_limit)
        )

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
        """Return the 15D torso-frame tokens around the current stance.

        Rung indices are 0-based. The rows run from one rung below the feet
        through one rung above the hands, ``RUNG_WINDOW_COUNT`` of them. That
        set is fixed for the attempt.

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
        torso_speed = torch.linalg.vector_norm(self.torso_com_vel_w, dim=-1)
        torso_ang = torch.linalg.vector_norm(self.torso_ang_vel_w, dim=-1)
        pelvis_ang = torch.linalg.vector_norm(self.pelvis_ang_vel_w, dim=-1)
        self.metrics["attached_hands"] += self.attached.sum(dim=1).float() * dt
        self.metrics["supported_feet"] += self.foot_support.sum(dim=1).float() * dt
        on_next = self._limbs_on_next_rung()
        self.metrics["next_rung_hands"] += on_next[:, :2].sum(dim=1).float() * dt
        self.metrics["next_rung_feet"] += on_next[:, 2:].sum(dim=1).float() * dt
        self.metrics["hold_progress"] += (
            torch.clamp(
                self.hold_count.float() / float(self.cfg.stabilization_dwell_steps),
                max=1.0,
            )
            * dt
        )
        self.metrics["successes"] += self.just_completed.float()
        latched = self.attached & (self.held_rung == self.baseline_hand_rung + 1)
        phase = self._reward_phase
        self.metrics["first_hand_reached"] += (
            (phase == CLIMB_PHASE_FIRST) & latched[:, self._lead_hand]
        ).float()
        self.metrics["second_hand_reached"] += (
            (phase == CLIMB_PHASE_SECOND) & latched[:, self._trail_hand]
        ).float()
        self.metrics["first_hand_phase"] += (phase == CLIMB_PHASE_FIRST).float() * dt
        self.metrics["second_hand_phase"] += (phase == CLIMB_PHASE_SECOND).float() * dt
        self.metrics["lead_leg_phase"] += (phase == CLIMB_PHASE_LEAD_LEG).float() * dt
        self.metrics["trail_leg_phase"] += (phase == CLIMB_PHASE_TRAIL_LEG).float() * dt
        self.metrics["hold_phase"] += (phase == CLIMB_PHASE_HOLD).float() * dt
        self.metrics["supports_one_rung_higher"] += (
            self.supports_one_rung_higher.float() * dt
        )
        self.metrics["stabilization_torso_speed"] += torso_speed * dt
        self.metrics["stabilization_torso_angular_speed"] += torso_ang * dt
        self.metrics["stabilization_pelvis_angular_speed"] += pelvis_ang * dt
        self.metrics["hold_quiet"] += self._hold_quiet.float() * dt

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.attached[env_ids] = False
        self.grip_strength[env_ids] = 0.0
        self.grip_progress[env_ids] = 0.0
        self.grip_request[env_ids] = 0.0
        self.held_rung[env_ids] = -1
        self.initialized[env_ids] = False
        self.finished[env_ids] = False
        self.just_completed[env_ids] = False
        self.hold_count[env_ids] = 0
        self._clear_quiet_window(env_ids)
        self._hold_quiet[env_ids] = False
        self.successes[env_ids] = 0
        self._climb_distance[env_ids] = 0.0
        self._climb_lateral[env_ids] = 0.0
        self._climb_paid[env_ids] = 0.0
        self._leg_cross_paid[env_ids] = 0.0
        self._leg_cross_step = -1
        self._side_cross_paid[env_ids] = 0.0
        self._side_cross_step = -1
        self._latch_pressure[env_ids] = 0.0
        self._flight_paid[env_ids] = 0.0
        self._flight_step = -1
        self._approach_paid[env_ids] = 0.0
        self._approach_peak[env_ids] = 0.0
        if hasattr(self, "_plane_baseline"):
            self._plane_baseline[env_ids] = 0.0
        self._climb_valid[env_ids] = False
        self._hold_count_prev[env_ids] = 0
        self._detach_progress_prev[env_ids] = 0.0
        self._detach_tracked[env_ids] = False
        self._detach_held_prev[env_ids] = -1
        self._detach_forced[env_ids] = False
        self._detach_commanded[env_ids] = False
        self._detach_done[env_ids] = False
        self.climb_phase[env_ids] = CLIMB_PHASE_FIRST
        self._reward_phase[env_ids] = CLIMB_PHASE_FIRST
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
        self._release_over_speed_welds()
        self._reward_phase.copy_(self.climb_phase)
        self._advance_climb_phase()

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
        self.climb_phase[env_ids] = CLIMB_PHASE_FIRST
        self._reward_phase[env_ids] = CLIMB_PHASE_FIRST
        self.initialized[env_ids] = True
        self.hold_count[env_ids] = 0
        self._clear_quiet_window(env_ids)
        self._hold_quiet[env_ids] = False
        self.successes[env_ids] = 0
        self.finished[env_ids] = False
        hand_pos = self.hand_pos_w[env_ids]
        foot_pos = self.foot_pos_w[env_ids]
        centroid = (hand_pos.sum(dim=1) + foot_pos.sum(dim=1)) / 4.0
        self._reference_torso_support_offset_w[env_ids] = (
            self.torso_com_pos_w[env_ids] - centroid
        )

    def _detect_foot_rungs(self) -> None:
        centers = self._env.sim.data.site_xpos[:, self._rung_site_ids]
        for foot_id in (0, 1):
            foot_pos = self.foot_pos_w[:, foot_id]
            indices, _distances = self._nearest_rung(foot_pos)
            chosen = _gather_rung_centers(centers, indices.unsqueeze(1)).squeeze(1)
            half_span = self._rung_half_lengths[indices]
            on_tread = foot_on_tread(foot_pos, chosen, half_span)
            supported = self.foot_contact[:, foot_id] & on_tread
            detected = torch.where(supported, indices, torch.full_like(indices, -1))
            self.foot_rung[:, foot_id] = torch.where(
                self._skip_foot_detection,
                self.foot_rung[:, foot_id],
                detected,
            )

    def _advance_hold(self) -> None:
        """Count a planted hold, then continue from the pose that completed it.

        The counter runs only while every support is on the next rung. The
        trailing window has to be full and quiet before that pose counts.
        A planted window that is still fast stays full and waits.
        """

        self.just_completed.zero_()
        placed = (
            self.initialized
            & ~self.finished
            & self.supports_one_rung_higher
            & (self.climb_phase == CLIMB_PHASE_HOLD)
        )
        torso_speed = torch.linalg.vector_norm(self.torso_com_vel_w, dim=-1)
        torso_ang = torch.linalg.vector_norm(self.torso_ang_vel_w, dim=-1)
        pelvis_ang = torch.linalg.vector_norm(self.pelvis_ang_vel_w, dim=-1)
        leaving = torch.where(~placed)[0]
        self._clear_quiet_window(leaving)
        self.hold_count = torch.where(
            placed, self.hold_count, torch.zeros_like(self.hold_count)
        )
        self._record_quiet_sample(placed, torso_speed, torso_ang, pelvis_ang)
        self.hold_count = torch.where(placed, self._quiet_fill, self.hold_count)
        quiet = self._quiet_window_satisfied()
        self._hold_quiet = quiet
        complete = placed & quiet
        if not torch.any(complete):
            return
        self.just_completed[complete] = True
        self.baseline_hand_rung[complete] = self.held_rung[complete]
        self.baseline_foot_rung[complete] = self.foot_rung[complete]
        self.hold_count[complete] = 0
        self._clear_quiet_window(torch.where(complete)[0])
        self.successes[complete] += 1
        self.finished[complete] = (
            self.successes[complete] >= self.cfg.successes_per_rollout
        )

    def _advance_climb_phase(self) -> None:
        """Move a rung through the hands, then each leg, then the hold.

        The reward for this step still uses ``_reward_phase``. A hand phase
        ends when that hand latches the next rung. A leg phase ends when
        that foot is on the next rung and carries 20% of body weight. A
        finished hold returns to the lead hand for the next rung.
        """

        latched = self.attached & (self.held_rung == self.baseline_hand_rung + 1)
        loaded = self._foot_loaded_on_next()
        phase = self.climb_phase
        phase = torch.where(
            (phase == CLIMB_PHASE_FIRST) & latched[:, self._lead_hand],
            torch.full_like(phase, CLIMB_PHASE_SECOND),
            phase,
        )
        phase = torch.where(
            (phase == CLIMB_PHASE_SECOND) & latched[:, self._trail_hand],
            torch.full_like(phase, CLIMB_PHASE_LEAD_LEG),
            phase,
        )
        phase = torch.where(
            (phase == CLIMB_PHASE_LEAD_LEG) & loaded[:, self._lead_hand],
            torch.full_like(phase, CLIMB_PHASE_TRAIL_LEG),
            phase,
        )
        phase = torch.where(
            (phase == CLIMB_PHASE_TRAIL_LEG) & loaded[:, self._trail_hand],
            torch.full_like(phase, CLIMB_PHASE_HOLD),
            phase,
        )
        self.climb_phase = torch.where(
            self.just_completed, torch.zeros_like(phase), phase
        )

    def _foot_loaded_on_next(self) -> torch.Tensor:
        """Whether each foot is on the next rung with 20% of body weight."""

        on_next = self.foot_rung == self.baseline_foot_rung + 1
        return on_next & (self.foot_load_fraction() >= FOOT_WEIGHT_FRACTION)

    def _climbing_hand(self) -> torch.Tensor:
        """Hand index earning the climb, or -1 during the leg and hold phases."""

        phase = self._reward_phase
        hand = torch.full_like(phase, -1)
        hand = torch.where(
            phase == CLIMB_PHASE_FIRST,
            torch.full_like(phase, self._lead_hand),
            hand,
        )
        return torch.where(
            phase == CLIMB_PHASE_SECOND,
            torch.full_like(phase, self._trail_hand),
            hand,
        )

    def _climbing_foot(self) -> torch.Tensor:
        """Foot index earning the climb, or -1 outside the leg phases.

        The lead foot is the same side as the lead hand.
        """

        phase = self._reward_phase
        foot = torch.full_like(phase, -1)
        foot = torch.where(
            phase == CLIMB_PHASE_LEAD_LEG,
            torch.full_like(phase, self._lead_hand),
            foot,
        )
        return torch.where(
            phase == CLIMB_PHASE_TRAIL_LEG,
            torch.full_like(phase, self._trail_hand),
            foot,
        )

    def _active_limbs(self) -> torch.Tensor:
        """Which climb columns pay. Shape ``(E, 4)``.

        A hand or foot turns on when its phase starts and stays on through
        the later phases of this rung. The lead foot follows ``lead_hand``.
        Motion before a column turns on is marked paid, so it does not
        arrive later.
        """

        phase = self._reward_phase
        lead_hand = torch.ones_like(phase, dtype=torch.bool)
        trail_hand = phase >= CLIMB_PHASE_SECOND
        lead_leg = phase >= CLIMB_PHASE_LEAD_LEG
        trail_leg = phase >= CLIMB_PHASE_TRAIL_LEG
        if self._lead_hand == 0:
            left_arm, right_arm = lead_hand, trail_hand
            left_leg, right_leg = lead_leg, trail_leg
        else:
            left_arm, right_arm = trail_hand, lead_hand
            left_leg, right_leg = trail_leg, lead_leg
        return torch.stack((left_arm, right_arm, left_leg, right_leg), dim=1)

    def _limbs_finishing_phase(self) -> torch.Tensor:
        """Columns whose phase ended this step. Shape ``(E, 4)``.

        The lead hand, the other hand, the lead foot, then the other foot.
        A finished hold moves the phase backward, so it fills nothing.
        """

        started = self._reward_phase
        ended = self.climb_phase
        lead_hand = (started <= CLIMB_PHASE_FIRST) & (ended > CLIMB_PHASE_FIRST)
        trail_hand = (started <= CLIMB_PHASE_SECOND) & (ended > CLIMB_PHASE_SECOND)
        lead_leg = (started <= CLIMB_PHASE_LEAD_LEG) & (ended > CLIMB_PHASE_LEAD_LEG)
        trail_leg = (started <= CLIMB_PHASE_TRAIL_LEG) & (ended > CLIMB_PHASE_TRAIL_LEG)
        if self._lead_hand == 0:
            left_arm, right_arm = lead_hand, trail_hand
            left_leg, right_leg = lead_leg, trail_leg
        else:
            left_arm, right_arm = trail_hand, lead_hand
            left_leg, right_leg = trail_leg, lead_leg
        return torch.stack((left_arm, right_arm, left_leg, right_leg), dim=1)

    def _measured_latch_pressure(self) -> torch.Tensor:
        """How hard the latched body is loading the weld, in trip units.

        1 is the over-speed release. Joint speed and pelvis spin each have
        their own trip, and the pressure is the larger of the two ratios.
        An open grip has pressure 0: there is no weld to break.
        """

        joint_limit = max(float(self.cfg.weld_release_joint_speed), 1.0e-8)
        pelvis_limit = max(
            float(self.cfg.weld_release_speed_factor)
            * float(self.cfg.max_stabilization_body_angular_speed),
            1.0e-8,
        )
        joint_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_vel), dim=1)
        )
        pelvis_ang = torch.linalg.vector_norm(self.pelvis_ang_vel_w, dim=-1)
        pressure = torch.maximum(joint_rms / joint_limit, pelvis_ang / pelvis_limit)
        return torch.where(
            self.attached.any(dim=1), pressure, torch.zeros_like(pressure)
        )

    def _release_over_speed_welds(self) -> None:
        """Open both grips when the body is spinning hard enough to blow the weld.

        The check runs after the physics step, so the next step is unwelded.
        Joint speed uses an absolute trip. Pelvis spin uses the quiet-window
        limit times the release factor, so a quiet hold keeps its grips.
        Pressure is recorded before the release, so the step that breaks the
        weld still counts.
        """

        self._latch_pressure = self._measured_latch_pressure()
        factor = float(self.cfg.weld_release_speed_factor)
        joint_speed_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_vel), dim=1)
        )
        pelvis_ang = torch.linalg.vector_norm(self.pelvis_ang_vel_w, dim=-1)
        over_speed = (joint_speed_rms > float(self.cfg.weld_release_joint_speed)) | (
            pelvis_ang
            > factor * float(self.cfg.max_stabilization_body_angular_speed)
        )
        env_ids = torch.where(over_speed & self.attached.any(dim=1))[0]
        if env_ids.numel() == 0:
            return
        for hand_id in (0, 1):
            self._release(env_ids, hand_id)

    def _limbs_on_next_rung(self) -> torch.Tensor:
        """Whether each limb is attached or supported on exactly baseline + 1."""

        on_hand = self.attached & (self.held_rung == self.baseline_hand_rung + 1)
        on_foot = self.foot_support & (self.foot_rung == self.baseline_foot_rung + 1)
        return torch.cat((on_hand, on_foot), dim=1)

    def _limb_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(d, ell)`` for both hands and both feet, shape ``(E, 4)``."""

        centers = self._env.sim.data.site_xpos[:, self._rung_site_ids]
        on_target = self._limbs_on_next_rung()
        hand_next = torch.clamp(
            self.baseline_hand_rung + 1, min=0, max=self.num_rungs - 1
        )
        foot_next = torch.clamp(
            self.baseline_foot_rung + 1, min=0, max=self.num_rungs - 1
        )
        origin, plane_lateral, climb, degenerate = self.bottom_rung_plane()
        normal = torch.cross(plane_lateral, climb, dim=-1)
        edge = _gather_rung_centers(self._rung_far_top_w(), hand_next)
        plane_gap, along = hand_approach_gaps(
            self.hand_pos_w, edge, origin, climb, normal, degenerate
        )
        # The climb to the edge stays in the approach account. The plane gap is
        # penalized on its own, so leaving the ladder is not floored away.
        self._hand_plane_gap = plane_gap
        hand_distance = along
        hand_lateral = torch.zeros_like(hand_distance)
        next_foot_center = _gather_rung_centers(centers, foot_next)
        foot_distance, foot_lateral = foot_tread_features(
            self.foot_sole_points_w,
            _gather_rung_centers(centers, self.baseline_foot_rung),
            next_foot_center,
            on_target[:, 2:],
        )
        return (
            torch.cat((hand_distance, foot_distance), dim=1),
            torch.cat((hand_lateral, foot_lateral), dim=1),
        )

    def _torso_rungs_along_rail(self) -> torch.Tensor:
        """Torso COM position along the rail, measured in rung pitches."""

        centers = self._env.sim.data.site_xpos[:, self._rung_site_ids]
        axis = centers[:, 1] - centers[:, 0]
        pitch = torch.linalg.vector_norm(axis, dim=-1).clamp_min(1.0e-8)
        direction = axis / pitch.unsqueeze(-1)
        along = (self.torso_com_pos_w * direction).sum(dim=-1)
        return along / pitch

    def _ground_contact(self, like: torch.Tensor) -> torch.Tensor:
        """Whether this step's termination is the body touching the ground."""

        manager = getattr(self._env, "termination_manager", None)
        get_term = getattr(manager, "get_term", None)
        if not callable(get_term):
            return torch.zeros(like.shape[0], dtype=torch.bool, device=like.device)
        try:
            done = get_term("touched_ground")
        except (KeyError, ValueError):
            return torch.zeros(like.shape[0], dtype=torch.bool, device=like.device)
        return done.to(device=like.device, dtype=torch.bool).reshape(like.shape[0])

    def _episode_ending(self) -> torch.Tensor:
        """Envs whose reward this step is the last of the episode.

        Terminations are computed before the reward, so ``reset_buf`` is the
        terminal step. The kept fraction of a drop is paid here, once.
        """

        reset = getattr(self._env, "reset_buf", None)
        if isinstance(reset, torch.Tensor):
            return reset.to(dtype=torch.bool)
        return torch.zeros(self._climb_valid.shape[0], dtype=torch.bool)

    def reward_terms(self) -> dict[str, torch.Tensor]:
        """Agent-visible climb, hold, and alive terms.

        Cached on ``common_step_counter`` so the reward terms share one sample.
        A hand or foot's climb turns on when its phase starts and stays on
        for the rest of the rung. The lead foot is the same side as the
        lead hand.
        """

        step_id = int(getattr(self._env, "common_step_counter", -1))
        if self._reward_cache is not None and self._reward_cache_step == step_id:
            return self._reward_cache

        # A hand's approach distance is how far it still is along the rail from
        # the far top edge of the next rung. Leaving the ladder plane is
        # penalized separately, at 0.5 per meter from the attached pose.
        # A foot's distance falls until the sole reaches the next tread.
        # A limb's approach account stops at 0, so moving far away only gives
        # back what that limb earned. Contact accrues
        # separately and stops at +3 per limb, including during the hold.
        # A latched hand on the next rung pays 0.25 per second. An unlatched
        # hand in that rung's zone pays the same rate times grip progress.
        # A foot on the next tread pays 0.10 per second for the contact, plus
        # 1 per second times its share of body weight. That share stops at 0.35.
        # Completion pays each active planted limb the unpaid part of its cap.
        dt = float(self._env.step_dt)
        ending = self._episode_ending()
        keep_share = 1.0 - REFUND_FRACTION
        distance, lateral = self._limb_features()
        on_next = self._limbs_on_next_rung()
        load = self.foot_load_fraction().clamp(0.0, FOOT_LOAD_CAP)
        hand_scale = self._hand_contact_scale()
        contact_rate = torch.cat(
            (
                torch.full_like(load[:, :2], CLIMB_CONTACT_RATE) * hand_scale,
                FOOT_STEP_RATE + FOOT_PRESSURE_RATE * load,
            ),
            dim=1,
        )
        # Feet still require the next tread. A hand earns while its scale is
        # positive, which is the latch on that rung or progress inside its zone.
        contact_open = torch.cat((hand_scale > 0.0, on_next[:, 2:]), dim=1)
        episode_fresh = ~self._climb_valid
        hold_fresh = episode_fresh | self.just_completed
        # Approach is the gap closed since the reset sample, or since the last
        # completed hold. The account stops at 0, so a limb that wanders far
        # gives back only what it earned, and coming back re-earns that same
        # gap rather than the extra distance.
        score = CLIMB_DISTANCE_PER_M * (
            (self._climb_distance - distance) + (self._climb_lateral - lateral)
        )
        shown = score.clamp(min=0.0)
        approach = torch.where(
            hold_fresh.unsqueeze(1),
            torch.zeros_like(shown),
            shown - self._approach_paid,
        )
        new_approach = torch.where(
            hold_fresh.unsqueeze(1), torch.zeros_like(shown), shown
        )
        new_approach_peak = torch.where(
            hold_fresh.unsqueeze(1),
            torch.zeros_like(self._approach_peak),
            torch.maximum(self._approach_peak, shown),
        )
        approach_kept = keep_share * (new_approach_peak - new_approach)
        approach = approach + torch.where(
            ending.unsqueeze(1) & ~hold_fresh.unsqueeze(1),
            approach_kept,
            torch.zeros_like(approach_kept),
        )
        room = (CLIMB_CONTACT_CAP - self._climb_paid).clamp(min=0.0)
        contact = torch.minimum(room, contact_rate * dt)
        contact = torch.where(
            contact_open & ~hold_fresh.unsqueeze(1),
            contact,
            torch.zeros_like(contact),
        )
        # A finished hold pays whatever is left of each planted limb's cap,
        # then the reset below starts the next rung at zero. The rate slice
        # is already zero on this step, so the remainder is not counted twice.
        contact = torch.where(
            self.just_completed.unsqueeze(1) & on_next,
            room,
            contact,
        )
        # A hand or foot stays on after its phase starts. Motion before a
        # column turns on is marked paid below.
        active = self._active_limbs()
        approach = torch.where(active, approach, torch.zeros_like(approach))
        contact = torch.where(active, contact, torch.zeros_like(contact))
        # The step a phase ends pays that limb whatever is left of its cap,
        # instead of the rate slice. Later steps on this rung pay no more
        # contact for it. A skip fills every limb whose phase was crossed.
        contact = torch.where(
            self._limbs_finishing_phase() & ~hold_fresh.unsqueeze(1),
            room,
            contact,
        )
        # -0.5 per meter out of the ladder plane, measured from the pose at
        # the reset or the last completed hold. Attached to the ladder, that
        # pose is the zero. Moving out pays the meters. Moving back returns them.
        plane_gap = getattr(self, "_hand_plane_gap", None)
        plane_penalty = torch.zeros_like(distance)
        if isinstance(plane_gap, torch.Tensor):
            if (
                not hasattr(self, "_plane_baseline")
                or self._plane_baseline.shape != plane_gap.shape
            ):
                self._plane_baseline = plane_gap.detach().clone()
            opened = plane_gap - self._plane_baseline
            plane_penalty[:, :2] = torch.where(
                hold_fresh.unsqueeze(1),
                torch.zeros_like(opened),
                -CLIMB_DISTANCE_PER_M * opened,
            )
            self._plane_baseline.copy_(plane_gap)
        plane_penalty = torch.where(
            active, plane_penalty, torch.zeros_like(plane_penalty)
        )
        # One column per limb: left arm, right arm, left leg, right leg.
        # The reward manager reads each column on its own.
        climb = approach + contact + plane_penalty
        self._climb_distance.copy_(
            torch.where(hold_fresh.unsqueeze(1), distance, self._climb_distance)
        )
        self._climb_lateral.copy_(
            torch.where(hold_fresh.unsqueeze(1), lateral, self._climb_lateral)
        )
        self._approach_paid.copy_(new_approach)
        self._approach_peak.copy_(new_approach_peak)
        paid = torch.where(
            hold_fresh.unsqueeze(1),
            torch.zeros_like(self._climb_paid),
            self._climb_paid + contact,
        )
        self._climb_paid.copy_(paid)
        self._climb_valid[:] = True

        dwell = float(self.cfg.stabilization_dwell_steps)
        reached = torch.full_like(self._hold_count_prev, int(self.cfg.stabilization_dwell_steps))
        hold_step = torch.where(
            self.just_completed,
            (reached - self._hold_count_prev).clamp(min=0),
            self.hold_count - self._hold_count_prev,
        )
        hold = HOLD_REWARD * hold_step.float() / dwell
        hold = torch.where(
            self._reward_phase == CLIMB_PHASE_HOLD,
            hold,
            torch.zeros_like(hold),
        )
        self._hold_count_prev.copy_(self.hold_count)
        climb_total = climb.sum(dim=-1)
        elapsed = self._env.episode_length_buf.to(
            device=climb_total.device, dtype=climb_total.dtype
        ) * dt
        on_ground = self._ground_contact(climb_total)
        alive = torch.where(
            (elapsed <= ALIVE_WINDOW_S) & ~on_ground,
            torch.full_like(climb_total, ALIVE_RATE),
            torch.zeros_like(climb_total),
        )
        self._reward_cache = {
            "climb": climb,
            "hold": hold,
            "alive": alive,
        }
        self._reward_cache_step = step_id
        return self._reward_cache

    def _hand_contact_scale(self) -> torch.Tensor:
        """Full contact when the latch holds the next rung, else progress there.

        An open hand away from its next rung pays nothing. Progress is already
        0 outside every latch zone, and this also refuses the rung the hand
        is already standing on.
        """

        on_next = self.attached & (self.held_rung == self.baseline_hand_rung + 1)
        partial = self._hands_in_next_rung_zone() & ~self.attached
        progress = self.grip_progress
        return torch.where(
            on_next,
            torch.ones_like(progress),
            torch.where(partial, progress, torch.zeros_like(progress)),
        )

    def _hands_in_next_rung_zone(self) -> torch.Tensor:
        """Whether each open reach is inside the next rung's latch radius."""

        distances = self._axis_distances(self.hand_pos_w)
        nxt = self._next_rung(self.baseline_hand_rung)
        chosen = distances.gather(2, nxt.unsqueeze(-1)).squeeze(-1)
        return chosen <= float(self.cfg.attach_distance)

    def feet_on_flight_rungs(self) -> torch.Tensor:
        """Feet touching the top face of their baseline rung or the next one."""

        data = self._env.sim.data
        centers = data.site_xpos[:, self._rung_site_ids]
        baseline = self.baseline_foot_rung
        landing = self._next_rung(baseline)
        return self.foot_contact & (
            self._feet_on_rung_face(centers, baseline)
            | self._feet_on_rung_face(centers, landing)
        )

    def _feet_on_rung_face(
        self,
        centers: torch.Tensor,
        rung_indices: torch.Tensor,
    ) -> torch.Tensor:
        chosen = _gather_rung_centers(centers, rung_indices)
        half_span = self._rung_half_lengths[rung_indices]
        return foot_on_tread(self.foot_pos_w, chosen, half_span)

    def apply_grip_requests(self, request: torch.Tensor) -> None:
        """Advance grip progress. A command of ±1 covers the ramp in one second.

        Progress stays 0 while an open hand is outside the latch radius, so
        leaving that radius before the weld closes drops the ramp. The weld
        closes when progress reaches 1 and opens when it reaches 0.
        """

        request = request.to(device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        self.grip_request = request
        step = float(self._env.step_dt) / GRIP_TRAVEL_S
        for hand_id in (0, 1):
            indices, distances = self._nearest_rung(self.hand_pos_w[:, hand_id])
            in_zone = distances <= float(self.cfg.attach_distance)
            latched = self.attached[:, hand_id]
            progress = self.grip_progress[:, hand_id] + request[:, hand_id] * step
            progress = progress.clamp(0.0, 1.0)
            progress = torch.where(
                latched | in_zone,
                progress,
                torch.zeros_like(progress),
            )
            progress = torch.where(
                progress >= 1.0 - 1.0e-4,
                torch.ones_like(progress),
                progress,
            )
            self.grip_progress[:, hand_id] = progress
            self.grip_strength[:, hand_id] = progress
            # A full second of -1 lands on 0, aside from float dust.
            self._release(
                torch.where(latched & (progress <= 1.0e-4))[0],
                hand_id,
                forced=False,
            )
            close = (
                ~self.attached[:, hand_id]
                & (self.grip_progress[:, hand_id] >= 1.0)
                & in_zone
                & ~self.finished
            )
            env_ids = torch.where(close)[0]
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
        self.grip_progress[env_ids, hand_id] = 1.0
        self.held_rung[env_ids, hand_id] = rung_indices

    def _release(
        self, env_ids: torch.Tensor, hand_id: int, *, forced: bool = True
    ) -> None:
        if env_ids.numel() == 0:
            return
        if not hasattr(self, "_detach_forced"):
            self._detach_forced = torch.zeros(
                self.attached.shape, dtype=torch.bool, device=self.attached.device
            )
        if not hasattr(self, "_detach_commanded"):
            self._detach_commanded = torch.zeros(
                self.attached.shape, dtype=torch.bool, device=self.attached.device
            )
        if forced:
            was_on = self.attached[env_ids, hand_id]
            forced_ids = env_ids[was_on]
            if forced_ids.numel() > 0:
                self._detach_forced[forced_ids, hand_id] = True
                self._detach_commanded[forced_ids, hand_id] = False
        else:
            self._detach_commanded[env_ids, hand_id] = True
            self._detach_forced[env_ids, hand_id] = False
        weld_id = int(self._weld_ids[hand_id].item())
        self._env.sim.data.eq_active[env_ids, weld_id] = False
        self.attached[env_ids, hand_id] = False
        self.grip_strength[env_ids, hand_id] = 0.0
        self.grip_progress[env_ids, hand_id] = 0.0
        self.held_rung[env_ids, hand_id] = -1

    def _next_rung(self, baseline: torch.Tensor) -> torch.Tensor:
        return torch.clamp(baseline + 1, min=0, max=self.num_rungs - 1)

    def foot_span_plane(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Face spanned by the foot baseline rung and the landing rung.

        Returns the baseline center, the rung axis, the climb axis, and a
        mask that is true when those two rungs do not define a plane. The
        axes are orthonormal. On this ladder every rung is parallel, so any
        pair gives the same face; the indices follow the current attempt.
        """

        data = self._env.sim.data
        centers = data.site_xpos[:, self._rung_site_ids]
        matrices = _as_rotation_matrix(data.site_xmat[:, self._rung_site_ids])
        start = self.baseline_foot_rung.amin(dim=1, keepdim=True)
        landing = self._next_rung(start)
        origin = _gather_rung_centers(centers, start).squeeze(1)
        rung_axis = _gather_rung_centers(matrices[..., :, 2], start).squeeze(1)
        landing_center = _gather_rung_centers(centers, landing).squeeze(1)
        return ladder_face_axes(origin, landing_center, rung_axis)

    def bottom_rung_plane(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Face spanned by the two lowest rungs.

        Returns the lowest center, the rung axis, the climb axis, and a mask
        that is true when those two rungs do not define a plane. The rails of
        the climbing face lie in this plane.
        """

        if self.num_rungs < 2:
            origin = torch.zeros(self.num_envs, 3, device=self.device)
            lateral = torch.zeros_like(origin)
            lateral[:, 1] = 1.0
            climb = torch.zeros_like(origin)
            climb[:, 2] = 1.0
            degenerate = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            return origin, lateral, climb, degenerate
        data = self._env.sim.data
        centers = data.site_xpos[:, self._rung_site_ids]
        matrices = _as_rotation_matrix(data.site_xmat[:, self._rung_site_ids])
        origin = centers[:, 0]
        rung_axis = matrices[:, 0, :, 2]
        return ladder_face_axes(origin, centers[:, 1], rung_axis)

    def _rung_far_top_w(self) -> torch.Tensor:
        """World midpoint of each rung box's far top edge, shape ``(E, R, 3)``.

        The half-extents come from the rung geom, so a rebuilt ladder moves
        this edge. Local +x is the far side and local +z is the top.
        """

        data = self._env.sim.data
        centers = data.geom_xpos[:, self._rung_geom_ids]
        rotation = _as_rotation_matrix(data.geom_xmat[:, self._rung_geom_ids])
        return rung_far_top_edge(centers, rotation, self._rung_box_half)

    def rail_endpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World endpoints of the climbing-face side rails.

        Each result has shape ``(num_envs, num_rails, 3)``. The capsule axis
        is the geom's local z, and the stored size is the half-length.
        """

        data = self._env.sim.data
        centers = data.geom_xpos[:, self._rail_geom_ids]
        matrices = _as_rotation_matrix(data.geom_xmat[:, self._rail_geom_ids])
        axes = matrices[..., :, 2]
        half_lengths = self._rail_half_lengths.view(1, -1, 1)
        return centers - axes * half_lengths, centers + axes * half_lengths

    def _axis_distances(self, points_w: torch.Tensor) -> torch.Tensor:
        """Distance from each point to each rung axis.

        ``points_w`` has shape ``(E, P, 3)``. The result has shape ``(E, P, R)``.
        """

        data = self._env.sim.data
        centers = data.site_xpos[:, self._rung_site_ids]
        matrices = _as_rotation_matrix(data.site_xmat[:, self._rung_site_ids])
        axes = matrices[..., :, 2]
        half_spans = self._rung_half_lengths.view(1, 1, -1)
        delta = points_w.unsqueeze(2) - centers.unsqueeze(1)
        offset = (delta * axes.unsqueeze(1)).sum(dim=-1)
        offset = torch.maximum(torch.minimum(offset, half_spans), -half_spans)
        closest = centers.unsqueeze(1) + offset.unsqueeze(-1) * axes.unsqueeze(1)
        return torch.linalg.vector_norm(points_w.unsqueeze(2) - closest, dim=-1)

    def _nearest_rung(
        self,
        points_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the closest rung index and distance for each environment."""

        distances = self._axis_distances(points_w.unsqueeze(1)).squeeze(1)
        indices = torch.argmin(distances, dim=-1)
        chosen = distances.gather(1, indices[:, None]).squeeze(1)
        return indices, chosen

    def _debug_vis_impl(self, visualizer) -> None:
        """Paint a sphere on each grip site from the latch and its progress."""

        self._grip_marker_latched = self.attached.detach().cpu().numpy()
        self._grip_marker_progress = self.grip_progress.detach().cpu().numpy()
        self._update_viser_gui()
        if hasattr(visualizer, "server"):
            self._install_viser_grip_markers(visualizer)
            return

        env_indices = list(visualizer.get_env_indices(self.num_envs))
        radius = self._hand_marker_radius
        for batch in env_indices:
            for hand_id in (0, 1):
                rgba = grip_marker_rgba(
                    bool(self.attached[batch, hand_id].item()),
                    float(self.grip_progress[batch, hand_id].item()),
                )
                visualizer.add_sphere(
                    self.hand_pos_w[batch, hand_id].detach().cpu().numpy(),
                    radius,
                    rgba,
                )

    def _install_viser_grip_markers(self, visualizer) -> None:
        """Redraw the spheres in the same update that places the robot.

        The viewer subtracts the tracked body position from every mesh. Doing
        that here, after the scene offset is chosen, keeps the spheres on the
        hands. The viewer's own debug spheres share one opacity.
        """

        if getattr(visualizer, "_grip_marker_hook_installed", False):
            return
        original = visualizer._update_visualization_locked

        def _update_visualization_locked(*args, **kwargs):
            original(*args, **kwargs)
            body_xpos = args[0] if args else kwargs["body_xpos"]
            body_xmat = args[1] if len(args) > 1 else kwargs["body_xmat"]
            scene_offset = args[5] if len(args) > 5 else kwargs["scene_offset"]
            self._paint_viser_grip_markers(
                visualizer, body_xpos, body_xmat, scene_offset
            )

        visualizer._update_visualization_locked = _update_visualization_locked
        visualizer._grip_marker_hook_installed = True

    def _paint_viser_grip_markers(
        self,
        visualizer,
        body_xpos: np.ndarray,
        body_xmat: np.ndarray,
        scene_offset: np.ndarray,
    ) -> None:
        enabled = (
            self.cfg.debug_vis
            and self._debug_vis_enabled
            and getattr(visualizer, "debug_visualization_enabled", True)
        )
        handle = self._grip_marker_handle
        self._update_viser_gui()
        if not enabled:
            if handle is not None:
                handle.visible = False
            self._set_grip_labels_visible(False)
            return

        env_indices = list(visualizer.get_env_indices(body_xpos.shape[0]))
        if not env_indices:
            if handle is not None:
                handle.visible = False
            self._set_grip_labels_visible(False)
            return

        latched = self._grip_marker_latched
        progress = self._grip_marker_progress
        if latched is None or progress is None:
            return

        offset = np.asarray(scene_offset, dtype=np.float64)
        positions: list[np.ndarray] = []
        colors: list[tuple[int, int, int]] = []
        opacities: list[float] = []
        label_keys: list[tuple[int, int]] = []
        label_texts: list[str] = []
        label_positions: list[np.ndarray] = []
        sides = self._hand_side_labels()
        for batch in env_indices:
            for hand_id in (0, 1):
                body_id = int(self._hand_site_body_ids[hand_id])
                rotation = np.asarray(body_xmat[batch, body_id], dtype=np.float64)
                if rotation.shape == (9,):
                    rotation = rotation.reshape(3, 3)
                center = (
                    np.asarray(body_xpos[batch, body_id], dtype=np.float64)
                    + rotation @ self._hand_site_local_pos[hand_id]
                    + offset
                )
                rgba = grip_marker_rgba(
                    bool(latched[batch, hand_id]),
                    float(progress[batch, hand_id]),
                )
                positions.append(center.astype(np.float32))
                colors.append(
                    tuple(int(round(channel * 255.0)) for channel in rgba[:3])
                )
                opacities.append(rgba[3])
                label_keys.append((int(batch), hand_id))
                label_texts.append(
                    grip_latch_caption(
                        sides[hand_id],
                        bool(latched[batch, hand_id]),
                        float(progress[batch, hand_id]),
                    )
                )
                label_positions.append(
                    (center + np.array([0.0, 0.0, 0.08], dtype=np.float64)).astype(
                        np.float32
                    )
                )
        self._sync_viser_grip_markers(visualizer, positions, colors, opacities)
        self._sync_viser_grip_labels(
            visualizer, label_keys, label_texts, label_positions
        )
        if self._grip_marker_handle is not None:
            self._grip_marker_handle.visible = True

    def _sync_viser_grip_markers(
        self,
        visualizer,
        positions: list[np.ndarray],
        colors: list[tuple[int, int, int]],
        opacities: list[float],
    ) -> None:
        """One viser batch so each hand keeps its own color and opacity.

        The viewer's debug spheres share a single opacity, and that value is
        only applied when the batch is created.
        """

        count = len(positions)
        pos = np.stack(positions).astype(np.float32)
        cols = np.asarray(colors, dtype=np.uint8)
        alphas = np.asarray(opacities, dtype=np.float32)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
        scales = np.full((count, 3), self._hand_marker_radius, dtype=np.float32)
        handle = self._grip_marker_handle
        if handle is None or self._grip_marker_count != count:
            if handle is not None:
                handle.remove()
            vertices, faces = self._grip_marker_mesh()
            self._grip_marker_handle = visualizer.server.scene.add_batched_meshes_simple(
                "/debug/grip_markers",
                vertices,
                faces,
                batched_wxyzs=quats,
                batched_positions=pos,
                batched_scales=scales,
                batched_colors=cols,
                batched_opacities=alphas,
                opacity=1.0,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
                lod="off",
            )
            self._grip_marker_count = count
            return
        handle.batched_positions = pos
        handle.batched_wxyzs = quats
        handle.batched_scales = scales
        handle.batched_colors = cols
        handle.batched_opacities = alphas

    def _hand_side_labels(self) -> tuple[str, ...]:
        names = getattr(self.cfg, "hand_site_names", ())
        labels: list[str] = []
        for index, name in enumerate(names):
            short = str(name).rsplit("/", 1)[-1].lower()
            if "left" in short:
                labels.append("Left")
            elif "right" in short:
                labels.append("Right")
            else:
                labels.append(f"Hand {index}")
        if len(labels) < 2:
            labels.extend(f"Hand {index}" for index in range(len(labels), 2))
        return tuple(labels[:2])

    def create_gui(
        self,
        name: str,
        server,
        get_env_idx,
        on_change=None,
        request_action=None,
    ) -> None:
        """Latch progress and episode return in the viser controls panel."""

        del name, on_change, request_action
        self._grip_gui_env = get_env_idx
        self._grip_gui_bars = []
        self._grip_gui_captions = []
        with server.gui.add_folder("Hand latch"):
            for side in self._hand_side_labels():
                self._grip_gui_captions.append(
                    server.gui.add_markdown(grip_latch_caption(side, False, 0.0))
                )
                self._grip_gui_bars.append(server.gui.add_progress_bar(0.0))
        with server.gui.add_folder("Reward"):
            self._reward_total_caption = server.gui.add_markdown(
                episode_reward_caption(0.0)
            )
        self._reward_total_shown = 0.0
        self._reward_last_episode = None
        self._reward_total_env = -1

    def _update_viser_gui(self) -> None:
        self._update_grip_progress_gui()
        self._update_reward_total_gui()

    def _gui_env_idx(self) -> int:
        env_idx = 0
        if self._grip_gui_env is not None:
            env_idx = int(self._grip_gui_env())
        last_env = int(self.grip_progress.shape[0]) - 1
        return min(max(env_idx, 0), last_env)

    def _update_grip_progress_gui(self) -> None:
        bars = self._grip_gui_bars
        if not bars:
            return
        env_idx = self._gui_env_idx()
        for hand_id, bar in enumerate(bars):
            progress = float(self.grip_progress[env_idx, hand_id].item())
            latched = bool(self.attached[env_idx, hand_id].item())
            clamped = min(max(progress, 0.0), 1.0)
            bar.value = clamped * 100.0
            bar.color = (220, 50, 50) if latched else (40, 90, 220)
            self._grip_gui_captions[hand_id].content = grip_latch_caption(
                self._hand_side_labels()[hand_id], latched, clamped
            )

    def _episode_reward_total(self, env_idx: int) -> float:
        manager = getattr(self._env, "reward_manager", None)
        sums = getattr(manager, "_episode_sums", None)
        if not sums:
            return 0.0
        total = 0.0
        for value in sums.values():
            total += float(value[env_idx].item())
        return total

    def _update_reward_total_gui(self) -> None:
        handle = self._reward_total_caption
        if handle is None:
            return
        env_idx = self._gui_env_idx()
        total = self._episode_reward_total(env_idx)
        if env_idx != self._reward_total_env:
            self._reward_total_env = env_idx
        elif (
            total + 1.0e-6 < self._reward_total_shown
            and self._reward_total_shown > 1.0e-6
        ):
            self._reward_last_episode = self._reward_total_shown
        self._reward_total_shown = total
        handle.content = episode_reward_caption(total, self._reward_last_episode)

    def _set_grip_labels_visible(self, visible: bool) -> None:
        for handle in self._grip_label_handles.values():
            handle.visible = visible

    def _sync_viser_grip_labels(
        self,
        visualizer,
        keys: list[tuple[int, int]],
        texts: list[str],
        positions: list[np.ndarray],
    ) -> None:
        """A screen-space label above each drawn hand."""

        wanted = set(keys)
        for key, text, position in zip(keys, texts, positions, strict=True):
            handle = self._grip_label_handles.get(key)
            if handle is None:
                batch, hand_id = key
                handle = visualizer.server.scene.add_label(
                    f"/debug/grip_labels/{batch}_{hand_id}",
                    text,
                    position=tuple(float(channel) for channel in position),
                    font_size_mode="screen",
                    font_screen_scale=0.7,
                    anchor="bottom-center",
                    depth_test=False,
                )
                self._grip_label_handles[key] = handle
                continue
            handle.text = text
            handle.position = position
            handle.visible = True
        for key, handle in self._grip_label_handles.items():
            if key not in wanted:
                handle.visible = False

    def _grip_marker_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self._grip_marker_vertices is None or self._grip_marker_faces is None:
            import trimesh

            sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            self._grip_marker_vertices = np.asarray(sphere.vertices, dtype=np.float32)
            self._grip_marker_faces = np.asarray(sphere.faces, dtype=np.uint32)
        return self._grip_marker_vertices, self._grip_marker_faces

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

    # Which hand climbs first. The same foot leads after the hands.
    lead_hand: str = "left"
    start_rung: int = 0
    initialize_on_reset: bool = False
    initial_foot_rung: int = 0
    successes_per_rollout: int = 4
    stabilization_dwell_steps: int = 50
    max_stabilization_torso_speed: float = 0.20
    max_stabilization_body_angular_speed: float = 0.40
    weld_release_joint_speed: float = 4.0
    weld_release_speed_factor: float = 4.0
    attach_distance: float = 0.14
    max_attach_speed: float = 0.30
    # No longer the support test. A foot counts only on the top face of a rung.
    foot_support_distance: float = 0.08
    grip_half_span: float | None = None

    def build(self, env: ManagerBasedRlEnv) -> LadderClimbCommand:
        return LadderClimbCommand(self, env)


@dataclass(kw_only=True)
class LadderGripActionCfg(ActionTermCfg):
    """Two grip rates, one for each hand, appended after the joint targets.

    A command of +1 closes an open hand in one second. A command of -1 opens
    a latched hand in one second.
    """

    command_name: str = "ladder"

    def build(self, env: ManagerBasedRlEnv) -> LadderGripAction:
        return LadderGripAction(self, env)


class LadderGripAction(ActionTerm):
    """Forward each hand's grip rate to the ladder command."""

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
        command.apply_grip_requests(self._raw_actions)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


def _ladder_command(env: ManagerBasedRlEnv, command_name: str) -> LadderClimbCommand:
    return cast(LadderClimbCommand, env.command_manager.get_term(command_name))


# PPO sees these values after the reward manager multiplies by the 0.02 s step.
# Approach pays 0.5 per meter. A full second latched to the next rung is
# +0.25. An unlatched hand in that zone pays the same rate times its grip
# progress. A foot on the next tread pays 0.10 per second for the contact,
# plus 1 per second times its share of body weight. The share stops at 0.35,
# so a fully loaded foot pays 0.45 per second. Each limb stops at +3, and a
# completed hold pays whatever is left of that cap. Ascent pays the change in
# min(torso rise, 1) on the step the torso moves.
# A saturated grip command covers 0 to 1 in this many seconds.
GRIP_TRAVEL_S = 1.0
CLIMB_DISTANCE_PER_M = 0.5
# The planted ankle site sits about 8 mm above the tread, 0.043 m above the
# rung center. The box half-height is 0.035 m, so the underside is below the
# center and this band only covers the top face.
FOOT_TREAD_ABOVE_CENTER_M = 0.043
FOOT_TREAD_Z_MIN = 0.02
FOOT_TREAD_Z_MAX = 0.08
FOOT_TREAD_HALF_DEPTH_M = 0.07
# Canonical sole, both feet. Capsules run from x=-0.054 to x=0.132 at
# z=-0.035, and the foot site sits at x=0.04. These are the site-frame
# offsets of those two ends.
FOOT_SOLE_END_OFFSETS = (
    (-0.094, 0.0, 0.0),
    (0.092, 0.0, 0.0),
)
HOLD_REWARD = 0.5
# A rung is five stages: lead hand, other hand, lead foot, other foot, hold.
# The lead foot is the same side as the lead hand.
CLIMB_PHASE_FIRST = 0
CLIMB_PHASE_SECOND = 1
CLIMB_PHASE_LEAD_LEG = 2
CLIMB_PHASE_TRAIL_LEG = 3
CLIMB_PHASE_HOLD = 4
# Impulse when a hand or foot phase ends. The manager multiplies by dt.
# Both limbs of a pair finishing on one step pays this twice.
HAND_PHASE_REWARD = 4.0
LEG_PHASE_REWARD = 4.0
# Per hand that is not the one climbing. Progress 1 pays 0. Progress 0
# pays -0.3 per second. The manager multiplies by dt. A foot that is not
# the one stepping uses the same rate on its weight share: 20% of body
# weight pays 0, and no load pays -0.3 per second.
UNGRIP_RATE = 0.3
# Climbing-hand detach potential, only while that hand still holds the
# baseline rung. The potential is (1 - progress) times this scale, so a
# commanded full release is +1 and closing gives it back. A commanded
# release that reaches progress 0 also pays DETACH_COMPLETE. Both the +1
# and the +0.5 pay once per rung; a later open on that same rung is 0. A
# weld that breaks without that commanded open pays 0, including the 0.5.
# The manager multiplies by dt. Latch load on a closed weld is latch_pressure.
DETACH_POTENTIAL = 1.0
DETACH_COMPLETE = 0.5
# Off the ground. The manager multiplies by dt. The rate is 1 per second
# until the episode has lasted 2 s, then 0. The ground-contact step pays 0.
ALIVE_RATE = 1.0
ALIVE_WINDOW_S = 2.0
# Share of body weight. Weight is mass times the live simulator gravity.
FOOT_WEIGHT_FRACTION = 0.20
# Flat rate for a foot standing on the next tread, paid on top of the load
# term so a light contact is worth taking before the foot bears weight.
FOOT_STEP_RATE = 0.10
# Per unit of body-weight share on a foot. The share is the upward contact
# force divided by mass times the live gravity, and it stops rising at 0.35,
# so the pressure term stops at 0.35 per second. With the flat contact rate,
# a fully loaded foot pays 0.45 per second.
FOOT_PRESSURE_RATE = 1.0
FOOT_LOAD_CAP = 0.35
# Per limb while it is on the next rung. reward_terms stores rate * dt, and
# the public function divides by dt, so one open step returns the rate.
# Completion pays the unpaid remainder of this cap, then clears it.
CLIMB_CONTACT_RATE = 0.25
CLIMB_CONTACT_CAP = 3.0
# Share of an unbanked approach drop that is given back. 1 gives it all back,
# 0 keeps it. Applied once, to the net drop from the highest point, on the
# terminal step.
REFUND_FRACTION = 0.5
# Any projected crossing, wherever it sits on the bones. The manager
# multiplies by dt with weight 1. The episode integral stops at -1.2.
LEG_CROSS_RATE = -0.5
LEG_CROSS_TOTAL = -1.2
# A projected crossing of a leg and a climbing-face rail. Four times the
# leg-cross rate and budget. The manager multiplies by dt with weight 1.
SIDE_CROSS_RATE = 4.0 * LEG_CROSS_RATE
SIDE_CROSS_TOTAL = 4.0 * LEG_CROSS_TOTAL
# Latch load in trip units. Half the release speed is free, and the slope
# there is zero. The rate is this coefficient times the squared excess, so
# the trip itself (excess 0.5) costs 8 per second. The manager multiplies by dt.
LATCH_PRESSURE_ONSET = 0.5
LATCH_PRESSURE_RATE = -32.0
# Two or more limbs off the climb. The manager multiplies by dt. The episode
# integral stops at -1, so a long fall costs the same as one second.
FLIGHT_RATE = -1.0
FLIGHT_TOTAL = -1.0
# Thigh and shin bones, left then right. Each leg is hip roll -> knee -> ankle.
LEG_LINK_NAMES = (
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
)
# Climbing-face stiles. The back face shares these lateral lines once
# projected, so it does not add crossings.
SIDE_RAIL_NAMES = (
    "left_ladder_rail_01",
    "left_ladder_rail_02",
)


def _gather_rung_centers(centers: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe = indices.clamp(0, centers.shape[1] - 1)
    return torch.gather(centers, 1, safe.unsqueeze(-1).expand(-1, -1, 3))


def rung_far_top_edge(
    center: torch.Tensor,
    rotation: torch.Tensor,
    half_sizes: torch.Tensor,
) -> torch.Tensor:
    """Midpoint of the far top edge of each rung box.

    ``center`` has shape ``(E, R, 3)`` and ``rotation`` has shape
    ``(E, R, 3, 3)``, with the geom's local axes as columns. ``half_sizes``
    has shape ``(R, 3)`` and is the box half-extent stored on the rung geom.
    Local +x is the far side and local +z is the top, matching the ladder
    spec. The result has shape ``(E, R, 3)``.
    """

    local = torch.stack(
        (
            half_sizes[..., 0],
            torch.zeros_like(half_sizes[..., 0]),
            half_sizes[..., 2],
        ),
        dim=-1,
    )
    offset = (rotation * local.view(1, -1, 1, 3)).sum(dim=-1)
    return center + offset


def hand_approach_gaps(
    hand: torch.Tensor,
    edge: torch.Tensor,
    origin: torch.Tensor,
    climb: torch.Tensor,
    normal: torch.Tensor,
    degenerate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(plane, along)`` for each hand.

    ``hand`` and ``edge`` have shape ``(E, H, 3)``. ``origin``, ``climb``,
    and ``normal`` have shape ``(E, 3)``. ``degenerate`` has shape ``(E,)``.
    ``plane`` is the absolute distance out of the ladder plane. ``along`` is
    the climb left to reach the far top edge, and it stops at 0 once the hand
    passes that edge. A degenerate plane gives 0 for both.
    """

    delta = hand - origin.unsqueeze(1)
    plane = (delta * normal.unsqueeze(1)).sum(dim=-1).abs()
    ahead = ((edge - origin.unsqueeze(1)) * climb.unsqueeze(1)).sum(dim=-1)
    here = (delta * climb.unsqueeze(1)).sum(dim=-1)
    along = (ahead - here).clamp(min=0.0)
    cleared = torch.zeros_like(plane)
    plane = torch.where(degenerate.unsqueeze(1), cleared, plane)
    along = torch.where(degenerate.unsqueeze(1), cleared, along)
    return plane, along


def hand_approach_distance(
    hand: torch.Tensor,
    edge: torch.Tensor,
    origin: torch.Tensor,
    climb: torch.Tensor,
    normal: torch.Tensor,
    degenerate: torch.Tensor,
) -> torch.Tensor:
    """Plane gap plus the climb left to reach the far top edge."""

    plane, along = hand_approach_gaps(hand, edge, origin, climb, normal, degenerate)
    return plane + along


def foot_bears_load(upward_n: torch.Tensor, body_weight_n: torch.Tensor) -> torch.Tensor:
    """Whether each upward force is at least 20% of body weight.

    ``body_weight_n`` is mass times the live gravity magnitude. A stance that
    carries the same share of the robot crosses this line when gravity
    changes, because the contact force scales with gravity too.
    """

    return upward_n >= FOOT_WEIGHT_FRACTION * body_weight_n


def foot_step_load(upward_n: torch.Tensor, body_weight_n: torch.Tensor) -> torch.Tensor:
    """Share of body weight on each foot, clamped to ``[0, FOOT_LOAD_CAP]``.

    ``body_weight_n`` is mass times the live gravity magnitude. A stance that
    carries the same share of the robot pays the same value when gravity
    changes, because the contact force scales with gravity too.
    """

    share = upward_n / body_weight_n.clamp_min(1.0e-8)
    return share.clamp(0.0, FOOT_LOAD_CAP)


def foot_on_tread(
    foot: torch.Tensor,
    rung_center: torch.Tensor,
    half_span: torch.Tensor | float,
) -> torch.Tensor:
    """Whether each foot site is on the top face of its rung.

    The rung box is axis-aligned. ``z`` above the center by 0.02–0.08 m is
    the top face (the underside is below the center). ``x`` within 0.07 m
    keeps the foot over the tread, and ``y`` within ``half_span`` keeps it
    on the rung rather than out on the rail.
    """

    delta = foot - rung_center
    on_top = (delta[..., 2] >= FOOT_TREAD_Z_MIN) & (delta[..., 2] <= FOOT_TREAD_Z_MAX)
    over_tread = delta[..., 0].abs() <= FOOT_TREAD_HALF_DEPTH_M
    on_span = delta[..., 1].abs() <= half_span
    return on_top & over_tread & on_span


def foot_tread_features(
    sole_points: torch.Tensor,
    base_center: torch.Tensor,
    next_center: torch.Tensor,
    on_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remaining climb of the sole point furthest behind the next tread.

    ``sole_points`` has shape ``(E, F, P, 3)``. The axis runs from the
    baseline rung center to the next rung center, the same axis as a hand.
    The target on that axis is 0.043 m above the next rung center. ``d`` is
    the largest remaining distance among the sole points, clamped at 0, so
    an ankle that has arrived does not hide a heel still a rung below.
    ``d = 0`` once the foot is on that next tread. The lateral term is zero.
    """

    axis = next_center - base_center
    length = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    direction = axis / length.clamp_min(1.0e-8)
    degenerate = length.squeeze(-1) < 1.0e-6
    standing = next_center.clone()
    standing[..., 2] = standing[..., 2] + FOOT_TREAD_ABOVE_CENTER_M
    gap = standing.unsqueeze(-2) - sole_points
    remaining = (gap * direction.unsqueeze(-2)).sum(dim=-1).clamp_min(0.0)
    distance = remaining.amax(dim=-1)
    distance = torch.where(degenerate, torch.zeros_like(distance), distance)
    distance = torch.where(on_target, torch.zeros_like(distance), distance)
    return distance, torch.zeros_like(distance)


def _ladder_climb_limb(
    env: ManagerBasedRlEnv, command_name: str, limb: int
) -> torch.Tensor:
    """One column of the climb account, divided by ``dt``.

    Columns are left arm, right arm, left leg, right leg. ``reward_terms``
    is cached on the step counter, so reading every column updates the
    account once.
    """

    command = _ladder_command(env, command_name)
    return command.reward_terms()["climb"][:, limb] / env.step_dt


def ladder_climb_left_arm(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Left-hand approach and contact. The four limb terms sum to ``ladder_climb``."""

    return _ladder_climb_limb(env, command_name, 0)


def ladder_climb_right_arm(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Right-hand approach and contact. The four limb terms sum to ``ladder_climb``."""

    return _ladder_climb_limb(env, command_name, 1)


def ladder_climb_left_leg(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Left-foot approach and contact. The four limb terms sum to ``ladder_climb``."""

    return _ladder_climb_limb(env, command_name, 2)


def ladder_climb_right_leg(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Right-foot approach and contact. The four limb terms sum to ``ladder_climb``."""

    return _ladder_climb_limb(env, command_name, 3)


def ladder_climb(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sum of the four limb climb terms.

    Approach toward the next rung, plus contact time on that rung.

    A hand's approach distance is how far the grip site still is along the
    plane through the two lowest rungs, up to the far top edge of the next
    rung. That edge is the far top corner of the rung box, read from the rung
    geom. Passing the edge adds nothing. The distance out of that plane is a
    separate penalty, ``-0.5`` per meter from the pose sampled at the reset
    or the last completed hold. Attached to the ladder, that sample is zero.
    Moving out pays the meters, and moving back returns them. A foot's
    distance is the rung-to-rung gap of the sole point furthest behind the
    standing height on the next tread, and it keeps paying until the foot is
    on that tread.
    ``0.5`` per meter. The account is the gap closed since the reset sample,
    or since the last completed hold, and it stops at 0. Moving far away
    gives back only what that limb earned, and coming back re-earns that same
    gap. On the last step of the episode,
    ``1 - REFUND_FRACTION`` of the net approach given back since that limb's
    best point is returned, once.
    On top of that, a latched hand pays ``0.25`` per second while it is welded
    to the next rung. That rate starts when the hand's phase starts and keeps
    paying through the later phases until the cap or the next rung. An
    unlatched hand inside that rung's latch zone pays the same rate times its
    grip progress, on the same schedule. A foot on the next tread pays
    ``0.10`` per second for the contact, plus ``1`` per second times the
    share of body weight on
    that foot. The share is the upward contact force divided by body mass
    times the simulator gravity, clamped to ``[0, 0.35]``, so the foot rate
    rises from ``0.10`` to ``0.45`` per second as the foot takes weight.
    Each limb caps at ``+3``. Letting go pauses that limb until its phase
    ends. The step the phase ends pays whatever is left of that limb's cap
    and the cap stays full for the rest of the rung, so later contact on
    that limb pays nothing. Both limbs of a pair ending on one step fill
    both caps. A finished hold still pays any remainder left on a limb that
    is on the next rung, then clears every cap.
    ``reward_terms`` stores ``rate * dt`` for one open step, and this
    divides by ``dt``, so the step returns the rate. The completion
    remainder is stored as the amount, and this divides by ``dt`` too.
    The first sample after a reset pays 0 and clears the caps.
    """

    command = _ladder_command(env, command_name)
    return command.reward_terms()["climb"].sum(dim=-1) / env.step_dt


def _hand_ungrip(env: ManagerBasedRlEnv, command_name: str, hand: int) -> torch.Tensor:
    """``(progress - 1) * 0.3`` per second while this hand is not climbing.

    A closed weld pays 0. A fully open hand pays -0.3 per second. The
    climbing hand is exempt. Both hands pay during the leg and hold phases. The
    manager multiplies by ``dt``.
    """

    command = _ladder_command(env, command_name)
    active = command._climbing_hand() != hand
    rate = (command.grip_progress[:, hand] - 1.0) * UNGRIP_RATE
    return torch.where(active, rate, torch.zeros_like(rate))


def _hand_detach(env: ManagerBasedRlEnv, command_name: str, hand: int) -> torch.Tensor:
    """Potential on releasing the climbing hand from the baseline rung.

        r = 1 * ((1 - p_t) - (1 - p_{t-1}))

    Opening pays the rise. Closing pays it back. The first sample after this
    hand becomes the one climbing pays 0. A commanded grip that reaches
    progress 0 also pays ``+0.5``. That full open, and the ``+0.5``, each pay
    once per rung. A later open on the same rung pays 0. A weld that breaks
    without that commanded open pays 0, including the ``+0.5``. A hand that
    is not climbing, or is already on the next rung, pays 0. The manager
    multiplies by ``dt``, so this returns the step amount divided by ``dt``.
    """

    command = _ladder_command(env, command_name)
    progress = command.grip_progress[:, hand]
    if not hasattr(command, "_detach_progress_prev"):
        command._detach_progress_prev = command.grip_progress.clone()
        command._detach_tracked = torch.zeros(
            command.grip_progress.shape,
            dtype=torch.bool,
            device=command.grip_progress.device,
        )
    if not hasattr(command, "_detach_held_prev"):
        command._detach_held_prev = command.held_rung.clone()
    if not hasattr(command, "_detach_forced"):
        command._detach_forced = torch.zeros(
            command.grip_progress.shape,
            dtype=torch.bool,
            device=command.grip_progress.device,
        )
    if not hasattr(command, "_detach_commanded"):
        command._detach_commanded = torch.zeros(
            command.grip_progress.shape,
            dtype=torch.bool,
            device=command.grip_progress.device,
        )
    if not hasattr(command, "_detach_done"):
        command._detach_done = torch.zeros(
            command.grip_progress.shape,
            dtype=torch.bool,
            device=command.grip_progress.device,
        )
    fresh = command.just_completed
    if fresh.any():
        command._detach_done[fresh] = False
    prev = command._detach_progress_prev[:, hand]
    tracked = command._detach_tracked[:, hand]
    on_base = command._detach_held_prev[:, hand] == command.baseline_hand_rung[:, hand]
    forced = command._detach_forced[:, hand]
    commanded = command._detach_commanded[:, hand]
    done = command._detach_done[:, hand]
    active = command._climbing_hand() == hand
    eligible = active & tracked & on_base & ~forced & ~done & ~fresh
    opened = DETACH_POTENTIAL * (prev - progress)
    hit_zero = (prev > 1.0e-4) & (progress <= 1.0e-4)
    opened = torch.where(hit_zero & ~commanded, torch.zeros_like(opened), opened)
    step = torch.where(eligible, opened, torch.zeros_like(progress))
    complete = eligible & commanded & hit_zero
    step = step + complete.float() * DETACH_COMPLETE
    command._detach_progress_prev[:, hand] = progress
    command._detach_held_prev[:, hand] = command.held_rung[:, hand]
    command._detach_tracked[:, hand] = active
    command._detach_forced[:, hand] = False
    command._detach_commanded[:, hand] = False
    command._detach_done[:, hand] = done | complete
    return step / float(env.step_dt)


def ladder_detach_left(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Left-hand detach potential. Zero unless the left hand is the one climbing."""

    return _hand_detach(env, command_name, 0)


def ladder_detach_right(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Right-hand detach potential. Zero unless the right hand is the one climbing."""

    return _hand_detach(env, command_name, 1)


def ladder_ungrip_left(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Left-hand ungrip cost. Zero during the left-hand phase."""

    return _hand_ungrip(env, command_name, 0)


def ladder_ungrip_right(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Right-hand ungrip cost. Zero while the right hand is the one climbing."""

    return _hand_ungrip(env, command_name, 1)


def _foot_unstep(env: ManagerBasedRlEnv, command_name: str, foot: int) -> torch.Tensor:
    """``(planted - 1) * 0.3`` per second while this foot is not stepping.

    ``planted`` is the weight share divided by 20% of body weight, clamped
    to 1. A foot carrying at least that share pays 0. A foot with no load
    pays -0.3 per second. The stepping foot is exempt. Both feet pay during
    the hand and hold phases. The manager multiplies by ``dt``.
    """

    command = _ladder_command(env, command_name)
    active = command._climbing_foot() != foot
    share = command.foot_load_fraction()[:, foot]
    planted = (share / FOOT_WEIGHT_FRACTION).clamp(max=1.0)
    rate = (planted - 1.0) * UNGRIP_RATE
    return torch.where(active, rate, torch.zeros_like(rate))


def ladder_unstep_left(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Left-foot unstep cost. Zero during the left-foot phase."""

    return _foot_unstep(env, command_name, 0)


def ladder_unstep_right(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Right-foot unstep cost. Zero while the right foot is the one stepping."""

    return _foot_unstep(env, command_name, 1)


def ladder_hold(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Potential on the planted dwell. One finished second is worth 0.5.

        r_hold = 0.5 * (n_t − n_{t−1}) / N_dwell

    ``n`` counts frames with every support on the next rung, and it stops at
    ``N_dwell``. Leaving a support sets ``n_t = 0`` and pays the accumulated
    count back. The counter can sit at ``N_dwell`` while the trailing window
    is still fast; those steps pay 0. Completion clears ``n``. The step that
    first reaches ``N_dwell`` pays the last slice, and a completion from a
    counter that was already full pays 0. ``r_done`` carries the separate
    ``+1`` impulse. The reward manager multiplies by ``dt``, so this returns
    ``r_hold / dt``.
    """

    command = _ladder_command(env, command_name)
    return command.reward_terms()["hold"] / env.step_dt


def ladder_face_axes(
    origin: torch.Tensor,
    landing: torch.Tensor,
    rung_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal axes of the plane through two rung centers.

    ``rung_axis`` is the rung direction. The climb axis is the component of
    ``landing - origin`` perpendicular to it. The mask is true when the two
    centers do not span a plane.
    """

    lateral = rung_axis / torch.linalg.vector_norm(
        rung_axis, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    climb = landing - origin
    climb = climb - lateral * (climb * lateral).sum(dim=-1, keepdim=True)
    length = torch.linalg.vector_norm(climb, dim=-1, keepdim=True)
    degenerate = length.squeeze(-1) < 1.0e-6
    climb = climb / length.clamp_min(1.0e-8)
    return origin, lateral, climb, degenerate


def _segments_intersect_2d(
    origin: torch.Tensor,
    end: torch.Tensor,
    other_origin: torch.Tensor,
    other_end: torch.Tensor,
) -> torch.Tensor:
    """Whether two projected segments intersect, including at an endpoint.

    Parallel projections do not count. A crossing near either end counts the
    same as a crossing in the middle.
    """

    direction = end - origin
    other = other_end - other_origin
    offset = origin - other_origin
    denom = direction[..., 0] * other[..., 1] - direction[..., 1] * other[..., 0]
    safe = torch.where(denom.abs() < 1.0e-8, torch.ones_like(denom), denom)
    s_param = (other[..., 0] * offset[..., 1] - other[..., 1] * offset[..., 0]) / safe
    t_param = (
        direction[..., 0] * offset[..., 1] - direction[..., 1] * offset[..., 0]
    ) / safe
    return (
        (denom.abs() >= 1.0e-8)
        & (s_param >= 0.0)
        & (s_param <= 1.0)
        & (t_param >= 0.0)
        & (t_param <= 1.0)
    )


def _project_onto_face(
    points: torch.Tensor,
    origin: torch.Tensor,
    lateral: torch.Tensor,
    climb: torch.Tensor,
) -> torch.Tensor:
    """Drop the coordinate out of the ladder face.

    ``points`` has shape ``(E, N, 3)``. The result is ``(E, N, 2)`` in
    ``(lateral, climb)`` order.
    """

    delta = points - origin.unsqueeze(1)
    return torch.stack(
        (
            (delta * lateral.unsqueeze(1)).sum(dim=-1),
            (delta * climb.unsqueeze(1)).sum(dim=-1),
        ),
        dim=-1,
    )


def projected_legs_cross(
    link_pos: torch.Tensor,
    origin: torch.Tensor,
    lateral: torch.Tensor,
    climb: torch.Tensor,
) -> torch.Tensor:
    """Whether any thigh or shin segment crosses the other leg on the ladder face.

    ``link_pos`` has shape ``(E, 6, 3)``: hip, knee, ankle on the left, then
    the same three on the right. Coordinates out of the ladder face are
    dropped, so ankles may pass close in depth without a cost.
    """

    projected = _project_onto_face(link_pos, origin, lateral, climb)
    left_thigh = (projected[:, 0], projected[:, 1])
    left_shin = (projected[:, 1], projected[:, 2])
    right_thigh = (projected[:, 3], projected[:, 4])
    right_shin = (projected[:, 4], projected[:, 5])
    pairs = (
        _segments_intersect_2d(*left_thigh, *right_thigh),
        _segments_intersect_2d(*left_thigh, *right_shin),
        _segments_intersect_2d(*left_shin, *right_thigh),
        _segments_intersect_2d(*left_shin, *right_shin),
    )
    return torch.stack(pairs, dim=0).any(dim=0)


def projected_legs_cross_rails(
    link_pos: torch.Tensor,
    rail_start: torch.Tensor,
    rail_end: torch.Tensor,
    origin: torch.Tensor,
    lateral: torch.Tensor,
    climb: torch.Tensor,
) -> torch.Tensor:
    """Whether any thigh or shin crosses a side rail on the ladder face.

    ``link_pos`` has shape ``(E, 6, 3)``. ``rail_start`` and ``rail_end`` have
    shape ``(E, R, 3)``. A crossing near either end counts. Parallel
    projections do not. Coordinates out of the face are dropped.
    """

    projected = _project_onto_face(link_pos, origin, lateral, climb)
    rails = _project_onto_face(rail_start, origin, lateral, climb)
    rail_tips = _project_onto_face(rail_end, origin, lateral, climb)
    segments = (
        (projected[:, 0], projected[:, 1]),
        (projected[:, 1], projected[:, 2]),
        (projected[:, 3], projected[:, 4]),
        (projected[:, 4], projected[:, 5]),
    )
    hits = [
        _segments_intersect_2d(start, end, rails[:, index], rail_tips[:, index])
        for start, end in segments
        for index in range(rail_start.shape[1])
    ]
    return torch.stack(hits, dim=0).any(dim=0)


def _leg_link_positions(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    indexes, names = robot.find_bodies(LEG_LINK_NAMES, preserve_order=True)
    if tuple(names) != LEG_LINK_NAMES:
        raise ValueError(f"Expected leg links {LEG_LINK_NAMES}, found {tuple(names)}")
    positions = robot.data.body_link_pos_w
    link_ids = torch.as_tensor(indexes, dtype=torch.long, device=positions.device)
    return positions[:, link_ids]


def leg_cross(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Constant rate while the projected legs cross, within the episode budget.

    The manager scales by ``dt`` and the weight is 1. The integral stops at
    ``LEG_CROSS_TOTAL`` (-1.2). A later call in the same step returns the
    same rate and does not charge twice.
    """

    command = _ladder_command(env, command_name)
    step_id = int(getattr(env, "common_step_counter", -1))
    if command._leg_cross_step == step_id:
        return command._leg_cross_rate

    origin, lateral, climb, degenerate = command.foot_span_plane()
    crossing = projected_legs_cross(
        _leg_link_positions(env), origin, lateral, climb
    )
    crossing = crossing & ~degenerate
    rate = torch.where(
        crossing,
        torch.full_like(command._leg_cross_paid, LEG_CROSS_RATE),
        torch.zeros_like(command._leg_cross_paid),
    )
    dt = float(env.step_dt)
    step_cost = torch.maximum(rate * dt, LEG_CROSS_TOTAL - command._leg_cross_paid)
    command._leg_cross_paid = command._leg_cross_paid + step_cost
    command._leg_cross_step = step_id
    command._leg_cross_rate = step_cost / dt
    return command._leg_cross_rate


def side_cross(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Constant rate while a projected leg crosses a side rail.

    The plane is the one through the two lowest rungs. The manager scales by
    ``dt`` and the weight is 1. The integral stops at ``SIDE_CROSS_TOTAL``
    (-4.8), four times the leg-cross budget. A later call in the same step
    returns the same rate and does not charge twice.
    """

    command = _ladder_command(env, command_name)
    step_id = int(getattr(env, "common_step_counter", -1))
    if command._side_cross_step == step_id:
        return command._side_cross_rate

    origin, lateral, climb, degenerate = command.bottom_rung_plane()
    rail_start, rail_end = command.rail_endpoints()
    crossing = projected_legs_cross_rails(
        _leg_link_positions(env),
        rail_start,
        rail_end,
        origin,
        lateral,
        climb,
    )
    crossing = crossing & ~degenerate
    rate = torch.where(
        crossing,
        torch.full_like(command._side_cross_paid, SIDE_CROSS_RATE),
        torch.zeros_like(command._side_cross_paid),
    )
    dt = float(env.step_dt)
    step_cost = torch.maximum(rate * dt, SIDE_CROSS_TOTAL - command._side_cross_paid)
    command._side_cross_paid = command._side_cross_paid + step_cost
    command._side_cross_step = step_id
    command._side_cross_rate = step_cost / dt
    return command._side_cross_rate


def latch_pressure(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Squared excess over half the speed that breaks a closed weld.

    Pressure is measured in units of the over-speed trip and stored before
    that trip opens the weld. At and below half the trip the rate is 0 and
    the slope is 0. At the trip the excess is 0.5 and the rate is -8 per
    second. An intentional grip release is not load. The manager scales by
    ``dt`` and the weight is 1.
    """

    command = _ladder_command(env, command_name)
    excess = (command._latch_pressure - LATCH_PRESSURE_ONSET).clamp(min=0.0)
    return excess.square() * LATCH_PRESSURE_RATE


def ladder_flight(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Rate while two or more limbs are off the climb, within the episode budget.

    A hand is on while its latch is on. A foot is on while it touches the top
    face of its baseline rung or the next one. One hand reaching, or one foot
    stepping, pays nothing. The manager scales by ``dt`` and the weight is 1.
    The integral stops at ``FLIGHT_TOTAL`` (-1).
    """

    command = _ladder_command(env, command_name)
    step_id = int(getattr(env, "common_step_counter", -1))
    if command._flight_step == step_id:
        return command._flight_rate

    hands_off = (~command.attached).sum(dim=-1)
    feet_off = (~command.feet_on_flight_rungs()).sum(dim=-1)
    flying = hands_off + feet_off >= 2
    rate = torch.where(
        flying,
        torch.full_like(command._flight_paid, FLIGHT_RATE),
        torch.zeros_like(command._flight_paid),
    )
    dt = float(env.step_dt)
    step_cost = torch.maximum(rate * dt, FLIGHT_TOTAL - command._flight_paid)
    command._flight_paid = command._flight_paid + step_cost
    command._flight_step = step_id
    command._flight_rate = step_cost / dt
    return command._flight_rate


def ladder_alive(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Off the ground, for the first 2 seconds. The manager scales by ``dt``.

    The rate is 1 per second while the episode time is at most 2 s, then 0.
    The step that touches the ground pays nothing.
    """

    command = _ladder_command(env, command_name)
    return command.reward_terms()["alive"]


def _phases_crossed(
    started: torch.Tensor, ended: torch.Tensor, first: int, last: int
) -> torch.Tensor:
    """How many phases in ``[first, last]`` this step walked past.

    A finished hold sets ``ended`` back to the lead hand, which is behind
    ``started``, so that step counts as zero.
    """

    count = torch.zeros_like(started, dtype=torch.float32)
    for phase in range(first, last + 1):
        count = count + ((started <= phase) & (ended > phase)).float()
    return count


def ladder_hand_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Impulse when a hand phase finishes. The manager scales by ``dt``.

    The lead hand latching the next rung pays ``+4``. The trailing hand
    doing the same pays ``+4``. Both on one step pays ``+8``. A finished
    hold returns to the lead hand and pays 0.
    """

    command = _ladder_command(env, command_name)
    count = _phases_crossed(
        command._reward_phase,
        command.climb_phase,
        CLIMB_PHASE_FIRST,
        CLIMB_PHASE_SECOND,
    )
    return count * HAND_PHASE_REWARD / env.step_dt


def ladder_leg_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Impulse when a leg phase finishes. The manager scales by ``dt``.

    The lead foot on the next rung with 20% of body weight pays ``+4``.
    The trailing foot doing the same pays ``+4``. Both on one step pays
    ``+8``. Weight still on the rung the foot started from pays 0. A
    finished hold pays 0.
    """

    command = _ladder_command(env, command_name)
    count = _phases_crossed(
        command._reward_phase,
        command.climb_phase,
        CLIMB_PHASE_LEAD_LEG,
        CLIMB_PHASE_TRAIL_LEG,
    )
    return count * LEG_PHASE_REWARD / env.step_dt


def ladder_hold_completed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Completion impulse. The manager scales by ``dt``, so the buffer receives 1.

        r_done = 1

    when the trailing planted window is full and its mean torso speed and
    mean torso and pelvis angular speeds are inside the limits. A fall, a
    deadline, or a fast planted window pays 0.
    """

    command = _ladder_command(env, command_name)
    done = command.just_completed & (command._reward_phase == CLIMB_PHASE_HOLD)
    return done.float() / env.step_dt


def ladder_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """End the rollout after N completed holds."""

    return _ladder_command(env, command_name).finished


def ladder_touched_ground(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """The robot body is touching the ground plane. The ladder is not included."""

    sensor = cast(ContactSensor, env.scene[sensor_name])
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return found.reshape(env.num_envs, -1).any(dim=1)


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
    """Privileged critic features in the checkpoint layout.

    The order is fixed so a model trained before the hold change still loads.
    Thirteen values: hold progress, successes, torso speed, joint-speed RMS,
    torso angular speed, pelvis angular speed, waist joint speed, support
    offset, the one-rung placement bit, both hand attachments, and both foot
    supports. The hold reward does not read this vector.
    """

    command = _ladder_command(env, command_name)
    dwell = float(command.cfg.stabilization_dwell_steps)
    limit = float(command.cfg.successes_per_rollout)
    joint_speed_rms = torch.sqrt(
        torch.mean(torch.square(command.robot.data.joint_vel), dim=1)
    )
    waist_joint_speed = torch.amax(
        torch.abs(command.robot.data.joint_vel[:, command._waist_joint_ids]),
        dim=1,
    )
    scalar_terms = (
        command.hold_count.float() / dwell,
        command.successes.float() / limit,
        torch.linalg.vector_norm(command.torso_com_vel_w, dim=-1),
        joint_speed_rms,
        torch.linalg.vector_norm(command.torso_ang_vel_w, dim=-1),
        torch.linalg.vector_norm(command.pelvis_ang_vel_w, dim=-1),
        waist_joint_speed,
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
