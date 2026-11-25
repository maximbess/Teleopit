"""Track five body points with continuous penalty weights.

One limb target moves inside a ball. The other points stay where they were
measured, and the head weight is a soft tether. Latches are not used. The
grip action is still accepted so the policy width matches the ladder action,
and the welds are forced inactive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg

from .ladder import episode_reward_caption, install_episode_sum_reward_logs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.tracking.config.ladder_init import LadderPose


POINT_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot", "head")
SITE_NAMES = (
    "left_grip_site",
    "right_grip_site",
    "left_foot",
    "right_foot",
    "head_site",
)
# The head mesh is fixed on the torso. This is the site-frame offset of its center.
HEAD_SITE_OFFSET = (0.0, 0.0, 0.43)

TARGET_BALL_RADIUS_M = 0.25
# Per second, before the reward manager multiplies by the control step.
# The cap is one meter of weighted error, so a fresh target is still worth a
# positive rate and a drift of one meter of weighted error stays positive.
TARGET_ALIVE_RATE = 1.25
TARGET_DISTANCE_CAP_M = 1.0
TARGET_GAIN_SCALE = 2.0
# The first four sites are the limbs. The head is a tether, not a swing point.
NUM_LIMB_POINTS = 4
FIRM_WEIGHT = 0.5
FIRM_HOLD_DISTANCE_M = 0.035
FIRM_HOLD_STEPS = 12
SUBGOAL_TIME_RANGE_S = (1.5, 2.0)
COMMAND_DIM = 35
PRIVILEGED_DIM = 6


def sample_ball_offsets(
    count: int,
    radius: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Volume-uniform offsets inside a ball. Shape ``(count, 3)``."""

    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    direction = torch.randn(
        count, 3, generator=generator, device=device, dtype=dtype
    )
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    unit = torch.rand(count, 1, generator=generator, device=device, dtype=dtype)
    return direction * (radius * unit.pow(1.0 / 3.0))


def sample_active_indices(
    count: int,
    num_points: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """One point index per environment."""

    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")
    return torch.randint(
        0, num_points, (count,), generator=generator, device=device
    )


def assign_targets(
    points: torch.Tensor,
    active: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Copy ``points`` and move the active point by ``offsets``.

    ``points`` has shape ``(E, P, 3)``. ``active`` has shape ``(E,)``.
    ``offsets`` has shape ``(E, 3)``.
    """

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (E, P, 3), got {tuple(points.shape)}")
    targets = points.clone()
    if points.shape[0] == 0:
        return targets
    env_ids = torch.arange(points.shape[0], device=points.device)
    targets[env_ids, active] = points[env_ids, active] + offsets
    return targets


def point_distance_sum(points: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sum of Euclidean distances. Shape ``(E,)``."""

    return torch.linalg.vector_norm(points - targets, dim=-1).sum(dim=-1)


def weighted_distance_sum(
    distances: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """``sum_i w_i d_i``. ``distances`` and ``weights`` have shape ``(E, 5)``."""

    return (distances * weights).sum(dim=-1)


def sample_point_weights(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor-biased weights and the limb that receives the ball offset.

    Limb weights start in ``[0.75, 1]``. Half of the rows soften the swing
    limb into ``[0.05, 0.75]``. The head is usually a soft tether, and is
    exactly ``0`` on 5% of rows. Returns ``(weights, swing)`` with shapes
    ``(count, 5)`` and ``(count,)``. ``swing`` is in ``0..3``.
    """

    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    weights = torch.empty(count, 5, device=device, dtype=dtype)
    swing = torch.empty(count, dtype=torch.long, device=device)
    if count == 0:
        return weights, swing
    weights[:, :NUM_LIMB_POINTS] = 0.75 + 0.25 * torch.rand(
        count, NUM_LIMB_POINTS, generator=generator, device=device, dtype=dtype
    )
    swing = torch.randint(0, NUM_LIMB_POINTS, (count,), generator=generator, device=device)
    soften = torch.rand(count, generator=generator, device=device) < 0.5
    soft = 0.05 + 0.70 * torch.rand(
        count, generator=generator, device=device, dtype=dtype
    )
    if bool(soften.any()):
        weights[soften, swing[soften]] = soft[soften]
    draw = torch.rand(count, generator=generator, device=device)
    low = 0.05 + 0.30 * torch.rand(
        count, generator=generator, device=device, dtype=dtype
    )
    high = 0.35 + 0.65 * torch.rand(
        count, generator=generator, device=device, dtype=dtype
    )
    head = torch.where(
        draw < 0.70,
        low,
        torch.where(draw < 0.95, high, torch.zeros(count, device=device, dtype=dtype)),
    )
    weights[:, 4] = head
    return weights.clamp(0.0, 1.0), swing


def sample_subgoal_times(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    low: float = SUBGOAL_TIME_RANGE_S[0],
    high: float = SUBGOAL_TIME_RANGE_S[1],
) -> torch.Tensor:
    """Uniform subgoal lifetimes in seconds. Shape ``(count,)``."""

    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if high < low:
        raise ValueError(f"high must be >= low, got {low} and {high}")
    return low + (high - low) * torch.rand(
        count, generator=generator, device=device
    )


def point_gain_valid(
    stored_valid: torch.Tensor, previous_weights: torch.Tensor
) -> torch.Tensor:
    """A point pays gain only after a weighted step with the same target."""

    return stored_valid & (previous_weights > 0.0)


def pack_body_target_command(
    weights: torch.Tensor,
    points_torso: torch.Tensor,
    targets_torso: torch.Tensor,
) -> torch.Tensor:
    """35D command. A zero weight copies the measured point and zeros its error."""

    if weights.shape[-1] != 5 or points_torso.shape[-2:] != (5, 3):
        raise ValueError(
            "expected weights (E, 5) and torso points (E, 5, 3), got "
            f"{tuple(weights.shape)} and {tuple(points_torso.shape)}"
        )
    if targets_torso.shape != points_torso.shape:
        raise ValueError(
            "targets_torso and points_torso must match, got "
            f"{tuple(targets_torso.shape)} and {tuple(points_torso.shape)}"
        )
    free = weights == 0.0
    targets = torch.where(free.unsqueeze(-1), points_torso, targets_torso)
    errors = torch.where(
        free.unsqueeze(-1),
        torch.zeros_like(points_torso),
        targets_torso - points_torso,
    )
    flat = weights.shape[0]
    return torch.cat(
        (weights, targets.reshape(flat, -1), errors.reshape(flat, -1)),
        dim=-1,
    )


def pack_privileged(distances: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Five raw distances and the weighted sum. Shape ``(E, 6)``."""

    weighted = weighted_distance_sum(distances, weights).unsqueeze(-1)
    return torch.cat((distances, weighted), dim=-1)


def target_position_rate(
    distance: torch.Tensor,
    on_ground: torch.Tensor,
    *,
    alive: float = TARGET_ALIVE_RATE,
    cap: float = TARGET_DISTANCE_CAP_M,
) -> torch.Tensor:
    """``alive - min(distance, cap)`` per second, and 0 on ground contact."""

    rate = alive - distance.clamp(max=cap)
    return torch.where(on_ground, torch.zeros_like(rate), rate)


def target_gain_rate(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    weights: torch.Tensor,
    valid: torch.Tensor,
    on_ground: torch.Tensor,
    dt: float,
    scale: float = TARGET_GAIN_SCALE,
) -> torch.Tensor:
    """Per-second rate for ``scale * sum_i w_i * valid_i * (previous_i - current_i)``.

    ``previous`` and ``current`` are per-point distances. Weight sits outside
    the difference, so a new weight on a stationary body pays 0. An invalid
    point pays 0. Ground contact pays 0.
    """

    if previous.shape != current.shape or weights.shape != previous.shape:
        raise ValueError(
            "previous, current, and weights must match, got "
            f"{tuple(previous.shape)}, {tuple(current.shape)}, {tuple(weights.shape)}"
        )
    safe_dt = max(float(dt), 1.0e-8)
    delta = (previous - current) * weights * valid.to(dtype=weights.dtype)
    rate = scale * delta.sum(dim=-1) / safe_dt
    return torch.where(on_ground.reshape(rate.shape), torch.zeros_like(rate), rate)


def joint_targets_from_patterns(
    joint_names: Sequence[str],
    patterns: Mapping[str, float],
) -> list[float]:
    """Expand regex joint targets. Unmatched joints stay at 0."""

    compiled = [(re.compile(pattern), value) for pattern, value in patterns.items()]
    values = [0.0] * len(joint_names)
    for index, name in enumerate(joint_names):
        short = name.rsplit("/", 1)[-1]
        for regex, value in compiled:
            if regex.fullmatch(short):
                values[index] = float(value)
    return values


def _as_rotation_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] == (3, 3):
        return value
    if value.shape[-1] == 9:
        return value.reshape(*value.shape[:-1], 3, 3)
    raise RuntimeError(f"Unexpected rotation shape: {tuple(value.shape)}")


class BodyPointTargetCommand(CommandTerm):
    """Sample a keyframe pose, then weighted targets for five body points.

    One limb target moves inside the ball. The other targets stay on the
    points measured after that pose is written. A later subgoal resamples
    the targets and weights without writing the pose again.
    ``latches_enabled`` is false: grip actions are stored and both welds are
    cleared every update.
    """

    cfg: BodyPointTargetCommandCfg

    def __init__(self, cfg: BodyPointTargetCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        if cfg.latches_enabled:
            raise ValueError(
                "Body-point targets keep the welds off. Latch attachment stays "
                "on LadderClimbCommand."
            )
        if len(cfg.site_names) != len(cfg.point_names):
            raise ValueError("site_names and point_names must have the same length")
        if len(cfg.site_names) == 0:
            raise ValueError("site_names must contain the tracked points")
        if cfg.target_ball_radius <= 0.0:
            raise ValueError("target_ball_radius must be > 0")
        if not cfg.pose_bank:
            raise ValueError("pose_bank must contain at least one pose")
        if not cfg.rung_site_names:
            raise ValueError("rung_site_names must contain the ladder rungs")

        self.robot = env.scene[cfg.entity_name]
        self.latches_enabled = False
        self.pose_bank = tuple(cfg.pose_bank)
        model = env.sim.mj_model
        self._site_ids = torch.tensor(
            [self._required_id(mujoco.mjtObj.mjOBJ_SITE, name) for name in cfg.site_names],
            dtype=torch.long,
            device=self.device,
        )
        self._rung_site_ids = torch.tensor(
            [
                self._required_id(mujoco.mjtObj.mjOBJ_SITE, name)
                for name in cfg.rung_site_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        half_lengths: list[float] = []
        for name, site_id in zip(cfg.rung_site_names, self._rung_site_ids.tolist()):
            if int(model.site_type[site_id]) != capsule_type:
                raise ValueError(f'Rung site {name!r} must have type="capsule"')
            half_length = float(model.site_size[site_id, 1])
            if half_length <= 0.0:
                raise ValueError(f"Rung site {name!r} has zero capsule length")
            half_lengths.append(half_length)
        self._rung_half_lengths = torch.tensor(
            half_lengths, dtype=torch.float32, device=self.device
        )
        self._torso_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY, cfg.torso_body_name
        )
        self._weld_ids = torch.tensor(
            [
                self._required_id(mujoco.mjtObj.mjOBJ_EQUALITY, name)
                for name in cfg.weld_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        joint_rows = [
            joint_targets_from_patterns(self.robot.joint_names, pose.joint_pos)
            for pose in self.pose_bank
        ]
        root_rows = [(*pose.root_pos, *pose.root_rot) for pose in self.pose_bank]
        self._pose_joints = torch.tensor(
            joint_rows, dtype=torch.float32, device=self.device
        )
        self._pose_root = torch.tensor(
            root_rows, dtype=torch.float32, device=self.device
        )
        self.pose_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.swing_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.point_weights = torch.ones(
            self.num_envs, self.num_points, dtype=torch.float32, device=self.device
        )
        self.targets_w = torch.zeros(
            self.num_envs, self.num_points, 3, dtype=torch.float32, device=self.device
        )
        self._pending_targets = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._point_valid = torch.zeros(
            self.num_envs, self.num_points, dtype=torch.bool, device=self.device
        )
        self._point_dist_prev = torch.zeros(
            self.num_envs, self.num_points, dtype=torch.float32, device=self.device
        )
        self._point_weights_prev = torch.zeros(
            self.num_envs, self.num_points, dtype=torch.float32, device=self.device
        )
        self._subgoal_time_left = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._firm_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reward_cache_step = -1
        self._reward_cache: dict[str, torch.Tensor] | None = None
        self.grip_request = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )
        self._target_caption = None
        self._gui_env = None
        self._current_marker_handle = None
        self._current_marker_count = 0
        self._target_marker_handle = None
        self._target_marker_count = 0
        self._target_marker_vertices: np.ndarray | None = None
        self._target_marker_faces: np.ndarray | None = None
        self.metrics["distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["active_point"] = torch.zeros(self.num_envs, device=self.device)
        install_episode_sum_reward_logs(env)

    @property
    def num_points(self) -> int:
        return int(self._site_ids.numel())

    @property
    def point_pos_w(self) -> torch.Tensor:
        """World positions of the five sites, shape ``(num_envs, 5, 3)``."""

        return self._env.sim.data.site_xpos[:, self._site_ids]

    @property
    def command(self) -> torch.Tensor:
        """Weights, torso-frame targets, and unscaled torso-frame errors."""

        points_t = self._points_to_torso(self.point_pos_w)
        targets_t = self._points_to_torso(self.targets_w)
        return pack_body_target_command(self.point_weights, points_t, targets_t)

    @property
    def rung_endpoints_torso(self) -> torch.Tensor:
        """Both ends of every rung capsule in the torso frame.

        Shape ``(num_envs, num_rungs, 6)``. There is no latch or phase bit.
        """

        data = self._env.sim.data
        centers = data.site_xpos[:, self._rung_site_ids]
        axes = _as_rotation_matrix(data.site_xmat[:, self._rung_site_ids])[..., :, 2]
        half = self._rung_half_lengths.view(1, -1, 1)
        endpoints = torch.stack(
            (
                self._points_to_torso(centers - axes * half),
                self._points_to_torso(centers + axes * half),
            ),
            dim=-2,
        )
        return endpoints.reshape(*centers.shape[:2], 6)

    def point_distances(self) -> torch.Tensor:
        """Per-point Euclidean error, shape ``(num_envs, num_points)``."""

        return torch.linalg.vector_norm(self.point_pos_w - self.targets_w, dim=-1)

    def privileged(self) -> torch.Tensor:
        """Five raw distances and the weighted sum."""

        return pack_privileged(self.point_distances(), self.point_weights)

    def apply_grip_requests(self, request: torch.Tensor) -> None:
        """Store the two grip actions and leave both welds inactive."""

        self.grip_request = request.to(device=self.device, dtype=torch.float32).clamp(
            -1.0, 1.0
        )
        self._force_welds_off()

    def refresh_after_reset(self) -> None:
        """Sample targets from the pose written by the reset, without latching."""

        self._update_command()

    def compute(self, dt: float) -> None:
        """Sample pending targets before metrics, so a reset is visible this step."""

        self.time_left -= dt
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        self._update_command()
        self._advance_subgoals(dt)
        self._update_metrics()

    def reward_terms(self) -> dict[str, torch.Tensor]:
        """Weighted position rate and per-point distance-gain rate.

        Both terms share one sample. Gain is 0 on the first call after a
        reset, after a target rewrite, and when the previous weight was 0.
        Changing the weights of a stationary body also pays 0. Ground contact
        zeros both rates.
        """

        step_id = int(getattr(self._env, "common_step_counter", -1))
        if self._reward_cache is not None and self._reward_cache_step == step_id:
            return self._reward_cache
        distances = self.point_distances()
        on_ground = self._on_ground()
        valid = point_gain_valid(self._point_valid, self._point_weights_prev)
        position = target_position_rate(
            weighted_distance_sum(distances, self.point_weights), on_ground
        )
        gain = target_gain_rate(
            self._point_dist_prev,
            distances,
            weights=self.point_weights,
            valid=valid,
            on_ground=on_ground,
            dt=float(self._env.step_dt),
        )
        self._point_dist_prev.copy_(distances)
        self._point_weights_prev.copy_(self.point_weights)
        self._point_valid[:] = True
        self._reward_cache = {"target_pos": position, "target_gain": gain}
        self._reward_cache_step = step_id
        return self._reward_cache

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._pending_targets[env_ids] = True
        self._point_valid[env_ids] = False
        self._point_dist_prev[env_ids] = 0.0
        self._point_weights_prev[env_ids] = 0.0
        self._firm_hold_steps[env_ids] = 0
        self._reward_cache_step = -1
        self.grip_request[env_ids] = 0.0
        self._write_pose(env_ids)
        self._force_welds_off(env_ids)

    def _update_command(self) -> None:
        self._force_welds_off()
        pending = torch.where(self._pending_targets)[0]
        if pending.numel() > 0:
            self._assign_sampled_targets(pending)
            self._pending_targets[pending] = False
        self._update_target_caption()

    def _assign_sampled_targets(self, env_ids: torch.Tensor) -> None:
        """Write a new weight vector and one limb offset. The pose stays put."""

        count = int(env_ids.numel())
        if count == 0:
            return
        points = self.point_pos_w[env_ids]
        weights, swing = sample_point_weights(count, device=self.device)
        offsets = sample_ball_offsets(
            count, float(self.cfg.target_ball_radius), device=self.device
        )
        self.targets_w[env_ids] = assign_targets(points, swing, offsets)
        self.point_weights[env_ids] = weights
        self.swing_index[env_ids] = swing
        self._point_valid[env_ids] = False
        self._firm_hold_steps[env_ids] = 0
        self._subgoal_time_left[env_ids] = sample_subgoal_times(count, device=self.device)
        self._reward_cache_step = -1

    def _advance_subgoals(self, dt: float) -> None:
        """Chain a subgoal when the firm points settle or the timer expires.

        This does not write a pose and does not reset observation history.
        """

        armed = ~self._pending_targets
        if not bool(armed.any()):
            return
        self._subgoal_time_left[armed] -= float(dt)
        distances = self.point_distances()
        firm = self.point_weights >= FIRM_WEIGHT
        close = distances <= FIRM_HOLD_DISTANCE_M
        held = armed & torch.where(firm, close, torch.ones_like(close)).all(dim=-1)
        zeros = torch.zeros_like(self._firm_hold_steps)
        self._firm_hold_steps = torch.where(
            held,
            self._firm_hold_steps + 1,
            torch.where(armed, zeros, self._firm_hold_steps),
        )
        advance = armed & (
            (self._subgoal_time_left <= 0.0)
            | (self._firm_hold_steps >= FIRM_HOLD_STEPS)
        )
        env_ids = advance.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        self._assign_sampled_targets(env_ids)

    def _update_metrics(self) -> None:
        distances = self.point_distances()
        self.metrics["distance"].copy_(
            weighted_distance_sum(distances, self.point_weights)
        )
        self.metrics["active_point"].copy_(self.swing_index.float())

    def _write_pose(self, env_ids: torch.Tensor) -> None:
        count = int(env_ids.numel())
        choice = torch.randint(0, len(self.pose_bank), (count,), device=self.device)
        self.pose_index[env_ids] = choice
        self.robot.write_root_link_pose_to_sim(self._pose_root[choice], env_ids)
        self.robot.write_root_link_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids
        )
        self.robot.write_joint_state_to_sim(
            self._pose_joints[choice],
            torch.zeros(count, self._pose_joints.shape[1], device=self.device),
            env_ids=env_ids,
        )

    def _force_welds_off(self, env_ids: torch.Tensor | None = None) -> None:
        data = self._env.sim.data
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        for weld_id in self._weld_ids.tolist():
            data.eq_active[ids, int(weld_id)] = False

    def _on_ground(self) -> torch.Tensor:
        sensor = self._env.scene[self.cfg.ground_contact_sensor_name]
        found = sensor.data.found
        if found is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return found.reshape(self.num_envs, -1).any(dim=1)

    def _points_to_torso(self, points_w: torch.Tensor) -> torch.Tensor:
        torso = self._env.sim.data.xpos[:, self._torso_body_id]
        rotation = _as_rotation_matrix(
            self._env.sim.data.xmat[:, self._torso_body_id]
        )
        while torso.ndim < points_w.ndim:
            torso = torso.unsqueeze(1)
            rotation = rotation.unsqueeze(1)
        world_to_torso = rotation.transpose(-1, -2)
        return torch.matmul(world_to_torso, (points_w - torso).unsqueeze(-1)).squeeze(-1)

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        qualified = name if "/" in name else f"{self.cfg.entity_name}/{name}"
        object_id = int(
            mujoco.mj_name2id(self._env.sim.mj_model, object_type, qualified)
        )
        if object_id < 0:
            raise ValueError(
                f"MJCF object {qualified!r} ({object_type.name}) was not found"
            )
        return object_id

    def _debug_vis_impl(self, visualizer) -> None:
        self._update_target_caption()
        if hasattr(visualizer, "server"):
            self._install_viser_target_markers(visualizer)
            return
        env_indices = list(visualizer.get_env_indices(self.num_envs))
        current = self.point_pos_w.detach().cpu().numpy()
        targets = self.targets_w.detach().cpu().numpy()
        weights = self.point_weights.detach().cpu().numpy()
        for batch in env_indices:
            for point_id in range(self.num_points):
                visualizer.add_sphere(
                    current[batch, point_id],
                    0.03,
                    (0.15, 0.55, 1.0, 0.95),
                )
                visualizer.add_sphere(
                    targets[batch, point_id],
                    0.045,
                    (1.0, 0.55, 0.1, float(weights[batch, point_id])),
                )

    def _install_viser_target_markers(self, visualizer) -> None:
        if getattr(visualizer, "_target_marker_hook_installed", False):
            return
        original = visualizer._update_visualization_locked

        def _update_visualization_locked(*args, **kwargs):
            original(*args, **kwargs)
            body_xpos = args[0] if args else kwargs["body_xpos"]
            scene_offset = args[5] if len(args) > 5 else kwargs["scene_offset"]
            self._paint_viser_targets(visualizer, body_xpos, scene_offset)

        visualizer._update_visualization_locked = _update_visualization_locked
        visualizer._target_marker_hook_installed = True

    def _paint_viser_targets(self, visualizer, body_xpos: np.ndarray, scene_offset) -> None:
        enabled = (
            self.cfg.debug_vis
            and self._debug_vis_enabled
            and getattr(visualizer, "debug_visualization_enabled", True)
        )
        if not enabled:
            self._set_marker_visible(self._current_marker_handle, False)
            self._set_marker_visible(self._target_marker_handle, False)
            return
        env_indices = list(visualizer.get_env_indices(body_xpos.shape[0]))
        if not env_indices:
            self._set_marker_visible(self._current_marker_handle, False)
            self._set_marker_visible(self._target_marker_handle, False)
            return
        offset = np.asarray(scene_offset, dtype=np.float64)
        current = self.point_pos_w.detach().cpu().numpy()
        targets = self.targets_w.detach().cpu().numpy()
        weights = self.point_weights.detach().cpu().numpy()
        current_positions: list[np.ndarray] = []
        target_positions: list[np.ndarray] = []
        target_opacities: list[float] = []
        for batch in env_indices:
            for point_id in range(self.num_points):
                current_positions.append(self._offset_point(current[batch, point_id], offset))
                target_positions.append(self._offset_point(targets[batch, point_id], offset))
                target_opacities.append(float(weights[batch, point_id]))
        self._current_marker_handle = self._sync_viser_spheres(
            visualizer,
            "/debug/body_points",
            self._current_marker_handle,
            self._current_marker_count,
            current_positions,
            color=(38, 140, 255),
            radius=0.03,
            opacity=0.95,
        )
        self._current_marker_count = len(current_positions)
        self._target_marker_handle = self._sync_viser_spheres(
            visualizer,
            "/debug/body_targets",
            self._target_marker_handle,
            self._target_marker_count,
            target_positions,
            color=(255, 140, 26),
            radius=0.045,
            opacity=1.0,
            opacities=np.asarray(target_opacities, dtype=np.float32),
        )
        self._target_marker_count = len(target_positions)

    @staticmethod
    def _offset_point(point: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return (np.asarray(point, dtype=np.float64) + offset).astype(np.float32)

    @staticmethod
    def _set_marker_visible(handle, visible: bool) -> None:
        if handle is not None:
            handle.visible = visible

    def _sync_viser_spheres(
        self,
        visualizer,
        name: str,
        handle,
        previous_count: int,
        positions: list[np.ndarray],
        *,
        color: tuple[int, int, int],
        radius: float,
        opacity: float,
        opacities: np.ndarray | None = None,
    ):
        count = len(positions)
        pos = np.stack(positions).astype(np.float32)
        cols = np.tile(np.asarray(color, dtype=np.uint8), (count, 1))
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
        scales = np.full((count, 3), radius, dtype=np.float32)
        alphas = (
            np.full(count, opacity, dtype=np.float32)
            if opacities is None
            else np.asarray(opacities, dtype=np.float32)
        )
        if handle is None or previous_count != count:
            if handle is not None:
                handle.remove()
            vertices, faces = self._target_marker_mesh()
            handle = visualizer.server.scene.add_batched_meshes_simple(
                name,
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
            return handle
        handle.batched_positions = pos
        handle.batched_wxyzs = quats
        handle.batched_scales = scales
        handle.batched_colors = cols
        handle.batched_opacities = alphas
        handle.visible = True
        return handle

    def _target_marker_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self._target_marker_vertices is None or self._target_marker_faces is None:
            import trimesh

            sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
            self._target_marker_vertices = np.asarray(sphere.vertices, dtype=np.float32)
            self._target_marker_faces = np.asarray(sphere.faces, dtype=np.uint32)
        return self._target_marker_vertices, self._target_marker_faces

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
        """Show the swing limb, the five weights, and the weighted distance."""

        del name, on_change, request_action
        self._gui_env = get_env_idx
        with server.gui.add_folder("Body targets"):
            self._target_caption = server.gui.add_markdown("**Target:** —")

    def _update_target_caption(self) -> None:
        handle = self._target_caption
        if handle is None:
            return
        env_idx = 0 if self._gui_env is None else int(self._gui_env())
        env_idx = min(max(env_idx, 0), self.num_envs - 1)
        point = self.cfg.point_names[int(self.swing_index[env_idx].item())]
        weights = self.point_weights[env_idx].detach().tolist()
        weight_lines = "\n\n".join(
            f"**{name.replace('_', ' ')}:** {weight:.2f}"
            for name, weight in zip(self.cfg.point_names, weights)
        )
        distance = float(self.metrics["distance"][env_idx].item())
        handle.content = (
            f"**Moving:** {point.replace('_', ' ')}\n\n"
            f"{weight_lines}\n\n"
            f"**Blue:** current points\n\n"
            f"**Orange:** targets, faded by weight\n\n"
            f"**Weighted distance:** {distance:.3f} m\n\n"
            f"{episode_reward_caption(self._episode_reward_total(env_idx))}"
        )

    def _episode_reward_total(self, env_idx: int) -> float:
        manager = getattr(self._env, "reward_manager", None)
        sums = getattr(manager, "_episode_sums", None)
        if not sums:
            return 0.0
        return float(sum(float(value[env_idx].item()) for value in sums.values()))


@dataclass(kw_only=True)
class BodyPointTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`BodyPointTargetCommand`."""

    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    site_names: tuple[str, ...] = SITE_NAMES
    point_names: tuple[str, ...] = POINT_NAMES
    torso_body_name: str = "torso_link"
    rung_site_names: tuple[str, ...] = ()
    weld_names: tuple[str, str] = ("left_grip_weld", "right_grip_weld")
    ground_contact_sensor_name: str = "ladder_ground_contact"
    pose_bank: tuple[LadderPose, ...] = ()
    target_ball_radius: float = TARGET_BALL_RADIUS_M
    latches_enabled: bool = False

    def build(self, env: ManagerBasedRlEnv) -> BodyPointTargetCommand:
        return BodyPointTargetCommand(self, env)


def _body_targets(env: ManagerBasedRlEnv, command_name: str) -> BodyPointTargetCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, BodyPointTargetCommand):
        raise TypeError(
            f"Command {command_name!r} is {type(command).__name__}, "
            "expected BodyPointTargetCommand"
        )
    return command


def target_position_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Alive bonus minus the capped weighted distance, per second."""

    return _body_targets(env, command_name).reward_terms()["target_pos"]


def target_gain_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Rate for twice the weighted drop in per-point distance since the previous step."""

    return _body_targets(env, command_name).reward_terms()["target_gain"]


def body_target_rung_endpoints(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Torso-frame rung endpoints, shape ``(num_envs, num_rungs, 6)``."""

    return _body_targets(env, command_name).rung_endpoints_torso


def body_target_privileged(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Critic-only distances: five point errors and the weighted sum."""

    return _body_targets(env, command_name).privileged()
