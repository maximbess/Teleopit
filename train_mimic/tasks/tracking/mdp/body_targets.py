"""Move one body point toward a nearby target. The other points stay put.

Latches are not used. The grip action is still accepted so the policy width
matches the ladder action, and the welds are forced inactive.
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

TARGET_BALL_RADIUS_M = 0.20
# Per second, before the reward manager multiplies by the control step.
# The cap is five points times the ball radius, so a fresh target is still
# worth a positive rate and a drift of one meter of total error stays positive.
TARGET_ALIVE_RATE = 1.25
TARGET_DISTANCE_CAP_M = 1.0
TARGET_GAIN_SCALE = 2.0


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
    fresh: torch.Tensor,
    on_ground: torch.Tensor,
    dt: float,
    scale: float = TARGET_GAIN_SCALE,
) -> torch.Tensor:
    """Per-second rate whose step integral is ``scale * (previous - current)``.

    The first sample after a reset, and a ground contact, pay 0.
    """

    safe_dt = max(float(dt), 1.0e-8)
    rate = scale * (previous - current) / safe_dt
    return torch.where(fresh | on_ground, torch.zeros_like(rate), rate)


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
    """Sample a pose, then a target for one of five body points.

    The other four targets stay on the points measured after that pose is
    written. ``latches_enabled`` is false: grip actions are stored and both
    welds are cleared every update.
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
        self.active_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.targets_w = torch.zeros(
            self.num_envs, self.num_points, 3, dtype=torch.float32, device=self.device
        )
        self._pending_targets = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._distance_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._distance_prev = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._reward_cache_step = -1
        self._reward_cache: dict[str, torch.Tensor] | None = None
        self.grip_request = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )
        self._target_caption = None
        self._gui_env = None
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
        """One-hot, torso-frame targets, and torso-frame point-to-target errors."""

        points_t = self._points_to_torso(self.point_pos_w)
        targets_t = self._points_to_torso(self.targets_w)
        one_hot = torch.nn.functional.one_hot(
            self.active_index, num_classes=self.num_points
        ).to(dtype=points_t.dtype)
        flat = points_t.shape[0]
        return torch.cat(
            (
                one_hot,
                targets_t.reshape(flat, -1),
                (targets_t - points_t).reshape(flat, -1),
            ),
            dim=-1,
        )

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
        """Five distances and their sum."""

        distances = self.point_distances()
        return torch.cat((distances, distances.sum(dim=-1, keepdim=True)), dim=-1)

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
        self._update_metrics()

    def reward_terms(self) -> dict[str, torch.Tensor]:
        """Position rate and distance-gain rate for this control step.

        Both terms share one sample. The gain is 0 on the first call after a
        reset. Ground contact zeros both rates.
        """

        step_id = int(getattr(self._env, "common_step_counter", -1))
        if self._reward_cache is not None and self._reward_cache_step == step_id:
            return self._reward_cache
        distance = point_distance_sum(self.point_pos_w, self.targets_w)
        on_ground = self._on_ground()
        fresh = ~self._distance_valid
        position = target_position_rate(distance, on_ground)
        gain = target_gain_rate(
            self._distance_prev,
            distance,
            fresh=fresh,
            on_ground=on_ground,
            dt=float(self._env.step_dt),
        )
        self._distance_prev.copy_(distance)
        self._distance_valid[:] = True
        self._reward_cache = {"target_pos": position, "target_gain": gain}
        self._reward_cache_step = step_id
        return self._reward_cache

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._pending_targets[env_ids] = True
        self._distance_valid[env_ids] = False
        self._distance_prev[env_ids] = 0.0
        self._reward_cache_step = -1
        self.grip_request[env_ids] = 0.0
        self._write_pose(env_ids)
        self._force_welds_off(env_ids)

    def _update_command(self) -> None:
        self._force_welds_off()
        pending = torch.where(self._pending_targets)[0]
        if pending.numel() == 0:
            self._update_target_caption()
            return
        points = self.point_pos_w[pending]
        active = sample_active_indices(
            int(pending.numel()), self.num_points, device=self.device
        )
        offsets = sample_ball_offsets(
            int(pending.numel()),
            float(self.cfg.target_ball_radius),
            device=self.device,
        )
        self.targets_w[pending] = assign_targets(points, active, offsets)
        self.active_index[pending] = active
        self._pending_targets[pending] = False
        self._update_target_caption()

    def _update_metrics(self) -> None:
        self.metrics["distance"].copy_(self.point_distances().sum(dim=-1))
        self.metrics["active_point"].copy_(self.active_index.float())

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
        for batch in env_indices:
            active = int(self.active_index[batch].item())
            for point_id in range(self.num_points):
                chosen = point_id == active
                rgba = (1.0, 0.75, 0.1, 0.95) if chosen else (0.2, 0.45, 0.95, 0.45)
                visualizer.add_sphere(
                    self.targets_w[batch, point_id].detach().cpu().numpy(),
                    0.04 if chosen else 0.025,
                    rgba,
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
        handle = self._target_marker_handle
        if not enabled:
            if handle is not None:
                handle.visible = False
            return
        env_indices = list(visualizer.get_env_indices(body_xpos.shape[0]))
        if not env_indices:
            if handle is not None:
                handle.visible = False
            return
        offset = np.asarray(scene_offset, dtype=np.float64)
        targets = self.targets_w.detach().cpu().numpy()
        active = self.active_index.detach().cpu().numpy()
        positions: list[np.ndarray] = []
        colors: list[tuple[int, int, int]] = []
        scales: list[float] = []
        for batch in env_indices:
            for point_id in range(self.num_points):
                positions.append(
                    (np.asarray(targets[batch, point_id], dtype=np.float64) + offset).astype(
                        np.float32
                    )
                )
                chosen = int(active[batch]) == point_id
                colors.append((242, 191, 26) if chosen else (51, 115, 242))
                scales.append(0.04 if chosen else 0.025)
        self._sync_viser_targets(visualizer, positions, colors, scales)
        if self._target_marker_handle is not None:
            self._target_marker_handle.visible = True

    def _sync_viser_targets(self, visualizer, positions, colors, scales) -> None:
        count = len(positions)
        pos = np.stack(positions).astype(np.float32)
        cols = np.asarray(colors, dtype=np.uint8)
        radii = np.asarray(scales, dtype=np.float32)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
        batched_scales = np.repeat(radii[:, None], 3, axis=1)
        handle = self._target_marker_handle
        if handle is None or self._target_marker_count != count:
            if handle is not None:
                handle.remove()
            vertices, faces = self._target_marker_mesh()
            self._target_marker_handle = visualizer.server.scene.add_batched_meshes_simple(
                "/debug/body_targets",
                vertices,
                faces,
                batched_wxyzs=quats,
                batched_positions=pos,
                batched_scales=batched_scales,
                batched_colors=cols,
                opacity=0.85,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
                lod="off",
            )
            self._target_marker_count = count
            return
        handle.batched_positions = pos
        handle.batched_wxyzs = quats
        handle.batched_scales = batched_scales
        handle.batched_colors = cols

    def _target_marker_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self._target_marker_vertices is None or self._target_marker_faces is None:
            import trimesh

            sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
            self._target_marker_vertices = np.asarray(sphere.vertices, dtype=np.float32)
            self._target_marker_faces = np.asarray(sphere.faces, dtype=np.uint32)
        return self._target_marker_vertices, self._target_marker_faces

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
        """Show which point is being moved and how far the five points are."""

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
        point = self.cfg.point_names[int(self.active_index[env_idx].item())]
        distance = float(self.metrics["distance"][env_idx].item())
        handle.content = (
            f"**Moving:** {point.replace('_', ' ')}\n\n"
            f"**Distance sum:** {distance:.3f} m\n\n"
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
    """Alive bonus minus the capped distance sum, per second."""

    return _body_targets(env, command_name).reward_terms()["target_pos"]


def target_gain_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Rate for twice the drop in the distance sum since the previous step."""

    return _body_targets(env, command_name).reward_terms()["target_gain"]


def body_target_rung_endpoints(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Torso-frame rung endpoints, shape ``(num_envs, num_rungs, 6)``."""

    return _body_targets(env, command_name).rung_endpoints_torso


def body_target_privileged(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Critic-only distances: five point errors and their sum."""

    return _body_targets(env, command_name).privileged()
