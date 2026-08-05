"""Automatic MuJoCo weld grips for ladder rungs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Mapping, cast

import mujoco
import numpy as np


HandName = Literal["left", "right"]
_HANDS: tuple[HandName, HandName] = ("left", "right")
_MISSING = object()


@dataclass(frozen=True)
class GripStatus:
    """Public state of one automatic hand grip."""

    requested: bool
    attached: bool
    rung_site: str | None


@dataclass(frozen=True)
class _HandSpec:
    hand_site_id: int
    anchor_body_id: int
    anchor_site_id: int
    anchor_mocap_id: int
    weld_id: int


@dataclass
class _HandState:
    requested: bool
    attached: bool = False
    rung_site_id: int | None = None
    rung_local_position: np.ndarray | None = None
    rung_local_rotation: np.ndarray | None = None


class AutomaticGrip:
    """Attach G1 hand sites to capsule-shaped ladder grip sites.

    A request is consumed when the hand attaches.  The weld stays active until
    :meth:`release` or :meth:`reset` is called.  While attached, the mocap
    anchor follows the rung frame, so this also works if a rung belongs to a
    moving body.

    ``attach_distance`` is an extra margin outside the capsule-site surface.
    Set it to ``0`` when the capsule-site itself already represents the entire
    allowed grip region.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        attach_distance: float,
        max_hand_speed: float,
        auto_attach: bool,
        rung_site_pattern: str,
        left_hand_site: str,
        right_hand_site: str,
        left_anchor_body: str,
        right_anchor_body: str,
        left_anchor_site: str,
        right_anchor_site: str,
        left_weld: str,
        right_weld: str,
        parking_position: np.ndarray | list[float] | tuple[float, float, float] = (
            0.0,
            0.0,
            -1.0,
        ),
    ) -> None:
        if attach_distance < 0.0:
            raise ValueError("grip.attach_distance must be >= 0")
        if max_hand_speed < 0.0:
            raise ValueError("grip.max_hand_speed must be >= 0")

        self.model = model
        self.data = data
        self.attach_distance = float(attach_distance)
        self.max_hand_speed = float(max_hand_speed)
        self.auto_attach = bool(auto_attach)
        self._parking_position = np.asarray(
            parking_position, dtype=np.float64
        ).reshape(-1)
        if self._parking_position.shape != (3,):
            raise ValueError("grip.parking_position must contain exactly 3 values")

        try:
            rung_pattern = re.compile(rung_site_pattern)
        except re.error as exc:
            raise ValueError(
                f"Invalid grip.rung_site_pattern {rung_site_pattern!r}: {exc}"
            ) from exc

        hand_site_names: dict[HandName, str] = {
            "left": left_hand_site,
            "right": right_hand_site,
        }
        anchor_body_names: dict[HandName, str] = {
            "left": left_anchor_body,
            "right": right_anchor_body,
        }
        anchor_site_names: dict[HandName, str] = {
            "left": left_anchor_site,
            "right": right_anchor_site,
        }
        weld_names: dict[HandName, str] = {
            "left": left_weld,
            "right": right_weld,
        }

        self._specs: dict[HandName, _HandSpec] = {}
        for hand in _HANDS:
            hand_site_id = self._required_id(
                mujoco.mjtObj.mjOBJ_SITE,
                hand_site_names[hand],
                f"grip.{hand}_hand_site",
            )
            anchor_body_id = self._required_id(
                mujoco.mjtObj.mjOBJ_BODY,
                anchor_body_names[hand],
                f"grip.{hand}_anchor_body",
            )
            anchor_site_id = self._required_id(
                mujoco.mjtObj.mjOBJ_SITE,
                anchor_site_names[hand],
                f"grip.{hand}_anchor_site",
            )
            weld_id = self._required_id(
                mujoco.mjtObj.mjOBJ_EQUALITY,
                weld_names[hand],
                f"grip.{hand}_weld",
            )

            mocap_id = int(self.model.body_mocapid[anchor_body_id])
            if mocap_id < 0:
                raise ValueError(
                    f"Body {anchor_body_names[hand]!r} must have mocap=\"true\""
                )
            if int(self.model.site_bodyid[anchor_site_id]) != anchor_body_id:
                raise ValueError(
                    f"Anchor site {anchor_site_names[hand]!r} is not inside "
                    f"body {anchor_body_names[hand]!r}"
                )
            if int(self.model.eq_type[weld_id]) != int(
                mujoco.mjtEq.mjEQ_WELD
            ):
                raise ValueError(
                    f"Equality {weld_names[hand]!r} must be type weld"
                )

            self._specs[hand] = _HandSpec(
                hand_site_id=hand_site_id,
                anchor_body_id=anchor_body_id,
                anchor_site_id=anchor_site_id,
                anchor_mocap_id=mocap_id,
                weld_id=weld_id,
            )

        if self._specs["left"].weld_id == self._specs["right"].weld_id:
            raise ValueError("Left and right grips must use different welds")
        if (
            self._specs["left"].anchor_mocap_id
            == self._specs["right"].anchor_mocap_id
        ):
            raise ValueError("Left and right grips must use different mocap bodies")

        self._rung_site_ids: tuple[int, ...] = self._find_rung_sites(rung_pattern)
        self._states: dict[HandName, _HandState] = {
            hand: _HandState(requested=self.auto_attach) for hand in _HANDS
        }

        # Ensure inactive equality constraints even if this object is created
        # after an externally modified simulation state.
        self.reset()

    @classmethod
    def from_config(
        cls,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cfg: Mapping[str, object] | object,
    ) -> AutomaticGrip:
        """Build from Teleopit's ``robot.grip`` DictConfig or a mapping."""

        return cls(
            model=model,
            data=data,
            attach_distance=float(_cfg_get(cfg, "attach_distance", 0.0)),
            max_hand_speed=float(_cfg_get(cfg, "max_hand_speed", 0.25)),
            auto_attach=bool(_cfg_get(cfg, "auto_attach", True)),
            rung_site_pattern=str(
                _cfg_get(cfg, "rung_site_pattern", _MISSING)
            ),
            left_hand_site=str(_cfg_get(cfg, "left_hand_site", _MISSING)),
            right_hand_site=str(_cfg_get(cfg, "right_hand_site", _MISSING)),
            left_anchor_body=str(
                _cfg_get(cfg, "left_anchor_body", _MISSING)
            ),
            right_anchor_body=str(
                _cfg_get(cfg, "right_anchor_body", _MISSING)
            ),
            left_anchor_site=str(
                _cfg_get(cfg, "left_anchor_site", _MISSING)
            ),
            right_anchor_site=str(
                _cfg_get(cfg, "right_anchor_site", _MISSING)
            ),
            left_weld=str(_cfg_get(cfg, "left_weld", _MISSING)),
            right_weld=str(_cfg_get(cfg, "right_weld", _MISSING)),
            parking_position=np.asarray(
                _cfg_get(cfg, "parking_position", (0.0, 0.0, -1.0)),
                dtype=np.float64,
            ),
        )

    def update(self) -> None:
        """Check and update grips immediately before one ``mj_step``."""

        for hand in _HANDS:
            state = self._states[hand]
            spec = self._specs[hand]

            # Keep an attached hand rigidly tied to the rung frame.  For a
            # static ladder this writes the same pose and is inexpensive.
            if state.attached:
                if not bool(self.data.eq_active[spec.weld_id]):
                    self._clear_attachment(state)
                else:
                    self._sync_anchor_to_rung(spec, state)
                    continue

            if not state.requested:
                continue
            if self._hand_speed(spec.hand_site_id) > self.max_hand_speed:
                continue

            rung_site_id = self._nearest_allowed_rung(spec.hand_site_id)
            if rung_site_id is not None:
                self._attach(hand, rung_site_id)

    def request_attach(self, hand: str) -> None:
        """Arm one hand; attachment occurs on a later :meth:`update`."""

        hand_name = _normalize_hand(hand)
        if not self._states[hand_name].attached:
            self._states[hand_name].requested = True

    def release(self, hand: str) -> None:
        """Disable one weld and disarm that hand."""

        hand_name = _normalize_hand(hand)
        state = self._states[hand_name]
        spec = self._specs[hand_name]
        self.data.eq_active[spec.weld_id] = 0
        state.requested = False
        self._clear_attachment(state)
        self._park_anchor(spec)

    def reset(self) -> None:
        """Disable both welds and restore the initial request state."""

        for hand in _HANDS:
            spec = self._specs[hand]
            state = self._states[hand]
            self.data.eq_active[spec.weld_id] = 0
            state.requested = self.auto_attach
            self._clear_attachment(state)
            self._park_anchor(spec)

    def status(self, hand: str) -> GripStatus:
        """Return an immutable snapshot of one hand's grip state."""

        hand_name = _normalize_hand(hand)
        state = self._states[hand_name]
        rung_name: str | None = None
        if state.rung_site_id is not None:
            rung_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                state.rung_site_id,
            )
        return GripStatus(
            requested=state.requested,
            attached=state.attached,
            rung_site=rung_name,
        )

    def is_attached(self, hand: str) -> bool:
        """Return whether one hand's weld is currently active."""

        return self.status(hand).attached

    def attached_rung(self, hand: str) -> str | None:
        """Return the attached rung site name, or ``None``."""

        return self.status(hand).rung_site

    def _required_id(self, obj_type: object, name: str, cfg_key: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise ValueError(
                f"{cfg_key} refers to missing MuJoCo object {name!r}"
            )
        return obj_id

    def _find_rung_sites(self, pattern: re.Pattern[str]) -> tuple[int, ...]:
        rung_ids: list[int] = []
        for site_id in range(self.model.nsite):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_id
            )
            if name is None or pattern.fullmatch(name) is None:
                continue
            if int(self.model.site_type[site_id]) != int(
                mujoco.mjtGeom.mjGEOM_CAPSULE
            ):
                raise ValueError(
                    f"Grip site {name!r} must have type=\"capsule\""
                )
            if float(self.model.site_size[site_id, 0]) <= 0.0:
                raise ValueError(f"Grip site {name!r} has a non-positive radius")
            if float(self.model.site_size[site_id, 1]) <= 0.0:
                raise ValueError(
                    f"Grip site {name!r} must have non-zero capsule length"
                )
            rung_ids.append(site_id)

        if not rung_ids:
            raise ValueError(
                "grip.rung_site_pattern did not match any capsule sites"
            )
        return tuple(rung_ids)

    def _nearest_allowed_rung(self, hand_site_id: int) -> int | None:
        hand_position = np.asarray(
            self.data.site_xpos[hand_site_id], dtype=np.float64
        )
        best_site_id: int | None = None
        best_surface_distance = np.inf

        for rung_site_id in self._rung_site_ids:
            start, end = self._capsule_axis(rung_site_id)
            closest = _closest_point_on_segment(hand_position, start, end)
            axis_distance = float(np.linalg.norm(hand_position - closest))
            radius = float(self.model.site_size[rung_site_id, 0])
            surface_distance = axis_distance - radius
            if surface_distance < best_surface_distance:
                best_surface_distance = surface_distance
                best_site_id = rung_site_id

        if best_surface_distance <= self.attach_distance:
            return best_site_id
        return None

    def _capsule_axis(self, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        center, rotation = self._site_pose(site_id)
        half_length = float(self.model.site_size[site_id, 1])
        axis = rotation[:, 2]
        return (
            center - axis * half_length,
            center + axis * half_length,
        )

    def _hand_speed(self, hand_site_id: int) -> float:
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            hand_site_id,
            spatial_velocity,
            0,
        )
        # MuJoCo spatial velocity is [angular_xyz, linear_xyz].
        return float(np.linalg.norm(spatial_velocity[3:6]))

    def _attach(self, hand: HandName, rung_site_id: int) -> None:
        spec = self._specs[hand]
        state = self._states[hand]

        hand_position, hand_rotation = self._site_pose(spec.hand_site_id)
        rung_position, rung_rotation = self._site_pose(rung_site_id)

        # Preserve the current hand pose relative to the selected rung.  This
        # prevents a large impulse when the equality constraint is enabled.
        state.rung_site_id = rung_site_id
        state.rung_local_position = (
            rung_rotation.T @ (hand_position - rung_position)
        )
        state.rung_local_rotation = rung_rotation.T @ hand_rotation

        self._set_anchor_site_pose(spec, hand_position, hand_rotation)
        mujoco.mj_forward(self.model, self.data)
        self.data.eq_active[spec.weld_id] = 1
        state.attached = True
        state.requested = False

    def _sync_anchor_to_rung(
        self, spec: _HandSpec, state: _HandState
    ) -> None:
        if (
            state.rung_site_id is None
            or state.rung_local_position is None
            or state.rung_local_rotation is None
        ):
            raise RuntimeError("Attached grip has incomplete rung-relative state")

        rung_position, rung_rotation = self._site_pose(state.rung_site_id)
        anchor_position = (
            rung_position + rung_rotation @ state.rung_local_position
        )
        anchor_rotation = rung_rotation @ state.rung_local_rotation
        self._set_anchor_site_pose(spec, anchor_position, anchor_rotation)

    def _set_anchor_site_pose(
        self,
        spec: _HandSpec,
        desired_site_position: np.ndarray,
        desired_site_rotation: np.ndarray,
    ) -> None:
        # The configured anchor site may have a local offset inside its mocap
        # body.  Solve the body pose that gives the desired world-space site
        # pose instead of assuming an identity local site frame.
        local_position = np.asarray(
            self.model.site_pos[spec.anchor_site_id], dtype=np.float64
        )
        local_rotation = _quat_to_matrix(
            np.asarray(
                self.model.site_quat[spec.anchor_site_id], dtype=np.float64
            )
        )
        body_rotation = desired_site_rotation @ local_rotation.T
        body_position = (
            desired_site_position - body_rotation @ local_position
        )

        self.data.mocap_pos[spec.anchor_mocap_id] = body_position
        self.data.mocap_quat[spec.anchor_mocap_id] = _matrix_to_quat(
            body_rotation
        )

    def _park_anchor(self, spec: _HandSpec) -> None:
        self.data.mocap_pos[spec.anchor_mocap_id] = self._parking_position
        self.data.mocap_quat[spec.anchor_mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

    def _site_pose(self, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy(),
            np.asarray(
                self.data.site_xmat[site_id], dtype=np.float64
            ).reshape(3, 3).copy(),
        )

    @staticmethod
    def _clear_attachment(state: _HandState) -> None:
        state.attached = False
        state.rung_site_id = None
        state.rung_local_position = None
        state.rung_local_rotation = None


def _cfg_get(
    cfg: Mapping[str, object] | object,
    key: str,
    default: object = _MISSING,
) -> object:
    value: object = _MISSING
    if isinstance(cfg, Mapping):
        value = cfg.get(key, _MISSING)
    else:
        getter = getattr(cfg, "get", None)
        if callable(getter):
            value = getter(key, _MISSING)
        elif hasattr(cfg, key):
            value = getattr(cfg, key)

    if value is _MISSING:
        if default is _MISSING:
            raise ValueError(f"Missing required robot.grip config key: {key}")
        return default
    return value


def _normalize_hand(hand: str) -> HandName:
    normalized = str(hand).strip().lower()
    if normalized not in _HANDS:
        raise ValueError("hand must be 'left' or 'right'")
    return cast(HandName, normalized)


def _closest_point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-16:
        return start.copy()
    fraction = float(np.dot(point - start, segment) / length_squared)
    return start + np.clip(fraction, 0.0, 1.0) * segment


def _quat_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion_wxyz)
    return matrix.reshape(3, 3)


def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion,
        np.ascontiguousarray(rotation, dtype=np.float64).reshape(9),
    )
    return quaternion
