"""Predeclared local batches on unchanged G6 geometry and native material."""

import re
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from teleopit.sim.g1_o6.grasp_pose import GraspPose

from .ladder_campaign import digest
from .g6_arm import ArmFeedback


@dataclass(frozen=True)
class G6Candidate:
    id: str
    pose: GraspPose = field(
        default_factory=lambda: GraspPose(hand_rungs=(8, 8), foot_rungs=(3, 3))
    )
    reset: dict = field(default_factory=dict)
    parent: str | None = None
    family: str = "control"
    controller: str | ArmFeedback = "reference"

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.id) or not (
            self.controller == "reference" or isinstance(self.controller, ArmFeedback)
        ):
            raise ValueError("unknown candidate/controller")
        limits = {"root_offset_x": 0.08, "root_offset_z": 0.06, "foot_offset_x": 0.02}
        if any(
            k not in limits or not np.isfinite(v) or abs(v) > limits[k]
            for k, v in self.reset.items()
        ):
            raise ValueError("reset parameters exceed the declared search bounds")
        if max((*self.pose.hand_rungs, *self.pose.foot_rungs)) > 8:
            raise ValueError("support tread absent from G6")

    def record(self):
        return json.loads(json.dumps(asdict(self)))

    def digest(self):
        return digest(
            {
                "pose": asdict(self.pose),
                "reset": self.reset,
                "controller": self.record()["controller"],
            }
        )

    @classmethod
    def from_dict(cls, value):
        controller = value.get("controller", "reference")
        if isinstance(controller, dict):
            controller = ArmFeedback(**controller)
        return cls(**{**value, "pose": GraspPose(**value["pose"]), "controller": controller})


def variant(parent, batch, index, *, pose=None, reset=None, family):
    return replace(
        parent,
        id=f"{batch.lower()}_{index:02d}",
        parent=parent.id,
        family=family,
        pose=replace(parent.pose, **(pose or {})),
        reset={**parent.reset, **(reset or {})},
    )


def batch_candidates(batch, parent):
    changes = {
        "A": root_changes,
        "B": support_changes,
        "C": grasp_changes,
        "D": closure_changes,
    }[batch](parent)
    return [variant(parent, batch, i, **change) for i, change in enumerate(changes)]


def root_changes(parent):
    result = [
        {"reset": {key: parent.reset.get(key, 0) + delta}, "family": key}
        for key in ("root_offset_x", "root_offset_z")
        for delta in (-0.04, -0.02, 0.02, 0.04)
    ]
    result.extend(
        {
            "pose": {"waist_lean_rad": parent.pose.waist_lean_rad + delta},
            "family": "waist",
        }
        for delta in (-0.10, -0.05, 0.05, 0.10)
    )
    return result


def support_changes(parent):
    result = [
        {
            "pose": {"hand_rungs": (hand, hand), "foot_rungs": (foot, foot)},
            "family": "support_pair",
        }
        for hand, foot in ((7, 2), (7, 3), (6, 1), (6, 2))
    ]
    result.extend(
        {
            "pose": {"hand_rungs": (hand, hand), "foot_rungs": (foot, foot)},
            "reset": {"foot_offset_x": delta},
            "family": "foot_placement",
        }
        for hand, foot in ((7, 2), (6, 1))
        for delta in (-0.01, 0.01)
    )
    return result


def grasp_changes(parent):
    result = [
        {"pose": {"hand_tilt_deg": parent.pose.hand_tilt_deg + delta}, "family": "tilt"}
        for delta in (-10, -5, 5, 10)
    ]
    for axis in (0, 2):
        for delta in (-0.005, 0.005):
            offset = list(parent.pose.grip_offset_m)
            offset[axis] += delta
            result.append({"pose": {"grip_offset_m": offset}, "family": "grip_offset"})
    result.extend(
        {
            "pose": {"hand_spacing_m": parent.pose.hand_spacing_m + delta},
            "family": "spacing",
        }
        for delta in (-0.02, 0.02)
    )
    result.extend(
        {"pose": {"right_grip_offset_m": (0, 0, delta)}, "family": "right_asymmetry"}
        for delta in (-0.005, 0.005)
    )
    return result


def closure_changes(parent):
    result = [
        {"pose": {"finger_preload_Nm": value}, "family": "preload"}
        for value in (0.10, 0.14)
    ]
    for delta in (-0.2, -0.1, 0.1, 0.2):
        closed = list(parent.pose.closed_pose)
        closed[1] += delta
        result.append({"pose": {"closed_pose": closed}, "family": "thumb_closure"})
    closed = list(parent.pose.closed_pose)
    closed[2:] = [v + 0.1 for v in closed[2:]]
    result.append({"pose": {"closed_pose": closed}, "family": "finger_closure"})
    return result
