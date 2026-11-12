"""Declarative, hashable candidate identities and bounded one-factor families."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace

from teleopit.sim.g1_o6.grasp_pose import GraspPose

from .pose_controller import MODES


@dataclass(frozen=True)
class Candidate:
    id: str
    parent: str | None = None
    family: str = "reference"
    kind: str = "geometry"
    pose: GraspPose = field(default_factory=GraspPose)
    controller: str = "reference"
    ik_seed: int | None = None
    reset_offsets: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.id):
            raise ValueError("candidate id must be a safe lowercase identifier")
        if (
            self.kind not in ("geometry", "controller", "robustness")
            or self.controller not in MODES
        ):
            raise ValueError("unknown candidate kind/controller")
        if self.ik_seed is not None and (
            type(self.ik_seed) is not int or self.ik_seed < 0
        ):
            raise ValueError("invalid deterministic IK seed")
        if set(self.reset_offsets) - {"root_offset_x", "wrist_offset_x"}:
            raise ValueError("only declared reset offsets are supported")

    def record(self):
        return asdict(self)

    def digest(self):
        value = {
            k: v
            for k, v in self.record().items()
            if k not in ("id", "parent", "family", "kind", "changes")
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["pose"] = GraspPose(**value.get("pose", {}))
        return cls(**value)


def variant(parent, name, family, change, **other):
    return replace(
        parent,
        id=name,
        parent=parent.id,
        family=family,
        pose=replace(parent.pose, **change),
        changes=change,
        **other,
    )


def initial_geometry():
    reference = Candidate("reference")
    lower = variant(reference, "rung10", "rung", {"hand_rungs": (10, 10)})
    result = [reference, lower]
    for rungs in ((10, 11), (11, 10)):
        result.append(
            variant(
                reference, f"split_{rungs[0]}_{rungs[1]}", "rung", {"hand_rungs": rungs}
            )
        )
    pairs = {
        "hand_tilt_deg": (40, 60),
        "waist_lean_rad": (0.2, 0.4),
        "hand_spacing_m": (0.16, 0.2),
    }
    for parent in (reference, lower):
        for parameter, values in pairs.items():
            for value in values:
                suffix = str(value).replace(".", "p")
                result.append(
                    variant(
                        parent,
                        f"{parent.id}_{parameter}_{suffix}",
                        parameter,
                        {parameter: value},
                    )
                )
    return result


def offset_and_branch_candidates(parents):
    result = []
    for parent in parents[:2]:
        for axis in (0, 2):
            for delta in (-0.005, 0.005):
                offset = list(parent.pose.grip_offset_m)
                offset[axis] += delta
                name = f"{parent.id}_offset_{axis}_{'plus' if delta > 0 else 'minus'}"
                result.append(
                    variant(parent, name, "offset", {"grip_offset_m": tuple(offset)})
                )
        for seed in (17, 23):
            result.append(
                variant(parent, f"{parent.id}_ik_{seed}", "ik", {}, ik_seed=seed)
            )
    return result


def controller_candidates(parents, hand_names):
    yaw = [
        next(i for i, n in enumerate(names) if n.endswith("_thumb_cmc_yaw"))
        for names in hand_names.values()
    ]
    if len(set(yaw)) != 1:
        raise ValueError("inconsistent authentic thumb command mapping")
    result = []
    for parent in parents[:2]:
        for mode in MODES[1:]:
            result.append(
                variant(
                    parent,
                    parent.id + "_" + mode,
                    mode,
                    {},
                    kind="controller",
                    controller=mode,
                )
            )
        for preload in (0.10, 0.14):
            name = parent.id + "_preload_" + str(preload).replace(".", "p")
            result.append(
                variant(
                    parent,
                    name,
                    "preload",
                    {"finger_preload_Nm": preload},
                    kind="controller",
                )
            )
        closed = list(parent.pose.closed_pose)
        closed[yaw[0]] = 0.40
        result.append(
            variant(
                parent,
                parent.id + "_thumb_0p40",
                "thumb_target",
                {"closed_pose": tuple(closed)},
                kind="controller",
            )
        )
    return result
