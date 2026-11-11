"""Current-run contract validation; historical measurements stay read-only."""

from dataclasses import asdict

import numpy as np

from teleopit.sim.g1_o6.config import Config
from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.g1_o6.metadata import SCHEMA, controller_settings


def validate_contract(metadata):
    try:
        if metadata["schema"] != SCHEMA:
            raise ValueError(
                "historical controller; reproduce it at its archived revision"
            )
        config, pose = resolved_settings(metadata)
        expected = {
            "controller": controller_settings(pose, config),
            "support_targets": pose.support_targets(),
            "command_joint_mapping": {
                side: metadata["model"]["hands"][side]["active"]
                for side in ("left", "right")
            },
        }
        for name, value in expected.items():
            if metadata[name] != value:
                raise ValueError(f"inconsistent {name}")
        for name in ("initial", "assets", "code_revision"):
            if name not in metadata:
                raise ValueError(f"missing {name}")
        geometry_fields(metadata)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid source-run contract: {exc}") from exc


def resolved_settings(metadata):
    config = Config(**metadata["config"])
    config.validate()
    pose = GraspPose(**metadata["grasp_pose"])
    for name, value in (("config", config), ("grasp_pose", pose)):
        if metadata[name].keys() != asdict(value).keys():
            raise ValueError(f"incomplete resolved {name}")
    return config, pose


def geometry_fields(metadata):
    initial = metadata["initial"]
    base = np.asarray(initial["base_pose"], dtype=float)
    joints = np.asarray(initial["joint_positions"], dtype=float)
    if base.shape != (7,) or joints.ndim != 1 or joints.size == 0:
        raise ValueError("incomplete initial geometry")
    if not np.isfinite(np.r_[base, joints]).all():
        raise ValueError("nonfinite initial geometry")
    for hand in metadata["model"]["hands"].values():
        if len(hand["active"]) != 6:
            raise ValueError("expected six named hand commands")
        for field in ("mount", "palm", "base"):
            if not hand[field]:
                raise ValueError(f"missing hand {field}")
