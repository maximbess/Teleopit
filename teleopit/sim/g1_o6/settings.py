"""JSON boundary shared by the coupled CLI and nominal diagnostics."""

import json
from dataclasses import fields
from pathlib import Path

from .config import Config
from .grasp_pose import GraspPose


def load_settings(config_path=None, grasp_path=None, *, ladder=None):
    from teleopit.sim.ladder_config import LadderConfig

    config, pose = load_config(Config, config_path), load_config(GraspPose, grasp_path)
    rung_count = (ladder or LadderConfig()).rung_count
    if max(*pose.hand_rungs, *pose.foot_rungs) > rung_count:
        raise ValueError("support rung absent from selected ladder")
    return config, pose


def load_config(cls, path):
    values = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(values, dict):
        raise TypeError(f"{cls.__name__} JSON must be an object")
    unknown = values.keys() - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    result = cls(**values)
    result.validate()
    return result
