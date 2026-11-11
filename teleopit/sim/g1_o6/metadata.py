"""Resolved geometry and controller contract shared by every coupled runner."""

import platform
import subprocess
from dataclasses import asdict

import mujoco
import numpy as np
import scipy

from .holding import COMMAND_RATE_RAD_S, FINGER_LAW
from .model import provenance

SCHEMA = "g1_o6_run_v2"


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def controller_settings(pose, config):
    return {
        "schema": FINGER_LAW,
        "closed_pose_rad": list(pose.closed_pose),
        "finger_preload_Nm": pose.finger_preload_Nm,
        "command_rate_rad_s": COMMAND_RATE_RAD_S,
        "delay_steps": config.delay_steps,
        "reset_servo_preload": True,
        "body_compensation": "static_bias_minus_equal_foot_load",
        "acquisition_assistance": "pose_stabilization_only",
        "hold_external_assistance": False,
    }


def resolved(env, initial):
    model = dict(env.audit, initial_com_world=initial["com"])
    return {
        "schema": SCHEMA,
        "code_revision": revision(),
        "code_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        ),
        "config": asdict(env.config),
        "grasp_pose": asdict(env.pose),
        "controller": controller_settings(env.pose, env.config),
        "units": {
            "position": "m",
            "joint_angle": "rad",
            "hand_tilt": "deg",
            "time": "s",
            "force": "N",
            "moment": "N*m",
        },
        "command_joint_mapping": {s: h["active"] for s, h in env.hands.items()},
        "support_targets": env.support_targets,
        "hold_only": True,
        "mujoco_version": mujoco.__version__,
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "assets": provenance(),
        "model": model,
        "ik": env.ik,
        "initial": initial,
    }
