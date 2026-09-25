"""Climbing initial pose, with no simulator imports.

The mjlab ladder robot uses these values as its reset state. A prototype can
read the same pose without compiling MuJoCo.
"""

from __future__ import annotations

LADDER_INITIAL_ROOT_POS = (-0.913384, 0.0, 1.364509)
LADDER_INITIAL_JOINT_POS = {
    ".*_hip_pitch_joint": -0.228917,
    ".*_knee_joint": 0.657862,
    ".*_ankle_pitch_joint": -0.333096,
    "waist_pitch_joint": -0.184737,
    ".*_shoulder_pitch_joint": -0.309773,
    "left_shoulder_roll_joint": 0.160023,
    "right_shoulder_roll_joint": -0.160023,
    "left_shoulder_yaw_joint": -0.034580,
    "right_shoulder_yaw_joint": 0.034580,
    ".*_elbow_joint": 0.569849,
    ".*_wrist_pitch_joint": -0.076011,
}
