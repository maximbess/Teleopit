"""Climbing initial pose, with no simulator imports.

The mjlab ladder robot uses these values as its reset state. A prototype can
read the same pose without compiling MuJoCo.

Feet stand on 0-based rung 3 and hands hold rung 7, four rungs above. The
root pitch follows the rail, the knees stay bent, and both feet can still
reach the next rung.
"""

from __future__ import annotations

LADDER_INITIAL_ROOT_POS = (-1.079482, 0.0, 1.385411)
# wxyz. A pitch about +Y of about 0.360 rad, the rail's lean from vertical.
LADDER_INITIAL_ROOT_ROT = (0.983835, 0.0, 0.179079, 0.0)
LADDER_INITIAL_JOINT_POS = {
    ".*_hip_pitch_joint": -1.472870,
    "left_hip_roll_joint": 0.003396,
    "right_hip_roll_joint": -0.003396,
    ".*_knee_joint": 1.107530,
    ".*_ankle_pitch_joint": -0.037076,
    "waist_pitch_joint": 0.142894,
    ".*_shoulder_pitch_joint": -1.494941,
    "left_shoulder_roll_joint": 0.519626,
    "right_shoulder_roll_joint": -0.519626,
    "left_shoulder_yaw_joint": -0.245816,
    "right_shoulder_yaw_joint": 0.245816,
    ".*_elbow_joint": 1.402475,
    "left_wrist_roll_joint": -0.603839,
    "right_wrist_roll_joint": 0.603839,
    ".*_wrist_pitch_joint": 0.067013,
    "left_wrist_yaw_joint": -1.521873,
    "right_wrist_yaw_joint": 1.521873,
}
