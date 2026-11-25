"""Climbing initial pose, with no simulator imports.

The mjlab ladder robot uses these values as its reset state. A prototype can
read the same pose without compiling MuJoCo.

Feet stand on 0-based rung 3 with the mass over the tread. The knees are
nearly straight so the shins stay behind the rung above, and the torso
follows the rail. Hands hold rung 7 from the climber's side, left hand on
the left and right hand on the right. The elbows hang below the shoulders
and the forearms reach forward to the rung. The wrists stay near neutral.
The grip stays clear of the wood. Under the training position targets both
welds stay under 50 N and the pelvis stays under the overspeed trip.

``LADDER_POSE_BANK`` is the set of reset poses for body-point targets. It
starts with this keyframe. Add further poses to that tuple; the robot
default state stays the first one.
"""

from __future__ import annotations

from dataclasses import dataclass

LADDER_INITIAL_ROOT_POS = (-0.816186, 0.0, 1.576016)
# wxyz. A pitch about +Y of about 0.082 rad, with the waist sharing the rail lean.
LADDER_INITIAL_ROOT_ROT = (0.999150, 0.0, 0.041219, 0.0)
LADDER_INITIAL_JOINT_POS = {
    ".*_hip_pitch_joint": -0.135494,
    "left_hip_roll_joint": 0.003396,
    "right_hip_roll_joint": -0.003396,
    ".*_knee_joint": 0.063312,
    ".*_ankle_pitch_joint": -0.014484,
    "waist_pitch_joint": 0.060937,
    ".*_shoulder_pitch_joint": 0.065580,
    "left_shoulder_roll_joint": 0.569310,
    "right_shoulder_roll_joint": -0.569310,
    "left_shoulder_yaw_joint": -0.392490,
    "right_shoulder_yaw_joint": 0.392490,
    ".*_elbow_joint": 0.754930,
    "left_wrist_roll_joint": 0.003620,
    "right_wrist_roll_joint": -0.003620,
    ".*_wrist_pitch_joint": -0.165270,
    "left_wrist_yaw_joint": -0.018650,
    "right_wrist_yaw_joint": 0.018650,
}


@dataclass(frozen=True)
class LadderPose:
    """One reset pose. Joint values are regex patterns, as in the robot cfg."""

    root_pos: tuple[float, float, float]
    root_rot: tuple[float, float, float, float]
    joint_pos: dict[str, float]


LADDER_INITIAL_POSE = LadderPose(
    root_pos=LADDER_INITIAL_ROOT_POS,
    root_rot=LADDER_INITIAL_ROOT_ROT,
    joint_pos=dict(LADDER_INITIAL_JOINT_POS),
)
LADDER_POSE_BANK = (LADDER_INITIAL_POSE,)
