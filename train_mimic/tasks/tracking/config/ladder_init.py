"""Climbing initial pose, with no simulator imports.

The mjlab ladder robot uses these values as its reset state. A prototype can
read the same pose without compiling MuJoCo.

Feet stand on 0-based rung 3 with the mass over the tread. The knees are
nearly straight so the shins stay behind the rung above, and the torso
follows the rail. Hands hold rung 7 from the climber's side, left hand on
the left and right hand on the right, the way a person grips a rung. The
grip stays clear of the wood. Under the training position targets both
welds stay under 100 N and the pelvis stays under the overspeed trip.
"""

from __future__ import annotations

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
    ".*_shoulder_pitch_joint": -1.029680,
    "left_shoulder_roll_joint": 0.310190,
    "right_shoulder_roll_joint": -0.310190,
    "left_shoulder_yaw_joint": -0.277790,
    "right_shoulder_yaw_joint": 0.277790,
    ".*_elbow_joint": 1.713410,
    "left_wrist_roll_joint": -0.619780,
    "right_wrist_roll_joint": 0.619780,
    ".*_wrist_pitch_joint": 1.271040,
    "left_wrist_yaw_joint": 1.321060,
    "right_wrist_yaw_joint": -1.321060,
}
