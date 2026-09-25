"""Pinned OmniRetarget surfaces and the generated ladder collision asset contract."""
from pathlib import Path

REVISION = "bccd4d7451640a2800ddc77e469d911a84f91994"
REPOSITORY = "amazon-far/holosoma"
MODEL_PATH = "src/holosoma_retargeting/holosoma_retargeting/models/g1"
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets/robots/unitree_g1/omniretarget_collision"
SCHEMA_VERSION = 1
# Canonical rev. 1.0 has a different waist assembly and torso surface from the
# donor. Keep its torso mesh and account for the head's changed fixed offset.
CANONICAL_MESH_OVERRIDES = {"torso_link": "meshes/g1/torso_link_rev_1_0.STL"}
DONOR_TO_CANONICAL_TRANSLATIONS = {"head_link": (0.0, 0.0, 0.010)}
CANONICAL_JOINT_ORIGIN_DELTAS = {
    "waist_roll_link": (0.0, 0.0, 0.009),
    "torso_link": (0.0, 0.0, -0.019),
    "left_shoulder_pitch_link": (0.0, 0.0, 0.010),
    "right_shoulder_pitch_link": (0.0, 0.0, 0.010),
}
# Fixed accessory links are mapped into their canonical moving parent.
BODY_MAP = {"pelvis_contour_link": "pelvis", "torso_link": "torso_link", "head_link": "torso_link"}
for side in ("left", "right"):
    for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
                 "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"):
        BODY_MAP[f"{side}_{part}_link"] = f"{side}_{part}_link"

# Only these primitive geoms are replaced. Canonical hands remain unchanged.
REPLACED_GEOMS = {"pelvis_collision", "torso_collision", "head_collision"}
for side in ("left", "right"):
    REPLACED_GEOMS.update(f"{side}_{name}_collision" for name in
                         ("hip", "thigh", "shin", "linkage_brace", "shoulder_yaw", "elbow_yaw", "wrist"))
    REPLACED_GEOMS.update(f"{side}_foot{i}_collision" for i in range(1, 8))
