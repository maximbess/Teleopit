"""Shared CPU-only ladder geometry and reset pose for training and reference preview."""
from __future__ import annotations

from pathlib import Path
import re

import mujoco

from teleopit.runtime.assets import UNITREE_G1_XML
from teleopit.runtime.g1_collision import apply_g1_collision_overlay


def build_ladder_preview_model(robot_xml: str | Path | None = None):
    """Compile the training geometry without importing the GPU training stack."""
    spec = mujoco.MjSpec.from_file(str(robot_xml or UNITREE_G1_XML))
    for item in list(spec.actuators):
        spec.delete(item)
    for item in list(spec.keys):
        spec.delete(item)
    _remove_embedded_ladder_floor(spec)
    _remove_embedded_ladder_lights(spec)
    _add_ladder_to_g1_spec(spec)
    apply_g1_collision_overlay(spec)
    spec.worldbody.add_geom(name="preview_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=(5, 5, .01), rgba=(.18, .22, .26, 1))
    spec.worldbody.add_light(pos=(-2, -2, 4), dir=(.4, .3, -1), castshadow=False)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = _LADDER_INITIAL_ROOT_POS
    data.qpos[3:7] = (1, 0, 0, 0)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        value = 0.0
        for pattern, candidate in _LADDER_INITIAL_JOINT_POS.items():
            if re.fullmatch(pattern, model.joint(j).name):
                value = candidate
        data.qpos[model.jnt_qposadr[j]] = value
    data.eq_active[:] = False
    mujoco.mj_forward(model, data)
    return model, data

_LADDER_NUM_RUNGS = 9
_LADDER_HALF_BASE = 1.05
_LADDER_HEIGHT = 2.80
_LADDER_HALF_WIDTH = 0.35
_LADDER_RAIL_RADIUS = 0.050
_LADDER_RUNG_HALF_DEPTH = 0.055
_LADDER_RUNG_HALF_HEIGHT = 0.035
_LADDER_GRIP_RADIUS = 0.075
_LADDER_START_RUNG = 4
_LADDER_INITIAL_FOOT_RUNG = 1
_LADDER_FOOT_CONTACT_SENSOR = "ladder_foot_contact"
_LADDER_ARM_EFFORT_SCALE = 0.70
_LADDER_CURRICULUM_MIN_PHASE_STEPS = (36_000, 120_000, 120_000, 120_000)
_LADDER_INITIAL_ROOT_POS = (-0.913384, 0.0, 1.364509)
_LADDER_INITIAL_JOINT_POS = {
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


def _required_spec_body(spec: mujoco.MjSpec, name: str):
    for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
        if body.name == name:
            return body
    raise ValueError(f"G1 MuJoCo XML is missing required body {name!r}")


def _add_ladder_to_g1_spec(spec: mujoco.MjSpec) -> None:
    """Augment the canonical G1 model with an A-frame ladder and grip welds."""

    existing_sites = {
        site.name for site in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_SITE)
    }
    generated_site_names = {"left_grip_site", "right_grip_site"}
    if generated_site_names & existing_sites:
        raise ValueError(
            "The ladder task requires the canonical G1 XML without pre-added grip "
            "sites. Remove custom left_grip_site/right_grip_site definitions."
        )

    left_wrist = _required_spec_body(spec, "left_wrist_yaw_link")
    right_wrist = _required_spec_body(spec, "right_wrist_yaw_link")
    left_wrist.add_site(
        name="left_grip_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(0.10, 0.0, 0.0),
        size=(0.05,),
        group=3,
        rgba=(1.0, 0.0, 0.0, 0.7),
    )
    right_wrist.add_site(
        name="right_grip_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(0.10, 0.0, 0.0),
        size=(0.05,),
        group=3,
        rgba=(0.0, 0.0, 1.0, 0.7),
    )

    world = spec.worldbody
    for side, base_x in (("left", -_LADDER_HALF_BASE), ("right", _LADDER_HALF_BASE)):
        side_body = world.add_body(name=f"{side}_ladder_body")
        for rail_index, y in enumerate(
            (-_LADDER_HALF_WIDTH, _LADDER_HALF_WIDTH),
            start=1,
        ):
            side_body.add_geom(
                name=f"{side}_ladder_rail_{rail_index:02d}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=(base_x, y, 0.0, 0.0, y, _LADDER_HEIGHT),
                size=(_LADDER_RAIL_RADIUS,),
                contype=1,
                conaffinity=1,
                condim=4,
                friction=(1.4, 0.02, 0.002),
                solref=(0.005, 1.0),
                solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                margin=0.0,
                rgba=(0.55, 0.55, 0.58, 1.0),
            )

        for rung_index in range(1, _LADDER_NUM_RUNGS + 1):
            fraction = rung_index / (_LADDER_NUM_RUNGS + 1)
            x = base_x * (1.0 - fraction)
            z = _LADDER_HEIGHT * fraction
            fromto = (
                x,
                -_LADDER_HALF_WIDTH,
                z,
                x,
                _LADDER_HALF_WIDTH,
                z,
            )
            rung_name = f"{side}_ladder_rung_{rung_index:02d}"
            side_body.add_geom(
                name=rung_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, 0.0, z),
                size=(
                    _LADDER_RUNG_HALF_DEPTH,
                    _LADDER_HALF_WIDTH,
                    _LADDER_RUNG_HALF_HEIGHT,
                ),
                contype=1,
                conaffinity=1,
                condim=4,
                friction=(1.8, 0.02, 0.002),
                solref=(0.005, 1.0),
                solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
                margin=0.0,
                rgba=(0.55, 0.55, 0.58, 1.0),
            )
            side_body.add_site(
                name=f"{rung_name}_grip",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=fromto,
                size=(_LADDER_GRIP_RADIUS,),
                group=3,
                rgba=(0.15, 0.95, 0.25, 0.25),
            )

    for hand, color in (
        ("left", (1.0, 0.5, 0.0, 0.8)),
        ("right", (0.0, 1.0, 1.0, 0.8)),
    ):
        anchor = world.add_body(
            name=f"{hand}_grip_anchor_body",
            mocap=True,
            pos=(0.0, 0.0, -1.0),
        )
        anchor.add_site(
            name=f"{hand}_grip_anchor",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.012,),
            rgba=color,
        )
        spec.add_equality(
            name=f"{hand}_grip_weld",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"{hand}_grip_site",
            name2=f"{hand}_grip_anchor",
            active=False,
            data=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05),
            solref=(0.03, 1.0),
            solimp=(0.90, 0.95, 0.001, 0.5, 2.0),
        )


def _remove_embedded_ladder_floor(spec: mujoco.MjSpec) -> None:
    """Keep the scene-owned terrain as the only ladder ground plane.

    The canonical G1 XML contains a world ``floor`` plane, while the ladder
    ``SceneCfg`` also creates its own plane terrain.  Leaving both at z=0
    produces duplicate contacts and visible z-fighting in rendered playback.
    """

    for geom in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        if geom.name != "floor":
            continue
        if geom.type != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError(
                "The canonical G1 geom named 'floor' must be a plane so the "
                "ladder scene can replace it with SceneCfg terrain"
            )
        spec.delete(geom)
        return


def _remove_embedded_ladder_lights(spec: mujoco.MjSpec) -> None:
    """Use only the scene-owned light for stable ladder video rendering."""

    for light in list(spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_LIGHT)):
        spec.delete(light)
