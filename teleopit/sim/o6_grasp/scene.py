"""Translate the pinned manufacturer model without changing its kinematics/inertia."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).with_name("assets")
ACTIVE_NAMES = (
    "rh_thumb_cmc_pitch",
    "rh_thumb_cmc_yaw",
    "rh_index_mcp_pitch",
    "rh_middle_mcp_pitch",
    "rh_ring_mcp_pitch",
    "rh_pinky_mcp_pitch",
)


def numbers(values):
    return " ".join(str(float(v)) for v in values)


def quat(rpy):
    return numbers(
        Rotation.from_euler("xyz", np.fromstring(rpy, sep=" ")).as_quat()[[3, 0, 1, 2]]
    )


def model_definition():
    root = ET.parse(ASSETS / "linkerhand_o6_right.urdf").getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    active = [joints[name] for name in ACTIVE_NAMES]
    passive = [j for j in root.findall("joint") if j.find("mimic") is not None]
    return root, active, passive


def _link(parent, link, joint, config, asset):
    attributes = {"name": link.get("name")}
    if joint is not None:
        origin = joint.find("origin")
        attributes.update(pos=origin.get("xyz"), quat=quat(origin.get("rpy")))
    body = ET.SubElement(parent, "body", attributes)
    inertial = link.find("inertial")
    inertia = inertial.find("inertia")
    ET.SubElement(
        body,
        "inertial",
        pos=inertial.find("origin").get("xyz"),
        mass=inertial.find("mass").get("value"),
        fullinertia=" ".join(
            inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    )
    if joint is not None:
        limit = joint.find("limit")
        ET.SubElement(
            body,
            "joint",
            name=joint.get("name"),
            axis=joint.find("axis").get("xyz"),
            range=f"{limit.get('lower')} {limit.get('upper')}",
            damping=str(config.joint_damping),
            armature=str(config.armature),
        )
    mesh_name = link.get("name")
    filename = link.find("collision/geometry/mesh").get("filename")
    ET.SubElement(asset, "mesh", name=mesh_name, file=str(ASSETS / filename))
    # Separate render/collision geoms. Low-poly source is deliberately used for rendering too.
    ET.SubElement(
        body,
        "geom",
        name=mesh_name + "_visual",
        type="mesh",
        mesh=mesh_name,
        contype="0",
        conaffinity="0",
        group="2",
        rgba="0.65 0.69 0.74 1",
        mass="0",
    )
    ET.SubElement(
        body,
        "geom",
        name=mesh_name + "_collision",
        type="mesh",
        mesh=mesh_name,
        group="3",
        rgba="0.2 0.6 0.8 0.25",
        mass="0",
    )
    if mesh_name.endswith("_distal"):
        ET.SubElement(
            body,
            "site",
            name=mesh_name.replace("rh_", "") + "_tip",
            pos="0.006 0 0.036",
            size="0.002",
            rgba="0 1 0 1",
        )
    return body


def build_scene(config):
    config.validate()
    source, active, passive = model_definition()
    root = ET.Element("mujoco", model="standalone_linkerhand_o6")
    ET.SubElement(
        root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false"
    )
    ET.SubElement(
        root,
        "option",
        timestep=str(config.physics_dt),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        solver="Newton",
        cone="elliptic",
        iterations=str(config.solver_iterations),
        tolerance=str(config.solver_tolerance),
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        friction=f"{config.friction} 0.001 0.0001",
        condim="4",
        solref=f"{config.contact_timeconst} 1",
        solimp="0.95 0.99 0.001",
    )
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    links = {link.get("name"): link for link in source.findall("link")}
    base = _link(world, links["rh_hand_base_link"], None, config, asset)
    base.set("pos", numbers(config.palm_pos))
    base.set("quat", numbers(config.palm_quat))
    ET.SubElement(base, "freejoint", name="floating_wrist")
    ET.SubElement(base, "site", name="wrist_mount", pos="0 0 0", size="0.003")
    ET.SubElement(base, "site", name="palm_frame", pos="0 0 0.06", size="0.003")
    bodies = {"rh_hand_base_link": base}
    contact = ET.SubElement(root, "contact")
    for joint in source.findall("joint"):
        parent, child = (
            joint.find("parent").get("link"),
            joint.find("child").get("link"),
        )
        bodies[child] = _link(bodies[parent], links[child], joint, config, asset)
        # Only adjacent connected links are excluded; all nonadjacent pairs remain enabled.
        ET.SubElement(contact, "exclude", body1=parent, body2=child)
    rung = ET.SubElement(
        world,
        "body",
        name="rung",
        pos=numbers(config.rung_pos),
        quat=numbers(config.rung_quat),
    )
    ET.SubElement(rung, "site", name="rung_frame", size="0.003")
    if config.rung_enabled:
        _rung_geom(rung, config)
    equality = ET.SubElement(root, "equality")
    for joint in passive:
        mimic = joint.find("mimic")
        ET.SubElement(
            equality,
            "joint",
            joint1=joint.get("name"),
            joint2=mimic.get("joint"),
            polycoef=f"{mimic.get('offset')} {mimic.get('multiplier')} 0 0 0",
            solref="0.004 1",
            solimp="0.9999 0.99999 0.001",
        )
    actuators = ET.SubElement(root, "actuator")
    for joint in active:
        limit = joint.find("limit")
        ET.SubElement(
            actuators,
            "position",
            name=joint.get("name") + "_servo",
            joint=joint.get("name"),
            kp=str(config.kp),
            kv=str(config.kv),
            ctrlrange=f"{limit.get('lower')} {limit.get('upper')}",
            forcerange=f"{-config.effort_limit} {config.effort_limit}",
        )
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml


def _rung_geom(body, config):
    attributes = {"name": "rung_collision", "rgba": "0.85 0.45 0.12 1"}
    for name in ("friction", "solref", "solimp"):
        value = getattr(config, "rung_" + name)
        if value is not None:
            attributes[name] = numbers(value)
    if config.cross_section == "circular":
        attributes.update(
            type="cylinder",
            size=numbers((config.radius, config.length / 2)),
            quat="0.7071067811865476 0 0.7071067811865476 0",
        )
    else:
        attributes.update(
            type="box",
            size=numbers((config.length / 2, config.width / 2, config.height / 2)),
        )
    ET.SubElement(body, "geom", attributes)
