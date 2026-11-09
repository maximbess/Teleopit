"""Translate the pinned manufacturer model without changing its kinematics/inertia."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).with_name("o6_grasp") / "assets"
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


def model_definition(side="right", prefix=""):
    if side not in ("left", "right"):
        raise ValueError("hand side must be left or right")
    folder = ASSETS if side == "right" else ASSETS / "left"
    path = folder / f"linkerhand_o6_{side}.urdf"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pinned {side} O6 assets: {path}; reinstall the complete teleopit package"
        )
    root = ET.parse(path).getroot()
    namespace(root, prefix)
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    active = [
        joints[prefix + name.replace("rh_", "lh_" if side == "left" else "rh_")]
        for name in ACTIVE_NAMES
    ]
    passive = [j for j in root.findall("joint") if j.find("mimic") is not None]
    return root, active, passive


def namespace(root, prefix):
    for element in root.iter():
        for key in ("name", "link", "joint"):
            if key in element.attrib:
                element.set(key, prefix + element.get(key))


def _link(parent, link, joint, config, asset, folder):
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
    ET.SubElement(asset, "mesh", name=mesh_name, file=str(folder / filename))
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


def section(root, name):
    found = root.find(name)
    return ET.SubElement(root, name) if found is None else found


def attach_hand(
    root,
    parent,
    config,
    *,
    side="right",
    prefix="",
    pos=(0, 0, 0),
    orientation=(1, 0, 0, 0),
    floating=False,
):
    source, active, passive = model_definition(side, prefix)
    folder = ASSETS if side == "right" else ASSETS / "left"
    base_name = prefix + ("lh_" if side == "left" else "rh_") + "hand_base_link"
    asset = section(root, "asset")
    links = {link.get("name"): link for link in source.findall("link")}
    base = _link(parent, links[base_name], None, config, asset, folder)
    base.set("pos", numbers(pos))
    base.set("quat", numbers(orientation))
    if floating:
        ET.SubElement(base, "freejoint", name=prefix + "floating_wrist")
    ET.SubElement(base, "site", name=prefix + "wrist_mount", pos="0 0 0", size="0.003")
    ET.SubElement(
        base, "site", name=prefix + "palm_frame", pos="0 0 0.06", size="0.003"
    )
    bodies = {base_name: base}
    contact = section(root, "contact")
    for joint in source.findall("joint"):
        parent_name, child = (
            joint.find("parent").get("link"),
            joint.find("child").get("link"),
        )
        bodies[child] = _link(
            bodies[parent_name], links[child], joint, config, asset, folder
        )
        # Only adjacent connected links are excluded; all nonadjacent pairs remain enabled.
        ET.SubElement(contact, "exclude", body1=parent_name, body2=child)
    add_drives(root, active, passive, config)
    return joint_mapping(base_name, prefix, active, passive)


def add_drives(root, active, passive, config):
    equality = section(root, "equality")
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
    actuators = section(root, "actuator")
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


def joint_mapping(base_name, prefix, active, passive):
    return {
        "base": base_name,
        "active": [j.get("name") for j in active],
        "passive": [
            (
                j.get("name"),
                j.find("mimic").get("joint"),
                float(j.find("mimic").get("multiplier")),
            )
            for j in passive
        ],
        "limits": [
            (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
            for j in active
        ],
        "mount": prefix + "wrist_mount",
        "palm": prefix + "palm_frame",
    }
