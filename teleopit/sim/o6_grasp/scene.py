"""Standalone world around the reusable physical O6 component."""

import xml.etree.ElementTree as ET

import mujoco

__all__ = ["ACTIVE_NAMES", "ASSETS", "build_scene", "model_definition"]

from teleopit.sim.o6_hand import (
    ACTIVE_NAMES,
    ASSETS,
    attach_hand,
    model_definition,
    numbers,
)


def build_scene(config):
    config.validate()
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
    ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    attach_hand(
        root,
        world,
        config,
        pos=config.palm_pos,
        orientation=config.palm_quat,
        floating=True,
    )
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
