import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from teleopit.sim.o6_grasp import GraspConfig, O6GraspEnv
from teleopit.sim.o6_grasp.evaluate import scenarios
from teleopit.sim.o6_grasp.presets import ladder_rung_config
from teleopit.sim.o6_hand import attach_hand


def test_two_authoritative_hands_attach_with_unique_names():
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    world = ET.SubElement(root, "worldbody")
    for side, y in [("left", 0.2), ("right", -0.2)]:
        parent = ET.SubElement(world, "body", name=side, pos=f"0 {y} 0")
        mapping = attach_hand(root, parent, GraspConfig(), side=side, prefix=side + "_")
        assert len(mapping["active"]) == 6 and len(mapping["passive"]) == 5
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nu == 12 and model.nq == 22 and model.neq == 10
    assert model.joint("left_lh_thumb_cmc_pitch").axis[0] == -1
    assert model.joint("right_rh_thumb_cmc_pitch").axis[0] == 1


def test_low_friction_changes_compiled_contact_coefficients():
    cfg = scenarios(ladder_rung_config())["friction_low"]
    env = O6GraspEnv(cfg)
    for _ in range(150):
        env.step(np.zeros(6))
    contacts = [c for c in env.data.contact if env.rung_geom in (c.geom1, c.geom2)]
    assert contacts
    assert all(c.friction[0] <= 0.25 for c in contacts)
