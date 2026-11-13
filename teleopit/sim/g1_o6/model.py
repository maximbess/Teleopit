"""Compose canonical assets without training imports or grip attachments."""

import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco

from teleopit.runtime.assets import UNITREE_G1_XML
from teleopit.runtime.g1_collision_assets import ASSET_DIR
from teleopit.sim.g1_collision import apply_g1_collision_overlay
from teleopit.sim.ladder_geometry import add_ladder
from teleopit.sim.o6_grasp.config import GraspConfig
from teleopit.sim.o6_hand import ASSETS, attach_hand, section

MOUNT_POS = (0.0415, 0, 0)
MOUNT_QUAT = (2**-0.5, 0, 2**-0.5, 0)  # O6 +Z along G1 wrist +X


def canonical_tree():
    if not UNITREE_G1_XML.is_file():
        raise FileNotFoundError(
            "Run conda run -n teleopit python scripts/setup/install_climb_assets.py"
        )
    root = ET.parse(UNITREE_G1_XML).getroot()
    compiler = root.find("compiler")
    meshdir = UNITREE_G1_XML.parent / compiler.get("meshdir", "")
    for mesh in root.findall("asset/mesh"):
        path = (meshdir / mesh.get("file")).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing canonical G1 mesh {path}; run conda run -n teleopit python scripts/setup/install_climb_assets.py"
            )
        mesh.set("file", str(path))
    compiler.set("meshdir", "")
    return root


def install_hand(root, side):
    wrist = root.find(f'.//body[@name="{side}_wrist_yaw_link"]')
    for geom in list(wrist.findall("geom")):
        if (
            geom.get("name") == side + "_hand_collision"
            or geom.get("mesh") == side + "_rubber_hand"
        ):
            wrist.remove(geom)
    mapping = attach_hand(
        root,
        wrist,
        GraspConfig(),
        side=side,
        prefix=side + "_o6_",
        pos=MOUNT_POS,
        orientation=MOUNT_QUAT,
    )
    base = wrist.find(f'body[@name="{mapping["base"]}"]')
    for geom in base.iter("geom"):
        if geom.get("group") == "3":
            geom.attrib.update(
                contype="2",
                conaffinity="3",
                condim="4",
                friction="0.8 0.001 0.0001",
                solref="0.006 1",
                solimp="0.95 0.99 0.001",
            )
    for kind in ("force", "torque"):
        ET.SubElement(
            section(root, "sensor"), kind, name=side + "_" + kind, site=mapping["mount"]
        )
    # Adapter and its own wrist housing intersect as one assembly only.
    for link in ("wrist_roll_link", "wrist_pitch_link"):
        ET.SubElement(
            section(root, "contact"),
            "exclude",
            body1=mapping["base"],
            body2=side + "_" + link,
        )
    return mapping


def build_model(cfg, *, ladder=None, material=None):
    cfg.validate()
    root = canonical_tree()
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    apply_g1_collision_overlay(spec)
    add_ladder(spec, ladder)
    root = ET.fromstring(spec.to_xml())
    hands = {side: install_hand(root, side) for side in ("left", "right")}
    section(root, "option").attrib.update(
        timestep=str(cfg.physics_dt),
        integrator="implicitfast",
        solver="Newton",
        cone="elliptic",
        iterations="200",
        tolerance="1e-11",
        gravity="0 0 -9.81",
    )
    # Canonical keyframes have only 29 joints; derive reset explicitly instead.
    for key in list(root.findall("keyframe")):
        root.remove(key)
    # Canonical visual geoms can be anonymous. Give every diagnostic row a
    # stable identity before configuring and serializing compiled contacts.
    for index, geom in enumerate(root.find("worldbody").iter("geom")):
        if geom.get("name") is None:
            geom.set("name", f"g1_original_geom_{index:03d}")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    configure_contacts(root, model, cfg.friction_scale)
    # condim is structural: mutating its compiled array is not retained by
    # mj_saveLastXML. Configure the source, then compile and audit the result.
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    from .material import configure_material

    if material is not None and cfg.friction_scale != 1:
        raise ValueError("selective material requires unchanged global friction scale")
    material_audit = configure_material(root, model, material)
    if material_audit is not None:
        xml = ET.tostring(root, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml)
    audit = inventory(original, model, hands)
    if material_audit is not None:
        audit["material"] = material_audit
    return model, xml, hands, audit


def configure_contacts(root, model, scale):
    for element in root.find("worldbody").iter("geom"):
        geom = model.geom(element.get("name")).id
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            element.set("condim", "4")
            element.set("priority", "0")
            element.set(
                "friction", " ".join(str(v) for v in model.geom_friction[geom] * scale)
            )


def inventory(before, after, hands):
    actuators = []
    for i in range(after.nu):
        j = int(after.actuator_trnid[i, 0])
        actuators.append(
            {
                "name": after.actuator(i).name,
                "joint": after.joint(j).name,
                "gain": after.actuator_gainprm[i].tolist(),
                "bias": after.actuator_biasprm[i].tolist(),
                "effort_range": after.actuator_forcerange[i].tolist(),
                "joint_range": after.jnt_range[j].tolist(),
            }
        )
    return {
        **inertial_audit(before, after),
        "mass_before_kg": float(before.body_subtreemass[before.body("pelvis").id]),
        "mass_after_kg": float(after.body_subtreemass[after.body("pelvis").id]),
        "retained_wrist_mass_kg": {
            s: float(before.body(s + "_wrist_yaw_link").mass[0]) for s in hands
        },
        "mass_policy": "retain aggregated wrist/rubber-hand inertia: no separable source inertia available",
        "mount_position": MOUNT_POS,
        "mount_quaternion_wxyz": MOUNT_QUAT,
        "hands": hands,
        "actuators": actuators,
        "contacts": [
            {
                "name": after.geom(i).name,
                "mask": [int(after.geom_contype[i]), int(after.geom_conaffinity[i])],
                "friction": after.geom_friction[i].tolist(),
                "condim": int(after.geom_condim[i]),
            }
            for i in range(after.ngeom)
        ],
    }


def inertial_audit(before, after):
    result = {
        "com_reference": "canonical qpos0; identical G1 joint/base pose in both models"
    }
    for label, model in (("before", before), ("after", after)):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        result[f"com_{label}_world_m"] = data.subtree_com[
            model.body("pelvis").id
        ].tolist()
    result["retained_wrist_inertia"] = {
        side: {
            "mass_kg": float(before.body(side + "_wrist_yaw_link").mass[0]),
            "principal_inertia_kg_m2": before.body(
                side + "_wrist_yaw_link"
            ).inertia.tolist(),
            "inertial_position_m": before.body(side + "_wrist_yaw_link").ipos.tolist(),
            "inertial_quaternion_wxyz": before.body(
                side + "_wrist_yaw_link"
            ).iquat.tolist(),
        }
        for side in ("left", "right")
    }
    return result


def provenance():
    paths = [
        UNITREE_G1_XML,
        ASSET_DIR / "manifest.json",
        ASSETS / "manifest.json",
        ASSETS / "left/manifest.json",
    ]
    paths += list((UNITREE_G1_XML.parent / "meshes/g1").glob("*"))
    return {
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
            if p.is_file()
        },
        "hand_source": json.loads((ASSETS / "manifest.json").read_text()),
        "left_hand_source": json.loads((ASSETS / "left/manifest.json").read_text()),
        "collision_source": json.loads((ASSET_DIR / "manifest.json").read_text()),
        "canonical_revision": "external asset distribution has no upstream commit; exact XML/mesh SHA256 recorded",
    }
