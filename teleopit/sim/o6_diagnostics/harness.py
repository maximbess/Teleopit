"""One authentic hand, exact coupled rung/materials/pose, no artificial surfaces."""

import json
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.holding import finger_targets
from teleopit.sim.o6_grasp.config import GraspConfig
from teleopit.sim.o6_grasp.wrenches import fixture_wrench
from teleopit.sim.o6_hand import attach_hand, numbers

from .mechanics import subtree
from .source import validate_contract
from .telemetry import Telemetry


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "resolved.json").read_text())
        validate_contract(self.metadata)
        self.model = mujoco.MjModel.from_xml_path(str(self.path / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        initial = self.metadata["initial"]
        self.data.qpos[:7] = initial["base_pose"]
        self.data.qpos[7:] = initial["joint_positions"]
        mujoco.mj_forward(self.model, self.data)
        self.hands = self.metadata["model"]["hands"]
        self.targets = self.metadata["support_targets"]


def build_hand(source, side, *, fixed=False, gravity=True, dt=None):
    original = source.hands[side]
    m, d = source.model, source.data
    root = ET.Element("mujoco", model="o6_diagnostic_" + side)
    ET.SubElement(
        root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false"
    )
    ET.SubElement(
        root,
        "option",
        timestep=str(dt or m.opt.timestep),
        gravity=numbers(m.opt.gravity if gravity else np.zeros(3)),
        integrator="implicitfast",
        solver="Newton",
        cone="elliptic",
        iterations=str(m.opt.iterations),
        tolerance=str(m.opt.tolerance),
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3")
    mount = d.site(original["mount"])
    rotation = Rotation.from_matrix(mount.xmat.reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
    hand = attach_hand(
        root,
        world,
        GraspConfig(),
        side=side,
        prefix=side + "_o6_",
        pos=mount.xpos,
        orientation=rotation,
        floating=not fixed,
    )
    copy_contact_settings(root, m)
    copy_actuators(root, m)
    if not fixed:
        add_source_rung(world, source, side)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, hand


def copy_contact_settings(root, source):
    for geom in root.findall(".//geom"):
        idx = source.geom(geom.get("name")).id
        for field in ("contype", "conaffinity", "condim", "priority"):
            geom.set(field, str(int(getattr(source, "geom_" + field)[idx])))
        for field in ("friction", "solref", "solimp"):
            geom.set(field, numbers(getattr(source, "geom_" + field)[idx]))
        geom.set("margin", str(source.geom_margin[idx]))


def copy_actuators(root, source):
    for actuator in root.findall("actuator/position"):
        i = source.actuator(actuator.get("name")).id
        actuator.set("kp", str(source.actuator_gainprm[i, 0]))
        actuator.set("kv", str(-source.actuator_biasprm[i, 2]))
        actuator.set("forcerange", numbers(source.actuator_forcerange[i]))
        actuator.set(
            "forcelimited", "true" if source.actuator_forcelimited[i] else "false"
        )
        actuator.set("ctrlrange", numbers(source.actuator_ctrlrange[i]))
        actuator.set(
            "ctrllimited", "true" if source.actuator_ctrllimited[i] else "false"
        )


def add_source_rung(world, source, side):
    name = source.targets[side + "_hand"]
    idx = source.model.geom(name).id
    body = ET.SubElement(
        world,
        "body",
        name="target_rung",
        pos=numbers(source.data.geom(name).xpos),
        quat=numbers(
            Rotation.from_matrix(source.data.geom(name).xmat.reshape(3, 3)).as_quat()[
                [3, 0, 1, 2]
            ]
        ),
    )
    geom = ET.SubElement(
        body,
        "geom",
        name=name,
        type="box",
        size=numbers(source.model.geom_size[idx]),
        mass="0",
    )
    for field in ("contype", "conaffinity", "condim", "priority"):
        geom.set(field, str(int(getattr(source.model, "geom_" + field)[idx])))
    for field in ("friction", "solref", "solimp"):
        geom.set(field, numbers(getattr(source.model, "geom_" + field)[idx]))
    geom.set("margin", str(source.model.geom_margin[idx]))


class HandHarness:
    def __init__(self, source, side, *, fixed=False, gravity=True, dt=None):
        self.model, self.hand = build_hand(
            source, side, fixed=fixed, gravity=gravity, dt=dt
        )
        self.data = mujoco.MjData(self.model)
        self.side = side
        self.fixed = fixed
        self.control_dt = source.metadata["config"]["control_dt"]
        self.dt = self.model.opt.timestep
        self.base = self.model.body(self.hand["base"]).id
        self.bodies = subtree(self.model, self.base)
        self.joints = [self.model.joint(n).id for n in self.hand["active"]]
        self.qpos = self.model.jnt_qposadr[self.joints]
        self.limits = self.model.jnt_range[self.joints]
        self.target = np.zeros(6)
        self.desired = np.zeros(6)
        self.controller = source.metadata["controller"]
        self.closed = np.array(self.controller["closed_pose_rad"])
        self.preload = self.controller["finger_preload_Nm"]
        self.kp = self.model.actuator_gainprm[:, 0].copy()
        self.queue = deque(
            [self.target.copy() for _ in range(self.controller["delay_steps"])]
        )
        self.fixture_config = GraspConfig()
        self.config = self.fixture_config
        self.palm_site_name = self.hand["palm"]
        self.fixture = np.zeros(6)
        self.external = np.zeros(6)
        self.gain = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.fixture_position = self.data.site(self.hand["palm"]).xpos.copy()
        self.fixture_rotation = (
            self.data.site(self.hand["palm"]).xmat.reshape(3, 3).copy()
        )
        targets = {} if fixed else {side + "_hand": source.targets[side + "_hand"]}
        self.telemetry = Telemetry(
            self.model,
            self.data,
            self.base,
            self.bodies,
            {side + "_hand": self.bodies},
            targets,
            {side: self.hand},
        )

    def command(self, amount, only=None):
        desired = finger_targets(
            amount, self.closed, self.data.qpos[self.qpos], self.kp, self.preload
        )
        if only is not None:
            desired = np.where(np.arange(6) == only, desired, 0.0)
        self.desired = np.clip(desired, self.limits[:, 0], self.limits[:, 1])
        delta = self.controller["command_rate_rad_s"] * self.control_dt
        self.target += np.clip(self.desired - self.target, -delta, delta)
        self.queue.append(self.target.copy())
        self.data.ctrl[:] = self.queue.popleft()

    def apply(self, gain, wrench):
        m, d = self.model, self.data
        self.gain = gain
        self.external = np.asarray(wrench)
        self.fixture = self.fixture_force(gain)
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_applyFT(
            m,
            d,
            self.fixture[:3],
            self.fixture[3:],
            d.site(self.hand["palm"]).xpos,
            self.base,
            d.qfrc_applied,
        )
        mujoco.mj_applyFT(
            m,
            d,
            self.external[:3],
            self.external[3:],
            d.site(self.hand["mount"]).xpos,
            self.base,
            d.qfrc_applied,
        )

    def fixture_force(self, gain):
        if gain == 0:
            return np.zeros(6)
        return fixture_wrench(self, gain)

    def palm_velocity(self, *, local):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.model.site(self.palm_site_name).id,
            velocity,
            int(local),
        )
        return velocity

    def sample(self):
        applied = [
            (self.fixture, self.data.site(self.hand["palm"]).xpos),
            (self.external, self.data.site(self.hand["mount"]).xpos),
        ]
        row = self.telemetry.sample(self.data, self.desired, self.target, applied)
        if self.fixed:
            row["momentum_rate"] = {
                "status": "not_evaluated_fixed_base_reaction_unmeasured",
                "linear_N": None,
                "angular_Nm": None,
                "balance_residual_N_Nm": None,
                "interval_s": None,
            }
        row.update(
            fixture_gain=float(self.gain),
            fixture_wrench_world_at_palm=self.fixture.tolist(),
            fixture_stiffness=(
                self.gain * np.array(self.fixture_config.fixture_stiffness)
            ).tolist(),
            fixture_damping=(
                self.gain * np.array(self.fixture_config.fixture_damping)
            ).tolist(),
            external_wrench_world_at_mount=self.external.tolist(),
            side=self.side,
            fixed_base=self.fixed,
        )
        row["max_joint_error_rad"] = max(
            (
                max(j["lower_violation_rad"], j["upper_violation_rad"])
                for j in row["joints"].values()
            ),
            default=0.0,
        )
        row["max_mimic_error_rad"] = max(
            (abs(j["residual_rad"]) for j in row["mimic"]), default=0.0
        )
        return row
