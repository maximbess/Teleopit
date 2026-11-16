"""Opt-in acquisition feedback through unchanged G1 arm position servos."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.diagnostics import failure_reasons, snapshot
from teleopit.sim.g1_o6.env import Environment, bounded

from .pose_arm import ArmReference, DEFAULT_RATE_RAD_S, OFFSET_BOUND_RAD, rung_pose
from .pose_contacts import LOADED_N, grasp_quality, hand_contacts

CARTESIAN_LIMIT_M = .005
ANGULAR_LIMIT_RAD = .03
ACQUIRE_START_S = 2.0
GAIN_PER_S = 2.0
MAX_RATE_RAD_S = .02


@dataclass(frozen=True)
class ArmFeedback:
    reference: str = "retain"
    rate_rad_s: float = DEFAULT_RATE_RAD_S

    def __post_init__(self):
        if self.reference not in ("retain", "seek"):
            raise ValueError("unknown arm-reference policy")
        if not np.isfinite(self.rate_rad_s) or not 0 < self.rate_rad_s <= MAX_RATE_RAD_S:
            raise ValueError("arm feedback rate must be finite and in (0, .02]")


def feedback_limits():
    return {"cartesian_command_bound_m": CARTESIAN_LIMIT_M,
            "angular_command_bound_rad": ANGULAR_LIMIT_RAD,
            "offset_bound_rad": OFFSET_BOUND_RAD,
            "acquire_start_s": ACQUIRE_START_S,
            "feedback_gain_per_s": GAIN_PER_S}


class ArmEnvironment(Environment):
    def __init__(self, *args, arm_feedback, **kwargs):
        self.arm_feedback = arm_feedback
        super().__init__(*args, **kwargs)

    def reset(self):
        row = super().reset()
        self.hold_controller = AcquisitionArmReference(self, self.arm_feedback)
        return row


def feedback_contacts(env):
    row = snapshot(env)
    hands = {
        side: grasp_quality(hand_contacts(env, row["contacts"], side),
            env.data.geom(env.support_targets[side + "_hand"]).xpos[2])
        for side in env.hands
    }
    structural = {"drift", "angle", "penetration", "joint_limits", "mimic",
                  "effort", "numerical", "unintended_support"}
    # Complete original gates still score every HOLD step. Assistance makes
    # static balance unsuitable as a controller inhibition during acquisition.
    reasons = sorted(set(failure_reasons(row, env.config, ())) & structural)
    reasons.extend(side + "_slip" for side in env.hands
        if row["supports"][side + "_hand"]["max_slip"] >= env.config.max_slip_m_s)
    return hands, reasons


def thumb_gap(env, data, side):
    target = env.support_targets[side + "_hand"]
    rung = env.model.geom(target).id
    measurements = []
    for geom in range(env.model.ngeom):
        name = env.model.geom(geom).name
        if not (name.startswith(side + "_o6_") and "thumb" in name
                and env.model.geom_contype[geom]):
            continue
        segment = np.zeros(6)
        distance = mujoco.mj_geomDistance(env.model, data, geom, rung, .1, segment)
        vector = segment[3:] - segment[:3]
        if 1e-9 < distance < .1 and np.linalg.norm(vector) > 1e-9:
            measurements.append((float(distance), name, vector))
    if not measurements:
        return {"target": target, "status": "NO_RELIABLE_GAP", "vector_world_m": [0, 0, 0]}
    distance, name, vector = min(measurements, key=lambda value: value[0])
    return {"target": target, "status": "MEASURED", "thumb_geom": name,
            "distance_m": distance, "vector_world_m": vector.tolist()}


class AcquisitionArmReference(ArmReference):
    def __init__(self, env, settings):
        self.settings = settings
        self.command_scratch = mujoco.MjData(env.model)
        super().__init__(env)

    def reset(self, env):
        super().reset(env)
        self.anchor, self.seek, self.quality = {}, {}, {}
        self.inhibition = []
        self.displacement, self.angle = {}, {}

    def active(self, env):
        if env.phase in ("REMOVE_ASSISTANCE", "HOLD"):
            return True
        return (self.settings.reference == "seek" and env.phase == "ACQUIRE"
                and env.data.time - env.phase_start >= ACQUIRE_START_S - 1e-9)

    def desired(self, env, amount):
        desired = self.base.desired(env, amount)
        if self.active(env):
            mujoco.mj_copyData(self.scratch, env.model, env.data)
            mujoco.mj_forward(env.model, self.scratch)
            self.quality, self.inhibition = feedback_contacts(env)
            self.latch(env)
            for side, indices in self.indices.items():
                self.update(env, desired, side, indices)
        self.raw = desired.copy()
        return desired

    def latch(self, env):
        if self.reference:
            return
        self.anchor = {s: rung_pose(env, self.scratch, s) for s in env.hands}
        self.reference = {s: (p.copy(), r.copy()) for s, (p, r) in self.anchor.items()}
        self.latch_time = float(env.data.time)
        if self.settings.reference == "seek":
            for side in env.hands:
                self.seek_reference(env, side)

    def seek_reference(self, env, side):
        measured = thumb_gap(env, self.scratch, side)
        vector = np.asarray(measured["vector_world_m"])
        shift = np.zeros(3)
        if self.quality[side]["thumb_normal_N"] < LOADED_N and np.linalg.norm(vector) > 0:
            # At most 0.5 mm past the geometric witness to establish contact;
            # actual penetration, effort and anatomical gates stay unchanged.
            shift = bounded(vector + .0005 * vector / np.linalg.norm(vector), CARTESIAN_LIMIT_M)
        rotation = self.scratch.geom(measured["target"]).xmat.reshape(3, 3)
        shift = rotation.T @ shift
        self.reference[side] = (self.anchor[side][0] + shift, self.anchor[side][1])
        self.seek[side] = {**measured, "target_shift_rung_m": shift.tolist()}

    def step_offset(self, env, desired, side, indices, rate):
        if self.inhibition:
            rate = -self.offset[side] / env.config.control_dt
        else:
            rate = GAIN_PER_S * rate
        candidate = self.bounded_step(env, desired, side, indices, rate, self.settings.rate_rad_s)
        candidate, distance, angle = self.project_command(env, desired, side, candidate, self.offset[side])
        self.displacement[side], self.angle[side] = distance, angle
        return candidate

    def project_command(self, env, desired, side, offset, previous=None):
        data, model = self.command_scratch, env.model
        mujoco.mj_copyData(data, model, env.data)
        command = desired[self.indices[side]]
        origin = self.command_pose(env, side, command)
        delta = self.command_motion(env, side, command + offset, origin)
        if delta[0] <= CARTESIAN_LIMIT_M and delta[1] <= ANGULAR_LIMIT_RAD:
            return offset, *delta
        previous = np.zeros_like(offset) if previous is None else previous
        low, high = 0.0, 1.0
        for _ in range(16):
            scale = (low + high) / 2
            candidate = previous + scale * (offset - previous)
            delta = self.command_motion(env, side, command + candidate, origin)
            if delta[0] <= CARTESIAN_LIMIT_M and delta[1] <= ANGULAR_LIMIT_RAD:
                low = scale
            else:
                high = scale
        candidate = previous + low * (offset - previous)
        delta = self.command_motion(env, side, command + candidate, origin)
        return candidate, *delta

    def command_pose(self, env, side, command):
        indices = self.indices[side]
        joints = env.model.actuator_trnid[indices, 0]
        limits = env.model.jnt_range[joints]
        self.command_scratch.qpos[env.act_qpos[indices]] = np.clip(command, limits[:, 0], limits[:, 1])
        mujoco.mj_kinematics(env.model, self.command_scratch)
        return rung_pose(env, self.command_scratch, side)

    def command_motion(self, env, side, command, origin):
        position, rotation = self.command_pose(env, side, command)
        return (float(np.linalg.norm(position - origin[0])),
                float(Rotation.from_matrix(rotation @ origin[1].T).magnitude()))

    def state(self):
        state = super().state()
        state["reference_at_activation"] = state.pop("reference_at_hold_entry")
        return {**state, "mode": "g6_acquisition_arm_feedback",
            "settings": asdict(self.settings), "activation_time_s": self.latch_time,
            "thumb_seek": self.seek, "grasp_quality": self.quality,
            "inhibition": self.inhibition, "command_displacement_m": self.displacement,
            "command_rotation_rad": self.angle}

    def metadata(self, env):
        return {**super().metadata(env), "schema": "g6_acquisition_arm_feedback_v1",
            "mode": "g6_acquisition_arm_feedback", "settings": asdict(self.settings),
            "offset_rate_rad_s": self.settings.rate_rad_s, **feedback_limits(),
            "reference": "measured wrist at activation; optional fixed thumb-gap target shift <=5 mm",
            "cartesian_bound": "scratch FK of original arm commands versus commands plus correction; both translation and rotation bounded",
            "inhibition": "back off on structural physical violations or intended-hand slip; all scored gates unchanged"}
