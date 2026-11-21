"""Regulate a captured distal-thumb contact through the bounded G1 arm and O6 grip."""

from dataclasses import dataclass

import mujoco
import numpy as np

from teleopit.sim.g1_o6.env import bounded
from .g6_acquisition import StabilizedAcquisition
from .g6_grip_preview import PreviewEnvironment, PreviewGrip
from .g6_knee_retention import capture_ready
from .g6_whole_body import damped_step

MODES = {"right": ("right",), "both": ("left", "right"), "distributed": ("left", "right")}
THUMB_TARGET_N = .65
NORMAL_GAIN = .004  # m/(N s)
TANGENT_GAIN = .8  # 1/s
SPEED_M_S = .002
RECENTER_GAIN = .4  # 1/s


@dataclass(frozen=True)
class WristGrip(PreviewGrip):
    wrist_mode: str = "right"

    def __post_init__(self):
        super().__post_init__()
        if self.wrist_mode not in MODES:
            raise ValueError("unknown bounded wrist feedback mode")


def pad_velocity(error_rung_m, load, opposed):
    """Tangential position and underside-normal load feedback in the G6 tread frame."""
    tangent = bounded(TANGENT_GAIN * error_rung_m[:2], SPEED_M_S)
    toward_tread = NORMAL_GAIN * (THUMB_TARGET_N - (load if opposed else 0.))
    return np.r_[tangent, np.clip(toward_tread, -SPEED_M_S, SPEED_M_S)]


def joint_rate(jacobian, velocity, offset, distributed):
    # Recenter in the redundant directions. Only the selected joints correct the pad velocity.
    rate = -RECENTER_GAIN * offset
    active = slice(None) if distributed else slice(-3, None)
    rate[active] += damped_step(jacobian[:, active], velocity - jacobian @ rate)
    return rate


@dataclass
class ThumbLandmark:
    body: int
    rung: int
    local: np.ndarray
    target: np.ndarray

    @classmethod
    def capture(cls, env, data, side, quality):
        thumb_ip = env.model.joint(f"{side}_o6_{side[0]}h_thumb_ip").id
        body = env.model.jnt_bodyid[thumb_ip]
        rung = env.model.geom(env.support_targets[side+"_hand"]).id
        target = np.array(quality["centroid_by_group_m"]["thumb"])
        world = data.geom_xpos[rung] + data.geom_xmat[rung].reshape(3,3) @ target
        local = data.xmat[body].reshape(3,3).T @ (world - data.xpos[body])
        return cls(body, rung, local, target)

    def kinematics(self, model, data, dofs):
        point = data.xpos[self.body] + data.xmat[self.body].reshape(3,3) @ self.local
        rotation = data.geom_xmat[self.rung].reshape(3,3).T
        position = rotation @ (point - data.geom_xpos[self.rung])
        linear = np.zeros((3,model.nv))
        mujoco.mj_jac(model, data, linear, None, point, self.body)
        return self.target - position, rotation @ linear[:,dofs]


class WristReference(StabilizedAcquisition):
    def __init__(self, env, settings):
        super().__init__(env, settings)
        self.mode = env.grasp_acquisition.wrist_mode
        self.sides = MODES[self.mode]
        self.landmarks, self.requests = {}, {}
        self.updates = dict.fromkeys(("left","right"), 0)
        self.capture_s = None
        self.dofs = {s: env.model.jnt_dofadr[env.model.actuator_trnid[ids,0]] for s,ids in self.indices.items()}
        for side, indices in self.indices.items():
            names = [env.model.joint(i).name for i in env.model.actuator_trnid[indices[-3:],0]]
            if names != [f"{side}_wrist_{axis}_joint" for axis in ("roll", "pitch", "yaw")]:
                raise ValueError("unexpected G1 wrist joint order")

    def step_offset(self, env, desired, side, indices, rate):
        elapsed = env.data.time - env.phase_start
        if self.capture_s is None and capture_ready(elapsed, self.quality, self.inhibition):
            self.landmarks = {s: ThumbLandmark.capture(env, self.scratch, s, self.quality[s]) for s in self.sides}
            self.capture_s = elapsed
        if side not in self.landmarks or self.inhibition:
            self.requests.pop(side, None)
            return super().step_offset(env, desired, side, indices, rate)
        error, jacobian = self.landmarks[side].kinematics(env.model, self.scratch, self.dofs[side])
        hand = self.quality[side]
        velocity = pad_velocity(error, hand["thumb_normal_N"], hand["thumb_opposition"])
        rate = self.tracking_rate(side, jacobian, velocity)
        self.requests[side] = {"error_rung_m": error.tolist(), "velocity_rung_m_s": velocity.tolist(),
                               "joint_rate_rad_s": rate.tolist()}
        self.updates[side] += 1
        return self.project_rate(env, desired, side, indices, rate)

    def tracking_rate(self, side, jacobian, velocity):
        return joint_rate(jacobian, velocity, self.offset[side], self.mode == "distributed")

    def project_rate(self, env, desired, side, indices, rate):
        previous = self.offset[side]
        limit = self.settings.rate_rad_s
        # A feasible inward step allows a new direction at the 5 mm Cartesian boundary.
        inward = self.bounded_step(env, desired, side, indices, -RECENTER_GAIN*previous, limit)
        inward, _, _ = self.project_command(env, desired, side, inward, previous)
        candidate = self.bounded_step(env, desired, side, indices, rate, limit)
        candidate, self.displacement[side], self.angle[side] = self.project_command(env, desired, side, candidate, inward)
        return candidate

    def state(self):
        return {**super().state(), "wrist_grip": {"mode": self.mode, "capture_s": self.capture_s,
                "updates": self.updates.copy(), "requests": self.requests.copy(),
                "landmarks": {s: {"local_m": p.local.tolist(), "target_rung_m": p.target.tolist()}
                              for s,p in self.landmarks.items()}}}

    def metadata(self, env):
        return {**super().metadata(env), "wrist_grip": {"mode": self.mode, "thumb_target_N": THUMB_TARGET_N,
                "normal_gain_m_per_Ns": NORMAL_GAIN, "tangent_gain_per_s": TANGENT_GAIN,
                "speed_bound_per_component_m_s": SPEED_M_S, "recenter_gain_per_s": RECENTER_GAIN}}


class WristEnvironment(PreviewEnvironment):
    stabilizer_type = WristReference
