"""Named thumb targets and bounded closure timing through original servos."""

from dataclasses import asdict, dataclass, replace

import numpy as np

from teleopit.sim.g1_o6.env import Environment
from teleopit.sim.g1_o6.holding import finger_targets
from teleopit.sim.g1_o6.metadata import controller_settings
from teleopit.sim.o6_grasp.controller import smooth
from teleopit.sim.o6_hand import ACTIVE_NAMES, model_definition

from .g6_arm import AcquisitionArmReference, ArmFeedback, MAX_RATE_RAD_S
from .pose_arm import DEFAULT_RATE_RAD_S

THUMB_DURATION_FRACTION = .5
FINGER_DELAY_FRACTION = .25


@dataclass(frozen=True)
class GraspAcquisition:
    order: str = "together"
    arm_rate_rad_s: float = 0.0

    def __post_init__(self):
        if self.order not in ("together", "thumb_first"):
            raise ValueError("unknown acquisition order")
        if self.arm_rate_rad_s not in (0, DEFAULT_RATE_RAD_S, MAX_RATE_RAD_S):
            raise ValueError("arm rate is outside the declared experiment")


def thumb_pose(pose, *, pitch, yaw):
    requested = {"rh_thumb_cmc_pitch": pitch, "rh_thumb_cmc_yaw": yaw}
    _, active, _ = model_definition()
    targets = dict(zip(ACTIVE_NAMES, pose.closed_pose))
    for joint in active:
        name = joint.get("name")
        if name in requested:
            value, limit = requested[name], joint.find("limit")
            if not np.isfinite(value) or not float(limit.get("lower")) <= value <= float(limit.get("upper")):
                raise ValueError("thumb target exceeds native range: " + name)
            targets[name] = value
    return replace(pose, closed_pose=tuple(targets[name] for name in ACTIVE_NAMES))


def closure_fractions(names, elapsed, duration):
    thumb = smooth(elapsed / (THUMB_DURATION_FRACTION * duration))
    fingers = smooth((elapsed / duration - FINGER_DELAY_FRACTION) / (1 - FINGER_DELAY_FRACTION))
    return np.array([thumb if "_thumb_" in name else fingers for name in names])


class FingerAcquisition:
    def __init__(self, env):
        self.settings, self.base = env.grasp_acquisition, env.hold_controller
        self.fractions = {}

    def desired(self, env, amount):
        desired = self.base.desired(env, amount)
        if self.settings.order == "thumb_first" and env.phase == "ACQUIRE":
            for side, indices in env.hand_act.items():
                fractions = closure_fractions(env.hands[side]["active"], env.data.time - env.phase_start, env.config.acquire_seconds)
                desired[indices] = finger_targets(fractions, self.base.closed,
                    env.data.qpos[env.act_qpos[indices]], self.base.kp[indices], self.base.preload)
                self.fractions[side] = fractions.tolist()
        return desired

    def state(self):
        return {"mode": "g6_grasp_acquisition", "settings": asdict(self.settings),
                "closure_fractions_at_last_acquire_update": self.fractions.copy()}

    def metadata(self, env):
        return {**controller_settings(env.pose, env.config),
                "schema": "g6_grasp_acquisition_v1", "mode": "g6_grasp_acquisition",
                "grasp_acquisition": asdict(self.settings),
                "thumb_duration_fraction": THUMB_DURATION_FRACTION,
                "finger_delay_fraction": FINGER_DELAY_FRACTION,
                "joint_mapping": {side: hand["active"] for side, hand in env.hands.items()},
                "reference": "native named finger joints; original bounded preload and servo limiter",
                "isolated_runner_support": False}


class StabilizedAcquisition(AcquisitionArmReference):
    def active(self, env):
        return env.phase == "HOLD"

    def metadata(self, env):
        return {**super().metadata(env), "grasp_acquisition": self.base.metadata(env),
                "activation": "HOLD entry after measured grasp acquisition"}

    def state(self):
        return {**super().state(), "grasp_acquisition": self.base.state()}


class GraspEnvironment(Environment):
    def __init__(self, *args, grasp_acquisition, **kwargs):
        self.grasp_acquisition = grasp_acquisition
        super().__init__(*args, **kwargs)

    def reset(self):
        row = super().reset()
        self.hold_controller = FingerAcquisition(self)
        if self.grasp_acquisition.arm_rate_rad_s:
            settings = ArmFeedback("retain", self.grasp_acquisition.arm_rate_rad_s)
            self.hold_controller = StabilizedAcquisition(self, settings)
        return row
