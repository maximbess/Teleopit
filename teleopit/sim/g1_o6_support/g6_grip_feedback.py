"""Regulate a settled G6 grasp through the named O6 servos."""

from dataclasses import dataclass

import numpy as np

from .g6_knee_retention import RetainedEnvironment, RetainedKnee, RetainedReference
from .pose_contacts import FINGERS

SIDES = ("left", "right")
FLEXORS = ("thumb_cmc_pitch", "thumb_cmc_yaw", *(name + "_mcp_pitch" for name in FINGERS[1:]))
RATES = (.04, .12)
MAX_RATE = .04
MAX_TRAVEL = .12
DEADBAND_N = .08
THUMB_N, FINGER_N, RESERVE_N = .70, 1.40, 1.
HEIGHT_GAIN = 6.


@dataclass(frozen=True)
class GripFeedback(RetainedKnee):
    grip_law: str = "pressure"
    pressure_gain: float = RATES[0]

    def __post_init__(self):
        super().__post_init__()
        if (self.grip_law, self.pressure_gain) not in (("pressure", RATES[0]), ("pressure", RATES[1]), ("palm", RATES[1])):
            raise ValueError("unknown bounded grasp feedback profile")


def pressure_rates(profile, loads, opposed, height_error):
    """Two hands by six native servo targets, in rad/s. Thumb yaw keeps its captured limit."""
    thumb = THUMB_N - np.where(opposed, loads[:, 0], 0.)
    fingers = FINGER_N - loads[:, 1:]
    error = np.column_stack((thumb, np.zeros(2), fingers))
    rates = profile.pressure_gain * np.sign(error) * np.maximum(abs(error) - DEADBAND_N, 0.)
    if profile.grip_law == "palm":
        rates[:, 2:] = (HEIGHT_GAIN * height_error[:, None]
                        + profile.pressure_gain * np.maximum(RESERVE_N - loads[:, 1:], 0.))
    return np.clip(rates, -MAX_RATE, MAX_RATE)


def advance(cap, anchor, rates, native, dt):
    # Avoid integrating closure when the native finite-preload command already binds.
    available = np.where((rates > 0.) & (cap > native), 0., rates)
    return np.clip(cap + dt * available, anchor - MAX_TRAVEL, anchor + MAX_TRAVEL)


class GripReference(RetainedReference):
    def __init__(self, env):
        super().__init__(env)
        # The force vector and the two thumb axes must use the applied actuator order.
        suffixes = [part + "_servo" for _ in SIDES for part in FLEXORS]
        if not all(env.model.actuator(i).name.endswith(s) for i, s in zip(self.fingers, suffixes, strict=True)):
            raise ValueError("unrecognized O6 feedback actuator order")
        self.cap = self.height_at_capture = None

    def grip_targets(self, env, elapsed):
        hands = self.parent.quality
        self.grip.capture(elapsed, hands, self.parent.inhibition, env.data.ctrl[self.fingers])
        if self.grip.ceiling is None:
            return self.uncapped
        anchor = self.grip.ceiling.reshape(2, 6)
        heights = np.array([env.data.site(env.hands[s]["palm"]).xpos[2] -
                            env.data.geom(env.support_targets[s + "_hand"]).xpos[2] for s in SIDES])
        if self.cap is None:
            self.cap, self.height_at_capture = anchor.copy(), heights.copy()
        if not self.parent.inhibition:
            loads = np.array([[hands[s]["normal_by_group_N"][digit] for digit in FINGERS] for s in SIDES])
            opposed = [hands[s]["thumb_opposition"] for s in SIDES]
            rates = pressure_rates(self.profile, loads, opposed, self.height_at_capture - heights)
            self.cap = advance(self.cap, anchor, rates, self.uncapped.reshape(2, 6), env.config.control_dt)
        return np.minimum(self.uncapped, self.cap.ravel())

    def state(self):
        return {**super().state(), "grip_feedback_cap_rad": None if self.cap is None else self.cap.ravel().tolist()}

    def metadata(self, env):
        return {**super().metadata(env), "grip_feedback": {
            "law": self.profile.grip_law, "pressure_gain_rad_per_Ns": self.profile.pressure_gain,
            "max_rate_rad_s": MAX_RATE, "max_travel_rad": MAX_TRAVEL, "deadband_N": DEADBAND_N,
            "thumb_target_N": THUMB_N, "finger_target_N": FINGER_N, "finger_reserve_N": RESERVE_N,
            "height_gain_rad_per_m_per_s": HEIGHT_GAIN}}


class GripEnvironment(RetainedEnvironment):
    reference_type = GripReference
