"""Hold an acquired closure and retreat the pelvis as the right knee approaches G6."""

from dataclasses import dataclass

import numpy as np

from .g6_ground_reaction import FINGER_RESERVE_N, GroundReaction, GroundReactionEnvironment, GroundReactionReference
from .g6_knee import KneeRung

CAPTURE_S = 5.
BUFFER_M = .035
MAX_RETREAT_M = .008


@dataclass(frozen=True)
class RetainedKnee(GroundReaction):
    governor: str = "individual"
    grip_retention: str = "settled"

    def __post_init__(self):
        super().__post_init__()
        if (self.governor, self.guard_N, self.grip_retention) != ("individual", 16., "settled"):
            raise ValueError("retention requires the settled grasp and foot-specific 16 N parent")


def retreat_m(gap):
    return float(np.clip(BUFFER_M - gap, 0., MAX_RETREAT_M)) if np.isfinite(gap) else MAX_RETREAT_M


def loaded_grasp(hands):
    return all(hands[side]["valid"] and hands[side]["minimum_finger_normal_N"] >= FINGER_RESERVE_N
               for side in ("left", "right"))


class SettledGrip:
    def __init__(self):
        self.ceiling, self.capture_s = None, None

    def command(self, elapsed, hands, inhibition, original, applied):
        if self.ceiling is None and elapsed >= CAPTURE_S - 1e-9 and not inhibition and loaded_grasp(hands):
            self.ceiling, self.capture_s = applied.copy(), elapsed
        return original if self.ceiling is None else np.minimum(original, self.ceiling)


class RetainedReference(GroundReactionReference):
    def __init__(self, env):
        super().__init__(env)
        self.knee, self.grip = KneeRung(env.model), SettledGrip()
        self.fingers = np.concatenate(list(env.hand_act.values()))
        self.gap, self.retreat, self.uncapped = None, 0., None

    def grasp_command(self, env, amount):
        command = super().grasp_command(env, amount)
        if env.phase == "HOLD":
            self.uncapped = command[self.fingers]
            elapsed = env.data.time - env.phase_start
            command[self.fingers] = self.grip.command(elapsed, self.parent.quality, self.parent.inhibition,
                                                       self.uncapped, env.data.ctrl[self.fingers])
        return command

    def wrench_targets(self, env, pelvis, translation, rotation, levers):
        self.gap = self.knee.distance(env.data)
        self.retreat = retreat_m(self.gap)
        target = translation.copy()
        target[0] -= self.retreat
        return super().wrench_targets(env, pelvis, target, rotation, levers)

    def state(self):
        return {**super().state(), "settled_grip": {"capture_s": self.grip.capture_s, "gap_m": self.gap,
                "retreat_m": self.retreat, "uncapped_targets_rad": None if self.uncapped is None else self.uncapped.tolist(),
                "ceiling_rad": None if self.grip.ceiling is None else self.grip.ceiling.tolist()}}

    def metadata(self, env):
        return {**super().metadata(env), "settled_grip": {"earliest_capture_s": CAPTURE_S,
                "knee_buffer_m": BUFFER_M, "max_pelvis_retreat_m": MAX_RETREAT_M,
                "minimum_finger_N_at_capture": FINGER_RESERVE_N,
                "finger_actuator_order": [env.model.actuator(i).name for i in self.fingers]}}


class RetainedEnvironment(GroundReactionEnvironment):
    reference_type = RetainedReference
