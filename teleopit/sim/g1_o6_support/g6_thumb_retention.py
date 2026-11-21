"""Two single-variable right-arm interventions for the recorded G6 thumb loss."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .g6_whole_body import damped_step
from .g6_wrist_grip import RECENTER_GAIN, WristEnvironment, WristGrip, WristReference

# Median unserved rotation over the last intact 12 s, in the baseline tread frame.
UNMET_DIRECTION = np.array([.6314176853214674, .5459109627335651, .5507203713597241])
PREORIENT_RAD = .010
PROXIMAL_RATE_RAD_S = .005
ENGAGE_RAD = .024


@dataclass(frozen=True)
class ThumbRetention(WristGrip):
    right_refinement: str = "preorient"

    def __post_init__(self):
        super().__post_init__()
        if self.wrist_mode != "right" or self.right_refinement not in ("preorient", "recenter"):
            raise ValueError("expected a single right-arm retention intervention")


def proximal_rate(jacobian, velocity, offset):
    """Give the wrist an inward velocity and let the proximal joints take up pad motion."""
    arm, wrist = jacobian[:, :4], jacobian[:, 4:]
    recenter = -RECENTER_GAIN * offset[4:]
    proximal = np.clip(damped_step(arm, velocity - wrist @ recenter), -PROXIMAL_RATE_RAD_S, PROXIMAL_RATE_RAD_S)
    remainder = velocity - arm @ proximal - wrist @ recenter
    return np.r_[proximal, recenter + damped_step(wrist, remainder)]


class RetentionReference(WristReference):
    def __init__(self, env, settings):
        super().__init__(env, settings)
        self.refinement = env.grasp_acquisition.right_refinement
        self.proximal_updates = 0

    def tracking_rate(self, side, jacobian, velocity):
        if self.refinement == "recenter" and self.angle[side] >= ENGAGE_RAD:
            self.proximal_updates += 1
            return proximal_rate(jacobian, velocity, self.offset[side])
        return super().tracking_rate(side, jacobian, velocity)

    def state(self):
        return {**super().state(), "right_proximal_updates": self.proximal_updates}

    def metadata(self, env):
        return {**super().metadata(env), "right_retention": {
            "intervention": self.refinement, "preorientation_rung_rad":
            (UNMET_DIRECTION * PREORIENT_RAD if self.refinement == "preorient" else np.zeros(3)).tolist(),
            "proximal_engage_rad": ENGAGE_RAD, "proximal_rate_rad_s": PROXIMAL_RATE_RAD_S}}


class RetentionEnvironment(WristEnvironment):
    stabilizer_type = RetentionReference

    def orient_right_target(self, rung, rotation):
        if self.grasp_acquisition.right_refinement == "preorient":
            frame = self.data.geom(f"left_ladder_rung_{rung:02d}").xmat.reshape(3, 3)
            return Rotation.from_rotvec(frame @ (PREORIENT_RAD * UNMET_DIRECTION)).as_matrix() @ rotation
        return rotation
