"""Sagittal foot and knee feedback on the acquired G6 pose."""

from dataclasses import dataclass

import numpy as np

from teleopit.sim.g1_o6.env import bounded
from teleopit.sim.g1_o6.posture import geom_clearance

from .g6_ground_reaction import GroundReaction, GroundReactionEnvironment, GroundReactionReference
from .g6_observer import body_groups
from .g6_whole_body import attitude_error, foot_wrenches

RUNG = "left_ladder_rung_03"
GAP_TARGET_M = .040
CONFIGURATIONS = {"stance_soft": ("stance", .75), "stance_firm": ("stance", 1.5),
                  "palm_coordinated": ("palms", 1.5)}
SETTLE_S = 2.
PALM_SHIFT_M = .012
PALM_TILT_RAD = .020


@dataclass(frozen=True)
class KneeClearance(GroundReaction):
    governor: str = "individual"
    coordination: str = "stance"
    stance_gain: float = .75

    def __post_init__(self):
        super().__post_init__()
        if (self.governor, self.guard_N) != ("individual", 16.):
            raise ValueError("knee refinement requires the foot-specific 16 N parent")
        if (self.coordination, self.stance_gain) not in CONFIGURATIONS.values():
            raise ValueError("unknown bounded G6 knee configuration")


class KneeRung:
    def __init__(self, model):
        self.model = model
        self.geoms = body_groups(model)["right_knee"]
        self.names = {model.geom(g).name for g in self.geoms}
        self.rung = model.geom(RUNG).id

    def distance(self, data):
        return min(geom_clearance(self.model, data, g, self.rung)[0] for g in self.geoms)

    def contacts(self, rows):
        return [c for c in rows if RUNG in c["geoms"] and self.names.intersection(c["geoms"])]


def foot_trim(advance_m, pitch_rad, gap_shortfall_m, gain):
    # A positive desired ground reaction commands an opposing foot servo torque.
    force = np.clip(gain * 800. * (advance_m + gap_shortfall_m), -8., 8.)
    moment = np.clip(gain * 24. * pitch_rad, -2., 2.)
    return np.array([force, 0., 0., 0., moment, 0.])


def distribute_trim(levers, trims):
    force = sum(v[:3] for v in trims.values())
    moment = sum(np.cross(levers[s], v[:3]) + v[3:] for s, v in trims.items())
    common = foot_wrenches(force, moment, levers["left"], levers["right"])
    return {s: v - common[s] for s, v in trims.items()}, force, moment


def world_pose(frame):
    return frame.xpos.copy(), frame.xmat.reshape(3, 3).copy()


class KneeReference(GroundReactionReference):
    def __init__(self, env):
        super().__init__(env)
        self.knee = KneeRung(env.model)
        self.feet, self.palms = {}, {}
        self.internal = {s: np.zeros(6) for s in env.hands}
        self.palm_shift, self.palm_tilt = np.zeros(3), np.zeros(3)
        self.gap, self.trims = None, {}

    def latch_supports(self, env):
        for side, hand in env.hands.items():
            self.feet[side] = world_pose(env.data.body(self.chains[side].body))
            self.palms[side] = world_pose(env.data.site(hand["palm"]))

    def stance_trims(self, env):
        self.gap = self.knee.distance(env.data)
        trims = {}
        for side, (position, rotation) in self.feet.items():
            foot = env.data.body(self.chains[side].body)
            pitch = attitude_error(foot.xmat.reshape(3, 3), rotation)[1]
            shortfall = max(0., GAP_TARGET_M - self.gap) if side == "right" else 0.
            trims[side] = foot_trim(foot.xpos[0] - position[0], pitch, shortfall, self.profile.stance_gain)
        return trims

    def follow_palms(self, env):
        positions, rotations = [], []
        for side, (position, rotation) in self.palms.items():
            palm = env.data.site(env.hands[side]["palm"])
            positions.append(position - palm.xpos)
            rotations.append(attitude_error(rotation, palm.xmat))
        dt = env.config.control_dt
        self.palm_shift = bounded(self.palm_shift + .4 * dt * np.mean(positions, axis=0), PALM_SHIFT_M)
        self.palm_tilt = bounded(self.palm_tilt + .4 * dt * np.mean(rotations, axis=0), PALM_TILT_RAD)

    def wrench_targets(self, env, pelvis, translation, rotation, levers):
        if not self.feet:
            self.latch_supports(env)
        if env.data.time - self.latch_time < SETTLE_S:
            return super().wrench_targets(env, pelvis, translation, rotation, levers)
        if self.profile.coordination == "palms":
            self.follow_palms(env)
        self.trims = self.stance_trims(env)
        self.internal, extra_force, extra_moment = distribute_trim(levers, self.trims)
        force, moment = self.pelvis_wrench(env, pelvis, translation + self.palm_shift, rotation + self.palm_tilt)
        return self.leg_targets(env, levers, bounded(force + extra_force, 30.), bounded(moment + extra_moment, 5.))

    def reaction_targets(self, env, reactions):
        return super().reaction_targets(env, {s: v + self.internal[s] for s, v in reactions.items()})

    def state(self):
        return {**super().state(), "knee_stance": {"right_knee_gap_m": self.gap,
                "foot_trim_wrenches": {s: v.tolist() for s, v in self.trims.items()},
                "palm_shift_m": self.palm_shift.tolist(), "palm_tilt_rad": self.palm_tilt.tolist()}}

    def metadata(self, env):
        return {**super().metadata(env), "knee_stance": {"rung": RUNG, "knee_geoms": sorted(self.knee.names),
                "target_clearance_m": GAP_TARGET_M, "settle_s": SETTLE_S, "foot_trim_limits": [8., 2.],
                "palm_shift_bound_m": PALM_SHIFT_M, "palm_tilt_bound_rad": PALM_TILT_RAD}}


class KneeEnvironment(GroundReactionEnvironment):
    reference_type = KneeReference
