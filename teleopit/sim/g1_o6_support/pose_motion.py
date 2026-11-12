"""Named joint travel/work and ladder clearances, without simulation writes."""

import numpy as np

from teleopit.sim.g1_o6.posture import geom_clearance

from .pose_contacts import hand_body


class HandMotion:
    def __init__(self, env, side):
        hand, model = env.hands[side], env.model
        self.names = [*hand["active"], *(p for p, _, _ in hand["passive"])]
        joints = [model.joint(name).id for name in self.names]
        self.q = model.jnt_qposadr[joints]
        self.v = model.jnt_dofadr[joints]
        self.limits = model.jnt_range[joints]
        self.act = env.hand_act[side]
        self.previous = env.data.qpos[self.q].copy()
        self.previous_target = env.data.ctrl[self.act].copy()
        self.travel = np.zeros(len(joints))
        self.work = np.zeros(len(joints))
        self.hold_reference = None
        self.mimics = hand["passive"]

    def start_hold(self):
        if self.hold_reference is None:
            self.hold_reference = self.travel.copy(), self.work.copy()

    def observe(self, env):
        q = env.data.qpos[self.q]
        self.travel += np.abs(q - self.previous)
        self.work += self.power(env) * env.config.physics_dt
        self.previous = q.copy()

    def power(self, env):
        return env.data.qfrc_actuator[self.v] * env.data.qvel[self.v]

    def sample(self, env):
        data = env.data
        q = data.qpos[self.q]
        margin = np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q)
        base = self.hold_reference or (self.travel, self.work)
        return {
            "joint_names": self.names,
            "position_rad": q.tolist(),
            "velocity_rad_s": data.qvel[self.v].tolist(),
            "joint_stop_margin_rad": margin.tolist(),
            "generalized_actuator_torque_Nm": data.qfrc_actuator[self.v].tolist(),
            "actuator_power_W": self.power(env).tolist(),
            "accumulated_work_J": self.work.tolist(),
            "cumulative_travel_rad": self.travel.tolist(),
            "hold_travel_rad": (self.travel - base[0]).tolist(),
            "hold_work_J": (self.work - base[1]).tolist(),
            "mimic_error_rad": {
                p: float(data.joint(p).qpos[0] - ratio * data.joint(a).qpos[0])
                for p, a, ratio in self.mimics
            },
            **command_sample(env, self.act),
        }

    def control_sample(self, env):
        result = self.sample(env)
        target = env.data.ctrl[self.act]
        result["applied_target_increment_rad"] = (
            target - self.previous_target
        ).tolist()
        self.previous_target = target.copy()
        return result


def command_sample(env, indices):
    data, model = env.data, env.model
    raw = getattr(
        env.hold_controller, "raw", getattr(env, "desired_target", env.target)
    )
    clipped = getattr(env, "desired_target", env.target)
    limits = np.max(np.abs(model.actuator_forcerange[indices]), axis=1)
    return {
        "command_joint_names": [
            model.joint(model.actuator_trnid[i, 0]).name for i in indices
        ],
        "raw_target_rad": raw[indices].tolist(),
        "clipped_target_rad": clipped[indices].tolist(),
        "rate_limited_target_rad": env.target[indices].tolist(),
        "applied_target_rad": data.ctrl[indices].tolist(),
        "target_error_rad": (
            data.ctrl[indices] - data.qpos[env.act_qpos[indices]]
        ).tolist(),
        "actual_effort_Nm": data.actuator_force[indices].tolist(),
        "effort_margin_Nm": (limits - np.abs(data.actuator_force[indices])).tolist(),
    }


class LadderClearance:
    def __init__(self, env):
        m = env.model
        self.obstacles = [
            g for g in range(m.ngeom) if m.geom(g).name.startswith("left_ladder_")
        ]
        self.hands = {
            s: [
                g
                for g in range(m.ngeom)
                if m.geom_contype[g] and hand_body(m.body(m.geom_bodyid[g]).name, s)
            ]
            for s in env.hands
        }

    def sample(self, env, side):
        return {
            env.model.geom(o).name: min(
                capped_clearance(env, g, o) for g in self.hands[side]
            )
            for o in self.obstacles
        }


def capped_clearance(env, first, second):
    m, d = env.model, env.data
    bound = (
        np.linalg.norm(d.geom_xpos[first] - d.geom_xpos[second])
        - m.geom_rbound[first]
        - m.geom_rbound[second]
    )
    if bound >= 0.1:
        return 0.1
    return geom_clearance(m, d, first, second)[0]
