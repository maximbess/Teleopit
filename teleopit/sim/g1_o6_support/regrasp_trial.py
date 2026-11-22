"""Servo-driven, same-rung trajectories and native confirmation clocks."""

from collections import defaultdict

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.o6_grasp.controller import smooth

from .config import TransferCriteria
from .g6_grip_feedback import SIDES
from .guards import detached, measured_opening
from .pose_arm import correction_rate, rung_pose
from .regrasp_measure import STATES, NativeTrace

CRITERIA = TransferCriteria()
MOVE_S = 1.2
POSITION_TOL_M = 0.003
ANGLE_TOL_RAD = 0.035


class Confirmation:
    def __init__(self, seconds, dt):
        self.required, self.count = round(seconds / dt), 0

    def update(self, values):
        for present in values:
            self.count = self.count + 1 if present else 0
        return self.count >= self.required


class Unmet(Exception):
    """A measured stage outcome; keep the trajectory."""


class WristPath:
    def __init__(self, env, policy, side, offset_mm):
        self.env, self.side = env, side
        self.digit = policy.digits[side]
        self.origin, self.rotation = rung_pose(env, env.data, side)
        self.offset = np.array([0.0, 0.001 * offset_mm, 0.0])
        self.ids = policy.parent.indices[side]
        joints = env.model.actuator_trnid[self.ids, 0]
        self.dofs = env.model.jnt_dofadr[joints]
        self.bounds = env.model.jnt_range[joints]
        self.addr = env.act_qpos[self.ids]
        self.anchor = env.data.qpos[self.addr].copy()
        self.bounds = np.clip(
            self.bounds, self.anchor[:, None] - 0.65, self.anchor[:, None] + 0.65
        )
        self.kp = env.model.actuator_gainprm[self.ids, 0]
        self.contact_preload = (
            env.target[self.ids] - self.anchor - env.data.qfrc_bias[self.dofs] / self.kp
        )
        self.scratch = mujoco.MjData(env.model)
        self.goal = self.origin.copy()
        rung = env.data.geom(env.support_targets[side + "_hand"])
        self.world_home = env.data.site(env.hands[side]["mount"]).xpos.copy()
        self.outward = -rung.xmat.reshape(3, 3)[:, 0].copy()

    @property
    def preload(self):
        return self.digit.amount * self.contact_preload

    def target(self, distance, lateral=1.0):
        env, data = self.env, self.scratch
        self.goal = self.origin + lateral * self.offset - [distance, 0.0, 0.0]
        data.qpos[:] = env.data.qpos
        data.qpos[self.addr] = self.anchor
        # Warm start at the same reference posture to avoid redundant-arm IK jumps.
        for _ in range(10):
            mujoco.mj_kinematics(env.model, data)
            mujoco.mj_comPos(env.model, data)
            dq, error = correction_rate(
                env, data, self.side, (self.goal, self.rotation), self.dofs
            )
            if np.linalg.norm(error) < 0.00005:
                break
            scale = min(1.0, 0.12 / max(np.max(abs(dq)), 1e-12))
            data.qpos[self.addr] = np.clip(
                data.qpos[self.addr] + scale * dq, *self.bounds.T
            )
        gravity = env.data.qfrc_bias[self.dofs] / self.kp
        return data.qpos[self.addr] + gravity + self.preload

    def error(self):
        position, rotation = rung_pose(self.env, self.env.data, self.side)
        return (
            np.linalg.norm(position - self.goal),
            Rotation.from_matrix(self.rotation @ rotation.T).magnitude(),
        )

    def arrived(self):
        position, angle = self.error()
        return position <= POSITION_TOL_M and angle <= ANGLE_TOL_RAD


class Trial:
    def __init__(self, env, policy, side, mode, stance, offset_mm):
        self.env, self.policy = env, policy
        self.side, self.i, self.mode = side, SIDES.index(side), mode
        self.readings, self.ticks = NativeTrace(env, side), defaultdict(list)
        self.stance, self.offset_mm = stance, offset_mm
        self.events, self.windows, self.handoffs = {}, {}, []
        self.interruptions = {}
        self.path = self.error = None
        self.fixed_stance = None
        self.dt, self.tick_s = env.config.physics_dt, env.config.control_dt
        self.n = round(self.tick_s / self.dt)

    @property
    def time(self):
        return round(self.env.data.time, 9)

    def last(self, key):
        return np.asarray(self.readings.samples[key][-self.n :])

    def event(self, name):
        self.events[name] = self.time
        print(f"{self.mode} {self.side} {self.time:.3f} s  {name}", flush=True)

    def set_stance(self):
        p = self.policy
        if self.fixed_stance is not None:
            p.stance = self.fixed_stance
        elif self.path is not None and self.mode == "free":
            index = round(self.time / self.tick_s) + 1
            if index >= len(self.stance):
                raise Unmet("reference stance recording exhausted")
            p.stance = self.stance[index, p.indices]

    def actions(self, arm, request):
        arms, grips = [None] * 2, [None] * 2
        arms[self.i], grips[self.i] = arm, request
        return arms, grips

    def tick(self, phase, arm=None, request=None, *, gap=False):
        e, p = self.env, self.policy
        arms, grips = self.actions(arm, request)
        self.readings.phase, self.readings.check_gap = phase, gap
        self.set_stance()
        row = p.step(arms, grips, physics_observer=self.readings)
        self.record_command(row, phase, arm, grips)
        if e.phase == "ABORT":
            raise Unmet(", ".join(e.abort_reasons))
        return row

    def record_command(self, row, phase, arm, grips):
        e, obs = self.env, row["physical_hands"]
        sample = {
            "time": self.time,
            "phase": phase,
            "qpos": e.data.qpos.copy(),
            "qvel": e.data.qvel.copy(),
            "arm_request": np.full(7, np.nan) if arm is None else arm.copy(),
            "grip_request": np.array(grips, dtype=float),
            "desired": e.desired_target.copy(),
            "target": e.target.copy(),
            "ctrl": e.data.ctrl.copy(),
            "goal_rung": np.full(3, np.nan) if arm is None else self.path.goal.copy(),
            "position_rung": [obs[s]["position_rung_m"] for s in SIDES],
            "rotation_rung": [obs[s]["rotation_rung"] for s in SIDES],
            "twist_rung": [obs[s]["twist_rung_rad_s_m_s"] for s in SIDES],
            "wrist_wrench": [row["wrist_world_on_hand"][s] for s in SIDES],
        }
        for key, value in sample.items():
            self.ticks[key].append(value)

    def support_ready(self):
        return (
            (self.last("state") == STATES.index("grasp")).all(axis=1)
            & self.last("loaded").all(axis=1)
            & (self.last("applied") == 0)
            & (self.last("assist_gain") == 0)
            & (self.last("wrong") == 0)
            & (self.last("slip").max(axis=1) < self.env.config.max_slip_m_s)
        )

    def acquire(self, event):
        timer = Confirmation(2.0, self.dt)
        while self.time < 16.0:
            if (
                self.env.phase == "HOLD"
                and self.mode == "fixed"
                and self.fixed_stance is None
            ):
                self.fixed_stance = self.env.data.ctrl[self.policy.indices].copy()
            self.tick("setup")
            engaged = self.policy.parent.updates["right"] > 0
            if timer.update(self.support_ready() & engaged):
                self.event(event)
                return
        raise Unmet(
            "two continuous seconds of unassisted four-support hold with wrist feedback"
        )

    def begin_measurement(self):
        self.path = WristPath(self.env, self.policy, self.side, self.offset_mm)
        self.windows["baseline"] = [self.time - 2.0, self.time]

    def settle(self):
        self.acquire("settled")
        self.begin_measurement()

    def opened(self):
        return measured_opening(self.start_angles, self.last("digits")[:, self.i])

    def clear(self):
        return detached(self.opened(), self.last("external_count"), self.last("gap"))

    def open_retreat(self, cycle):
        prefix, start = str(cycle), self.time
        self.start_angles = self.env.data.qpos[
            self.env.act_qpos[self.env.hand_act[self.side]]
        ].copy()
        self.event("open_request" + prefix)
        while self.time - start < 3.0:
            self.tick(
                "opening" + prefix,
                self.path.target(0.0, float(cycle != 1)),
                -1.0,
                gap=True,
            )
            if self.opened()[-1]:
                break
        else:
            raise Unmet("measured opening within 0.1 rad")
        self.event("opened" + prefix)
        self.windows["opening" + prefix] = [start, self.time]
        start = self.time
        timer = Confirmation(CRITERIA.persistence_s, self.dt)
        while self.time - start < 4.0:
            f = smooth((self.time - start) / MOVE_S)
            self.tick(
                "retreat" + prefix,
                self.path.target(CRITERIA.retraction_m * f, f if cycle == 1 else 1.0),
                -1.0,
                gap=True,
            )
            displacement = (
                self.last("wrist")[:, self.i] - self.path.world_home
            ) @ self.path.outward
            outward = displacement >= CRITERIA.retraction_m - POSITION_TOL_M
            if timer.update(self.clear() & outward & (f == 1.0)):
                self.event("clear_verified" + prefix)
                self.windows["retreat" + prefix] = [start, self.time]
                return
        self.windows["retreat" + prefix + "_transient"] = [start, self.time]
        raise Unmet(
            f"retreat confirmation lasted {timer.count * self.dt:.3f} s; "
            f"last displacement {displacement[-1] * 1000:.1f} mm, gap {self.last('gap')[-1] * 1000:.1f} mm, "
            f"external contacts {self.last('external_count')[-1]}"
        )

    def dwell(self, stage, seconds, command, predicate, missing, *, gap=False):
        start = self.time
        for _ in range(round(seconds / self.tick_s)):
            self.tick(stage, *command(), gap=gap)
            lost = np.flatnonzero(~predicate())
            if len(lost) and stage not in self.interruptions:
                self.interruptions[stage] = {
                    "first_s": float(self.last("time")[lost[0]]),
                    "condition": missing,
                }
        name = stage + "_transient" if stage in self.interruptions else stage
        self.windows[name] = [start, self.time]

    def approach(self, prefix):
        start = self.time
        self.event("approach" + prefix)
        while self.time - start < 4.0:
            f = smooth((self.time - start) / MOVE_S)
            self.tick(
                "approach" + prefix,
                self.path.target(CRITERIA.retraction_m * (1.0 - f)),
                -1.0,
                gap=True,
            )
            approach_clear = (
                self.last("external_count")[-1] == 0 and self.last("gap")[-1] >= 0.0
            )
            if (
                f == 1.0
                and self.path.arrived()
                and self.opened()[-1]
                and approach_clear
            ):
                break
        else:
            pos, angle = self.path.error()
            raise Unmet(
                f"open approach missed pose or clearance: position error {1000 * pos:.1f} mm, "
                f"angle {angle:.3f} rad, external contacts {self.last('external_count')[-1]}"
            )

    def grasp(self, cycle):
        prefix = str(cycle)
        self.approach(prefix)
        self.event("close_request" + prefix)
        timer = Confirmation(CRITERIA.persistence_s, self.dt)
        start = self.time
        while self.time - start < 4.0:
            self.tick("closing" + prefix, self.path.target(0.0), 1.0)
            if timer.update(self.opposed()):
                break
        else:
            raise Unmet(
                "0.5 s continuously loaded thumb below and four fingers above rung 7"
            )
        self.event("grasp_verified" + prefix)
        previous = self.env.target.copy()
        self.dwell(
            "hold" + prefix,
            2.0,
            lambda: (None, None),
            self.opposed,
            "uncommanded loss of moving-hand opposition in 2 s hold",
        )
        first = self.ticks["ctrl"][-round(2 / self.tick_s)]
        ids = np.r_[self.path.ids, self.env.hand_act[self.side]]
        self.handoffs.append(
            {
                "time": self.events["grasp_verified" + prefix],
                "max_first_applied_change_rad": float(
                    max(abs(first[ids] - previous[ids]))
                ),
                "landmark": sorted(self.policy.parent.landmarks),
            }
        )
        if "hold" + prefix not in self.interruptions:
            self.event("recovered" + prefix)

    def opposed(self):
        return self.last("state")[:, self.i] == STATES.index("grasp")

    def release(self):
        for cycle in (1, 2):
            self.open_retreat(cycle)
            if cycle == 1:
                self.dwell(
                    "clear1",
                    2.0,
                    lambda: (self.path.target(CRITERIA.retraction_m), -1.0),
                    self.clear,
                    "lost certified clearance during 2 s window",
                    gap=True,
                )
            self.grasp(cycle)

    def control(self, matched):
        self.event("control_start")
        ends = [
            (start, matched["events"].get("opened" + str(i), matched["end_s"]), i)
            for i in (1, 2)
            if (start := matched["events"].get("open_request" + str(i))) is not None
        ]
        while self.time < matched["end_s"] - self.tick_s / 2:
            cycle = next(
                (i for start, end, i in ends if start <= self.time < end), None
            )
            arm = (
                self.path.target(0.0, float(cycle != 1)) if cycle is not None else None
            )
            self.tick("control", arm)
        self.windows.update(matched["windows"])

    def run(self, matched=None):
        try:
            self.settle()
            if matched is None:
                self.release()
            else:
                self.control(matched)
        except Unmet as reason:
            self.error = str(reason)
        terminal = self.error
        if self.interruptions:
            self.error = next(iter(self.interruptions.values()))["condition"]
        return {
            "events": self.events,
            "windows": self.windows,
            "end_s": self.time,
            "missing": self.error,
            "handoffs": self.handoffs,
            "interruptions": self.interruptions,
            "terminal_condition": terminal,
            "last_phase": self.readings.phase,
            "wrist_capture_hold_s": self.policy.parent.capture_s,
        }
