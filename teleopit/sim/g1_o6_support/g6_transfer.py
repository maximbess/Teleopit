"""Scratch-planned upward hand motion through bounded existing arm actuators."""

import numpy as np

from teleopit.sim.o6_grasp.controller import smooth

from .control import TransferController
from .g6_reach import adjacent_targets, plan_reach, trajectory_point
from .kinematics import Clearance, arm_target


class AdjacentController(TransferController):
    def __init__(self, env, original, protocol, target):
        super().__init__(env, original, protocol, target)
        self.destinations = adjacent_targets(env)
        self.plans = {}
        self.clearance = {
            side: Clearance(env, side, include_arm=True) for side in env.hands
        }

    def enter(self, env, row):
        changed = self.previous != self.protocol.index
        super().enter(env, row)
        if not changed or self.protocol.phase in ("COMPLETE", "ABORT"):
            return
        phase, side = self.protocol.phase, self.protocol.step.side
        if phase == "REACH":
            env.support_targets[side + "_hand"] = self.destinations[side]
            self.clearance[side].target = self.destinations[side]
            self.plans[side] = plan_reach(env, side, self.destinations[side])
            self.arm_previous = env.target.copy()
            if self.plans[side]["status"] != "FEASIBLE":
                self.failure = "adjacent_swept_reach_infeasible"
        if phase == "CLOSE":
            plan = self.plans[side]
            env.support_targets[side + "_hand"] = self.destinations[side]
            self.mounts[side] = (
                np.asarray(plan["waypoints_world_m"][-1]),
                np.asarray(plan["rotation_world"]),
            )

    def desired(self, env, amount):
        desired = super().desired(env, amount)
        if env.phase == "HOLD":
            for side in self.plans:
                if side != self.protocol.step.side and side in self.mounts:
                    self.move_arm(env, desired, side, "AT_RUNG")
        if env.phase == "HOLD" and self.protocol.phase == "REACH" and not self.failure:
            self.reach_command(env, desired)
        return desired

    def reach_command(self, env, desired):
        side = self.protocol.step.side
        plan = self.plans[side]
        progress = smooth(self.protocol.elapsed(env.data.time) / 15)
        goal = trajectory_point(np.asarray(plan["waypoints_world_m"]), progress)
        solution = arm_target(env, side, goal, np.asarray(plan["rotation_world"]))
        clearance = self.clearance[side].measure(solution["scratch"])
        self.motion = {
            "goal_world_m": goal.tolist(),
            "planned_clearance": clearance,
            "position_error_m": solution["position_error_m"],
            "angle_error_rad": solution["angle_error_rad"],
        }
        if (
            solution["position_error_m"] > 0.001
            or solution["angle_error_rad"] > 0.03
            or clearance["clearance_m"] < 0
        ):
            self.failure = "adjacent_command_infeasible_or_colliding"
            return
        ids = [
            int(np.flatnonzero(env.model.actuator_trnid[:, 0] == j)[0])
            for j in solution["joints"]
        ]
        limit = 0.05 * env.config.control_dt
        desired[ids] = self.arm_previous[ids] + np.clip(
            solution["qpos"] - self.arm_previous[ids], -limit, limit
        )
        self.arm_previous[ids] = desired[ids]
