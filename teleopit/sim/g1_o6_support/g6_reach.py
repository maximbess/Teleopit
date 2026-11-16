"""Adjacent-tread targets and scratch-data swept arm plans; never teleport."""

import numpy as np

from teleopit.sim.g1_o6.grasp_pose import mount_target

from .kinematics import Clearance, arm_target


def adjacent_targets(env):
    if env.ladder is None or env.ladder.identity != "G6":
        raise ValueError("adjacent transfer requires resolved G6 geometry")
    result = {}
    for side in env.hands:
        rung = int(env.support_targets[side + "_hand"].rsplit("_", 1)[1]) + 1
        if rung > env.ladder.rung_count:
            raise ValueError("no adjacent higher tread for " + side)
        result[side] = f"left_ladder_rung_{rung:02d}"
    return result


def target_mount(env, side, destination):
    rung = int(destination.rsplit("_", 1)[1])
    return mount_target(env, env.pose, side, rung)


def next_tread_reach(env):
    try:
        targets = adjacent_targets(env)
    except ValueError as error:
        return {"status": "UNAVAILABLE", "reason": str(error)}
    result = {}
    for side, target in targets.items():
        pose = arm_target(env, side, *target_mount(env, side, target))
        result[side] = {
            "position_error_m": pose["position_error_m"],
            "angle_error_rad": pose["angle_error_rad"],
        }
    feasible = all(
        r["position_error_m"] < 0.001 and r["angle_error_rad"] < 0.03
        for r in result.values()
    )
    return {
        "status": "REACHABLE" if feasible else "UNREACHABLE",
        "targets": targets,
        "endpoint_ik": result,
        "scope": "reset endpoint IK only; swept reach must be checked after actual unloading",
    }


def reach_waypoints(start, goal):
    outside = min(start[0], goal[0]) - 0.08
    return np.array(
        [start, [outside, start[1], start[2]], [outside, goal[1], goal[2]], goal]
    )


def trajectory_point(points, progress):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(distances)]
    return np.array(
        [
            np.interp(progress * cumulative[-1], cumulative, points[:, axis])
            for axis in range(3)
        ]
    )


def plan_reach(env, side, destination, *, samples=None):
    position, rotation = target_mount(env, side, destination)
    points = reach_waypoints(
        env.data.site(env.hands[side]["mount"]).xpos.copy(), position
    )
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    count = samples or max(3, int(np.ceil(length / 0.005)) + 1)
    clearance = Clearance(env, side, include_arm=True)
    rows = [
        plan_sample(env, side, trajectory_point(points, t), rotation, clearance)
        for t in np.linspace(0, 1, count)
    ]
    return {
        "status": "FEASIBLE" if all(r["valid"] for r in rows) else "INFEASIBLE",
        "destination": destination,
        "waypoints_world_m": points.tolist(),
        "rotation_world": rotation.tolist(),
        "samples": rows,
        "maximum_cartesian_sample_interval_m": float(length / (count - 1)),
        "execution": "scratch IK and collision queries; bounded live actuator commands with physics-step contact guards",
    }


def plan_sample(env, side, position, rotation, clearance):
    solution = arm_target(env, side, position, rotation)
    distance = clearance.measure(solution["scratch"])
    return {
        "position_world_m": position.tolist(),
        "position_error_m": solution["position_error_m"],
        "angle_error_rad": solution["angle_error_rad"],
        "clearance": distance,
        "valid": bool(
            solution["position_error_m"] < 0.001
            and solution["angle_error_rad"] < 0.03
            and distance["clearance_m"] >= 0
        ),
    }
