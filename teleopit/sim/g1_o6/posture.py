"""Physical arm alignment and clearance, independent of Euler joint conventions."""

from itertools import product

import mujoco
import numpy as np


def direction_angle(first, second):
    cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def posture_metrics(env):
    m, d = env.model, env.data
    bends = {}
    elbows = {}
    margins = {}
    for side, hand in env.hands.items():
        ids = [
            m.joint(side + "_" + name + "_joint").id
            for name in ("shoulder_roll", "elbow", "wrist_roll")
        ]
        shoulder, elbow, wrist = d.xanchor[ids]
        bends[side] = direction_angle(
            wrist - elbow, d.site(hand["mount"]).xmat.reshape(3, 3)[:, 2]
        )
        elbows[side] = direction_angle(elbow - shoulder, wrist - elbow)
        for name in ("wrist_roll", "wrist_pitch", "wrist_yaw"):
            joint = m.joint(side + "_" + name + "_joint").id
            q = d.qpos[m.jnt_qposadr[joint]]
            low, high = m.jnt_range[joint]
            margins[m.joint(joint).name] = float(min(q - low, high - q))
    return {
        "wrist_bend_deg": bends,
        "elbow_bend_deg": elbows,
        "wrist_joint_margin_rad": margins,
        "housing_clearance": housing_clearance(m, d),
    }


def housing_clearance(model, data):
    obstacles = [
        g for g in range(model.ngeom) if "_ladder_" in model.geom(g).name
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    result = {}
    for side in ("left", "right"):
        distance, method, geom, obstacle = min(
            (*geom_clearance(model, data, g, o), g, o)
            for g in housing_geoms(model, side)
            for o in obstacles
        )
        result[side] = {
            "distance_m": distance,
            "method": method,
            "housing": model.geom(geom).name,
            "obstacle": model.geom(obstacle).name,
        }
    return result


def housing_geoms(model, side):
    return [
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name.startswith(side + "_wrist_")
        and model.geom_contype[g]
    ]


def geom_clearance(model, data, first, second):
    distance = float(mujoco.mj_geomDistance(model, data, first, second, 0.1, None))
    if distance != 0:
        return distance, "native"
    # MuJoCo can return exact zero for separated convex meshes (upstream #3383).
    # A projection gap is an independent conservative certificate, never a
    # replacement for a negative distance or the simulator's actual contacts.
    vertices = [world_vertices(model, data, g) for g in (first, second)]
    directions = np.vstack([data.geom_xmat[g].reshape(3, 3).T for g in (first, second)])
    return min(
        0.1, separating_gap(*vertices, directions)
    ), "separating_plane_lower_bound"


def world_vertices(model, data, geom):
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[start : start + count]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array(list(product((-1, 1), repeat=3))) * model.geom_size[geom]
    else:
        raise ValueError("housing certificate requires mesh or box geometry")
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def separating_gap(first, second, directions):
    index = np.arange(64) + 0.5
    z = 1 - 2 * index / 64
    angle = np.pi * (1 + np.sqrt(5)) * index
    radial = np.sqrt(1 - z * z)
    axes = np.vstack(
        (directions, np.c_[radial * np.cos(angle), radial * np.sin(angle), z])
    )
    a, b = first @ axes.T, second @ axes.T
    gaps = np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0))
    return float(max(0.0, gaps.max()))
