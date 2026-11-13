"""Predeclared geometric reset candidates; selection never sees HOLD results."""

from dataclasses import asdict, replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from teleopit.sim.g1_o6.grasp_pose import GraspPose
from teleopit.sim.ladder_config import geometry_matrix


def reset_candidates(ladder):
    """Rank at most twelve same-rung hand/foot pairs by reference displacement.

    Score (metres): Euclidean hand-foot displacement error plus 0.1 times
    foot-height error. G0–G3 preserve 11/4 as the controlled comparison;
    G6 starts at the predeclared 8/3 pair. Ties use foot then hand index.
    """
    reference = geometry_matrix()["G0"].centers()
    displacement = reference[10] - reference[3]
    centers = ladder.centers()
    candidates = []
    for foot in range(1, ladder.rung_count):
        for hand in range(foot + 1, ladder.rung_count + 1):
            error = np.linalg.norm(centers[hand - 1] - centers[foot - 1] - displacement)
            height_error = abs(centers[foot - 1, 2] - reference[3, 2])
            candidates.append(
                {
                    "hand": hand,
                    "foot": foot,
                    "score_m": float(error + 0.1 * height_error),
                }
            )
    candidates.sort(key=lambda c: (c["score_m"], c["foot"], c["hand"]))
    preferred = {
        "G0": (11, 4),
        "G1": (11, 4),
        "G2": (11, 4),
        "G3": (11, 4),
        "G6": (8, 3),
    }.get(ladder.identity)
    if preferred:
        candidates.sort(key=lambda c: (c["hand"], c["foot"]) != preferred)
    return [
        dict(
            order=i,
            **c,
            hand_spacing_m=min(0.18, ladder.length_m / 2 - 0.06),
            mapping="front_top_face_clearances_v1",
        )
        for i, c in enumerate(candidates[:12])
    ]


def candidate_pose(candidate):
    return replace(
        GraspPose(),
        hand_rungs=(candidate["hand"],) * 2,
        foot_rungs=(candidate["foot"],) * 2,
        hand_spacing_m=candidate["hand_spacing_m"],
    )


def inspect_reset(env):
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.posture_report import annotate

    from .pose_trial import reset_report

    row = annotate(env, snapshot(env))
    report = reset_report(env)
    reasons = list(env.initial_errors)
    if report["minimum_joint_margin_rad"] < -1e-9:
        reasons.append("initial_joint_range")
    if any(
        r["distance_m"] < -1e-6 for r in row["posture"]["housing_clearance"].values()
    ):
        reasons.append("initial_housing_penetration")
    return {
        "reasons": sorted(set(reasons)),
        "feasible": not reasons,
        "report": report,
        "root_pose": env.data.qpos[:7].tolist(),
        "qpos": env.data.qpos.tolist(),
        "penetration_m": row["penetration_m"],
        "posture": row["posture"],
        "support_positions_world_m": {
            k: env.data.geom(v).xpos.tolist() for k, v in env.support_targets.items()
        },
        "foot_positions_world_m": {
            side: env.data.body(side + "_ankle_roll_link").xpos.tolist()
            for side in env.hands
        },
    }


def select_reset(ladder):
    from teleopit.sim.g1_o6.env import Environment

    shortlist = reset_candidates(ladder)
    inspections = []
    exact = exact_pose_inspection(ladder)
    for candidate in shortlist:
        pose = candidate_pose(candidate)
        env = Environment(ladder=ladder, pose=pose)
        inspection = dict(candidate=candidate, **inspect_reset(env))
        inspections.append(inspection)
        if inspection["feasible"]:
            return {
                "status": "FEASIBLE",
                "rule": "first feasible in frozen geometric ordering",
                "shortlist": shortlist,
                "exact_reference": exact,
                "inspections": inspections,
                "pose": asdict(pose),
                "selected": candidate,
            }
    return {
        "status": "INFEASIBLE",
        "rule": "first feasible in frozen geometric ordering",
        "shortlist": shortlist,
        "exact_reference": exact,
        "inspections": inspections,
        "pose": None,
        "selected": None,
    }


def exact_pose_inspection(ladder, reference=None):
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6.grasp_pose import mount_target

    if ladder.rung_count < 11:
        return {"status": "NOT RUN", "reason": "reference support indices absent"}
    reference = reference or Environment()
    env = Environment(ladder=ladder, adapt_reset=False)
    env.data.qpos[:] = reference.data.qpos
    mujoco.mj_forward(env.model, env.data)
    for side, rung in zip(env.hands, env.pose.hand_rungs):
        target, rotation = mount_target(env, env.pose, side, rung)
        mount = env.data.site(env.hands[side]["mount"])
        env.ik[side] = {
            "position_error_m": float(np.linalg.norm(mount.xpos - target)),
            "angle_error_rad": float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        rotation @ mount.xmat.reshape(3, 3).T
                    ).as_rotvec()
                )
            ),
        }
    env.initial_errors = []
    if snapshot(env)["penetration_m"] > 1e-6:
        env.initial_errors.append("initial_penetration")
    if any(v["position_error_m"] > 0.005 for v in env.ik.values()):
        env.initial_errors.append("exact_pose_misses_support_target")
    return dict(inspect_reset(env), mapping="exact_reference_qpos_no_ik_adaptation")


def exact_pose_inspections():
    from teleopit.sim.g1_o6.env import Environment

    reference = Environment()
    return {
        identity: exact_pose_inspection(ladder, reference)
        for identity, ladder in geometry_matrix().items()
    }
