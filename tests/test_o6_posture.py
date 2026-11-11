import mujoco
import numpy as np

from teleopit.sim.g1_o6.env import Environment


def test_current_grasp_has_bounded_elbows_and_opposed_finger_geometry():
    env = Environment()
    assert not env.initial_errors
    for side in env.hands:
        ids = [
            env.model.joint(side + "_" + n + "_joint").id
            for n in ("shoulder_roll", "elbow", "wrist_roll")
        ]
        a, b, c = env.data.xanchor[ids]
        bend = np.arccos(
            np.clip(
                np.dot(b - a, c - b) / np.linalg.norm(b - a) / np.linalg.norm(c - b),
                -1,
                1,
            )
        )
        assert bend < np.deg2rad(60)
    # Kinematic endpoint probe only: the dynamic trial closes using actuators.
    for hand in env.hands.values():
        for j, v in zip(hand["active"], env.pose.closed_pose):
            env.data.joint(j).qpos[0] = v
        for p, a, ratio in hand["passive"]:
            env.data.joint(p).qpos[0] = ratio * env.data.joint(a).qpos[0]
    mujoco.mj_forward(env.model, env.data)
    height = env.data.geom(env.support_targets["left_hand"]).xpos[2]
    for side, code in [("left", "lh"), ("right", "rh")]:
        assert (
            env.data.geom(f"{side}_o6_{code}_thumb_distal_collision").xpos[2] < height
        )
        assert all(
            env.data.geom(f"{side}_o6_{code}_{finger}_distal_collision").xpos[2]
            > height
            for finger in ("index", "middle", "ring", "pinky")
        )
