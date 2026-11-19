"""Reconstruct the captured thumb point, bounded wrist requests and executed arm commands."""

import argparse
from collections import deque
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.g1_o6_g6_arm_report import arm_mapping, command_pose
from scripts.verify_g1_o6_g6_grip_feedback import grasp_sample
from scripts.verify_g1_o6_g6_grip_preview import verify_run as verify_grip
from scripts.verify_g1_o6_g6_whole_body import read, verify
from teleopit.sim.g1_o6_support.pose_contacts import contact_group


def check(value, expected, label, tolerance=1e-8):
    if not np.allclose(value, expected, rtol=0., atol=tolerance):
        raise ValueError("incorrect wrist " + label)


def contact_centroid(contacts, side, target):
    points, weights = [], []
    for c in contacts:
        if target in c["geoms"] and side+"_hand" in c["regions"] and c["normal_force"] >= .05:
            i = c["regions"].index(side+"_hand")
            if contact_group(c["bodies"][i]) == "thumb":
                points.append(c["position"])
                weights.append(c["normal_force"])
    if not points:
        raise ValueError("missing loaded thumb at wrist capture")
    return np.average(points, axis=0, weights=weights)


def check_request(saved, jacobian, error, previous, load, opposed, mode):
    tangent = .8 * error[:2]
    velocity = np.r_[tangent / max(1., np.linalg.norm(tangent)/.002),
                     np.clip(.004*(.65 - (load if opposed else 0.)), -.002, .002)]
    rates = -.4 * previous
    active = slice(None) if mode == "distributed" else slice(-3,None)
    j = jacobian[:,active]
    matrix = np.vstack((j, .01*np.eye(j.shape[1])))
    rhs = np.r_[velocity - jacobian @ rates, np.zeros(j.shape[1])]
    rates[active] += np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    for name, expected in (("error_rung_m",error), ("velocity_rung_m_s",velocity), ("joint_rate_rad_s",rates)):
        check(saved[name], expected, name)
    return rates


def joint_step(previous, rate, baseline, bounds, effort, dt):
    delta = np.clip(rate, -.02, .02) * dt
    limits = bounds - (baseline + previous)[:,None]
    delta = np.clip(delta, np.minimum(limits[:,0],0.), np.maximum(limits[:,1],0.))
    delta[effort >= .98] = 0.
    return np.clip(previous + delta, -.03, .03)


def project(start, destination, feasible):
    if feasible(destination):
        return destination
    low, high = 0., 1.
    for _ in range(16):
        fraction = (low + high) / 2
        if feasible(start + fraction*(destination-start)):
            low = fraction
        else:
            high = fraction
    return start + low*(destination-start)


class WristReplay:
    def __init__(self, model, meta, baseline):
        self.model, self.meta, self.base = model, meta, np.asarray(baseline)
        self.mode = meta["controller"]["wrist_grip"]["mode"]
        self.sides = {"right":("right",), "both":("left","right"), "distributed":("left","right")}[self.mode]
        self.ids = arm_mapping(model, meta)
        self.landmarks, self.updates = {}, dict.fromkeys(("left","right"),0)
        self.data, self.scratch = mujoco.MjData(model), mujoco.MjData(model)
        self.dt = meta["config"]["control_dt"]
        self.max_tangent_error = self.max_travel = 0.
        self.delay = deque([self.base.copy()] * meta["config"]["delay_steps"])

    def applied(self, row, state):
        proposed = self.base.copy()
        for side, ids in self.ids.items():
            proposed[ids] += state["arm_correction_rad"][side]
        self.delay.append(proposed)
        expected = self.delay.popleft()
        ids = np.concatenate(list(self.ids.values()))
        check(np.asarray(row["applied_position_targets_rad"])[ids], expected[ids], "applied arm target")

    def landmark(self, side, raw, saved):
        data = self.data
        body = data.body(f"{side}_o6_{side[0]}h_thumb_distal")
        rung = data.geom(self.meta["support_targets"][side+"_hand"])
        rotation = rung.xmat.reshape(3,3).T
        if side not in self.landmarks:
            world = contact_centroid(raw["contacts"], side, rung.name)
            self.landmarks[side] = (body.xmat.reshape(3,3).T @ (world-body.xpos), rotation @ (world-rung.xpos))
        local, target = self.landmarks[side]
        check(saved["local_m"], local, "captured material point")
        check(saved["target_rung_m"], target, "captured target")
        point = body.xpos + body.xmat.reshape(3,3) @ local
        j = np.zeros((3,self.model.nv))
        mujoco.mj_jac(self.model, data, j, None, point, body.id)
        dofs = self.model.jnt_dofadr[self.model.actuator_trnid[self.ids[side],0]]
        return target - rotation @ (point-rung.xpos), rotation @ j[:,dofs]

    def projected(self, side, old, rate, previous):
        model, ids = self.model, self.ids[side]
        command = self.base.copy()
        mount = self.meta["model"]["hands"][side]["mount"]
        self.scratch.qpos[:] = self.data.qpos
        origin, orientation = command_pose(model, self.scratch, mount, ids, self.base)
        limits = model.jnt_range[model.actuator_trnid[ids,0]]
        def feasible(offset):
            command[ids] = np.clip(self.base[ids]+offset, limits[:,0], limits[:,1])
            pos, mat = command_pose(model, self.scratch, mount, ids, command)
            return np.linalg.norm(pos-origin) <= .005 and Rotation.from_matrix(mat@orientation.T).magnitude() <= .03
        args = self.base[ids], limits, np.asarray(previous["effort_fraction"])[ids], self.dt
        inward = project(old, joint_step(old, -.4*old, *args), feasible)
        return project(inward, joint_step(old, rate, *args), feasible)

    def update_side(self, side, previous, state, wrist):
        raw = {"contacts":previous["contacts"],"support_targets":self.meta["support_targets"]}
        error, jac = self.landmark(side, raw, wrist["landmarks"][side])
        self.max_tangent_error = max(self.max_tangent_error, float(np.linalg.norm(error[:2])))
        loads, opposed = grasp_sample(raw, side)
        old = np.array(previous["pose_search"]["controller_state"]["arm_correction_rad"][side])
        rates = check_request(wrist["requests"][side], jac, error, old, loads[0], opposed, self.mode)
        actual = state["arm_correction_rad"][side]
        check(actual, self.projected(side, old, rates, previous), "projected servo correction", 3e-8)
        self.max_travel = max(self.max_travel, float(np.max(np.abs(actual))))
        self.updates[side] += 1

    def observe(self, previous, row):
        state = row["pose_search"]["controller_state"]
        self.applied(row, state)
        wrist = state["wrist_grip"]
        assert wrist["mode"] == self.mode and wrist["capture_s"] == state["settled_grip"]["capture_s"]
        captured = self.sides if wrist["capture_s"] is not None else ()
        active = captured if row["phase"]=="HOLD" and not state["inhibition"] else ()
        assert set(wrist["requests"]) == set(active) and set(wrist["landmarks"]) == set(captured)
        self.data.qpos[:] = [*previous["base_pose"], *previous["joint_positions"]]
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_comPos(self.model,self.data)
        for side in active:
            self.update_side(side, previous, state, wrist)
        assert self.updates == wrist["updates"]


def verify_run(path, report):
    checked = verify_grip(path, report)
    meta = read(path/"resolved.json")
    if "wrist_grip" not in meta["controller"]:
        return checked
    model = mujoco.MjModel.from_binary_path(str(path/"scene.mjb"))
    with (path/"steps.jsonl").open() as stream:
        rows = map(json.loads,stream)
        previous = next(rows)
        replay = WristReplay(model, meta, previous["applied_position_targets_rad"])
        for row in rows:
            replay.observe(previous, row)
            previous = row
    summary = read(path/"summary.json")["physics_audit"]["wrist_grip"]
    assert summary["mode"] == replay.mode and summary["updates"] == replay.updates
    return {**checked, "wrist_grip":{**summary,"maximum_tangential_error_m":replay.max_tangent_error,
                                      "maximum_active_joint_correction_rad":replay.max_travel}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_wrist_grip"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    verify(args.root, args.run_id, verify_function=verify_run)
