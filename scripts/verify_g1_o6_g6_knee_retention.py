"""Verify the settled hand commands against native preload, slew, delay and raw poses."""

import argparse
import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from scripts.verify_g1_o6_g6_knee import RUNG, verify_run as verify_knee
from scripts.verify_g1_o6_g6_whole_body import read, verify


def equal(first, second):
    return np.allclose(first, second, rtol=0., atol=1e-10)


def captured_ceiling(saved, previous, ceiling):
    if saved is None:
        if ceiling is not None:
            raise ValueError("captured ceiling disappeared")
        return None
    expected = previous if ceiling is None else ceiling
    if not equal(saved, expected):
        raise ValueError("grip ceiling changed or differs from the preceding applied command")
    return expected.copy()


def check_grip(state, previous, native, limited, ceiling, step, previous_limited=None, limits=None, feedback_cap=None):
    if not equal(state["uncapped_targets_rad"], native):
        raise ValueError("original native preload differs")
    ceiling = captured_ceiling(state["ceiling_rad"], previous, ceiling)
    effective = ceiling if feedback_cap is None else feedback_cap
    desired = native if effective is None else np.minimum(native, effective)
    if limits is not None:
        desired = np.clip(desired, limits[:, 0], limits[:, 1])
    start = previous if previous_limited is None else previous_limited
    if not equal(limited, start + np.clip(desired-start, -step, step)):
        raise ValueError("executed hand target differs from bounded native closure")
    return ceiling


def limited_targets(row):
    return np.concatenate([row["pose_search"]["hands"][s]["motion"]["rate_limited_target_rad"] for s in ("left","right")])


def eligible_grasp(previous, control):
    elapsed = previous["hold_time_s"]
    if elapsed is None or elapsed < 5. - 1e-9 or control["inhibition"]:
        return False
    hands = previous["pose_search"]["hands"]
    return all(hands[side]["quality"]["valid"] and hands[side]["quality"]["minimum_finger_normal_N"] >= .35
               for side in ("left", "right"))


def check_capture(previous, control, captured):
    saved = control["settled_grip"]["capture_s"]
    if captured is not None:
        if saved != captured:
            raise ValueError("grasp capture time changed")
        return captured
    eligible = eligible_grasp(previous, control)
    if saved is None:
        if eligible:
            raise ValueError("first eligible bilateral grasp was not captured")
        return None
    assert eligible and abs(saved - previous["hold_time_s"]) < 1e-9
    return saved


def check_retreat(previous, control):
    state = control["settled_grip"]
    if not control["inhibition"]:
        measured = previous["g6"]["body_clearance"]["right_knee"][RUNG]["signed_distance_m"]
        assert equal(state["gap_m"], measured)
        assert equal(state["retreat_m"], np.clip(.035 - measured, 0., .008))
    return state["retreat_m"]


class GripAudit:
    def __init__(self, model, meta):
        names = [n+"_servo" for s in ("left","right") for n in meta["model"]["hands"][s]["active"]]
        assert names == meta["controller"]["settled_grip"]["finger_actuator_order"]
        self.act = np.array([model.actuator(n).id for n in names])
        joints = model.actuator_trnid[self.act, 0]
        self.qpos, self.limits = model.jnt_qposadr[joints], model.jnt_range[joints]
        self.closed = np.tile(meta["grasp_pose"]["closed_pose"], 2)
        self.preload = meta["grasp_pose"]["finger_preload_Nm"] / model.actuator_gainprm[self.act, 0]
        self.step = meta["controller"]["grasp_acquisition"]["command_rate_rad_s"] * meta["config"]["control_dt"]
        self.capture, self.ceiling = None, None
        self.max_retreat, self.max_prevented = 0., 0.
        self.held, self.capped = 0, 0

    def feedback_cap(self, previous, control, native):
        return None

    def observe(self, previous, row):
        control = row["pose_search"]["controller_state"]
        original = np.asarray([*previous["base_pose"], *previous["joint_positions"]])[self.qpos]
        native = np.minimum(self.closed, original+self.preload)
        last = np.asarray(previous["applied_position_targets_rad"])[self.act]
        feedback = self.feedback_cap(previous, control, native)
        self.ceiling = check_grip(control["settled_grip"], last, native, limited_targets(row), self.ceiling,
                                  self.step, limited_targets(previous), self.limits, feedback)
        self.capture = check_capture(previous, control, self.capture)
        assert (self.capture is None) == (self.ceiling is None)
        self.max_retreat = max(self.max_retreat, check_retreat(previous, control))
        if self.ceiling is not None:
            self.held += 1
            excess = float(np.max(native - (self.ceiling if feedback is None else feedback)))
            self.capped += int(excess > 1e-10)
            self.max_prevented = max(self.max_prevented, excess)

    def result(self):
        return {"capture_s": self.capture, "captured_intervals": self.held, "closure_limited_intervals": self.capped,
                "max_prevented_closure_rad": self.max_prevented, "peak_knee_retreat_m": self.max_retreat,
                "ceiling_rad": None if self.ceiling is None else self.ceiling.tolist()}


def verify_grip(path, meta, audit_type=GripAudit):
    model = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    audit = audit_type(model, meta)
    with (path / "steps.jsonl").open() as stream:
        rows = map(json.loads, stream)
        previous = next(rows)
        delay = deque([limited_targets(previous)] * meta["config"]["delay_steps"])
        for row in rows:
            delay.append(limited_targets(row))
            assert equal(delay.popleft(), np.asarray(row["applied_position_targets_rad"])[audit.act])
            if row["phase"] == "HOLD":
                audit.observe(previous, row)
            previous = row
    return audit.result()


def verify_run(path, report):
    result, meta = verify_knee(path, report), read(path / "resolved.json")
    if "settled_grip" in meta["controller"]:
        result["settled_grip"] = verify_grip(path, meta)
        assert result["settled_grip"]["capture_s"] == read(path/"summary.json")["physics_audit"]["settled_grip"]["capture_s"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_knee_retention"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    verify(args.root, args.run_id, verify_function=verify_run)
